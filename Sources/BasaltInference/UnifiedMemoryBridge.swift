import BasaltCore
import CoreAI
import Foundation
import MLX

package struct UnifiedPageBuffer<Scalar: HasDType & BitwiseCopyable>: ~Copyable {
    package static var pageSize: Int { 16_384 }

    package static func paddedBytes(_ byteCount: Int) -> Int {
        max(pageSize, (byteCount + pageSize - 1) / pageSize * pageSize)
    }

    package let shape: [Int]
    package let count: Int
    package let byteCount: Int
    package let allocatedByteCount: Int
    private let pointer: UnsafeMutableRawPointer
    private let storage: MLXArray

    package init(shape: [Int]) throws(BasaltError) {
        let count = shape.reduce(1, *)
        precondition(count > 0, "UnifiedPageBuffer requires at least one element")
        let byteCount = count * MemoryLayout<Scalar>.stride
        let allocated = Self.paddedBytes(byteCount)
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: allocated, alignment: Self.pageSize)
        let flat = MLXArray(rawPointer: pointer, [allocated], dtype: .uint8) { pointer.deallocate() }
        let storage = flat[0 ..< byteCount].view(dtype: Scalar.dtype).reshaped(shape)
        eval(storage)
        guard UnifiedMemoryBridge.baseAddress(of: flat) == UnsafeRawPointer(pointer),
              UnifiedMemoryBridge.baseAddress(of: storage) == UnsafeRawPointer(pointer)
        else {
            throw .inferenceError("unified memory bridge: Metal did not wrap page-aligned buffer in place")
        }
        self.shape = shape
        self.count = count
        self.byteCount = byteCount
        self.allocatedByteCount = allocated
        self.pointer = pointer
        self.storage = storage
    }

    package init(count: Int) throws(BasaltError) {
        try self.init(shape: [count])
    }

    package var rawBaseAddress: UnsafeRawPointer { UnsafeRawPointer(pointer) }

    package borrowing func withMutableSpan<R, E: Error>(
        _ body: (inout MutableSpan<Scalar>) throws(E) -> R
    ) throws(E) -> R {
        let typed = pointer.bindMemory(to: Scalar.self, capacity: count)
        let buffer = UnsafeMutableBufferPointer(start: typed, count: count)
        var span = buffer.mutableSpan
        return try body(&span)
    }

    package consuming func intoMLXArray(shape targetShape: [Int]? = nil) -> MLXArray {
        guard let targetShape, targetShape != shape else { return storage }
        precondition(targetShape.reduce(1, *) == count, "shape \(targetShape) does not match count \(count)")
        return storage.reshaped(targetShape)
    }
}

package enum UnifiedMemoryBridge {
    package static let pageSize = 16_384

    package static func baseAddress(of array: MLXArray) -> UnsafeRawPointer? {
        array.asData(access: .noCopy).data.withUnsafeBytes { $0.baseAddress }
    }

    package static func withContiguousRawSpan<R, E: Error>(
        of array: MLXArray,
        _ body: (RawSpan) throws(E) -> R
    ) throws(E) -> R {
        let prepared = contiguous(array)
        eval(prepared)
        let bytes = prepared.asData(access: .noCopyIfContiguous).data
        defer { withExtendedLifetime(prepared) {} }
        return try body(bytes.bytes)
    }

    package static func withContiguousSpan<Scalar: HasDType & BitwiseCopyable, R, E: Error>(
        of array: MLXArray,
        as _: Scalar.Type = Scalar.self,
        _ body: (Span<Scalar>) throws(E) -> R
    ) throws(E) -> R {
        let typed = array.dtype == Scalar.dtype ? array : array.asType(Scalar.dtype)
        let prepared = contiguous(typed)
        eval(prepared)
        let bytes = prepared.asData(access: .noCopyIfContiguous).data
        defer { withExtendedLifetime(prepared) {} }
        return try bytes.withUnsafeBytes { raw throws(E) -> R in
            let count = raw.count / MemoryLayout<Scalar>.stride
            guard let base = raw.baseAddress else {
                return try body(UnsafeBufferPointer<Scalar>(start: nil, count: 0).span)
            }
            let bound = base.assumingMemoryBound(to: Scalar.self)
            return try body(UnsafeBufferPointer(start: bound, count: count).span)
        }
    }

    package static func makeMLXArray<Scalar: HasDType & BitwiseCopyable>(
        shape: [Int],
        as _: Scalar.Type = Scalar.self,
        initializingWith populate: (inout MutableSpan<Scalar>) throws(BasaltError) -> Void
    ) throws(BasaltError) -> MLXArray {
        let count = shape.reduce(1, *)
        guard count > 0 else { return MLXArray.zeros(shape, dtype: Scalar.dtype) }
        let buffer = try UnifiedPageBuffer<Scalar>(shape: shape)
        try buffer.withMutableSpan { span throws(BasaltError) in
            try populate(&span)
        }
        return buffer.intoMLXArray()
    }

    package static func exportPlanarNDArray<Scalar: HasDType & BitwiseCopyable>(
        from array: MLXArray,
        shape: [Int],
        scalarType: NDArray.ScalarType,
        as _: Scalar.Type = Scalar.self
    ) -> NDArray {
        let count = shape.reduce(1, *)
        return withContiguousSpan(of: array, as: Scalar.self) { source in
            var output = NDArray(shape: shape, scalarType: scalarType)
            var copied = false
            do {
                if var target = output.mutableView(as: Scalar.self).contiguousElements,
                   target.count == count, source.count >= count
                {
                    copyScalars(from: source.extracting(first: count), into: &target)
                    copied = true
                }
            }
            if copied { return output }
            return source.extracting(first: min(count, source.count)).withUnsafeBufferPointer {
                NDArray(scalars: $0, shape: shape)
            }
        }
    }

    package static func importPlanarNDArray<Scalar: HasDType & BitwiseCopyable>(
        _ source: NDArray,
        rows: Int,
        columns: Int,
        as _: Scalar.Type = Scalar.self
    ) -> MLXArray {
        let count = rows * columns
        guard count > 0 else { return MLXArray.zeros([rows, columns], dtype: Scalar.dtype) }
        if let built = try? makeMLXArray(shape: [rows, columns], as: Scalar.self, initializingWith: { dst in
            fillPlanar(from: source, rows: rows, columns: columns, into: &dst)
        }) {
            return built
        }
        var fallback = [Scalar](repeating: viewZero(source), count: count)
        fallback.withUnsafeMutableBufferPointer { buf in
            var span = buf.mutableSpan
            fillPlanar(from: source, rows: rows, columns: columns, into: &span)
        }
        return MLXArray(fallback, [rows, columns])
    }

    private static func viewZero<Scalar: HasDType & BitwiseCopyable>(_ source: NDArray) -> Scalar {
        let coords: InlineArray<4, Int> = [0, 0, 0, 0]
        return source.view(as: Scalar.self)[scalarAt: coords]
    }

    private static func fillPlanar<Scalar: HasDType & BitwiseCopyable>(
        from source: NDArray,
        rows: Int,
        columns: Int,
        into dst: inout MutableSpan<Scalar>
    ) {
        let count = rows * columns
        let view = source.view(as: Scalar.self)
        if let elements = view.contiguousElements, elements.count >= count {
            copyScalars(from: elements.extracting(first: count), into: &dst)
        } else {
            for index in 0 ..< count {
                let coords: InlineArray<4, Int> = [0, index / columns, 0, index % columns]
                dst[index] = view[scalarAt: coords]
            }
        }
    }

    package static func copyScalars<Scalar: BitwiseCopyable>(
        from source: Span<Scalar>,
        into destination: inout MutableSpan<Scalar>
    ) {
        let count = min(source.count, destination.count)
        guard count > 0 else { return }
        destination.withUnsafeMutableBufferPointer { dst in
            source.withUnsafeBufferPointer { src in
                guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress else { return }
                UnsafeMutableRawPointer(dstBase).copyMemory(
                    from: srcBase, byteCount: count * MemoryLayout<Scalar>.stride)
            }
        }
    }
}
