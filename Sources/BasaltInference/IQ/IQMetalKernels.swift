import BasaltCore
import Foundation
import MLX

package enum IQMetalKernels {
    package static let supportedTypes: [GGMLType] = [
        .iq1S, .iq1M, .iq2XXS, .iq2XS, .iq2S, .iq3XXS, .iq3S, .iq4NL, .iq4XS,
        .q2K, .q3K, .q4K,
    ]

    package static let primarySub4Types: [GGMLType] = [
        .q3K, .q4K, .iq3S, .iq3XXS, .iq4XS, .iq2XS,
    ]

    package static let smallMRows = 2 ... 8
    package static let supportedDTypes: Set<DType> = [.float32, .float16, .bfloat16]
    private static let simdgroups = IQMetalShaders.simdgroups
    private static let rowsPerSimdgroup = 4

    private struct Entry: Sendable {
        let matvec: MLXFast.MLXFastKernel
        let smallM: MLXFast.MLXFastKernel
    }

    private static let kernels: [GGMLType: Entry] = Dictionary(
        uniqueKeysWithValues: supportedTypes.map { type in
            let header = IQMetalShaders.header(for: type)
            let mv = MLXFast.metalKernel(
                name: "iq_matvec_\(type.name)",
                inputNames: ["x", "w"],
                outputNames: ["out"],
                source: IQMetalShaders.matvecSource(for: type),
                header: header
            )
            let sm = MLXFast.metalKernel(
                name: "iq_smallm_\(type.name)",
                inputNames: ["x", "w"],
                outputNames: ["out"],
                source: IQMetalShaders.smallMSource(for: type),
                header: header
            )
            return (type, Entry(matvec: mv, smallM: sm))
        }
    )

    package static func supports(_ type: GGMLType) -> Bool {
        kernels[type] != nil
    }

    private static func validGeometry(_ packed: MLXArray, columns: Int, dtype: DType, type: GGMLType) -> Bool {
        let (be, bb) = type.block
        return columns > 0
            && columns.isMultiple(of: be)
            && packed.ndim == 2
            && packed.dtype == .uint8
            && packed.dim(0) > 0
            && packed.dim(1) == (columns / be) * bb
            && supportedDTypes.contains(dtype)
    }

    private static func gridFits(_ outputs: Int) -> Bool {
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        return simdgroups * groups <= Int(Int32.max)
    }

    private static func cpuMatmul(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        let outputs = packed.dim(0)
        let columns = x.dim(1)
        let (be, bb) = type.block
        precondition(
            columns.isMultiple(of: be) && packed.dim(1) == (columns / be) * bb,
            "IQMetalKernels invalid geometry for \(type.name): columns=\(columns), packed=\(packed.shape)")
        let bytesArr = packed.dtype == .uint8 ? packed : packed.asType(.uint8)
        eval(bytesArr)
        let bytes = bytesArr.asArray(UInt8.self)
        let weights = MLXArray(GGUFDequant.floats(bytes.span.bytes, type: type, count: outputs * columns), [outputs, columns])
        return matmul(x.asType(.float32), weights.T).asType(x.dtype)
    }

    package static func matvec(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        precondition(x.ndim == 1 || x.ndim == 2, "IQMetalKernels.matvec requires 1D or 2D input, got ndim=\(x.ndim)")
        precondition(packed.ndim == 2, "IQMetalKernels.matvec requires 2D packed weights, got ndim=\(packed.ndim)")
        if (x.dtype == .float64 || packed.dtype == .float64) && Device.defaultDevice() == .gpu {
            return Device.withDefaultDevice(.cpu) { matvec(x, packed, type: type) }
        }
        let vector = x.ndim == 1 ? x.reshaped(1, x.dim(0)) : x
        precondition(vector.dim(0) == 1, "IQMetalKernels.matvec requires M == 1, got \(vector.dim(0))")
        let columns = vector.dim(1)
        let outputs = packed.dim(0)
        if outputs == 0 || columns == 0 {
            return x.ndim == 1 ? MLXArray.zeros([outputs], dtype: vector.dtype) : MLXArray.zeros([1, outputs], dtype: vector.dtype)
        }
        guard Device.defaultDevice() == .gpu, validGeometry(packed, columns: columns, dtype: vector.dtype, type: type), gridFits(outputs) else {
            let fallback = cpuMatmul(vector, packed, type: type)
            return x.ndim == 1 ? fallback.reshaped(outputs) : fallback
        }
        guard let entry = kernels[type] else {
            let legacy = GGUFKernels.legacyMatvec(vector, packed, type: type)
            return x.ndim == 1 ? legacy.reshaped(outputs) : legacy
        }
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        let template: [(String, any KernelTemplateArg)] = [
            ("K", columns),
            ("BE", type.block.elements),
            ("BB", type.block.bytes),
            ("T", vector.dtype),
        ]
        let out = entry.matvec(
            [vector, packed],
            template: template,
            grid: (32, simdgroups * groups, 1),
            threadGroup: (32, simdgroups, 1),
            outputShapes: [[1, outputs]],
            outputDTypes: [vector.dtype]
        )[0]
        return x.ndim == 1 ? out.reshaped(outputs) : out
    }

    package static func smallM(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        precondition(x.ndim == 2, "IQMetalKernels.smallM requires 2D input, got ndim=\(x.ndim)")
        precondition(packed.ndim == 2, "IQMetalKernels.smallM requires 2D packed weights, got ndim=\(packed.ndim)")
        if (x.dtype == .float64 || packed.dtype == .float64) && Device.defaultDevice() == .gpu {
            return Device.withDefaultDevice(.cpu) { smallM(x, packed, type: type) }
        }
        let rows = x.dim(0)
        if rows == 1 { return matvec(x, packed, type: type) }
        let columns = x.dim(1)
        let outputs = packed.dim(0)
        if rows == 0 || outputs == 0 || columns == 0 {
            return MLXArray.zeros([rows, outputs], dtype: x.dtype)
        }
        guard Device.defaultDevice() == .gpu, validGeometry(packed, columns: columns, dtype: x.dtype, type: type) else {
            return cpuMatmul(x, packed, type: type)
        }
        guard let entry = kernels[type], smallMRows.contains(rows), gridFits(outputs) else {
            return GGUFKernels.matmul(x, packed, type: type, columns: columns)
        }
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        let template: [(String, any KernelTemplateArg)] = [
            ("K", columns),
            ("BE", type.block.elements),
            ("BB", type.block.bytes),
            ("M", rows),
            ("T", x.dtype),
        ]
        return entry.smallM(
            [x, packed],
            template: template,
            grid: (32, simdgroups * groups, 1),
            threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows, outputs]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
