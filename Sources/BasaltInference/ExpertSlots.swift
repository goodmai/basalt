import BasaltCore
import Foundation
import MLX

enum ExpertReader {
    struct Request {
        let descriptor: Int32
        let offset: Int64
        let destination: UnsafeMutableRawPointer
        let count: Int
    }

    static let batch = 16
    static let window = 128

    struct InFlight: ~Copyable {
        private var blocks: UnsafeMutableBufferPointer<aiocb>?

        init(_ requests: Span<Request>) {
            guard !requests.isEmpty else { return }
            let buffer = UnsafeMutableBufferPointer<aiocb>.allocate(capacity: requests.count)
            for index in 0 ..< requests.count {
                let request = requests[index]
                var block = aiocb()
                (block.aio_fildes, block.aio_offset, block.aio_buf) = (request.descriptor, off_t(request.offset), request.destination)
                (block.aio_nbytes, block.aio_lio_opcode) = (request.count, LIO_READ)
                buffer.initializeElement(at: index, to: block)
            }
            for start in stride(from: 0, to: requests.count, by: batch) {
                let count = min(batch, requests.count - start)
                var list: [16 of UnsafeMutablePointer<aiocb>?] = .init(repeating: nil)
                for offset in 0 ..< count { list[offset] = buffer.baseAddress! + start + offset }
                var span = list.mutableSpan
                _ = span.withUnsafeMutableBufferPointer { lio_listio(LIO_NOWAIT, $0.baseAddress, Int32(count), nil) }
            }
            blocks = buffer
        }

        init(_ requests: [Request]) {
            self.init(requests.span)
        }

        mutating func drain() {
            guard let buffer = blocks else { return }
            blocks = nil
            defer {
                buffer.deinitialize()
                buffer.deallocate()
            }
            for index in 0 ..< buffer.count {
                let block = buffer[index]
                let count = ExpertReader.wait(buffer.baseAddress! + index)
                guard count < block.aio_nbytes, let destination = block.aio_buf else { continue }
                let rest = UnsafeMutableRawBufferPointer(start: destination + count, count: block.aio_nbytes - count)
                let read = ExpertReader.pread(block.aio_fildes, into: rest, at: Int64(block.aio_offset) + Int64(count))
                precondition(read == rest.count, "expert streaming: read \(count + read) of \(block.aio_nbytes) bytes at \(block.aio_offset): \(String(cString: strerror(errno)))")
            }
        }

        consuming func finish() {
            drain()
        }

        deinit {
            guard let buffer = blocks else { return }
            for index in 0 ..< buffer.count { _ = ExpertReader.wait(buffer.baseAddress! + index) }
            buffer.deinitialize()
            buffer.deallocate()
        }
    }

    static func wait(_ block: UnsafeMutablePointer<aiocb>) -> Int {
        while aio_error(block) == EINPROGRESS {
            var pending: [1 of UnsafePointer<aiocb>?] = [UnsafePointer(block)]
            var span = pending.mutableSpan
            _ = span.withUnsafeMutableBufferPointer { aio_suspend($0.baseAddress, 1, nil) }
        }
        return max(0, aio_return(block))
    }

    static func read(_ requests: Span<Request>) {
        for start in stride(from: 0, to: requests.count, by: window) {
            InFlight(requests.extracting(start ..< min(start + window, requests.count))).finish()
        }
    }

    static func read(_ requests: [Request]) {
        read(requests.span)
    }

    static func pread(_ descriptor: Int32, into buffer: UnsafeMutableRawBufferPointer, at offset: Int64) -> Int {
        var done = 0
        while done < buffer.count {
            let count = Darwin.pread(descriptor, buffer.baseAddress! + done, buffer.count - done, off_t(offset) + off_t(done))
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { return done }
            done += count
        }
        return done
    }
}

struct SlotBuffer: ~Copyable {
    static let page = 16_384

    let tensor: ExpertTensor
    let slots: Int
    private let pointer: UnsafeMutableRawPointer
    let array: MLXArray
    let stride: Int

    init(slots: Int, tensor: ExpertTensor) throws(BasaltError) {
        precondition(slots > 0, "SlotBuffer requires at least one slot")
        let padded = Self.padded(slots * tensor.expertBytes)
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: padded, alignment: Self.page)
        let flat = MLXArray(rawPointer: pointer, [padded], dtype: .uint8) { pointer.deallocate() }
        let array = flat[0 ..< slots * tensor.expertBytes].view(dtype: tensor.dtype).reshaped([slots] + tensor.shape)
        eval(array)
        guard Self.address(of: flat) == UnsafeRawPointer(pointer), Self.address(of: array) == UnsafeRawPointer(pointer) else {
            throw .inferenceError("expert streaming: Metal did not wrap the expert slot buffer in place, so slots cannot be filled from disk")
        }
        self.tensor = tensor
        self.slots = slots
        self.pointer = pointer
        self.array = array
        stride = tensor.expertBytes
    }

    var byteCount: Int { slots * stride }

    static func padded(_ bytes: Int) -> Int { max(1, (bytes + page - 1) / page) * page }

    static func address(of array: MLXArray) -> UnsafeRawPointer? {
        array.asData(access: .noCopy).data.withUnsafeBytes { $0.baseAddress }
    }

    borrowing func destination(slot: Int) -> UnsafeMutableRawPointer {
        precondition(slot >= 0 && slot < slots, "expert slot \(slot) out of bounds 0..<\(slots)")
        return pointer + slot * stride
    }
}
