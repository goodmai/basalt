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
        private let requests: [Request]
        private var blocks: UnsafeMutableBufferPointer<aiocb>?

        init(_ requests: [Request]) {
            self.requests = requests
            guard !requests.isEmpty else { return }
            let buffer = UnsafeMutableBufferPointer<aiocb>.allocate(capacity: requests.count)
            for (index, request) in requests.enumerated() {
                var block = aiocb()
                (block.aio_fildes, block.aio_offset, block.aio_buf) = (request.descriptor, off_t(request.offset), request.destination)
                (block.aio_nbytes, block.aio_lio_opcode) = (request.count, LIO_READ)
                buffer.initializeElement(at: index, to: block)
            }
            for start in stride(from: 0, to: requests.count, by: batch) {
                let end = min(start + batch, requests.count)
                var list: [UnsafeMutablePointer<aiocb>?] = (start ..< end).map { buffer.baseAddress! + $0 }
                _ = lio_listio(LIO_NOWAIT, &list, Int32(list.count), nil)
            }
            blocks = buffer
        }

        mutating func drain() {
            guard let buffer = blocks else { return }
            blocks = nil
            defer {
                buffer.deinitialize()
                buffer.deallocate()
            }
            for (index, request) in requests.enumerated() {
                let count = ExpertReader.wait(buffer.baseAddress! + index)
                guard count < request.count else { continue }
                let rest = UnsafeMutableRawBufferPointer(start: request.destination + count, count: request.count - count)
                let read = ExpertReader.pread(request.descriptor, into: rest, at: request.offset + Int64(count))
                precondition(read == rest.count, "expert streaming: read \(count + read) of \(request.count) bytes at \(request.offset): \(String(cString: strerror(errno)))")
            }
        }

        consuming func finish() {
            drain()
        }

        deinit {
            guard let buffer = blocks else { return }
            for index in 0 ..< requests.count { _ = ExpertReader.wait(buffer.baseAddress! + index) }
            buffer.deinitialize()
            buffer.deallocate()
        }
    }

    static func wait(_ block: UnsafeMutablePointer<aiocb>) -> Int {
        while aio_error(block) == EINPROGRESS {
            var pending: [UnsafePointer<aiocb>?] = [UnsafePointer(block)]
            _ = aio_suspend(&pending, 1, nil)
        }
        return max(0, aio_return(block))
    }

    static func read(_ requests: [Request]) {
        for start in stride(from: 0, to: requests.count, by: window) {
            InFlight(Array(requests[start ..< min(start + window, requests.count)])).finish()
        }
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

struct SlotBuffer {
    static let page = 16_384

    let pointer: UnsafeMutableRawPointer
    let array: MLXArray
    let stride: Int

    init(slots: Int, tensor: ExpertTensor) throws(BasaltError) {
        let padded = Self.padded(slots * tensor.expertBytes)
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: padded, alignment: Self.page)
        let flat = MLXArray(rawPointer: pointer, [padded], dtype: .uint8) { pointer.deallocate() }
        let array = flat[0 ..< slots * tensor.expertBytes].view(dtype: tensor.dtype).reshaped([slots] + tensor.shape)
        eval(array)
        guard Self.address(of: flat) == UnsafeRawPointer(pointer), Self.address(of: array) == UnsafeRawPointer(pointer) else {
            throw .inferenceError("expert streaming: Metal did not wrap the expert slot buffer in place, so slots cannot be filled from disk")
        }
        (self.pointer, self.array, stride) = (pointer, array, tensor.expertBytes)
    }

    static func padded(_ bytes: Int) -> Int { max(1, (bytes + page - 1) / page) * page }

    static func address(of array: MLXArray) -> UnsafeRawPointer? {
        array.asData(access: .noCopy).data.withUnsafeBytes { $0.baseAddress }
    }

    func destination(slot: Int) -> UnsafeMutableRawPointer { pointer + slot * stride }
}
