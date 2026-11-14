import Foundation
import MLX
import Synchronization
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltUI

@Suite("Zero-hatch subprocess I/O, MCP line stream lifecycle, and ~Copyable SlotBuffer ownership (#490)")
struct ZeroHatchAndSlotOwnershipTests {
    private static func makeLargePayload(lines: Int = 32_768) -> String {
        var chunks: [String] = []
        chunks.reserveCapacity(lines)
        for index in 0 ..< lines {
            chunks.append("row-\(index)-0123456789abcdef-XYZ\n")
        }
        return chunks.joined()
    }

    private static func fillSlot(_ buffer: borrowing SlotBuffer, slot: Int, values: [Float]) {
        let destination = buffer.destination(slot: slot)
        values.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            destination.copyMemory(from: base, byteCount: raw.count)
        }
    }

    private static func consumeArray(_ buffer: consuming SlotBuffer) -> MLXArray {
        buffer.array
    }

    @Test("ClipboardManager streams a payload larger than 512 KB through /bin/cat without raw threads or pipe deadlock", .timeLimit(.minutes(1)))
    func largeClipboardPayloadRoundTrip() async throws {
        let largePayload = Self.makeLargePayload()
        #expect(largePayload.utf8.count > 512 * 1024)
        let echoed = try await ClipboardManager().runClipboardCommand(["/bin/cat"], input: largePayload)
        #expect(echoed == largePayload)
    }

    @Test("ClipboardManager drains stderr and avoids SIGPIPE when a child exits early during a multi-hundred-KB stdin write", .timeLimit(.minutes(1)))
    func childExitingEarlySurfacesStderrWithoutDeadlock() async throws {
        let largePayload = Self.makeLargePayload()
        #expect(largePayload.utf8.count > 512 * 1024)
        do {
            _ = try await ClipboardManager().runClipboardCommand(
                ["/bin/sh", "-c", "printf 'early-child-failure' >&2; exit 7"],
                input: largePayload
            )
            Issue.record("Expected ClipboardError.operationFailed")
        } catch ClipboardError.operationFailed(let message) {
            #expect(message.contains("early-child-failure"))
        }
    }

    @Test("ClipboardManager terminates a blocked child process cleanly when the calling Task cancels", .timeLimit(.minutes(1)))
    func clipboardCommandCancelsBlockedChild() async throws {
        let fifoPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("basalt-clip-\(UUID().uuidString).fifo").path
        #expect(mkfifo(fifoPath, 0o600) == 0)
        defer { unlink(fifoPath) }
        let fd = open(fifoPath, O_RDWR)
        #expect(fd >= 0)
        let reader = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        let task = Task {
            try await ClipboardManager().runClipboardCommand(
                ["/bin/sh", "-c", "printf '1' > '\(fifoPath)'; exec /bin/sleep 30"]
            )
        }
        var bytes = reader.bytes.makeAsyncIterator()
        _ = try await bytes.next()
        task.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
    }

    @Test("MCPServer.lineStream dispatches JSON-RPC lines until pipe EOF and exits cleanly", .timeLimit(.minutes(1)))
    func mcpLineStreamFinishesOnEOF() async throws {
        let pipe = Pipe()
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
        let captured = Mutex<[String]>([])
        let server = MCPServer(
            orchestrator: orchestrator,
            modelId: "mock",
            input: MCPServer.lineStream(from: pipe.fileHandleForReading)
        ) { reply in
            captured.withLock { $0.append(reply) }
        }
        let wire = """
        {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}

        {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
        """
        try pipe.fileHandleForWriting.write(contentsOf: Data((wire + "\n").utf8))
        try pipe.fileHandleForWriting.close()
        await server.run()
        let replies = captured.withLock { $0 }
        #expect(replies.count == 2)
        #expect(replies[0].contains("\"id\":1"))
        #expect(replies[1].contains("\"id\":2"))
    }

    @Test("MCPServer input stream cancels its reader task and unblocks run() when the consumer task cancels on an open pipe", .timeLimit(.minutes(1)))
    func mcpLineStreamCancelsCleanlyOnTermination() async throws {
        let pipe = Pipe()
        defer { try? pipe.fileHandleForWriting.close() }
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
        let (replies, repliesContinuation) = AsyncStream<String>.makeStream()
        let count = Mutex(0)
        let server = MCPServer(
            orchestrator: orchestrator,
            modelId: "mock",
            input: MCPServer.lineStream(from: pipe.fileHandleForReading)
        ) { reply in
            count.withLock { $0 += 1 }
            repliesContinuation.yield(reply)
        }
        let running = Task { await server.run() }
        let firstLine = #"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"# + "\n"
        try pipe.fileHandleForWriting.write(contentsOf: Data(firstLine.utf8))
        var iterator = replies.makeAsyncIterator()
        let firstReply = try #require(await iterator.next())
        #expect(firstReply.contains("\"id\":1"))
        running.cancel()
        await running.value
        let secondLine = #"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"# + "\n"
        try pipe.fileHandleForWriting.write(contentsOf: Data(secondLine.utf8))
        try pipe.fileHandleForWriting.close()
        #expect(count.withLock { $0 } == 1)
    }

    @Test("Noncopyable SlotBuffer borrows slot pointers for in-place writes and transfers ownership cleanly into MLXArray and ModuleSlots")
    func slotBufferSingleOwnerBorrowAndConsume() throws {
        let tensor = ExpertTensor(file: 0, offset: 0, expertBytes: 16, shape: [4], dtype: .float32)
        let buffer = try SlotBuffer(slots: 2, tensor: tensor)
        #expect(buffer.slots == 2)
        #expect(buffer.stride == 16)
        #expect(buffer.byteCount == 32)
        #expect(SlotBuffer.padded(32) == SlotBuffer.page)
        Self.fillSlot(buffer, slot: 0, values: [1.5, 2.5, 3.5, 4.5])
        Self.fillSlot(buffer, slot: 1, values: [5.0, 6.0, 7.0, 8.0])
        let array = Self.consumeArray(buffer)
        eval(array)
        #expect(array.shape == [2, 4])
        #expect(array.asArray(Float.self) == [1.5, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0])
        let layout = ExpertLayout(path: "model.layers.0.mlp.experts.up_proj", numExperts: 4, weight: tensor)
        let moduleSlots = try ModuleSlots(layout: layout, slots: 2)
        #expect(moduleSlots.slotBytes == 32)
        #expect(moduleSlots.arrays != nil)
    }
}
