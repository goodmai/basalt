import Testing
import Foundation
import NIOCore
@testable import BasaltServer
@testable import BasaltCLI

@Suite("Experimental zero-copy request body (#345)")
struct ZeroCopyBodyTests {
    @Test("Both modes give the readable bytes only, also after the reader index moved", arguments: [false, true])
    func readableBytes(zeroCopy: Bool) {
        var buffer = ByteBuffer()
        buffer.writeString("skip{\"a\":1}")
        buffer.moveReaderIndex(forwardBy: 4)
        #expect(ChatPipeline.data(of: buffer, zeroCopy: zeroCopy) == Data("{\"a\":1}".utf8))
    }

    @Test("Off unless --experimental-zero-copy-body is given")
    func flag() throws {
        #expect(try ServeCommand.parse([]).experimentalZeroCopyBody == false)
        #expect(try ServeCommand.parse(["--experimental-zero-copy-body"]).experimentalZeroCopyBody)
    }
}
