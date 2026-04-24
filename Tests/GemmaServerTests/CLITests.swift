import Testing
import Foundation
import ArgumentParser
@testable import GemmaServerCore

@Suite("CLI Command Parsing")
struct CLITests {

    @Test("ServeCommand default values")
    func testServeDefaults() throws {
        let cmd = try ServeCommand.parse([])
        #expect(cmd.port == 8080)
        #expect(cmd.host == "127.0.0.1")
        #expect(cmd.maxTokens == 65536)
        #expect(cmd.mcp == false)
        #expect(cmd.rest == false)
    }

    @Test("ServeCommand custom flags")
    func testServeCustom() throws {
        let cmd = try ServeCommand.parse([
            "--port", "9090",
            "--host", "0.0.0.0",
            "--mcp",
            "--max-tokens", "1024"
        ])
        #expect(cmd.port == 9090)
        #expect(cmd.host == "0.0.0.0")
        #expect(cmd.mcp == true)
        #expect(cmd.maxTokens == 1024)
    }

    @Test("ServeCommand model options")
    func testServeModel() throws {
        let cmd = try ServeCommand.parse(["--model", "google/gemma-7b"])
        #expect(cmd.model == "google/gemma-7b")
        #expect(cmd.modelPath == nil)
        
        let cmd2 = try ServeCommand.parse(["--model-path", "/tmp/model"])
        #expect(cmd2.modelPath == "/tmp/model")
    }
}
