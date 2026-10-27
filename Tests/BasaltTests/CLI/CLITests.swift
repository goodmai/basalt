import Testing
import Foundation
import ArgumentParser
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("CLI Command Parsing")
struct CLITests {

    @Test("ServeCommand default values")
    func testServeDefaults() throws {
        let cmd = try ServeCommand.parse([])
        // Unset on the command line; the defaults are applied by resolveNetwork.
        #expect(cmd.port == nil)
        #expect(cmd.host == nil)
        let network = try ServeCommand.resolveNetwork(cliPort: cmd.port, cliHost: cmd.host, cliApiKey: cmd.apiKey, configFile: nil)
        #expect(network.port == 8080)
        #expect(network.host == "127.0.0.1")
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

    @Test("--kv-bits takes auto or a bit width, on serve and on models check")
    func kvBitsSetting() throws {
        #expect(try ServeCommand.parse([]).kvBits == nil)
        #expect(try ServeCommand.parse(["--kv-bits", "auto"]).kvBits == .auto)
        #expect(try ServeCommand.parse(["--kv-bits", "8"]).kvBits == .bits(8))
        #expect(try CheckSubcommand.parse(["m", "--kv-bits", "auto"]).kvBits == .auto)
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", "fp8"]) }
        #expect(KVBitsSetting.auto.fixedBits == nil)
        #expect(KVBitsSetting.auto.budgetBits == 4)
        #expect(KVBitsSetting.bits(8).fixedBits == 8)
        #expect(KVBitsSetting.bits(8).budgetBits == 8)
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
