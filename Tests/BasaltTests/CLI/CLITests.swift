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

    @Test("--kv-bits takes auto, a bit width or k8v4, on serve and on models check")
    func kvBitsSetting() throws {
        #expect(try ServeCommand.parse([]).kvBits == nil)
        #expect(try ServeCommand.parse(["--kv-bits", "auto"]).kvBits == .auto)
        #expect(try ServeCommand.parse(["--kv-bits", "8"]).kvBits == .fixed(.q8))
        #expect(try ServeCommand.parse(["--kv-bits", "k8v4", "--kv-zones"]).kvBits == .fixed(.k8v4))
        #expect(try CheckSubcommand.parse(["m", "--kv-bits", "k8v4", "--kv-zones"]).kvBits == .fixed(.k8v4))
        #expect(try ServeCommand.parse(["--kv-bits", "k8v4h", "--kv-zones"]).kvBits == .fixed(.k8v4h))
        #expect(KVPrecision.k8v4h.rotatesValues && !KVPrecision.k8v4.rotatesValues)
        #expect(try CheckSubcommand.parse(["m", "--kv-bits", "auto"]).kvBits == .auto)
        for invalid in ["fp8", "k9v4", "k4v8", "k8v4H", "8h", "0", "7", "16"] {
            #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", invalid]) }
        }
        #expect(KVBitsSetting.auto.fixedPrecision == nil)
        #expect(KVBitsSetting.auto.budgetPrecision == .q4)
        #expect(KVBitsSetting.fixed(.q8).fixedPrecision == .q8)
        #expect(KVBitsSetting.fixed(.k8v4).budgetPrecision == .k8v4)
        #expect(KVPrecision.k8v4.uniform == .q8)
    }

    @Test("--kv-zones is off by default, parses on serve and models check, and needs --kv-bits; --kv-bits k8v4 needs --kv-zones")
    func kvZonesFlag() throws {
        #expect(try ServeCommand.parse([]).kvZones == false)
        #expect(try ServeCommand.parse(["--kv-bits", "4", "--kv-zones"]).kvZones)
        #expect(try ServeCommand.parse(["--kv-bits", "auto", "--kv-zones"]).kvZones)
        #expect(try CheckSubcommand.parse(["m", "--kv-bits", "8", "--kv-zones"]).kvZones)
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-zones"]) }
        #expect(throws: (any Error).self) { _ = try CheckSubcommand.parse(["m", "--kv-zones"]) }
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", "k8v4"]) }
        #expect(throws: (any Error).self) { _ = try CheckSubcommand.parse(["m", "--kv-bits", "k8v4"]) }
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", "k8v4h"]) }
        #expect(KVPrecision.accepted == "2, 3, 4, 5, 6, 8, k8v4, k8v4h")
        #expect(KVPrecision.flagConflict(kvBitsSet: false, fixed: nil, zones: true) != nil)
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: .k8v4h, zones: false) != nil)
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: .k8v4, zones: true) == nil)
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: nil, zones: true) == nil)
        #expect(KVPrecision.flagConflict(kvBitsSet: true, fixed: .q4, zones: false) == nil)
        #expect(KVBitsSetting.zones(true) == .standard)
        #expect(KVBitsSetting.zones(false) == nil)
    }

    @Test("--kv-evict takes a fraction of the zoned middle on serve and models check, needs --kv-zones and stays within 0...0.9")
    func kvEvictFlag() throws {
        #expect(try ServeCommand.parse([]).kvEvict == nil)
        #expect(try ServeCommand.parse(["--kv-bits", "k8v4", "--kv-zones", "--kv-evict", "0.5"]).kvEvict == 0.5)
        #expect(try CheckSubcommand.parse(["m", "--kv-bits", "8", "--kv-zones", "--kv-evict", "0.25"]).kvEvict == 0.25)
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", "8", "--kv-evict", "0.5"]) }
        #expect(throws: (any Error).self) { _ = try CheckSubcommand.parse(["m", "--kv-bits", "8", "--kv-evict", "0.5"]) }
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", "8", "--kv-zones", "--kv-evict", "0.95"]) }
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--kv-bits", "8", "--kv-zones", "--kv-evict", "-0.1"]) }
        #expect(KVBitsSetting.zones(true, evicting: 0.5) == KVZones.standard.evicting(0.5))
        #expect(KVBitsSetting.zones(true, evicting: 0.5)?.evictedMiddleFraction == 0.5)
        #expect(KVBitsSetting.zones(false, evicting: 0.5) == nil)
        #expect(KVBitsSetting.evictionSizing(0.25)?.hasPrefix("budget sized with 25% of each full-attention layer's quantized middle evicted") == true)
        #expect(KVBitsSetting.evictionSizing(0) == nil)
        #expect(KVBitsSetting.evictionSizing(nil) == nil)
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
