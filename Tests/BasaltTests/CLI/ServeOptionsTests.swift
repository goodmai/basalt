import Testing
import Foundation
import ArgumentParser
@testable import BasaltCore
@testable import BasaltCLI

private typealias CloudFlags = ServeCommand.CloudFallbackSettings.Flags

@Suite("Serve option groups resolve without a server")
struct ServeOptionsTests {
    func message(_ body: () throws -> Void) -> String? {
        do {
            try body()
            return nil
        } catch {
            return "\(error)"
        }
    }

    @Test("A config value of the wrong type names its key with the section prefix; JSON true is not the number 1")
    func configSectionTypes() throws {
        let section = ConfigSection(values: ["port": true, "limit": NSNull(), "name": "x"], keyPrefix: "cloud_fallback.")
        #expect(try section.value("limit", as: Int.self) == nil)
        #expect(try section.value("name", as: String.self) == "x")
        #expect(message { _ = try section.value("port", as: Int.self) }?.contains("\"cloud_fallback.port\" must be Int") == true)
        #expect(try section.value("port", as: Bool.self) == true)
    }

    @Test("Virtual keys need an id, a 16-character key, a non-negative budget and no duplicates")
    func virtualKeys() throws {
        let good: [String: Any] = ["id": "team", "key": String(repeating: "k", count: 16), "max_budget": 2.5]
        #expect(try NetworkSettings.virtualKey(good) == VirtualKey(key: String(repeating: "k", count: 16), id: "team", maxBudget: 2.5))
        #expect(throws: ValidationError.self) { try NetworkSettings.virtualKey(["id": "team", "key": "short"]) }
        #expect(throws: ValidationError.self) { try NetworkSettings.virtualKey(good.merging(["max_budget": true]) { $1 }) }
        #expect(throws: ValidationError.self) { try NetworkSettings.virtualKeys(in: ConfigSection(values: ["virtual_keys": [good, good]])) }
    }

    @Test("Rates are 0 or more, a blank CLI key counts as unset")
    func ratesAndBlankKey() {
        #expect(throws: Never.self) { try NetworkSettings.nonNegativeRate(0, key: "--rate-burst") }
        #expect(message { _ = try NetworkSettings.nonNegativeRate(-1, key: "rate_limit", unit: " of requests per minute") }
            == "rate_limit must be 0 (off) or a positive number of requests per minute, got -1")
        #expect(NetworkSettings.nonBlank(" \n") == nil)
        #expect(NetworkSettings.nonBlank("k") == "k")
    }

    @Test("Cloud base URL must be http(s) with a host", arguments: ["ftp://h/v1", "http:///v1", "not a url"])
    func badBaseURL(_ raw: String) {
        #expect(throws: ValidationError.self) { try ServeCommand.CloudFallbackSettings.endpoint(raw) }
    }

    @Test("The OpenRouter key comes from api_key, then the named variable, then the saved key; a self-hosted endpoint needs none")
    func cloudKeyOrder() throws {
        let openRouter = try #require(URL(string: "https://openrouter.ai/api/v1"))
        let selfHosted = try #require(URL(string: "http://127.0.0.1:9/v1"))
        let key = ServeCommand.CloudFallbackSettings.key
        #expect(try key(openRouter, ConfigSection(values: ["api_key": "a"]), ["OPENROUTER_API_KEY": "e"], { "s" }) == "a")
        #expect(try key(openRouter, ConfigSection(), ["OPENROUTER_API_KEY": "e"], { "s" }) == "e")
        #expect(try key(openRouter, ConfigSection(), [:], { "s" }) == "s")
        #expect(throws: ValidationError.self) { try key(openRouter, ConfigSection(), [:], { nil }) }
        #expect(try key(selfHosted, ConfigSection(), ["OPENROUTER_API_KEY": "e"], { "s" }) == "")
    }

    @Test("Cost routing needs router params, a probability threshold and non-negative stickiness and budget")
    func routing() throws {
                var settings = ServeCommand.CloudFallbackSettings(
            model: "m", baseURL: try #require(URL(string: "http://h/v1")), apiKey: "", localContextLimit: nil)
        #expect(throws: ValidationError.self) { try settings.resolveRouting(CloudFlags(costThreshold: 0.5), in: ConfigSection()) }
        #expect(throws: ValidationError.self) {
            try settings.resolveRouting(CloudFlags(routerParams: "p.json", costThreshold: 1.5), in: ConfigSection())
        }
        #expect(throws: ValidationError.self) {
            try settings.resolveRouting(CloudFlags(routerParams: "p.json"), in: ConfigSection(values: ["session_budget": -1.0]))
        }
        try settings.resolveRouting(
            CloudFlags(cacheStickiness: 0.2), in: ConfigSection(values: ["router_params": "~/p.json", "session_budget": 3.0]))
        #expect(settings.routerParams?.path.hasSuffix("/p.json") == true && settings.routerParams?.path.hasPrefix("~") == false)
        #expect(settings.cacheStickiness == 0.2 && settings.sessionBudget == 3)
    }

    @Test("CLI values win over the config file, which wins over the defaults")
    func networkPrecedence() throws {
        let file = ConfigSection(values: ["port": 9000, "host": "0.0.0.0", "api_key": "file", "allowed_origins": ["https://a.example"]])
        let settings = try NetworkSettings.resolve(NetworkSettings.Flags(port: 8081, apiKey: " ", allowedOrigins: ["http://b:1"]), file: file)
        #expect(settings == NetworkSettings(
            port: 8081, host: "0.0.0.0", apiKey: "file", rateLimit: nil, rateBurst: nil,
            allowedOrigins: ["http://b:1", "https://a.example"], virtualKeys: []))
        #expect(try NetworkSettings.resolve(NetworkSettings.Flags(), file: ConfigSection()).port == 8080)
    }

    @Test("A disabled or model-less cloud fallback resolves to nil before any key is needed")
    func cloudOff() throws {
        let off = ConfigSection(values: ["enabled": false, "model": "m"])
        #expect(try ServeCommand.CloudFallbackSettings.resolve(CloudFlags(), section: off, environment: [:], savedOpenRouterKey: { nil }) == nil)
        #expect(try ServeCommand.CloudFallbackSettings.resolve(
            CloudFlags(model: ""), section: ConfigSection(), environment: [:], savedOpenRouterKey: { nil }) == nil)
        #expect(throws: ValidationError.self) {
            try ServeCommand.CloudFallbackSettings.resolve(
                CloudFlags(model: "m", baseURL: "http://h/v1", localContextLimit: 0), section: ConfigSection(), environment: [:],
                savedOpenRouterKey: { nil })
        }
    }

    @Test("KV head split and Quest exclude quantized and zoned caches", arguments: [
        ["--kv-head-split", "0.01"], ["--kv-head-split", "0.5", "--kv-bits", "8"],
        ["--quest-top-k", "0"], ["--quest-top-k", "8", "--kv-head-split", "0.5"],
    ])
    func kvCombinations(_ arguments: [String]) throws {
        let options = try KVCacheOptions.parse(arguments)
        #expect(throws: ValidationError.self) { try options.validateCombination() }
    }

    @Test("KV zones carry the eviction fraction")
    func kvZones() throws {
        #expect(try KVCacheOptions.parse([]).zones == nil)
        #expect(try KVCacheOptions.parse(["--kv-bits", "4", "--kv-zones", "--kv-evict", "0.5"]).zones == KVZones.standard.evicting(0.5))
    }

    @Test("Session-log retention must not be negative")
    func retention() throws {
        #expect(throws: ValidationError.self) { try SessionLogOptions.parse(["--session-log-retention-days=-1"]).validateRetention() }
        #expect(throws: Never.self) { try SessionLogOptions.parse(["--session-log", "off"]).validateRetention() }
    }
}
