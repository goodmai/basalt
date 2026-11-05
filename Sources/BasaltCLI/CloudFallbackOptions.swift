import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltCloud

struct CloudFallbackOptions: ParsableArguments {
    @Option(
        name: .customLong("cloud-fallback"),
        help: "Model on an OpenAI-compatible server that answers when the local model cannot: the request does not fit in free memory, the prompt exceeds --local-context-limit, or the task rates beyond the local model (also cloud_fallback.model in the config; spend is capped by the `basalt cloud` daily/monthly budgets)"
    )
    var model: String?

    @Option(
        name: .customLong("cloud-base-url"),
        help: "OpenAI-compatible endpoint for --cloud-fallback, e.g. http://192.168.1.20:8000/v1 (default: https://openrouter.ai/api/v1; also cloud_fallback.base_url). cloud_fallback.tiers lists several endpoints [{base_url, model, api_key?, api_key_env?, free?}] tried in order on connection errors and 5xx, a free one before the paid cloud; this flag or --cloud-fallback replaces them with one"
    )
    var baseURL: String?

    @Flag(name: .customLong("no-cloud-fallback"), help: "Never switch to the cloud fallback, even if the config enables it")
    var disabled = false

    @Option(
        name: .customLong("router-params"),
        help: "JSON with trained Bradley–Terry router parameters (strongEmbedding, weakEmbedding, weights, projection); enables the capability reason of the cloud fallback (also cloud_fallback.router_params)"
    )
    var routerParams: String?

    @Option(
        name: .customLong("cost-threshold"),
        help: "Win probability θ of the cloud model above which a task counts as beyond the local model (default 0.5; also cloud_fallback.cost_threshold)"
    )
    var costThreshold: Double?

    @Option(
        name: .customLong("cache-stickiness"),
        help: "Penalty λ for leaving a warm session's model, scaled by its cached prefix (default 0.1; also cloud_fallback.cache_stickiness); sessions are told apart by the x-session-id header"
    )
    var cacheStickiness: Double?

    @Option(
        name: .customLong("local-context-limit"),
        help: "Prompt tokens beyond which the local model counts as degraded and the cloud fallback answers instead"
    )
    var localContextLimit: Int?

    func settings(configFile: String?) throws -> ServeCommand.CloudFallbackSettings? {
        try ServeCommand.resolveCloudFallback(
            ServeCommand.CloudFallbackSettings.Flags(
                model: model, baseURL: baseURL, localContextLimit: localContextLimit,
                routerParams: routerParams, costThreshold: costThreshold, cacheStickiness: cacheStickiness),
            disabled: disabled, configFile: configFile)
    }
}

extension ServeCommand {
    struct CloudFallbackSettings: Sendable, Equatable {
        let model: String
        let baseURL: URL
        let apiKey: String
        let localContextLimit: Int?
        var routerParams: URL? = nil
        var costThreshold: Double? = nil
        var cacheStickiness: Double? = nil
        var sessionBudget: Double? = nil
        var tiers: [Tier] = []

        struct Flags: Sendable, Equatable {
            var model: String?
            var baseURL: String?
            var localContextLimit: Int?
            var routerParams: String?
            var costThreshold: Double?
            var cacheStickiness: Double?
        }

        static func resolve(
            _ flags: Flags, section: ConfigSection, environment: [String: String], savedOpenRouterKey: () -> String?
        ) throws -> CloudFallbackSettings? {
            if flags.model == nil, try section.value("enabled", as: Bool.self) == false { return nil }
            let tiers = flags.model == nil && flags.baseURL == nil
                ? try Tier.list(in: section, environment: environment, savedOpenRouterKey: savedOpenRouterKey) : []
            guard let first = try tiers.first ?? primary(flags, in: section, environment: environment, savedOpenRouterKey: savedOpenRouterKey)
            else { return nil }
            let limit = try flags.localContextLimit ?? section.value("local_context_limit", as: Int.self)
            if let limit, limit <= 0 {
                throw ValidationError("local_context_limit must be a positive number of tokens, got \(limit)")
            }
            var settings = CloudFallbackSettings(model: first.model, baseURL: first.baseURL, apiKey: first.apiKey, localContextLimit: limit)
            settings.tiers = tiers
            try settings.resolveRouting(flags, in: section)
            return settings
        }

        static func primary(
            _ flags: Flags, in section: ConfigSection, environment: [String: String], savedOpenRouterKey: () -> String?
        ) throws -> Tier? {
            guard let model = try flags.model ?? section.value("model", as: String.self), !model.isEmpty else { return nil }
            let baseURL = try endpoint(flags.baseURL ?? section.value("base_url", as: String.self) ?? "https://openrouter.ai/api/v1")
            let apiKey = try key(for: baseURL, in: section, environment: environment, savedOpenRouterKey: savedOpenRouterKey)
            return Tier(model: model, baseURL: baseURL, apiKey: apiKey)
        }

        static func endpoint(_ raw: String) throws -> URL {
            guard let url = URL(string: raw), ["http", "https"].contains(url.scheme?.lowercased()), url.host != nil else {
                throw ValidationError("cloud fallback base URL must be http(s)://host[:port]/path, got \(SecretRedactor.redact(raw))")
            }
            guard url.user == nil, url.password == nil else {
                throw ValidationError("cloud fallback base URL must not carry credentials; use api_key or api_key_env")
            }
            return url
        }

        static func key(
            for baseURL: URL, in section: ConfigSection, environment: [String: String], savedOpenRouterKey: () -> String?
        ) throws -> String {
            let openRouter = OpenRouterClient.isOpenRouter(baseURL)
            let keyVariable = try section.value("api_key_env", as: String.self) ?? (openRouter ? "OPENROUTER_API_KEY" : nil)
            let apiKey = try section.value("api_key", as: String.self)
                ?? keyVariable.flatMap { environment[$0] }
                ?? (openRouter ? savedOpenRouterKey() : nil)
                ?? ""
            if openRouter, apiKey.isEmpty {
                throw ValidationError("cloud fallback on OpenRouter needs OPENROUTER_API_KEY, cloud_fallback.api_key or `basalt cloud configure`")
            }
            guard apiKey.unicodeScalars.allSatisfy({ (0x21...0x7E).contains($0.value) }) else {
                throw ValidationError("\(section.keyPrefix)api_key must be printable ASCII without spaces or line breaks")
            }
            return apiKey
        }

        mutating func resolveRouting(_ flags: Flags, in section: ConfigSection) throws {
            routerParams = try (flags.routerParams ?? section.value("router_params", as: String.self)).map(ConfigSection.fileURL)
            costThreshold = try flags.costThreshold ?? section.value("cost_threshold", as: Double.self)
            cacheStickiness = try flags.cacheStickiness ?? section.value("cache_stickiness", as: Double.self)
            if routerParams == nil, costThreshold != nil || cacheStickiness != nil {
                throw ValidationError("cost_threshold and cache_stickiness need router_params (trained Bradley–Terry parameters)")
            }
            if let costThreshold, !(0...1).contains(costThreshold) {
                throw ValidationError("cost_threshold must be a probability between 0 and 1, got \(costThreshold)")
            }
            if let cacheStickiness, cacheStickiness < 0 {
                throw ValidationError("cache_stickiness must not be negative, got \(cacheStickiness)")
            }
            sessionBudget = try section.value("session_budget", as: Double.self)
            if let sessionBudget, sessionBudget < 0 {
                throw ValidationError("session_budget must not be negative, got \(sessionBudget)")
            }
        }
    }

    static func cloudFallbackRouter(
        _ settings: CloudFallbackSettings, orchestrator: ModelOrchestratorActor, virtualKeys: [VirtualKey] = [],
        costURL: URL = CloudCommand.costURL(), session: URLSessionConfiguration = .default
    ) async throws -> FallbackTiers {
        let saved = CloudCommand.loadConfig()
        let budget = CostTracker.Budget(dailyLimit: saved?.dailyBudget ?? 5, monthlyLimit: saved?.monthlyBudget ?? 50)
        let tracker = try CostTracker(storageURL: costURL, budget: budget)
        await tracker.setDefaultSessionBudget(settings.sessionBudget)
        for key in virtualKeys {
            if let limit = key.maxBudget { await tracker.setMaxBudget(limit, for: .virtualKey(key.id)) }
        }
        let costRouting = try Self.costRouting(settings, costURL: costURL)
        let endpoints = settings.endpoints
        return FallbackTiers(try endpoints.enumerated().map { index, tier in
            ModelRouter(
                localOrchestrator: orchestrator,
                cloudClient: try OpenRouterClient(
                    config: .init(apiKey: tier.apiKey, baseURL: tier.baseURL, timeout: 120, maxRetries: index == endpoints.count - 1 ? 2 : 1),
                    configuration: session),
                options: .init(
                    strategy: .cloudOnly, costRouting: index == 0 ? costRouting : nil,
                    costTracker: tier.free ? nil : tracker, fallbackCloudModel: tier.model))
        })
    }

    static func costRouting(_ settings: CloudFallbackSettings, costURL: URL) throws -> ModelRouter.CostRouting? {
        guard let params = settings.routerParams else { return nil }
        let scorer: BradleyTerryRouter
        do {
            scorer = try BradleyTerryRouter.load(from: params)
        } catch {
            throw ValidationError("router_params \(params.path): \(error.localizedDescription)")
        }
        return ModelRouter.CostRouting(
            policy: StickyRoutingPolicy(
                strongModel: settings.model, weakModel: "local",
                threshold: settings.costThreshold ?? 0.5, cacheStickinessPenalty: settings.cacheStickiness ?? 0.1),
            scorer: scorer, outcomeBiasURL: CloudCommand.routerBiasURL(nextTo: costURL))
    }

    func attachCloudFallback(to orchestrator: ModelOrchestratorActor, virtualKeys: [VirtualKey]) async throws {
        guard let fallback = try cloud.settings(configFile: network.configPath) else { return }
        let tiers = try await Self.cloudFallbackRouter(fallback, orchestrator: orchestrator, virtualKeys: virtualKeys)
        for (tier, router) in zip(fallback.endpoints, tiers.tiers) {
            if OpenRouterClient.isOpenRouter(tier.baseURL) {
                do {
                    try await router.syncCloudModels()
                } catch {
                    log("\(yellow("[warn]")) Could not fetch OpenRouter prices: \(error.localizedDescription)")
                }
            }
            if !tier.free, await !router.knowsPrice(of: tier.model) {
                log("\(yellow("[warn]")) No price known for \(tier.model): its cloud spend is counted as $0 and the budget does not limit it")
            }
            log("Cloud fallback → \(bold(tier.model)) at \(tier.baseURL.absoluteString)\(tier.free ? " (free)" : "")")
        }
        await orchestrator.setCloudFallback(tiers, arbiter: HybridArbiter(localContextLimit: fallback.localContextLimit))
    }
}

extension ServeCommand.CloudFallbackSettings {
    struct Tier: Sendable, Equatable {
        let model: String
        let baseURL: URL
        let apiKey: String
        var free = false
    }

    var endpoints: [Tier] { tiers.isEmpty ? [Tier(model: model, baseURL: baseURL, apiKey: apiKey)] : tiers }
}

extension ServeCommand.CloudFallbackSettings.Tier {
    static func list(
        in section: ConfigSection, environment: [String: String], savedOpenRouterKey: () -> String?
    ) throws -> [Self] {
        guard let entries = try section.value("tiers", as: [Any].self) else { return [] }
        guard !entries.isEmpty else { throw ValidationError("\(section.keyPrefix)tiers must list at least one endpoint") }
        return try entries.enumerated().map { index, entry in
            let prefix = "\(section.keyPrefix)tiers[\(index)]."
            guard let values = entry as? [String: Any] else { throw ValidationError("\(prefix.dropLast()) must be an object") }
            let tier = ConfigSection(values: values, keyPrefix: prefix)
            guard let model = try tier.value("model", as: String.self), !model.isEmpty else {
                throw ValidationError("\(prefix)model is required")
            }
            guard let rawURL = try tier.value("base_url", as: String.self) else { throw ValidationError("\(prefix)base_url is required") }
            let baseURL = try ServeCommand.CloudFallbackSettings.endpoint(rawURL)
            let apiKey = try ServeCommand.CloudFallbackSettings.key(
                for: baseURL, in: tier, environment: environment, savedOpenRouterKey: savedOpenRouterKey)
            return Self(model: model, baseURL: baseURL, apiKey: apiKey, free: try tier.value("free", as: Bool.self) ?? false)
        }
    }
}
