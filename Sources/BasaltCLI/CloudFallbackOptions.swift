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
        help: "OpenAI-compatible endpoint for --cloud-fallback, e.g. http://192.168.1.20:8000/v1 (default: https://openrouter.ai/api/v1; also cloud_fallback.base_url)"
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
            guard let model = try flags.model ?? section.value("model", as: String.self), !model.isEmpty else { return nil }
            let baseURL = try endpoint(flags.baseURL ?? section.value("base_url", as: String.self) ?? "https://openrouter.ai/api/v1")
            let apiKey = try key(for: baseURL, in: section, environment: environment, savedOpenRouterKey: savedOpenRouterKey)
            let limit = try flags.localContextLimit ?? section.value("local_context_limit", as: Int.self)
            if let limit, limit <= 0 {
                throw ValidationError("local_context_limit must be a positive number of tokens, got \(limit)")
            }
            var settings = CloudFallbackSettings(model: model, baseURL: baseURL, apiKey: apiKey, localContextLimit: limit)
            try settings.resolveRouting(flags, in: section)
            return settings
        }

        static func endpoint(_ raw: String) throws -> URL {
            guard let url = URL(string: raw), ["http", "https"].contains(url.scheme?.lowercased()), url.host != nil else {
                throw ValidationError("cloud fallback base URL must be http(s)://host[:port]/path, got \(raw)")
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
        costURL: URL = CloudCommand.costURL()
    ) async throws -> ModelRouter {
        let saved = CloudCommand.loadConfig()
        let budget = CostTracker.Budget(dailyLimit: saved?.dailyBudget ?? 5, monthlyLimit: saved?.monthlyBudget ?? 50)
        let tracker = try CostTracker(storageURL: costURL, budget: budget)
        await tracker.setDefaultSessionBudget(settings.sessionBudget)
        for key in virtualKeys {
            if let limit = key.maxBudget { await tracker.setMaxBudget(limit, for: .virtualKey(key.id)) }
        }
        let costRouting: ModelRouter.CostRouting?
        if let params = settings.routerParams {
            let scorer: BradleyTerryRouter
            do {
                scorer = try BradleyTerryRouter.load(from: params)
            } catch {
                throw ValidationError("router_params \(params.path): \(error.localizedDescription)")
            }
            costRouting = ModelRouter.CostRouting(
                policy: StickyRoutingPolicy(
                    strongModel: settings.model, weakModel: "local",
                    threshold: settings.costThreshold ?? 0.5, cacheStickinessPenalty: settings.cacheStickiness ?? 0.1),
                scorer: scorer, outcomeBiasURL: CloudCommand.routerBiasURL(nextTo: costURL))
        } else {
            costRouting = nil
        }
        return ModelRouter(
            localOrchestrator: orchestrator,
            cloudClient: try OpenRouterClient(config: .init(
                apiKey: settings.apiKey, baseURL: settings.baseURL, timeout: 120, maxRetries: 2)),
            options: .init(strategy: .cloudOnly, costRouting: costRouting, costTracker: tracker, fallbackCloudModel: settings.model))
    }

    func attachCloudFallback(to orchestrator: ModelOrchestratorActor, virtualKeys: [VirtualKey]) async throws {
        guard let fallback = try cloud.settings(configFile: network.configPath) else { return }
        let router = try await Self.cloudFallbackRouter(fallback, orchestrator: orchestrator, virtualKeys: virtualKeys)
        if OpenRouterClient.isOpenRouter(fallback.baseURL) {
            do {
                try await router.syncCloudModels()
            } catch {
                log("\(yellow("[warn]")) Could not fetch OpenRouter prices: \(error.localizedDescription)")
            }
        }
        if await !router.knowsPrice(of: fallback.model) {
            log("\(yellow("[warn]")) No price known for \(fallback.model): its cloud spend is counted as $0 and the budget does not limit it")
        }
        await orchestrator.setCloudFallback(router, arbiter: HybridArbiter(localContextLimit: fallback.localContextLimit))
        log("Cloud fallback → \(bold(fallback.model)) at \(fallback.baseURL.absoluteString)")
    }
}
