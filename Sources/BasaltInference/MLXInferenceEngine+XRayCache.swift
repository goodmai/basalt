import BasaltCore

extension MLXInferenceEngine {
    public func promptCacheReport(namespace: String?) async -> PromptCacheReport {
        var report = readPromptCaches { $0.report(namespace: namespace) }
        report.leases = leases.count
        report.launches = launches.sorted { $0.key < $1.key }.map { $0.value.report }
        return report
    }
}

extension MLXInferenceEngine.GenerationLaunch {
    var report: LaunchCacheReport {
        LaunchCacheReport(
            reusedTokens: reusedTokens.rawValue,
            layers: cache?.layers.map { String(describing: type(of: $0)) } ?? [])
    }
}
