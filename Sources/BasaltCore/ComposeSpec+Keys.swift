import Foundation

extension ComposeSpec {
    public func missingKeys(_ environment: [String: String]) -> [String] {
        models.sorted { $0.key < $1.key }.compactMap { name, model in
            guard model.isCloud, let variable = model.apiKeyEnv, environment[variable]?.isEmpty != false else { return nil }
            return "\(name) reads its key from $\(variable), which is not set"
        }
    }
}
