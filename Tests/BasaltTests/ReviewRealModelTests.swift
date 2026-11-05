import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("basalt review on a real model (#35)", .serialized)
struct ReviewRealModelTests {
    static var model: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", let entry = env["BASALT_E2E_REVIEW_MODEL"] else { return nil }
        return FileManager.default.fileExists(atPath: entry) ? entry : ModelCache.resolve(repoId: entry)?.path
    }

    static let plantedBypass = """
        --- a/Sources/Auth/Authenticator.swift
        +++ b/Sources/Auth/Authenticator.swift
        @@ -18,9 +18,7 @@ public struct Authenticator {
             public func verify(user: String, password: String) async throws -> Bool {
        -        guard let stored = try await store.hash(for: user) else { return false }
        -        return BCrypt.matches(password, stored)
        +        return true
             }
        """

    @Test("A diff that makes verify(user:password:) always succeed is flagged, naming the symbol, fully offline",
          .enabled(if: model != nil, "set BASALT_E2E=1 and BASALT_E2E_REVIEW_MODEL"))
    func flagsAuthBypass() async throws {
        let path = try #require(Self.model)
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine())
        try await orchestrator.loadModel(path: path)
        let prompt = ReviewContext.prompt(diff: Self.plantedBypass, budgetChars: 48_000)
        let report = try await ReviewCommand.review(prompt: prompt, orchestrator: orchestrator, maxTokens: 512)
        await orchestrator.shutdown()
        print("[#35] \(path):\n\(report)")
        #expect(report.contains("verify"))
        #expect(!report.contains("No issues found"))
    }
}
