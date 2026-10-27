import Testing
import Foundation
@testable import BasaltCore

@Suite("Session log: redaction and prefix de-duplication (#106, part 3)", .serialized)
struct SessionLogRedactionTests {
    static func privateKeyLine(_ kind: String, _ edge: String = "BEGIN") -> String {
        "-----\(edge) \(kind) " + "PRIV" + "ATE KEY-----"
    }

    static let pem = [
        privateKeyLine("RSA"),
        "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy0AHB7MbmP3Fzp0Qe0Xg",
        "Proc-Type: 4,ENCRYPTED",
        privateKeyLine("RSA", "END"),
    ].joined(separator: "\n")
    static let openAIKey = "sk-" + String(repeating: "a1B2", count: 10)

    @Test("Each secret class is replaced by its label", arguments: [
        ("key: \(pem) done", "[REDACTED:private_key]", "MIIEpAIBAAK"),
        ("use \(openAIKey) now", "[REDACTED:openai_key]", "a1B2a1B2"),
        ("shortest sk-" + String(repeating: "q", count: 20), "[REDACTED:openai_key]", "qqqqqqqqqq"),
        ("sk-ant-api03-" + String(repeating: "x", count: 30), "[REDACTED:anthropic_key]", "api03"),
        ("token ghp_" + String(repeating: "Z9", count: 18), "[REDACTED:github_token]", "Z9Z9"),
        ("github_pat_" + String(repeating: "A", count: 45), "[REDACTED:github_token]", "AAAA"),
        ("aws " + "AKIA" + "IOSFODNN7EXAMPLE x", "[REDACTED:aws_access_key]", "IOSFODNN"),
        ("slack " + "xox" + "b-1234567890-abcdefghij", "[REDACTED:slack_token]", "1234567890"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload", "[REDACTED:bearer]", "eyJhbGci"),
        ("authorization: bearer eyJhbGciOiJIUzI1NiJ9.lower", "[REDACTED:bearer]", "eyJhbGci"),
    ] as [(String, String, String)])
    func redactsEachClass(input: String, marker: String, leaked: String) {
        let out = SecretRedactor.redact(input)
        #expect(out.contains(marker), "\(out)")
        #expect(!out.contains(leaked), "\(out)")
    }

    @Test("Ordinary text, short sk- words and a public key block are left alone")
    func leavesOrdinaryTextAlone() {
        for text in ["skip-list and sk-short", "-----BEGIN PUBLIC KEY-----\nMFkw\n-----END PUBLIC KEY-----",
                     "bearer of bad news", "AKIA is a prefix", "a - b -- c"] {
            #expect(SecretRedactor.redact(text) == text, "\(text)")
        }
        #expect(SecretRedactor.redact("a \(Self.pem) b").hasPrefix("a [REDACTED:private_key]"))
        #expect(SecretRedactor.redact("a \(Self.pem) b").hasSuffix(" b"))
        let publicBlock = "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQ\n-----END PUBLIC KEY-----"
        let mixed = SecretRedactor.redact(publicBlock + "\n" + Self.pem)
        #expect(mixed.hasPrefix(publicBlock))
        #expect(mixed.hasSuffix("[REDACTED:private_key]"))
    }

    @Test("A PEM header without its END line redacts to the end instead of leaking the key")
    func unterminatedPEM() {
        let out = SecretRedactor.redact("x " + Self.privateKeyLine("EC") + "\nMHcCAQEEIBkg secret")
        #expect(out == "x [REDACTED:private_key]")
    }

    @Test("A 1 MB adversarial prompt is redacted in under 200 ms")
    func linearTime() {
        let chunks = [String(repeating: Self.privateKeyLine("RSA") + "\n", count: 8_000),
                      String(repeating: "sk-ab ", count: 60_000), String(repeating: "Bearer x ", count: 40_000)]
        for adversarial in chunks {
            let input = String(adversarial.prefix(1_000_000))
            let start = ContinuousClock.now
            _ = SecretRedactor.redact(input)
            #expect(ContinuousClock.now - start < .milliseconds(200), "\(input.prefix(20))")
        }
    }

    static func lines(_ dir: URL) throws -> [[String: Any]] {
        try SessionLogTests.lines(in: dir)
    }

    @Test("full mode writes redacted prompts and replies; --session-log-redact off keeps them verbatim")
    func writerRedacts() async throws {
        for redacts in [true, false] {
            let dir = SessionLogTests.tempDirectory()
            defer { try? FileManager.default.removeItem(at: dir) }
            let writer = SessionLogWriter(mode: .full, directory: dir, redacts: redacts, defaultSession: "t")
            let request = GenerationRequest(prompt: "flat", messages: [ChatMessageInput(role: .user, text: "my key \(Self.openAIKey)")])
            await writer.logRequest(request, id: "r1", origin: .unspecified, modelRequested: nil)
            var result = SessionLogResult()
            result.absorb(.text("echo \(Self.openAIKey)"))
            await writer.logResponse(request, id: "r1", status: "ok", result: result, queueWait: .zero)
            await writer.flush()
            let text = SessionLogTests.rawText(in: dir)
            #expect(text.contains(Self.openAIKey) == !redacts)
            #expect(text.contains("[REDACTED:openai_key]") == redacts)
        }
    }

    static func turns(_ texts: [String]) -> [ChatMessageInput] {
        texts.enumerated().map { ChatMessageInput(role: $0.offset.isMultiple(of: 2) ? .user : .assistant, text: $0.element) }
    }

    static func rebuild(_ id: String, from requests: [String: [String: Any]]) throws -> [String] {
        let record = try #require(requests[id])
        let content = try #require(record["content"] as? [String: Any])
        let own = (content["messages"] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }
        guard let parent = content["parent"] as? String else { return own }
        let prefix = try rebuild(parent, from: requests)
        #expect(prefix.count >= content["prefix_count"] as? Int ?? -1)
        return Array(prefix.prefix(content["prefix_count"] as? Int ?? 0)) + own
    }

    @Test("A request that extends an earlier one stores only its new tail, a parent and a prefix count; the chain rebuilds it exactly")
    func prefixDeduplication() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .full, directory: dir, defaultSession: "t")
        let first = Self.turns(["hello", "hi"])
        let second = Self.turns(["hello", "hi", "how are you", "fine"])
        let third = Self.turns(["hello", "hi", "how are you", "fine", "bye"])
        let fork = Self.turns(["hello", "something else"])
        for (id, messages) in [("r1", first), ("r2", second), ("r3", third), ("r4", fork)] {
            await writer.logRequest(GenerationRequest(prompt: "", messages: messages), id: id, origin: .unspecified, modelRequested: nil)
        }
        await writer.flush()
        let requests = Dictionary(uniqueKeysWithValues: try Self.lines(dir).filter { $0["type"] as? String == "request" }
            .map { ($0["id"] as! String, $0) })
        let r3 = try #require(requests["r3"]?["content"] as? [String: Any])
        #expect(r3["parent"] as? String == "r2")
        #expect(r3["prefix_count"] as? Int == 4)
        #expect((r3["messages"] as? [[String: Any]])?.count == 1)
        #expect(try Self.rebuild("r3", from: requests) == third.map(\.text))
        #expect(try Self.rebuild("r2", from: requests) == second.map(\.text))
        let r4 = try #require(requests["r4"]?["content"] as? [String: Any])
        #expect(r4["parent"] == nil)
        #expect((r4["messages"] as? [[String: Any]])?.count == 2)
        #expect(requests["r1"]?["messages_count"] as? Int == 2)
        await writer.logRequest(GenerationRequest(prompt: "", messages: third), id: "r5", origin: .unspecified, modelRequested: nil)
        await writer.flush()
        let repeated = try #require(try Self.lines(dir).first { $0["id"] as? String == "r5" }?["content"] as? [String: Any])
        #expect(repeated["parent"] as? String == "r3")
        #expect(repeated["prefix_count"] as? Int == third.count)
        #expect((repeated["messages"] as? [[String: Any]])?.isEmpty == true)
        #expect(requests["r3"]?["messages_count"] as? Int == 5)
    }

    @Test("Only the last 8 conversations of a session are compared, and sessions do not share parents")
    func prefixWindowAndSessions() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .full, directory: dir, defaultSession: "t")
        func inT(_ messages: [ChatMessageInput]) -> GenerationRequest {
            GenerationRequest(prompt: "", messages: messages).withRouting(sessionId: "t", budgetKey: nil)
        }
        await writer.logRequest(inT(Self.turns(["base"])), id: "base", origin: .unspecified, modelRequested: nil)
        for i in 0..<SessionLogWriter.prefixWindow {
            await writer.logRequest(inT(Self.turns(["other \(i)"])), id: "o\(i)", origin: .unspecified, modelRequested: nil)
        }
        await writer.logRequest(inT(Self.turns(["base", "more"])), id: "late", origin: .unspecified, modelRequested: nil)
        await writer.logRequest(GenerationRequest(prompt: "", messages: Self.turns(["other 7", "x"])).withRouting(sessionId: "s2", budgetKey: nil),
                                id: "elsewhere", origin: .unspecified, modelRequested: nil)
        await writer.flush()
        let requests = Dictionary(uniqueKeysWithValues: try Self.lines(dir).filter { $0["type"] as? String == "request" }
            .map { ($0["id"] as! String, $0) })
        #expect((requests["late"]?["content"] as? [String: Any])?["parent"] == nil)
        #expect((requests["elsewhere"]?["content"] as? [String: Any])?["parent"] == nil)
    }
}
