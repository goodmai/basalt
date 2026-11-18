import Foundation
import MLX
import Testing
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Retrospective audit Wave 2 fixes across Core, Inference, Cloud, Server, CLI, and App")
struct RetroWave2Tests {
    @Test func audioClipRejectsNonFiniteSamplesAndExtremeDuration() {
        #expect(throws: BasaltError.self) {
            _ = try AudioClip(samples: [.nan])
        }
        #expect(throws: BasaltError.self) {
            _ = try AudioClip(samples: [.infinity])
        }
        let err = AudioClip.tooLong(.infinity)
        #expect(err.localizedDescription.contains("1801 s"))
    }

    @Test func kvCacheGeometryUsesZonesWithEvictionUnderPrecision() {
        let zonesOnly = KVZones.standard
        let evicting = KVZones.standard.evicting(0.5)
        let geo = KVCacheGeometry(parameterProxy: 7_000_000_000)
        let bytesNoEvict = geo.bytes(tokens: TokenCount(2048), kvPrecision: .q8, zones: zonesOnly)
        let bytesEvict = geo.bytes(tokens: TokenCount(2048), kvPrecision: .q8, zones: evicting)
        #expect(bytesEvict < bytesNoEvict)
        #expect(FreshCache.plan(kvBitsSet: false, headMask: nil, questTopK: nil, zones: evicting.keepingMiddleFullPrecision(), precision: nil) == .stock)
        #expect(FreshCache.plan(kvBitsSet: true, headMask: nil, questTopK: nil, zones: evicting.keepingMiddleFullPrecision(), precision: .q8) == .evicting(evicting.keepingMiddleFullPrecision()))
    }

    @Test func sessionLogRequiresAssistantTurnForSingleMessagePrefix() {
        var conv = SessionConversations()
        let clientA = SessionConversations.digests([ChatMessageInput(role: .system, text: "You are helpful.")])
        conv.remember(clientA, id: "r1", session: "s1")
        let clientB = SessionConversations.digests([
            ChatMessageInput(role: .system, text: "You are helpful."),
            ChatMessageInput(role: .user, text: "Secret prompt from client B"),
        ])
        let matched = conv.session(continuing: clientB, hasAssistantTurn: false)
        #expect(matched == nil)
        let clientA2 = SessionConversations.digests([
            ChatMessageInput(role: .system, text: "You are helpful."),
            ChatMessageInput(role: .assistant, text: "Ready"),
            ChatMessageInput(role: .user, text: "Next"),
        ])
        #expect(conv.session(continuing: clientA2, hasAssistantTurn: true) == "s1")
    }

    @Test func generationPlanEvictingCacheTakesPriorityAndBlocksBatching() {
        var settings = GenerationPlan.Settings()
        settings.evictsThroughUpdate = true
        settings.kvZones = KVZones.standard.evicting(0.5)
        settings.kvPrecision = .q8
        let request = GenerationRequest(prompt: "hi")
        let namespace = GenerationPlan.cacheNamespace(request, settings, cacheEnabled: true, media: .text, sameEpoch: true)
        #expect(namespace == nil)
        let plan = GenerationPlan.make(request, settings, GenerationPlan.Facts())
        #expect(plan.kvZones?.middleStaysFullPrecision == true)
        #expect(plan.kvZones?.evictedMiddleFraction == 0.5)
        #expect(plan.batchable == false)
    }

    @Test func mcpPingAndNotificationsAndBoolPreservation() throws {
        if case .ping(let id) = MCPRoute(line: #"{"jsonrpc":"2.0","id":7,"method":"ping"}"#) {
            let reply = MCPWire.result(id: id, [String: String]())
            #expect(reply == #"{"jsonrpc":"2.0","id":7,"result":{}}"#)
        } else {
            Issue.record("Expected ping route")
        }
        if case .ignored = MCPRoute(line: #"{"jsonrpc":"2.0","method":"notifications/cancelled"}"#) {
        } else {
            Issue.record("Expected ignored route for notification without id")
        }
        let parsed = try #require(try JSONSerialization.jsonObject(with: Data(#"{"flag":true,"count":1}"#.utf8)) as? [String: Any])
        let wrapped = AnyCodable(parsed)
        let encoded = try String(data: JSONEncoder().encode(wrapped), encoding: .utf8)
        #expect(encoded?.contains(#""flag":true"#) == true)
        #expect(encoded?.contains(#""count":1"#) == true)
    }

    @Test func mcpArgumentsRejectBooleanNumericSpoofing() {
        #expect(throws: MCPFailure.self) {
            _ = try MCPArguments.generation(["prompt": "hi", "maxTokens": true])
        }
        #expect(throws: MCPFailure.self) {
            _ = try MCPArguments.generation(["prompt": "hi", "temperature": false])
        }
        #expect(throws: MCPFailure.self) {
            _ = try HarnessInputArguments(["directory": "/tmp", "alwaysOnMaxChars": true])
        }
    }

    @Test func multipartFormIgnoresFalseBoundaryPrefixInsideBinaryPayload() throws {
        let boundary = "bound123"
        var raw = Data()
        raw.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.bin\"\r\n\r\n".utf8))
        let payload = Data("prefix\r\n--\(boundary)NOT_REAL_BOUNDARY_BYTES_HERE".utf8)
        raw.append(payload)
        raw.append(Data("\r\n--\(boundary)--\r\n".utf8))
        let form = try MultipartForm(raw, boundary: boundary)
        #expect(form.parts["file"]?.body == payload)
    }

    @Test func leJEPAMaxPoolPreservesAllNegativeUnmaskedFeatures() {
        let h = MLXArray([Float(-3.0), Float(-1.5), Float(-9.0), Float(-0.5)], [1, 4, 1])
        let mask = MLXArray([Int32(1), Int32(1), Int32(0), Int32(0)], [1, 4])
        let pooled = LeJEPABackbone.pool(.init(hiddenStates: h), mask: mask, strategy: .max)
        eval(pooled)
        #expect(abs(pooled.item(Float.self) - (-1.5)) < 1e-5)
    }

    @Test func weightConverterPreservesFalseQuantizationOverrides() throws {
        let raw = Data(#"{"quantization":{"bits":4,"group_size":64,"lm_head":false,"count":1}}"#.utf8)
        let json = try #require(try JSONSerialization.jsonObject(with: raw) as? [String: Any])
        let disabled = WeightConverter.disabledQuantizationEntries(in: json)
        #expect(disabled == ["lm_head": false])
        let merged = WeightConverter.mergingDisabled(disabled, into: ["bits": 4, "group_size": 32, "mode": "mxfp4"])
        #expect((merged["lm_head"] as? Bool) == false)
        #expect(merged["count"] == nil)
    }

    @Test func reviewRemoteValidatesGiteaOwnerRepoAndHandlesSpecialPaths() {
        #expect(throws: ReviewRemote.Failure.self) {
            _ = try ReviewRemote(forge: .gitea, instance: "https://gitea.example.com", repo: "singleSegment", number: 1, token: nil)
        }
        #expect(throws: ReviewRemote.Failure.self) {
            _ = try ReviewRemote(forge: .gitea, instance: "https://gitea.example.com", repo: "a/b/c", number: 1, token: nil)
        }
    }

    @Test func cloudStreamRequestRequestsUsageChunk() {
        let req = ModelRouter.chatRequest(GenerationRequest(prompt: "hi"), model: "openai/gpt-4o-mini", stream: true)
        #expect(req.usage?.include == true)
        let nonStream = ModelRouter.chatRequest(GenerationRequest(prompt: "hi"), model: "openai/gpt-4o-mini", stream: false)
        #expect(nonStream.usage == nil)
    }
}
