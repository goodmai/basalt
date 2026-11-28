@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

@testable import BasaltAppCore
@testable import BasaltCore
@testable import BasaltFoundationModels
@testable import BasaltInference

#if canImport(FoundationModels, _version: 2)
import FoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct ReleaseReadinessReport: Equatable, Sendable {
    @Guide(description: "Release name.")
    var title: String
    @Guide(description: "Whether verification gates passed.")
    var passedGates: Bool
    @Guide(description: "Measured token throughput.")
    var tokensPerSecond: Int
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct CityLookupArguments: Equatable, Sendable {
    @Guide(description: "Target city name.")
    var city: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct WeatherProbeTool: FoundationModels.Tool {
    let name = "lookup_weather"
    let description = "Looks up the current temperature in Celsius for a city."
    typealias Arguments = CityLookupArguments

    func call(arguments: CityLookupArguments) async throws -> String {
        "Temperature in \(arguments.city) is 21C and clear"
    }
}
#endif

actor IsolatedTensorProducer {
    func produceEmbeddings() -> MaterializedEmbeddingBatch {
        let raw = MLXArray([Float(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]) * Float(2.0)
        return MaterializedTensorBridge.freezeEmbeddings(raw, promptTokens: 9)
    }

    func produceLogits() -> MaterializedLogitSnapshot {
        let scores = MLXArray([Float(-1.2), 0.4, 3.9, 1.1, 4.5, 0.2, -0.3, 2.0], [2, 4])
        return MaterializedTensorBridge.freezeLogits(scores, stepIndex: 5)
    }

    func produceBatch() -> [MaterializedTensorSnapshot] {
        let first = MLXArray([Float(0.5), -0.5, 1.5], [3]) + Float(1.0)
        let second = MLXArray([Int32(10), 20, 30, 40], [2, 2])
        return MaterializedTensorBridge.freezeBatch([first, second])
    }
}

actor IsolatedTensorConsumer {
    func sumEmbeddingRow(_ batch: MaterializedEmbeddingBatch, row: Int) -> Float {
        batch.withRowSpan(row) { span in
            var sum: Float = 0
            for index in 0 ..< span.count { sum += span[index] }
            return sum
        }
    }

    func readByteCount(_ snapshot: MaterializedTensorSnapshot) -> Int {
        snapshot.withRawSpan { $0.byteCount }
    }
}

@Suite("MaterializedTensorBridge transfers evaluated MLX snapshots across actors without host array copies", .serialized)
struct MaterializedTensorBridgeTests {
    @Test("Cross-actor embedding, logit, and batch snapshots preserve exact values via Span and RawSpan")
    func crossActorMaterializedRoundTrip() async {
        let producer = IsolatedTensorProducer()
        let consumer = IsolatedTensorConsumer()

        let embeddings = await producer.produceEmbeddings()
        #expect(embeddings.count == 2)
        #expect(embeddings.dimensions == 3)
        #expect(embeddings.promptTokens == 9)
        let row0Sum = await consumer.sumEmbeddingRow(embeddings, row: 0)
        let row1Sum = await consumer.sumEmbeddingRow(embeddings, row: 1)
        #expect(abs(row0Sum - 12.0) < 1e-5)
        #expect(abs(row1Sum - 30.0) < 1e-5)
        #expect(embeddings.toEmbeddings().vectors == [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])

        let logits = await producer.produceLogits()
        #expect(logits.stepIndex == 5)
        #expect(logits.vocabularySize == 4)
        #expect(logits.greedyToken(row: 0) == 2)
        #expect(logits.greedyToken(row: 1) == 0)

        let batch = await producer.produceBatch()
        #expect(batch.count == 2)
        #expect(batch[0].shape == [3])
        #expect(batch[0].toArray(as: Float.self) == [1.5, 0.5, 2.5])
        #expect(batch[1].shape == [2, 2])
        #expect(batch[1].toArray(as: Int32.self) == [10, 20, 30, 40])
        #expect(await consumer.readByteCount(batch[1]) == 16)
    }

    @Test("MLXInferenceEngine materializedTensor extracts Sendable X-Ray weight slices from a TinyModel")
    func engineMaterializedXRayReadout() async throws {
        let engine = try await TinyModel.engine(.qwen3Small)
        let allTensors = try await engine.tensors(matching: "embed_tokens")
        let target = try #require(allTensors.first)
        let request = TensorRequest(name: target.name, slice: [0 ..< 4, 0 ..< 8])
        let readout = try await engine.materializedTensor(request)
        #expect(readout.info.name == target.name)
        #expect(readout.slice.shape == [4, 8])
        #expect(readout.slice.elementCount == 32)
        let values = readout.slice.toArray(as: Float.self)
        #expect(values.count == 32)
        #expect(values.allSatisfy { $0.isFinite })
    }
}

@Suite("GatedDeltaFastPath matches gatedDeltaStepReference and scalar CPU oracle within 1e-4 (#228)", .serialized)
struct GatedDeltaFastPathTests {
    static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        var worst: Float = 0
        for index in lhs.indices {
            worst = max(worst, abs(lhs[index] - rhs[index]))
        }
        return worst
    }

    static func makeFixture(
        shape: [Int],
        seed: UInt64,
        withState: Bool,
        maskValues: [Bool]? = nil
    ) -> (mlx: GatedDeltaInputs, cpu: GatedDeltaCPUInputs) {
        let (b, t, hk, hv, dk, dv) = (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5])
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let q = MLXRandom.normal([b, t, hk, dk]) * Float(0.25)
            let k = MLXRandom.uniform(low: -0.1, high: 0.1, [b, t, hk, dk])
            let v = MLXRandom.normal([b, t, hv, dv]) * Float(0.25)
            let gamma = MLXRandom.uniform(low: 0.55, high: 0.98, [b, t, hv])
            let beta = MLXRandom.uniform(low: 0.1, high: 0.85, [b, t, hv])
            let state = withState ? MLXRandom.normal([b, hv, dv, dk]) * Float(0.2) : nil
            let mask = maskValues.map { MLXArray($0, [b, t]) }
            eval(q, k, v, gamma, beta)
            if let state { eval(state) }
            if let mask { eval(mask) }
            let mlxInputs = GatedDeltaInputs(queries: q, keys: k, values: v, gamma: gamma, beta: beta)
                .withInitialState(state)
                .withMask(mask)
            let cpuInputs = GatedDeltaCPUInputs(
                shape: shape,
                queries: q.asArray(Float.self),
                keys: k.asArray(Float.self),
                values: v.asArray(Float.self),
                gamma: gamma.asArray(Float.self),
                beta: beta.asArray(Float.self)
            ).withStateAndMask(initialState: state?.asArray(Float.self), mask: maskValues)
            return (mlxInputs, cpuInputs)
        }
    }

    @Test("Fast path matches step reference and scalar CPU reference across equal-head, GQA, and masked sequences")
    func parityAgainstStepAndCPUReference() throws {
        let maskPattern = [true, true, false, true, false, true, false, true, true, false]
        let scenarios: [([Int], UInt64, Bool, [Bool]?)] = [
            ([2, 5, 4, 4, 8, 8], 101, false, nil),
            ([2, 6, 2, 4, 8, 8], 202, true, nil),
            ([2, 5, 2, 4, 8, 8], 303, true, maskPattern),
        ]
        for (shape, seed, withState, mask) in scenarios {
            let fixture = Self.makeFixture(shape: shape, seed: seed, withState: withState, maskValues: mask)
            let fast = try GatedDeltaFastPath.materializedUpdate(fixture.mlx)
            let ref = try GatedDeltaFastPath.gatedDeltaStepReference(fixture.mlx)
            eval(ref.output, ref.state)
            let cpu = GatedDeltaFastPath.gatedDeltaStepCPUReference(fixture.cpu)

            let fastOut = fast.output.toArray(as: Float.self)
            let fastState = fast.state.toArray(as: Float.self)
            let refOut = ref.output.asArray(Float.self)
            let refState = ref.state.asArray(Float.self)

            #expect(Self.maxAbsDiff(fastOut, refOut) < 1e-4)
            #expect(Self.maxAbsDiff(fastState, refState) < 1e-4)
            #expect(Self.maxAbsDiff(fastOut, cpu.output) < 1e-4)
            #expect(Self.maxAbsDiff(fastState, cpu.state) < 1e-4)
        }
    }

    @Test("Metal fused kernel dimensions Dk=128 Dv=128 and Qwen3.5 gate parameterization match reference within 1e-4")
    func metalKernelShapeAndQwen35GateParity() throws {
        let kernelFixture = Self.makeFixture(shape: [1, 3, 16, 32, 128, 128], seed: 404, withState: true)
        let kernelFast = try GatedDeltaFastPath.materializedUpdate(kernelFixture.mlx)
        let kernelRef = try GatedDeltaFastPath.gatedDeltaStepReference(kernelFixture.mlx)
        eval(kernelRef.output, kernelRef.state)
        #expect(Self.maxAbsDiff(kernelFast.output.toArray(as: Float.self), kernelRef.output.asArray(Float.self)) < 1e-4)
        #expect(Self.maxAbsDiff(kernelFast.state.toArray(as: Float.self), kernelRef.state.asArray(Float.self)) < 1e-4)

        let gateInputs = withRandomState(MLXRandom.RandomState(seed: 505)) {
            let q = MLXRandom.normal([1, 4, 2, 16]) * Float(0.2)
            let k = MLXRandom.uniform(low: -0.1, high: 0.1, [1, 4, 2, 16])
            let v = MLXRandom.normal([1, 4, 4, 16]) * Float(0.2)
            let gate = Qwen35GatedDeltaGate(
                a: MLXRandom.normal([1, 4, 4]) * Float(0.3),
                b: MLXRandom.normal([1, 4, 4]) * Float(0.3),
                aLog: MLXArray([Float(0.1), 0.2, 0.3, 0.4], [1, 1, 4]),
                dtBias: MLXArray([Float(0.05), -0.05, 0.1, 0.0], [1, 1, 4])
            )
            return GatedDeltaInputs.fromQwen35(queries: q, keys: k, values: v, gate: gate)
        }
        let qwenFast = try GatedDeltaFastPath.update(gateInputs)
        let qwenRef = try GatedDeltaFastPath.gatedDeltaStepReference(gateInputs)
        eval(qwenFast.output, qwenFast.state, qwenRef.output, qwenRef.state)
        #expect(Self.maxAbsDiff(qwenFast.output.asArray(Float.self), qwenRef.output.asArray(Float.self)) < 1e-4)
        #expect(Self.maxAbsDiff(qwenFast.state.asArray(Float.self), qwenRef.state.asArray(Float.self)) < 1e-4)
    }

    @Test("Invalid head divisibility or mismatched state shape throws BasaltError.invalidRequestStructure")
    func rejectsInvalidShapes() {
        let badHeads = GatedDeltaInputs(
            queries: MLXArray.zeros([1, 2, 3, 8]),
            keys: MLXArray.zeros([1, 2, 3, 8]),
            values: MLXArray.zeros([1, 2, 4, 8]),
            gamma: MLXArray.ones([1, 2, 4]),
            beta: MLXArray.ones([1, 2, 4])
        )
        #expect(throws: BasaltError.self) { _ = try GatedDeltaFastPath.update(badHeads) }

        let badState = GatedDeltaInputs(
            queries: MLXArray.zeros([1, 2, 2, 8]),
            keys: MLXArray.zeros([1, 2, 2, 8]),
            values: MLXArray.zeros([1, 2, 4, 8]),
            gamma: MLXArray.ones([1, 2, 4]),
            beta: MLXArray.ones([1, 2, 4])
        ).withInitialState(MLXArray.zeros([1, 4, 8, 4]))
        #expect(throws: BasaltError.self) { _ = try GatedDeltaFastPath.gatedDeltaStepReference(badState) }
    }
}

actor MockTransportRecorder {
    private(set) var requests: [BasaltFoundationRequest] = []
    private var scriptedReplies: [[ChatDelta]]

    init(replies: [[ChatDelta]]) {
        self.scriptedReplies = replies
    }

    func nextDeltas(for request: BasaltFoundationRequest) -> [ChatDelta] {
        requests.append(request)
        guard !scriptedReplies.isEmpty else { return [.finish("stop")] }
        return scriptedReplies.removeFirst()
    }
}

@Suite("BasaltFoundationModels bridges @Generable schemas, tool-call round-trips, and REST SSE requests")
struct BasaltFoundationModelsTests {
    static func mockTransport(recorder: MockTransportRecorder) -> BasaltFoundationTransport {
        BasaltFoundationTransport(
            stream: { request in
                AsyncThrowingStream { continuation in
                    let task = Task {
                        let deltas = await recorder.nextDeltas(for: request)
                        for delta in deltas { continuation.yield(delta) }
                        continuation.finish()
                    }
                    continuation.onTermination = { _ in task.cancel() }
                }
            },
            health: { true }
        )
    }

    @Test("Tool-calling envelope hoists nested $defs, structural tags validate, and REST serialization omits MLX")
    func schemaEnvelopeAndRESTRequestSerialization() throws {
        let toolA = ToolDefinition(
            name: "book_flight",
            description: "Book a trip",
            parametersJSON: ##"{"type":"object","required":["passenger"],"properties":{"passenger":{"$ref":"#/$defs/Passenger"}},"$defs":{"Passenger":{"type":"object","properties":{"name":{"type":"string"}}}}}"##
        )
        let toolB = ToolDefinition(
            name: "book_hotel",
            description: "Book a room",
            parametersJSON: ##"{"type":"object","required":["passenger"],"properties":{"passenger":{"$ref":"#/$defs/Passenger"}},"$defs":{"Passenger":{"type":"object","properties":{"vip":{"type":"boolean"}}}}}"##
        )
        let envelopeJSON = try BasaltFoundationSchemaAdapter.encodeToolCallingEnvelope(tools: [toolA, toolB])
        let envelopeObj = try #require(try JSONSerialization.jsonObject(with: Data(envelopeJSON.utf8)) as? [String: Any])
        let defs = try #require(envelopeObj["$defs"] as? [String: Any])
        #expect(defs["book_flight__Passenger"] != nil)
        #expect(defs["book_hotel__Passenger"] != nil)
        #expect(envelopeJSON.contains("#/$defs/book_flight__Passenger") || envelopeJSON.contains("#\\/$defs\\/book_flight__Passenger"))

        let tagJSON = try BasaltFoundationSchemaAdapter.encodeToolCallingStructuralTag(tools: [toolA, toolB])
        let validatedTag = try ResponseFormat.structuralTag(tagJSON).grammarSource()
        #expect(validatedTag.contains("<tool_call>"))

        var req = BasaltFoundationRequest(
            messages: [ChatMessageInput(role: .user, text: "Check flights")],
            model: "qwen3.5-27b"
        )
        req.tools = [toolA]
        req.toolChoice = ToolChoice(forcedTool: "book_flight")
        req.responseFormat = .jsonSchema(#"{"type":"object","properties":{"ok":{"type":"boolean"}}}"#, strict: true)
        let urlReq = try req.httpURLRequest(for: .loopback(port: 8123, token: "tok_abc"))
        #expect(urlReq.url?.absoluteString == "http://127.0.0.1:8123/v1/chat/completions")
        #expect(urlReq.value(forHTTPHeaderField: "Authorization") == "Bearer tok_abc")
        let httpBody = try #require(urlReq.httpBody)
        let bodyObj = try #require(try JSONSerialization.jsonObject(with: httpBody) as? [String: Any])
        #expect(bodyObj["model"] as? String == "qwen3.5-27b")
        #expect((bodyObj["tools"] as? [[String: Any]])?.count == 1)
        #expect(((bodyObj["response_format"] as? [String: Any])?["type"] as? String) == "json_schema")
    }

    @Test("Multi-turn tool loop handles both embedded structural-tag tool calls and native SSE toolCall deltas")
    func providerToolLoopRoundTrip() async throws {
        let recorder = MockTransportRecorder(replies: [
            [
                .text("<tool_call>\n{\"name\": \"lookup_weather\", \"arguments\": {\"city\": \"Zurich\"}}\n</tool_call>"),
                .usage(prompt: 18, completion: 12),
                .finish("tool_calls"),
            ],
            [
                .text("Zurich is currently 21C and clear."),
                .usage(prompt: 42, completion: 9),
                .finish("stop"),
            ],
        ])
        let provider = BasaltFoundationModelProvider(
            modelID: "basalt-mock",
            transport: Self.mockTransport(recorder: recorder)
        )
        let weatherTool = ToolDefinition(
            name: "lookup_weather",
            description: "Lookup weather",
            parametersJSON: #"{"type":"object","required":["city"],"properties":{"city":{"type":"string"}}}"#
        )
        let finalReply = try await provider.executeToolLoop(
            messages: [ChatMessageInput(role: .user, text: "Weather in Zurich?")],
            tools: [weatherTool]
        ) { call in
            #expect(call.name == "lookup_weather")
            #expect(call.arguments.contains("Zurich"))
            return #"{"tempC":21,"condition":"clear"}"#
        }
        #expect(finalReply.text == "Zurich is currently 21C and clear.")
        let recorded = await recorder.requests
        #expect(recorded.count == 2)
        #expect(recorded[1].messages.contains { $0.role == .tool && $0.toolName == "lookup_weather" })
    }

    #if canImport(FoundationModels, _version: 2)
    @Test("FoundationModels @Generable decoding and LanguageModelSession tool-calling round-trip succeed")
    func generableAndLanguageModelSessionRoundTrip() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let jsonPayload = #"{"title":"Basalt 0.33","passedGates":true,"tokensPerSecond":142}"#
        let recorder = MockTransportRecorder(replies: [
            [.text(jsonPayload), .usage(prompt: 20, completion: 14), .finish("stop")],
            [.text(jsonPayload), .usage(prompt: 24, completion: 14), .finish("stop")],
            [
                .toolCall(ToolCallOutput(id: "call_w1", name: "lookup_weather", arguments: #"{"city":"Tokyo"}"#)),
                .usage(prompt: 30, completion: 10),
                .finish("tool_calls"),
            ],
            [
                .text("Tokyo is 21C and clear today."),
                .usage(prompt: 52, completion: 8),
                .finish("stop"),
            ],
        ])
        let provider = BasaltFoundationModelProvider(
            modelID: "basalt-fm-test",
            transport: Self.mockTransport(recorder: recorder)
        )

        let direct = try await provider.generate(
            ReleaseReadinessReport.self,
            prompt: "Summarize release readiness",
            instructions: "Return strict JSON."
        )
        #expect(direct == ReleaseReadinessReport(title: "Basalt 0.33", passedGates: true, tokensPerSecond: 142))

        let structuredSession = provider.makeSession(instructions: "Be concise.")
        let sessionStructured = try await structuredSession.respond(
            to: "Generate the readiness report",
            generating: ReleaseReadinessReport.self
        )
        #expect(sessionStructured.content == direct)

        let toolSession = provider.makeSession(tools: [WeatherProbeTool()], instructions: "Use tools when asked about weather.")
        let toolAnswer = try await toolSession.respond(to: "What is the weather in Tokyo?")
        #expect(toolAnswer.content.contains("21C"))

        let captured = await recorder.requests
        #expect(captured.count == 4)
        if case .jsonSchema(let schemaText, _)? = captured[0].responseFormat {
            #expect(schemaText.contains("tokensPerSecond"))
        } else {
            Issue.record("Expected .jsonSchema responseFormat on @Generable request")
        }
    }
    #endif
}

@Suite("VideoFrameSampler enforces bounded uniform frame timestamps and registers new LLM/VLM families", .serialized)
struct VideoFrameSamplerAndAliasesTests {
    @Test("Uniform timestamp sampling enforces monotonic spacing, endpoints, minFrames floor, and maxFrames cap")
    func uniformTimestampInvariants() throws {
        let tenSeconds = CMTime(seconds: 10.0, preferredTimescale: 600)
        let cappedConfig = VideoSamplingConfiguration(maxFrames: 6, minFrames: 2, targetFPS: 2.0)
        let capped = try VideoFrameSampler.uniformTimestamps(duration: tenSeconds, configuration: cappedConfig)
        #expect(capped.count == 6)
        #expect(abs(capped.first?.seconds ?? -1) < 1e-6)
        #expect(abs((capped.last?.seconds ?? -1) - 10.0) < 1e-6)
        for index in 1 ..< capped.count {
            #expect(CMTimeCompare(capped[index - 1], capped[index]) < 0)
            #expect(abs((capped[index].seconds - capped[index - 1].seconds) - 2.0) < 1e-2)
        }

        let zeroTimes = try VideoFrameSampler.uniformTimestamps(
            duration: .zero,
            configuration: VideoSamplingConfiguration(maxFrames: 8, minFrames: 1, targetFPS: 1.0)
        )
        #expect(zeroTimes.count == 1)
        #expect(zeroTimes[0].value == 0)

        #expect(throws: BasaltError.self) {
            _ = try VideoSamplingConfiguration(maxFrames: 0).validated()
        }
        #expect(throws: BasaltError.self) {
            _ = try VideoFrameSampler.resolveVideoURL("https://example.com/clip.mp4")
        }
        let localURL = try VideoFrameSampler.resolveVideoURL("/tmp/sample_clip.mov")
        #expect(localURL.path == "/tmp/sample_clip.mov")
    }

    @Test("In-memory video frame subsampling caps frame count and converts cleanly to MLXVLM UserInput")
    func inMemoryFrameSubsamplingAndVLMBridge() throws {
        let sourceFrames = (0 ..< 20).map { index in
            let color = CIColor(red: CGFloat(index) / 20.0, green: 0.25, blue: 0.5)
            let image = CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16))
            let ts = CMTime(value: Int64(index * 300), timescale: 600)
            return SampledFrameInput(image: image, timestamp: ts)
        }
        let sampled = try VideoFrameSampler.sampleMemoryFrames(
            sourceFrames,
            configuration: VideoSamplingConfiguration(maxFrames: 5, minFrames: 2, targetFPS: 2.0)
        )
        #expect(sampled.frames.count == 5)
        #expect(sampled.timestamps.count == 5)
        #expect(sampled.timestamps.first?.value == 0)
        #expect(sampled.timestamps.last?.value == 5700)

        let vlmImages = sampled.asUserInputImages(labelPrefix: "frame")
        #expect(vlmImages.count == 5)
        if case .frames(let videoFrames) = sampled.asUserInputVideo().source {
            #expect(videoFrames.count == 5)
        } else {
            Issue.record("Expected .frames source in UserInput.Video")
        }
    }

    @Test("All 7 upstream and alias model families resolve in LLMTypeRegistry, VLMTypeRegistry, and ModelCapabilities")
    func modelFamilyAliasesAndCapabilities() async throws {
        await MLXInferenceEngine.registerAliases()
        let llm = LLMTypeRegistry.shared
        let vlm = VLMTypeRegistry.shared

        for type in ["lfm3", "lfm3_moe", "deepseek_v3", "deepseek-v3", "mimo", "mimo_v2_flash", "minimax", "minimax_m1", "bitnet", "falcon_h1", "falcon-h1", "hunyuan_mt", "hunyuan_v1_dense"] {
            #expect(await llm.contains(type), "Expected LLMTypeRegistry to contain \(type)")
        }
        for vlmType in ["lfm2_vl", "lfm3_vl", "lfm3-vl"] {
            #expect(await vlm.contains(vlmType), "Expected VLMTypeRegistry to contain \(vlmType)")
        }

        let probeTokens = MLXArray([Int32(1)]).reshaped(1, 1)
        let lfm3Data = Data(TinyModel.Family.lfm2.config.replacingOccurrences(of: "\"lfm2\"", with: "\"lfm3\"").utf8)
        let lfm3Model = try await llm.createModel(configuration: lfm3Data, modelType: "lfm3")
        #expect(lfm3Model(LMInput.Text(tokens: probeTokens), cache: nil, state: nil).logits.dim(-1) == 256)

        let hunyuanConfig = """
            {"model_type": "hunyuan_mt", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 256, "head_dim": 16}
            """
        let hunyuanModel = try await llm.createModel(configuration: Data(hunyuanConfig.utf8), modelType: "hunyuan_mt")
        #expect(hunyuanModel(LMInput.Text(tokens: probeTokens), cache: nil, state: nil).logits.dim(-1) == 256)

        #expect(ModelCapabilities.of(modelType: "lfm3")?.allows(.kvQuantization) == true)
        #expect(ModelCapabilities.of(modelType: "lfm3")?.allows(.cacheBlend) == false)
        #expect(ModelCapabilities.of(modelType: "mimo")?.allows(.cacheBlend) == true)
        #expect(ModelCapabilities.of(modelType: "mimo_v2_flash")?.allows(.cacheBlend) == false)
        #expect(ModelCapabilities.of(modelType: "minimax")?.allows(.batchedDecode) == true)
        #expect(ModelCapabilities.of(modelType: "hunyuan_mt")?.allows(.kvQuantization) == true)
        #expect(ModelCapabilities.of(modelType: "falcon_h1")?.allows(.kvQuantization) == true)
        #expect(ModelCapabilities.of(modelType: "gemma4_unified")?.allows(.kvQuantization) == true)
        #expect(ModelCapabilities.of(modelType: "gemma4_unified_text")?.allows(.speculativeLookup) == true)
        #expect(ModelCapabilities.of(modelType: "falcon_h1")?.allows(.batchedDecode) == false)
        #expect(ModelCapabilities.of(modelType: "deepseek_v3")?.allows(.promptCacheReuse) == true)
        #expect(ModelCapabilities.of(modelType: "deepseek-v3")?.allows(.kvQuantization) == false)
        #expect(ModelCapabilities.of(modelType: "deepseek_v2") == nil)
        #expect(ModelCapabilities.of(modelType: "bitnet") == nil)
    }
}
