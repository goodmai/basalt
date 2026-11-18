import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltUI

@Suite("Retrospective audit closure — Core & Gate (#544, #556, #557)")
struct RetroWave4Tests {
    @Test("PhasorProbeEvaluator keeps trainStates and trainLabels aligned when a prompt is missing from extracted vectors")
    func phasorMissingPromptAlignment() {
        var req = PhasorProbeRequest(
            train: [
                LabeledPrompt(prompt: "p0", sector: 0),
                LabeledPrompt(prompt: "missing_prompt", sector: 1),
                LabeledPrompt(prompt: "p1", sector: 1),
            ],
            heldOut: [
                LabeledPrompt(prompt: "missing_eval", sector: 0),
                LabeledPrompt(prompt: "p1", sector: 1),
            ],
            sectors: 2
        )
        req.maxEpochs = 4
        let extracted = ExtractedPhasorStates(
            layers: [0],
            vectorsByPrompt: [
                "p0": [0: [1.0, 0.0, 1.0, 0.0]],
                "p1": [0: [-1.0, 0.0, -1.0, 0.0]],
            ]
        )
        let summary = PhasorProbeEvaluator.evaluate(request: req, extracted: extracted)
        #expect(summary.layers.count == 1)
        #expect(summary.layers[0].trainAccuracy >= 0)
    }

    @Test("TelemetryCore caps distinct decision model buckets at maxDecisionModels + _other")
    func telemetryDecisionCardinalityCap() {
        var core = TelemetryCore()
        let usage = SystemOneUsage(
            prefixTokens: 4, candidateBranches: 2, prefixEvaluations: 1, branchForwardCalls: 2, suffixTokens: 2)
        for i in 0 ..< (TelemetryCore.maxDecisionModels + 12) {
            let resp = SystemOneResponse(model: "dynamic-model-\(i)", answers: [:], usage: usage, calibrated: false)
            core.recordDecision(resp, latencyMs: 1.5)
        }
        #expect(core.decisions.count == TelemetryCore.maxDecisionModels + 1)
        #expect(core.decisions["_other"] != nil)
    }

    @Test("ExpertSlotTable speculative prefetch hit does not double-count as a demand miss")
    func expertSlotSpeculativePrefetchHitCountsZeroMisses() throws {
        var table = ExpertSlotTable(slots: 4)
        let prefetched = table.place([0, 1], prefetch: true, speculative: true)
        _ = try #require(prefetched)
        let demandOpt = table.place([0, 1], prefetch: false, speculative: false)
        let demand = try #require(demandOpt)
        #expect(demand.loads.isEmpty)
        #expect(demand.misses == 0)
        #expect(demand.prefetchHits == 2)
    }

    @Test("HubSnapshot preserves double-hyphen model names and ignores .DS_Store")
    func hubSnapshotDoubleHyphenRepoAndDSStore() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("hub-\(UUID().uuidString)/models--org--model--v2/snapshots/abc123")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) }
        try Data("{}".utf8).write(to: base.appendingPathComponent("config.json"))
        try Data("junk".utf8).write(to: base.appendingPathComponent(".DS_Store"))
        let snap = HubSnapshot(directory: base)
        #expect(snap.repoId == "org/model--v2")
        #expect(snap.revision == "abc123")
        #expect(snap.files() == ["config.json"])
    }

    @Test("ChatMessageInput leadingSystem and alternating preserve images on system and leading assistant turns")
    func chatMessageInputPreservesImagesAcrossFolding() throws {
        let sysImg = try ImageInput(data: Data("sys".utf8), mime: "image/png")
        let asstImg = try ImageInput(data: Data("asst".utf8), mime: "image/png")
        let usrImg = try ImageInput(data: Data("usr".utf8), mime: "image/png")
        let turns = [
            ChatMessageInput(role: .system, text: "Rule", images: [sysImg]),
            ChatMessageInput(role: .assistant, text: "Hello", images: [asstImg]),
            ChatMessageInput(role: .user, text: "World", images: [usrImg]),
        ]
        let alt = ChatMessageInput.alternating(turns)
        #expect(alt.count == 2)
        #expect(alt[0].role == .system)
        #expect(alt[0].images == [sysImg])
        #expect(alt[1].role == .user)
        #expect(alt[1].images == [asstImg, usrImg])
    }

    @Test("GenerationGate admit(after:) refuses when closed and SpeechFallback transcription obeys GenerationGate")
    func generationGateClosedAdmitAfterAndSpeechFallbackGate() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 0)
        await gate.shutdown(deadline: .zero)
        await #expect(throws: BasaltError.self) {
            _ = try await gate.admit { }
        }

        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
        let speech = ScriptedTranscriber()
        await orchestrator.setSpeechFallback(speech)
        await orchestrator.shutdown(deadline: .zero)
        let clip = try AudioClip(samples: [0.1, 0.2])
        let req = try TranscriptionRequest(audio: clip)
        await #expect(throws: BasaltError.self) {
            _ = try await orchestrator.transcribe(req)
        }
        #expect(speech.requests.withLock { $0.isEmpty })
    }
}

@Suite("Retrospective audit closure — Inference, CLI & DiffParser (#544, #556)")
struct RetroWave4EngineCLITests {
    @Test("LogitProbe clamps negative topK and LogitProbeRecorder slices only the last sequence token")
    func logitProbeMultiRowAndZeroTopK() {
        let zeroProbe = LogitProbe(topK: -3)
        #expect(zeroProbe.topK == 0)
        let zeroRec = LogitProbeRecorder(probe: zeroProbe)
        _ = zeroRec.process(logits: MLXArray([Float(1), 2, 3, 4]).reshaped(1, 2, 2))
        #expect(zeroProbe.steps.count == 1)
        #expect(zeroProbe.steps[0].ids.isEmpty)

        let probe = LogitProbe(topK: 2)
        var rec = LogitProbeRecorder(probe: probe)
        let multi = MLXArray([Float(100), 0, -5, 1, 9, 2]).reshaped(1, 2, 3)
        _ = rec.process(logits: multi)
        rec.didSample(token: MLXArray(1))
        #expect(probe.steps.count == 1)
        #expect(probe.steps[0].ids == [1, 2])
        #expect(probe.steps[0].token == 1)
    }

    @Test("OrthogonalProcrustes rejects zero-column matrices and non-scalar mean_cosine")
    func orthogonalProcrustesValidation() throws {
        let emptyCols = MLXArray.zeros([4, 0], dtype: .float32)
        #expect(throws: BasaltError.self) {
            _ = try OrthogonalProcrustes.fit(x: emptyCols, y: emptyCols)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bad-proc-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(
            arrays: [
                "q": MLXArray.eye(2, dtype: .float32),
                "mean_x": MLXArray.zeros([2], dtype: .float32),
                "mean_y": MLXArray.zeros([2], dtype: .float32),
                "mean_cosine": MLXArray([Float(0.5), 0.6]),
            ],
            url: url
        )
        #expect(throws: BasaltError.self) {
            _ = try OrthogonalProcrustes.load(from: url)
        }
    }

    @Test("MLXInferenceEngine tool helpers map NSNull cleanly and preserve caller toolCall IDs")
    func toolHelpersNullAndIdPreservation() {
        let spec = MLXInferenceEngine.toolSpecs([
            ToolDefinition(name: "fn", description: "d", parametersJSON: #"{"type":"object","default":null}"#),
        ])
        #expect(spec?.count == 1)
        let preserved = MLXInferenceEngine.toolCallOutput(
            ToolCall(function: .init(name: "fn", arguments: ["x": .int(1)]), id: "call_fixed_42")
        )
        #expect(preserved.id == "call_fixed_42")
        let generated = MLXInferenceEngine.toolCallOutput(
            ToolCall(function: .init(name: "fn", arguments: [:]), id: "")
        )
        #expect(generated.id.hasPrefix("call_"))
    }

    @Test("CLI validators reject out-of-range ports, oversized expert cache, and non-finite YAML numbers")
    func cliBoundaryValidation() throws {
        #expect(throws: ValidationError.self) {
            _ = try NetworkSettings.resolve(NetworkSettings.Flags(port: -1), file: ConfigSection())
        }
        #expect(throws: ValidationError.self) {
            _ = try NetworkSettings.resolve(NetworkSettings.Flags(port: 70_000), file: ConfigSection())
        }
        #expect(try NetworkSettings.validPort(0, key: "--port") == 0)
        #expect(throws: (any Error).self) {
            _ = try ExpertStreamingOptions.parse(["--expert-streaming", "--expert-cache-gb", "100000"])
        }
        #expect(throws: ServeConfigError.self) {
            _ = try ServeConfigFile.tokens(
                yaml: "version: 1\ndefaults:\n  serve:\n    sampling:\n      min_p: .nan\n",
                file: "basalt.yaml",
                cli: []
            )
        }
    }

    @Test("DiffParser keeps --- and +++ body lines inside active hunks and skips no-newline markers")
    func diffParserKeepsDoubleDashInsideActiveHunks() {
        let diff = """
        diff --git a/schema.sql b/schema.sql
        --- a/schema.sql
        +++ b/schema.sql
        @@ -1,3 +1,3 @@
         SELECT 1;
        --- old sql comment
        +++ new header note
         SELECT 2;
        \\ No newline at end of file
        """
        let hunks = DiffParser.parseUnifiedDiff(diff)
        #expect(hunks.count == 1)
        #expect(hunks[0].oldFile == "a/schema.sql")
        #expect(hunks[0].newFile == "b/schema.sql")
        #expect(hunks[0].lines == [
            DiffLine(type: .context, text: "SELECT 1;"),
            DiffLine(type: .deletion, text: "-- old sql comment"),
            DiffLine(type: .addition, text: "++ new header note"),
            DiffLine(type: .context, text: "SELECT 2;"),
        ])
    }
}
