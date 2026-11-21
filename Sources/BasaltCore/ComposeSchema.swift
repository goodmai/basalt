import Foundation

public enum ComposeSchema {
    public enum Kind: Sendable, Equatable {
        case string, number, integer, boolean, duration, names, integers
        case choice([String])
        case section(String)
        case sections(String)
        case list(String)
        case nameOrSection(String)
        case namesOrSection(String)
        case suiteMap(item: String, section: String)
        case variables
    }

    public struct Field: Sendable, Equatable {
        public let name: String
        public let kind: Kind
        public let since: Int

        init(_ name: String, _ kind: Kind, since: Int = 1) {
            (self.name, self.kind, self.since) = (name, kind, since)
        }
    }

    public static let versions = 1 ... 2

    public static let sections: [String: [Field]] = [
        "root": [
            Field("version", .choice(["1", "2", "2.2"])), Field("name", .string), Field("memory", .choice(PipelineMemory.allCases.map(\.rawValue))),
            Field("budget", .section("budget")), Field("models", .sections("model")), Field("pipeline", .section("pipeline")),
            Field("pipelines", .sections("pipeline")), Field("experimental", .section("experimental")), Field("mcp", .section("mcp")),
            Field("variables", .variables, since: 2), Field("profiles", .sections("profile"), since: 2),
            Field("adapters", .sections("compose_adapter"), since: 2),
            Field("benchmarks", .section("benchmarks"), since: 2),
            Field("voice", .section("voice_v22"), since: 2),
            Field("music", .section("music_v22"), since: 2),
            Field("video", .section("video_v22"), since: 2),
            Field("train", .section("train_v22"), since: 2),
            Field("cluster", .section("cluster_v22"), since: 2),
        ],
        "experimental": [
            Field("zero_copy_body", .boolean), Field("smallm_kernel", .boolean), Field("q3_kernel", .boolean), Field("kv_shared_row", .boolean), Field("speech_analyzer", .boolean),
            Field("image_fallback", .choice(["ocr"])), Field("ane_vision_tower", .string),
            Field("telemetry_confidence_every", .integer), Field("adaptive_thinking_budget", .boolean),
            Field("samplers", .boolean), Field("mtp_draft_head", .boolean),
            Field("expert_streaming", .boolean), Field("expert_cache_gb", .number),
            Field("kv_head_split", .number), Field("quest_top_k", .integer),
        ],
        "mcp": [
            Field("enabled", .boolean), Field("rest", .boolean), Field("research_tools", .boolean),
            Field("enable", .names), Field("disable", .names), Field("image_dirs", .names),
        ],
        "model": [
            Field("source", .string), Field("keep", .choice(ComposeSpec.Keep.allCases.map(\.rawValue))), Field("idle_unload", .duration),
            Field("api_key_env", .string), Field("data", .choice([ComposeSpec.noPerceived])), Field("budget", .section("budget")),
            Field("profiles", .names, since: 2),
            Field("speculative", .section("model_speculative"), since: 2),
            Field("diffusion_text", .section("diffusion_text"), since: 2),
        ],
        "pipeline": [
            Field("perceive", .section("perceive")), Field("decide", .section("decide")),
            Field("answer", .nameOrSection("answer")), Field("speak", .namesOrSection("speak")),
            Field("imagine", .namesOrSection("imagine")),
            Field("profiles", .names, since: 2),
        ],
        "speak": [
            Field("chain", .names), Field("voice", .string), Field("language", .string),
            Field("reference_audio", .string),
            Field("on_gpu_busy", .choice(SpeechOnGPUBusy.allCases.map(\.rawValue))),
            Field("sentence_streaming", .boolean),
            Field("format", .choice(SpeechAudioFormat.allCases.map(\.rawValue))),
            Field("exaggeration", .number), Field("max_ttfa_ms", .integer),
        ],
        "imagine": [
            Field("chain", .names), Field("upsample_via", .string), Field("share_encoder", .string),
            Field("share_text_encoder_with", .string, since: 2), Field("kv_cache_references", .boolean, since: 2),
            Field("rgba", .boolean, since: 2), Field("upscale", .section("imagine_upscale"), since: 2),
            Field("decoder_via", .string), Field("steps", .integer), Field("size", .string),
            Field("format", .choice(ImageOutputFormat.allCases.map(\.rawValue))),
            Field("teacache_threshold", .number), Field("preview_every_steps", .integer),
            Field("candidates", .integer), Field("judge_via", .string),
            Field("on_gpu_busy", .choice(ImageOnGPUBusy.allCases.map(\.rawValue))),
        ],
        "perceive": [Field("documents", .names), Field("audio", .names)],
        "decide": [
            Field("cascade", .names), Field("escalate_below", .number), Field("abstain_below", .number), Field("min_margin", .number),
            Field("epistemic", .section("epistemic"), since: 2),
        ],
        "epistemic": [
            Field("enabled", .boolean), Field("support_threshold", .number), Field("refutation_threshold", .number),
            Field("tool_threshold", .number),
            Field("on_no_evidence", .choice(EpistemicGateSpec.NoEvidencePolicy.allCases.map(\.rawValue))),
            Field("on_conflict", .choice(EpistemicGateSpec.ConflictPolicy.allCases.map(\.rawValue))),
            Field("abstain_message", .string),
        ],
        "answer": [Field("model", .string), Field("adapter", .string, since: 2), Field("routes", .list("route"))],
        "route": [Field("when", .section("when")), Field("model", .string), Field("adapter", .string, since: 2), Field("fallback", .string)],
        "when": [
            Field("modality", .choice(AnswerRoute.Modality.allCases.map(\.rawValue))), Field("decide", .variables),
            Field("prompt_tokens", .section("over")), Field("escalated", .boolean), Field("cost_left", .boolean),
        ],
        "over": [Field("over", .integer)],
        "budget": [Field("usd_per_day", .number)],
        "profile": [Field("when", .section("machine")), Field("variables", .variables)],
        "machine": [Field("memory_gb", .section("range")), Field("chip", .string)],
        "range": [Field("under", .number), Field("over", .number)],
        "benchmarks": [
            Field("defaults", .section("benchmark_defaults"), since: 2),
            Field("suites", .suiteMap(item: "benchmark_task", section: "benchmark_suite"), since: 2),
        ],
        "benchmark_defaults": [
            Field("seed", .integer), Field("temperature", .number),
            Field("gpu_lock", .boolean), Field("output_dir", .string),
        ],
        "benchmark_suite": [
            Field("profiles", .names), Field("budget", .section("benchmark_budget")),
            Field("tasks", .list("benchmark_task")),
        ],
        "benchmark_budget": [Field("usd_max", .number), Field("max_minutes", .number)],
        "benchmark_task": [
            Field("adapter", .choice(BenchmarkAdapterID.known.sorted())), Field("dataset", .string),
            Field("target", .string), Field("params", .variables),
            Field("gates", .section("benchmark_gates")), Field("submit", .section("benchmark_submit")),
        ],
        "benchmark_gates": [
            Field("min_decode_tps", .number), Field("max_peak_gb", .number), Field("min_accuracy", .number),
            Field("max_p50_ms", .number), Field("max_ppl", .number), Field("max_ece", .number),
        ],
        "benchmark_submit": [
            Field("target", .choice(SubmissionTarget.allCases.map(\.rawValue))), Field("api_key_env", .string),
            Field("endpoint", .string), Field("subset", .string), Field("split", .string),
        ],
        "compose_adapter": [
            Field("base", .string), Field("method", .choice(["lora", "phm", "oft"])),
            Field("rank", .integer), Field("scale", .number),
            Field("hypercomplex_dim", .integer), Field("factor_rank", .integer), Field("shared_basis", .boolean),
            Field("block_size", .integer), Field("parameterization", .choice(OrthogonalParameterization.allCases.map(\.rawValue))),
            Field("block_shared", .boolean), Field("corpus", .section("adapter_corpus")),
            Field("output", .string), Field("layers", .integers),
            Field("iterations", .integer), Field("batch_size", .integer),
            Field("learning_rate", .number), Field("max_seq_len", .integer),
            Field("checkpoint_every", .integer), Field("gate", .section("adapter_gate")),
            Field("profiles", .names),
        ],
        "adapter_corpus": [
            Field("source", .string), Field("subset", .string), Field("split", .string),
            Field("max_rows", .integer), Field("min_pass_rate", .number),
            Field("include_reasoning", .boolean), Field("seed", .integer),
        ],
        "adapter_gate": [
            Field("max_kl", .number), Field("max_ppl_rise", .number), Field("held_out", .names),
        ],
        "model_speculative": [
            Field("mode", .choice(ModelSpeculativeSpec.validModes.sorted())),
            Field("drafter", .string), Field("block_size", .integer),
            Field("verify", .choice(ModelSpeculativeSpec.validVerifiers.sorted())),
        ],
        "diffusion_text": [
            Field("dual_kv_cache", .boolean), Field("confidence_threshold", .number), Field("contrastive_alpha", .number),
        ],
        "imagine_upscale": [
            Field("model", .string), Field("steps", .integer), Field("target_resolution", .integer),
        ],
        "voice_v22": [
            Field("mode", .choice(VoiceDuplexSpec.validModes.sorted())), Field("sts_model", .string),
            Field("denoise", .string), Field("turn_detection", .section("voice_turn_detection")),
            Field("tts_fallback", .string), Field("aligner", .string),
        ],
        "voice_turn_detection": [
            Field("model", .string), Field("diarization", .string),
        ],
        "music_v22": [
            Field("model", .string), Field("steps", .integer), Field("duration_seconds", .integer),
        ],
        "video_v22": [
            Field("model", .string), Field("vae_tiling", .boolean), Field("sta_sparse", .boolean),
            Field("fps", .integer), Field("frames", .integer),
        ],
        "train_v22": [
            Field("method", .choice(ComposeTrainSpec.validMethods.sorted())),
            Field("segments", .integer), Field("ema_decay", .number),
        ],
        "cluster_v22": [
            Field("topology", .choice(ComposeClusterSpec.validTopologies.sorted())),
            Field("peers", .names),
            Field("compression", .choice(ComposeClusterSpec.validCompressions.sorted())),
        ],
    ]

    public static func keys(_ section: String, version: Int = versions.upperBound) -> Set<String> {
        Set((sections[section] ?? []).filter { $0.since <= version }.map(\.name))
    }

    public static func since(_ key: String, in section: String) -> Int {
        sections[section]?.first { $0.name == key }?.since ?? 1
    }

    public static var allKeys: Set<String> { Set(sections.values.flatMap { $0.map(\.name) }) }
}

extension ComposeSchema {
    public static var jsonSchema: String {
        var definitions: [String: Any] = [:]
        for (name, fields) in sections {
            var entry = object(fields)
            if name == "benchmarks" {
                entry["additionalProperties"] = suiteValueSchema(item: "benchmark_task", section: "benchmark_suite")
            }
            definitions[name] = entry
        }
        let schema: [String: Any] = [
            "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "basalt compose file", "$ref": "#/$defs/root", "$defs": definitions,
        ]
        let data = (try? JSONSerialization.data(withJSONObject: schema, options: [.prettyPrinted, .sortedKeys])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    static func suiteValueSchema(item: String, section: String) -> [String: Any] {
        ["oneOf": [["type": "array", "items": ["$ref": "#/$defs/\(item)"]], ["$ref": "#/$defs/\(section)"]]]
    }

    static func object(_ fields: [Field]) -> [String: Any] {
        ["type": "object", "additionalProperties": false, "properties": Dictionary(uniqueKeysWithValues: fields.map { ($0.name, json($0.kind)) })]
    }

    static func json(_ kind: Kind) -> [String: Any] {
        if let leaf = scalarJson(kind) { return leaf }
        switch kind {
        case .choice(let values): return ["enum": values]
        case .section(let name): return ["$ref": "#/$defs/\(name)"]
        case .sections(let name): return ["type": "object", "additionalProperties": ["$ref": "#/$defs/\(name)"]]
        case .list(let name): return ["type": "array", "items": ["$ref": "#/$defs/\(name)"]]
        case .nameOrSection(let name): return ["oneOf": [["type": "string"], ["$ref": "#/$defs/\(name)"]]]
        case .namesOrSection(let name):
            return ["oneOf": [["type": "string"], ["type": "array", "items": ["type": "string"]], ["$ref": "#/$defs/\(name)"]]]
        case .suiteMap(let item, let section): return ["type": "object", "additionalProperties": suiteValueSchema(item: item, section: section)]
        case .variables: return ["type": "object", "additionalProperties": ["type": ["string", "number", "boolean"]]]
        case .string, .number, .integer, .boolean, .duration, .names, .integers: return [:]
        }
    }

    private static func scalarJson(_ kind: Kind) -> [String: Any]? {
        switch kind {
        case .string: ["type": "string"]
        case .number: ["type": "number"]
        case .integer: ["type": "integer"]
        case .boolean: ["type": "boolean"]
        case .duration: ["oneOf": [["type": "integer"], ["type": "string", "pattern": "^[0-9]+[smh]$"]]]
        case .names: ["type": "array", "items": ["type": "string"]]
        case .integers: ["type": "array", "items": ["type": "integer"]]
        default: nil
        }
    }
}
