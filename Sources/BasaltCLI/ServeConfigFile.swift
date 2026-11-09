import Foundation
import ArgumentParser
import Yams

enum ServeConfigError: Error, Equatable, CustomStringConvertible {
    struct Issue: Equatable, Sendable {
        var path: String
        var line: Int?
        var column: Int?
        var message: String
    }

    case unreadable(file: String, reason: String)
    case invalid(file: String, issues: [Issue])

    var description: String {
        switch self {
        case .unreadable(let file, let reason):
            return "\(file): \(reason)"
        case .invalid(let file, let issues):
            return issues.map { issue in
                let location = issue.line.map { ":\($0):\(issue.column ?? 1)" } ?? ""
                return "\(file)\(location): \(issue.path): \(issue.message)"
            }.joined(separator: "\n")
        }
    }
}

enum ServeConfigFile {
    enum Kind: Sendable {
        case int(String)
        case number(String)
        case text(String)
        case list(String)
        case flag(on: String?, off: String?)
    }

    static let maxBytes = 256 << 10

    static func expand(_ arguments: [String]) throws(ServeConfigError) -> [String] {
        guard arguments.first == "serve", let found = configArgument(in: arguments) else { return arguments }
        var cli = arguments
        cli.removeSubrange(found.range)
        cli.removeFirst()
        return ["serve"] + (try tokens(file: found.path, cli: cli)) + cli
    }

    static func configArgument(in arguments: [String]) -> (range: Range<Int>, path: String)? {
        for index in arguments.indices {
            let argument = arguments[index]
            if argument == "--config", index + 1 < arguments.endIndex, isYAML(arguments[index + 1]) {
                return (index ..< index + 2, arguments[index + 1])
            }
            if argument.hasPrefix("--config="), isYAML(String(argument.dropFirst(9))) {
                return (index ..< index + 1, String(argument.dropFirst(9)))
            }
        }
        return nil
    }

    static func isYAML(_ path: String) -> Bool {
        let lowered = path.lowercased()
        return lowered.hasSuffix(".yaml") || lowered.hasSuffix(".yml")
    }

    static func tokens(file: String, cli: [String]) throws(ServeConfigError) -> [String] {
        let data: Data
        do {
            let handle = try FileHandle(forReadingFrom: ConfigSection.fileURL(file))
            defer { try? handle.close() }
            data = try handle.read(upToCount: maxBytes + 1) ?? Data()
        } catch {
            throw .unreadable(file: file, reason: error.localizedDescription)
        }
        guard data.count <= maxBytes else { throw .unreadable(file: file, reason: "larger than \(maxBytes >> 10) KB") }
        return try tokens(yaml: String(decoding: data, as: UTF8.self), file: file, cli: cli)
    }

    static func tokens(yaml: String, file: String, cli: [String]) throws(ServeConfigError) -> [String] {
        let root: Node?
        do {
            root = try Yams.compose(yaml: yaml)
        } catch {
            throw .unreadable(file: file, reason: String(describing: error))
        }
        var walk = Walk(cli: cli)
        if let root { walk.root(root) }
        guard walk.issues.isEmpty else { throw .invalid(file: file, issues: walk.issues) }
        return walk.tokens
    }

    static let sections: Set<String> = Set(keys.keys.flatMap { path in
        let parts = path.split(separator: ".")
        return (1 ..< parts.count).map { parts.prefix($0).joined(separator: ".") }
    })

    struct Walk {
        let cli: [String]
        var tokens: [String] = []
        var issues: [ServeConfigError.Issue] = []

        mutating func root(_ node: Node) {
            guard let mapping = node.mapping else { return issue("", node, "the file must be a mapping of version and defaults") }
            for (key, value) in mapping {
                switch key.string {
                case "version": if value.int != 1 { issue("version", value, "must be 1") }
                case "defaults": section(value, path: "")
                default: issue(key.string ?? "", key, "unknown key")
                }
            }
        }

        mutating func section(_ node: Node, path: String) {
            if node.null != nil { return }
            guard let mapping = node.mapping else { return issue(Self.display(path), node, "must be a mapping") }
            for (key, value) in mapping {
                let name = key.string ?? ""
                let child = path.isEmpty ? name : "\(path).\(name)"
                if let kind = ServeConfigFile.keys[child] {
                    leaf(value, path: child, kind: kind)
                } else if ServeConfigFile.sections.contains(child) {
                    section(value, path: child)
                } else {
                    issue(Self.display(child), key, "unknown key")
                }
            }
        }

        mutating func leaf(_ node: Node, path: String, kind: Kind) {
            if node.null != nil { return }
            switch kind {
            case .int(let flag):
                guard let value = node.int else { return mismatch(path, node, "an integer") }
                tokens.append("\(flag)=\(value)")
            case .number(let flag):
                guard let value = node.int.map(Double.init) ?? node.float else { return mismatch(path, node, "a number") }
                tokens.append("\(flag)=\(value)")
            case .text(let flag):
                guard let value = node.scalar?.string else { return mismatch(path, node, "a string") }
                tokens.append("\(flag)=\(Self.expanded(value))")
            case .list(let flag):
                let items = node.sequence?.compactMap { $0.scalar?.string }
                guard let items, items.count == node.sequence?.count else { return mismatch(path, node, "a list of strings") }
                if !cli.contains(where: { $0 == flag || $0.hasPrefix("\(flag)=") }) { tokens += items.map { "\(flag)=\(Self.expanded($0))" } }
            case .flag(let on, let off):
                guard let value = node.bool else { return mismatch(path, node, "true or false") }
                if let token = value ? on : off { tokens.append(token) }
            }
        }

        mutating func mismatch(_ path: String, _ node: Node, _ expected: String) {
            let got = node.scalar.map { "\"\($0.string)\"" } ?? (node.mapping == nil ? "a list" : "a mapping")
            issue(Self.display(path), node, "expected \(expected), got \(got)")
        }

        mutating func issue(_ path: String, _ node: Node, _ message: String) {
            issues.append(.init(path: path, line: node.mark?.line, column: node.mark?.column, message: message))
        }

        static func expanded(_ value: String) -> String {
            value == "~" || value.hasPrefix("~/") ? (value as NSString).expandingTildeInPath : value
        }

        static func display(_ path: String) -> String {
            path.isEmpty ? "defaults" : "defaults.\(path)"
        }
    }
}

extension ServeConfigFile {
    static let keys: [String: Kind] = [
        "model.repo": .text("--model"),
        "model.path": .text("--model-path"),
        "model.quant": .text("--quant"),
        "model.embedding": .text("--embedding-model"),
        "model.embedding_adapter": .text("--embedding-adapter"),
        "model.rerank": .text("--rerank-model"),
        "serve.port": .int("--port"),
        "serve.host": .text("--host"),
        "serve.max_concurrent": .int("--max-concurrent"),
        "serve.max_queue": .int("--max-queue"),
        "serve.max_tokens": .int("--max-tokens"),
        "serve.log_level": .text("--log-level"),
        "serve.experimental_samplers": .flag(on: "--experimental-samplers", off: nil),
        "serve.experimental_zero_copy_body": .flag(on: "--experimental-zero-copy-body", off: nil),
        "serve.experimental_smallm_kernel": .flag(on: "--experimental-smallm-kernel", off: nil),
        "serve.speculative.experimental_mtp_draft_head": .flag(on: "--experimental-mtp-draft-head", off: nil),
        "serve.experimental_adaptive_thinking_budget": .flag(on: "--experimental-adaptive-thinking-budget", off: nil),
        "serve.image_fallback": .text("--image-fallback"),
        "serve.grammar_jump_forward": .flag(on: "--grammar-jump-forward", off: nil),
        "serve.expert_streaming.enabled": .flag(on: "--experimental-expert-streaming", off: nil),
        "serve.expert_streaming.cache_gb": .number("--expert-cache-gb"),
        "serve.kv.bits": .text("--kv-bits"),
        "serve.kv.zones": .flag(on: "--kv-zones", off: nil),
        "serve.kv.evict": .number("--kv-evict"),
        "serve.kv.head_split": .number("--kv-head-split"),
        "serve.kv.quest_top_k": .int("--quest-top-k"),
        "serve.prompt_cache.mb": .int("--prompt-cache-mb"),
        "serve.prompt_cache.disk_mb": .int("--prompt-cache-disk-mb"),
        "serve.prompt_cache.disk_days": .int("--prompt-cache-disk-days"),
        "serve.prompt_cache.dir": .text("--prompt-cache-dir"),
        "serve.prompt_cache.hybrid_anchors": .int("--hybrid-anchors"),
        "serve.cache_blend.enabled": .flag(on: "--cache-blend", off: nil),
        "serve.cache_blend.ratio": .number("--cache-blend-ratio"),
        "serve.speculative.lookup": .flag(on: "--speculative-lookup", off: nil),
        "serve.speculative.recycling": .flag(on: "--speculative-recycling", off: nil),
        "serve.speculative.mtp": .flag(on: "--mtp", off: nil),
        "serve.speculative.mtp_block": .int("--mtp-block"),
        "serve.sampling.top_k": .int("--top-k"),
        "serve.sampling.min_p": .number("--min-p"),
        "serve.sampling.repetition_penalty": .number("--repetition-penalty"),
        "serve.sampling.seed": .int("--seed"),
        "serve.sampling.reasoning_effort": .text("--reasoning-effort"),
        "serve.batching.slots": .int("--batch-slots"),
        "serve.memory.wired_limit": .flag(on: "--wired-limit", off: "--no-wired-limit"),
        "serve.memory.coordinator": .flag(on: nil, off: "--no-memory-coordinator"),
        "serve.memory.model_probe": .flag(on: "--model-probe", off: "--no-model-probe"),
        "serve.health.canary_minutes": .int("--canary-minutes"),
        "serve.health.stats": .flag(on: "--stats", off: nil),
        "serve.health.stats_interval": .int("--stats-interval"),
        "serve.session_log.mode": .text("--session-log"),
        "serve.session_log.redact": .text("--session-log-redact"),
        "serve.session_log.idle_minutes": .int("--session-idle-minutes"),
        "serve.session_log.dir": .text("--session-log-dir"),
        "serve.session_log.retention_days": .int("--session-log-retention-days"),
        "serve.session_log.max_total_mb": .int("--session-log-max-total-mb"),
        "serve.access.rate_limit": .int("--rate-limit"),
        "serve.access.rate_burst": .int("--rate-burst"),
        "serve.access.allow_origins": .list("--allow-origin"),
        "serve.access.auto_generate_key": .flag(on: "--auto-generate-key", off: nil),
        "serve.pipeline.ocr_model": .text("--pipeline-ocr-model"),
        "serve.pipeline.audio_model": .text("--pipeline-audio-model"),
        "serve.pipeline.decide": .text("--decision-model"),
        "serve.pipeline.abstain_below": .number("--pipeline-abstain-below"),
        "serve.pipeline.min_margin": .number("--pipeline-min-margin"),
        "serve.cloud.fallback": .text("--cloud-fallback"),
        "serve.cloud.base_url": .text("--cloud-base-url"),
        "serve.cloud.router_params": .text("--router-params"),
        "serve.cloud.cost_threshold": .number("--cost-threshold"),
        "serve.cloud.cache_stickiness": .number("--cache-stickiness"),
        "serve.cloud.local_context_limit": .int("--local-context-limit"),
        "mcp.enabled": .flag(on: "--mcp", off: nil),
        "mcp.rest": .flag(on: "--rest", off: nil),
        "mcp.research_tools": .flag(on: "--research-tools", off: nil),
        "mcp.image_dirs": .list("--image-dir"),
    ]
}

extension BasaltCLI {
    public static func configuredArguments(_ arguments: [String] = Array(CommandLine.arguments.dropFirst())) -> [String] {
        do {
            return try ServeConfigFile.expand(arguments)
        } catch {
            exit(withError: error)
        }
    }
}
