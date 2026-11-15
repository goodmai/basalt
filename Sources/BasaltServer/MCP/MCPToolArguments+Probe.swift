import Foundation
import BasaltCore

extension PhasorProbeRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let train = try Self.labeled("train", in: args, range: Self.trainCount, required: false)
        let heldOut = try Self.labeled("heldOut", in: args, range: 1...Self.heldOutCount.upperBound, required: false)
        let triples = try Self.roleFillerTriples(in: args)
        guard !train.isEmpty || !triples.isEmpty else {
            throw .invalidParams("phasor_probe requires train prompts or triples")
        }
        let sectors = try MCPArguments.integer("sectors", in: args, Self.sectorRange) ?? 2
        if !train.isEmpty {
            for item in train + heldOut where !(0 ..< sectors).contains(item.sector) {
                throw .invalidParams("sector must be in 0..<\(sectors)")
            }
        }
        var request = PhasorProbeRequest(
            train: train,
            heldOut: heldOut,
            triples: triples,
            sectors: sectors,
            layers: try MCPArguments.integers("layers", in: args)
        )
        request.folds = try MCPArguments.integer("folds", in: args, Self.foldRange) ?? request.folds
        request.continuous = try MCPArguments.optionalBoolean("continuous", in: args) ?? request.continuous
        request.encoding = try Self.parseEncoding(in: args)
        request.pooling = try Self.parsePooling(in: args)
        request.maxEpochs = try MCPArguments.integer("maxEpochs", in: args, Self.epochRange) ?? request.maxEpochs
        if let lr = try MCPArguments.number("learningRate", in: args) {
            guard lr > 0 else { throw .invalidParams("learningRate must be positive") }
            request.learningRate = lr
        }
        if let seed = try MCPArguments.integer("seed", in: args, ResearchCall.nonNegative) {
            request.seed = UInt64(seed)
        }
        self = request
    }

    static func labeled(_ key: String, in args: [String: Any]?, range: ClosedRange<Int>, required: Bool) throws(MCPFailure) -> [LabeledPrompt] {
        guard let raw = MCPArguments.present(key, in: args) else {
            if required { throw .invalidParams("Missing required argument: \(key)") }
            return []
        }
        guard let items = raw as? [[String: Any]], range.contains(items.count) else {
            throw .invalidParams("\(key) must be an array of \(range) {prompt, sector} objects")
        }
        var parsed: [LabeledPrompt] = []
        for item in items {
            let prompt = try MCPArguments.required("\(key).prompt", MCPArguments.string("prompt", in: item))
            guard !prompt.isEmpty else { throw .invalidParams("\(key).prompt must be non-empty") }
            let sector = try MCPArguments.required("\(key).sector", MCPArguments.integer("sector", in: item, ResearchCall.nonNegative))
            parsed.append(LabeledPrompt(prompt: prompt, sector: sector))
        }
        return parsed
    }

    static func roleFillerTriples(in args: [String: Any]?) throws(MCPFailure) -> [RoleFillerTriple] {
        guard let raw = MCPArguments.present("triples", in: args) else { return [] }
        guard let items = raw as? [[String: Any]], tripleCount.contains(items.count) else {
            throw .invalidParams("triples must be an array of \(tripleCount) {bound, role, filler} objects")
        }
        var parsed: [RoleFillerTriple] = []
        for item in items {
            let bound = try MCPArguments.required("triples.bound", MCPArguments.string("bound", in: item))
            let role = try MCPArguments.required("triples.role", MCPArguments.string("role", in: item))
            let filler = try MCPArguments.required("triples.filler", MCPArguments.string("filler", in: item))
            guard !bound.isEmpty, !role.isEmpty, !filler.isEmpty else {
                throw .invalidParams("triples bound, role, and filler must be non-empty strings")
            }
            parsed.append(RoleFillerTriple(bound: bound, role: role, filler: filler))
        }
        return parsed
    }

    static func parseEncoding(in args: [String: Any]?) throws(MCPFailure) -> PhasorInputEncoding {
        guard let raw = try MCPArguments.string("encoding", in: args) else { return .unitPhasors }
        guard let value = PhasorInputEncoding(rawValue: raw) else {
            let names = PhasorInputEncoding.allCases.map(\.rawValue).joined(separator: ", ")
            throw .invalidParams("encoding must be one of: \(names)")
        }
        return value
    }

    static func parsePooling(in args: [String: Any]?) throws(MCPFailure) -> PhasorPooling {
        guard let raw = try MCPArguments.string("pooling", in: args) else { return .lastToken }
        guard let value = PhasorPooling(rawValue: raw) else {
            let names = PhasorPooling.allCases.map(\.rawValue).joined(separator: ", ")
            throw .invalidParams("pooling must be one of: \(names)")
        }
        return value
    }
}
