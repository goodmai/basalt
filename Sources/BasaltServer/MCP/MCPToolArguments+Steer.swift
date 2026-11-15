import Foundation
import BasaltCore

extension SteerGeometry {
    static let names = Self.allCases.map(\.rawValue)

    static func parse(in args: [String: Any]?) throws(MCPFailure) -> SteerGeometry {
        guard let raw = try MCPArguments.string("geometry", in: args) else { return .additive }
        guard let geometry = SteerGeometry(rawValue: raw) else {
            throw .invalidParams("geometry must be one of: \(names.joined(separator: ", "))")
        }
        return geometry
    }
}

extension Intervention {
    static let kinds = ["steer", "ablate_head", "ablate_expert"]

    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let layer = try MCPArguments.required("layer", MCPArguments.integer("layer", in: args, ResearchCall.nonNegative))
        switch try MCPArguments.string("kind", in: args) {
        case "steer"?:
            let vector = try MCPArguments.required("vector", MCPArguments.numbers("vector", in: args))
            let scale = try MCPArguments.number("scale", in: args) ?? 1
            let geometry = try SteerGeometry.parse(in: args)
            self = .steer(layer: layer, vector: vector, scale: scale, geometry: geometry)
        case "ablate_head"?:
            self = .ablateHead(layer: layer, head: try MCPArguments.required("head", MCPArguments.integer("head", in: args, ResearchCall.nonNegative)))
        case "ablate_expert"?:
            self = .ablateExpert(layer: layer, expert: try MCPArguments.required("expert", MCPArguments.integer("expert", in: args, ResearchCall.nonNegative)))
        default:
            throw .invalidParams("kind must be one of: \(Self.kinds.joined(separator: ", "))")
        }
        if let when = MCPArguments.present("when", in: args) { self = .conditional(try ConditionalIntervention(self, mcp: when)) }
    }
}
