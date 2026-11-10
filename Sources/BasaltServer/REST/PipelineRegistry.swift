import Foundation
import BasaltCore

package struct PipelineSystem: Sendable {
    package var decisions: (any DecisionEngine)?
    package var stages: PipelineStages

    package init(decisions: (any DecisionEngine)?, stages: PipelineStages) {
        (self.decisions, self.stages) = (decisions, stages)
    }
}

package actor PipelineRegistry {
    private var systems: [String: PipelineSystem]

    package init(_ systems: [String: PipelineSystem]) { self.systems = systems }

    package var names: [String] { systems.keys.sorted() }

    package func system(_ name: String) -> PipelineSystem? { systems[name] }

    package func replace(_ next: [String: PipelineSystem]) { systems = next }
}
