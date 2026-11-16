import Foundation
import BasaltCore

package struct PipelineSystem: Sendable {
    package var decisions: (any DecisionEngine)?
    package var stages: PipelineStages
    package var speech: (any SpeechSynthesizer)?

    package init(
        decisions: (any DecisionEngine)?,
        stages: PipelineStages,
        speech: (any SpeechSynthesizer)? = nil
    ) {
        (self.decisions, self.stages, self.speech) = (decisions, stages, speech)
    }
}

package actor PipelineRegistry {
    private var systems: [String: PipelineSystem]

    package init(_ systems: [String: PipelineSystem]) { self.systems = systems }

    package var names: [String] { systems.keys.sorted() }

    package func system(_ name: String) -> PipelineSystem? { systems[name] }

    package func replace(_ next: [String: PipelineSystem]) { systems = next }
}
