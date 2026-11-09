import Foundation
import Testing
@testable import BasaltCore

@Suite("System One letter readout: the OpenJev prompt, letter softmax and the noul/score calibrations (#234)")
struct SystemOneReadoutTests {
    @Test("The text-lane prompt matches the published shim: State, Question, lettered options, the letter-only instruction; score adds the ordering note")
    func prompt() {
        let choice = SystemOneQuestion(type: .choice, instructions: "Which team?", criteria: .named(["shipping": "Parcels.", "billing": "Charges."]))
        #expect(SystemOneReadout.prefix(state: "s") == "State:\ns")
        #expect(SystemOneReadout.tail(choice) == "\n\nQuestion: Which team?\nOptions:\n[A] billing: Charges.\n[B] shipping: Parcels.\n\nAnswer with the letter of the best option only.")
        let score = SystemOneQuestion(type: .score, instructions: "How bad?", criteria: .levels(["low", "high"]))
        #expect(SystemOneReadout.tail(score).hasPrefix("\n\nQuestion: How bad? Rate along the ordered levels below (lowest first).\nOptions:\n[A] 0: low\n[B] 1: high"))
        let noul = SystemOneQuestion(type: .noul, instructions: "It rains.")
        #expect(SystemOneReadout.options(noul).map(\.key) == ["yes", "no"])
        #expect(SystemOneReadout.options(noul).map(\.description) == ["The statement is true.", "The statement is false."])
        #expect(SystemOneReadout.letters.count == 52 && SystemOneReadout.letters[26] == "a")
    }

    @Test("Letters become probabilities with temperature; noul applies the logit calibration; score is the expected level")
    func calibration() {
        let readout = SystemOneReadout(temperature: 2, noulTemperature: 2, noulBias: 0)
        let choice = readout.answer(SystemOneQuestion(type: .choice, instructions: "x", criteria: .named(["a": "A", "b": "B"])), letterLogits: [2, 0])
        #expect(abs((choice.probabilities?["a"] ?? 0) - 1 / (1 + exp(-1))) < 1e-9 && choice.choice == "a")
        let noul = readout.answer(SystemOneQuestion(type: .noul, instructions: "x"), letterLogits: [2, 0])
        let pYes = 1 / (1 + exp(-1.0))
        #expect(abs((noul.noul ?? 0) - 1 / (1 + exp(-log(pYes / (1 - pYes)) / 2))) < 1e-9)
        let score = SystemOneReadout.uncalibrated.answer(SystemOneQuestion(type: .score, instructions: "x", criteria: .levels(["a", "b", "c"])), letterLogits: [0, 0, 0])
        #expect(abs((score.score ?? 0) - 1) < 1e-9)
        #expect(SystemOneReadout.known("openjev/OpenJev-Flash-9B-MLX")?.temperature == 1.07)
        #expect(SystemOneReadout.known("org/other") == nil)
    }
}
