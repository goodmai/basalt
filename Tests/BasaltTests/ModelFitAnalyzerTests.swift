import Testing
@testable import BasaltCore

@Suite("ModelFitAnalyzer grid math")
struct ModelFitAnalyzerTests {

    @Test("Quality multiplier per grid width", arguments: [(8, 1.00), (6, 0.98), (4, 0.93), (3, 0.84), (2, 0.68)])
    func qualityMultiplier(bits: Int, expected: Double) {
        #expect(ModelFitAnalyzer.qualityMultiplier(bits: bits) == expected)
    }

    @Test("Fit score is flat to 70% of the budget, then falls, then halves past it")
    func fitScore() {
        #expect(ModelFitAnalyzer.fitScore(ramRatio: 0.5) == 1.0)
        #expect(ModelFitAnalyzer.fitScore(ramRatio: 0.7) == 1.0)
        #expect(abs(ModelFitAnalyzer.fitScore(ramRatio: 1.0) - 0.502) < 1e-12)
        #expect(ModelFitAnalyzer.fitScore(ramRatio: 1.5) == 0.25)
        #expect(ModelFitAnalyzer.fitScore(ramRatio: 3.0) == 0.0)
    }

    @Test("Weights in MiB from decimal billions with 15% overhead")
    func weightMB() {
        #expect(abs(ModelFitAnalyzer.weightMB(parametersBillions: 7.0, bits: 4) - 3838.539) < 0.001)
        #expect(ModelFitAnalyzer.weightMB(parametersBillions: 7.0, bits: 4) * 2 == ModelFitAnalyzer.weightMB(parametersBillions: 7.0, bits: 8))
    }
}
