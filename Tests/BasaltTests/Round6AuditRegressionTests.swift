import Foundation
import Testing
import MLX
import MLXNN
@testable import BasaltCore
@testable import BasaltBench
@testable import BasaltUI
@testable import BasaltInference

@Suite("Round 6 audit bench, UI & core regression tests (#696, #697)", .serialized)
struct Round6CoreUIBenchRegressionTests {
    @Test("TableRenderer handles negative caller widths without trapping (#696)")
    func tableRendererNegativeWidths() {
        let rendered = TableRenderer.render(headers: ["A", "B"], rows: [["hello", "world"]], widths: [-5, -1])
        #expect(!rendered.isEmpty)
    }

    @Test("BenchmarkComparator flags NaN candidate as regressed and encodes cleanly to JSON (#696)")
    func benchmarkComparatorNaNAndJSON() throws {
        let base = BenchmarkRunArtifact(
            runID: "b1", suiteID: "s", model: "m",
            scorecard: BenchmarkScorecard(metrics: ["accuracy": 0.9]), samples: []
        )
        let cand = BenchmarkRunArtifact(
            runID: "c1", suiteID: "s", model: "m",
            scorecard: BenchmarkScorecard(metrics: ["accuracy": Double.nan]), samples: []
        )
        let report = BenchmarkComparator.compare(baseline: base, candidate: cand)
        #expect(report.hasRegressions)
        let encoded = try JSONEncoder().encode(report)
        #expect(!encoded.isEmpty)
    }

    @Test("TerminalUI.Configuration.shared syncs with TerminalUI.colorsEnabled (#696)")
    func terminalUIConfigSyncsColorsEnabled() async {
        let prior = TerminalUI.colorsEnabled
        defer { TerminalUI.colorsEnabled = prior }
        await TerminalUI.Configuration.shared.setColorsEnabled(false)
        #expect(TerminalUI.colorsEnabled == false)
        await TerminalUI.Configuration.shared.setColorsEnabled(true)
        #expect(TerminalUI.colorsEnabled == true)
    }

    @Test("ARCAGIAdapter.buildPrompt selects requested testIndex (#696)")
    func arcAGIBuildPromptTestIndex() {
        let item = ARCAGITaskItem(
            id: "multi",
            train: [.init(input: [[1]], output: [[2]])],
            test: [.init(input: [[3]], output: [[4]]), .init(input: [[5]], output: [[6]])]
        )
        let prompt1 = ARCAGIAdapter.buildPrompt(item, testIndex: 1)
        #expect(prompt1.contains("[[5]]"))
        #expect(!prompt1.contains("[[3]]"))
    }

    @Test("HuggingFaceHub usesCustomProtocols is false for default URLSessionConfiguration (#697)")
    func hfHubDefaultConfigNotCustomProtocol() async {
        let hub = HuggingFaceHub(configuration: .default)
        #expect(await hub.usesCustomProtocols == false)
    }

    @Test("SecretRedactor redacts PEM private keys in JSON-escaped single-line strings (#697)")
    func secretRedactorJSONEscapedPEM() {
        let header = ["-----BEGIN", "PRIVATE", "KEY-----"].joined(separator: " ")
        let footer = ["-----END", "PRIVATE", "KEY-----"].joined(separator: " ")
        let json = "{\"key\":\"\(header)\\nMIIEvQIBADANBg\\n\(footer)\"}"
        let redacted = SecretRedactor.redact(json)
        #expect(redacted.contains("[REDACTED:private_key]"))
        #expect(!redacted.contains("MIIEvQIBADANBg"))
    }

    @Test("ReasoningPolicy trims whitespace and rejects empty strings (#697)")
    func reasoningPolicyTrimAndEmpty() {
        #expect(ReasoningPolicy(rawValue: "") == nil)
        #expect(ReasoningPolicy(rawValue: "   ") == nil)
        #expect(ReasoningPolicy(rawValue: " none ") == .disabled)
        #expect(ReasoningPolicy(rawValue: " high ") == .effort("high"))
    }

    @Test("ModelCache preserves -- inside repository names (#697)")
    func modelCachePreservesDoubleHyphenInRepoName() {
        #expect(ModelCache.hfRepoId(from: "claude-local/org--model--4bit") == "org/model--4bit")
        #expect(ModelCache.canonicalRepoId(from: "/cache/models--org--model--4bit/snapshots/main") == "org/model--4bit")
    }

    @Test("MCPToolFilter.allowsSpectrum respects disabled experiment tools (#697)")
    func mcpToolFilterSpectrumDisabled() {
        let filter = MCPToolFilter(enabled: ["weights_delta"], disabled: ["weights_delta"])
        #expect(!filter.allowsSpectrum(researchTools: false))
    }

    @Test("ChatTemplateCompatibility handles }} inside Jinja string literals (#697)")
    func chatTemplateHandlesBracesInStrings() {
        let tpl = "{% for call in calls %}{{ '{\"a\": ' + call.args + '}}' }}{% endfor %}"
        let out = ChatTemplateCompatibility.renamingReservedLoopVariables(tpl)
        #expect(out.contains("call_item.args"))
        #expect(out.contains("}}'"))
    }
}

@Suite("Round 6 audit adapter & surgery regression tests (#698)", .serialized)
struct Round6AdaptersSurgeryRegressionTests {
    @Test("QuantizedPHMLinear and QuantizedOFTLinear preserve base layer bias (#698)")
    func quantizedPHMAndOFTPreserveBias() throws {
        let dense = Linear(64, 64, bias: true)
        let ql = QuantizedLinear(dense, groupSize: 64, bits: 4, mode: .affine)
        let phm = try QuantizedPHMLinear(linear: ql, hypercomplexDim: 4, factorRank: 2)
        let oft = try QuantizedOFTLinear(linear: ql, blockSize: 16)
        #expect(phm.bias != nil)
        #expect(oft.bias != nil)
    }

    @Test("PackedLayer preserves MACaQQuantizedLinear rowScale and kernel flags (#698)")
    func packedLayerPreservesMACaQRowScale() {
        let dense = Linear(64, 64, bias: false)
        let spec = MACaQLayerSpec(bits: 4, groupSize: 64, mode: .affine, hadamardBlock: 32)
        let base = MACaQQuantizedLinear(replacing: dense, spec: spec)
        let rowScale = MLXArray.ones([64], dtype: .float32) * 2.0
        let withRow = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: base.weight, scales: base.scales, biases: base.biases),
            bias: nil, preScale: base.preScale, spec: spec, rowScale: rowScale)
        withRow.useSmallMKernel = true
        let packed = PackedLayer(path: "test", layer: withRow)
        let restored = packed.layer as? MACaQQuantizedLinear
        #expect(restored?.rowScale != nil)
        #expect(restored?.useSmallMKernel == true)
    }
}
