import Foundation
import Testing
import MLX
import MLXNN
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("Round 4 audit model & quantization regression tests (#687, #688)", .serialized)
struct Round4ModelQuantRegressionTests {
    @Test("RoPE2D preserves vector norm at off-diagonal positions where row != col (#687)")
    func rope2DPreservesNormOffDiagonal() {
        let (cos, sin) = RoPE2D.frequencies(gridH: 3, gridW: 4, headDim: 16)
        let values: [Float] = (0 ..< 192).map { i -> Float in
            Float((i % 7) - 3) * 0.25 + 0.1
        }
        let x = MLXArray(values, [1, 1, 12, 16])
        let rotated = RoPE2D.apply(x, cos: cos, sin: sin)
        let normBefore = sum(x * x, axis: -1)
        let normAfter = sum(rotated * rotated, axis: -1)
        let maxDiff = max(abs(normBefore - normAfter))
        eval(maxDiff)
        #expect(maxDiff.item(Float.self) < 1e-4)
    }

    @Test("AdaLNZeroModulation.modulatePerToken matches gates modulation for uniform timestep (#687)")
    func adaLNModulatePerTokenMatchesGates() {
        let mod = AdaLNZeroModulation(dimensions: 16)
        let normed = MLXArray(Array(repeating: Float(0.5), count: 32), [1, 2, 16])
        let cond = MLXArray(Array(repeating: Float(0.25), count: 16), [1, 16])
        let t: Float = 0.35
        let g: Float = 3.0
        let gates = mod.gates(timestep: t, guidance: g, condVec: cond)
        let expected = normed * (1.0 + gates.scaleMSA) + gates.shiftMSA
        let perToken = mod.modulatePerToken(
            normed,
            tokenTimesteps: MLXArray([t, t], [1, 2]),
            guidance: g,
            condVec: cond
        )
        let diff = max(abs(expected - perToken))
        eval(diff)
        #expect(diff.item(Float.self) < 1e-4)
    }

    @Test("BabaiLLLCompensator.lllReduce does not trap on ill-conditioned basis (#688)")
    func babaiLLLHandlesIllConditionedBasis() {
        let basis: [Float] = [1.0, 1e12, 0.0, 1e-6]
        let result = BabaiLLLCompensator.lllReduce(basis: basis, dim: 2)
        #expect(result.dimension == 2)
        #expect(result.reducedBasis.count == 4)
    }

    @Test("ThreeBitQMV handles zero tasks and out-of-bounds expert index safely (#688)")
    func threeBitQMVZeroTasksAndOOBExpert() {
        let wDense = MLXArray.ones([2, 32, 64], dtype: .float16)
        let (wq, s, b) = MLX.quantized(wDense, groupSize: 64, bits: 3, mode: .affine)
        let emptyX = MLXArray.zeros([0, 1, 64], dtype: .float16)
        let emptyOut = ThreeBitQMV.matmul(emptyX, indices: nil, weight: wq[0], scales: s[0], biases: b![0], groupSize: 64)
        #expect(emptyOut.shape == [0, 1, 32])

        let x = MLXArray.ones([1, 1, 64], dtype: .float16)
        let oobIdx = MLXArray([Int32(-1)], [1])
        let oobOut = ThreeBitQMV.matmul(x, indices: oobIdx, weight: wq, scales: s, biases: b!, groupSize: 64)
        let maxAbs = max(abs(oobOut))
        eval(maxAbs)
        #expect(maxAbs.item(Float.self) == 0)
    }

    @Test("FusedRMSNormQuantizedLinear skips input_layernorm when extra self_attn projections exist (#687)")
    func fusedRMSNormSkipsExtraSelfAttnProjections() {
        final class FakeAttn: Module {
            @ModuleInfo(key: "q_proj") var q: QuantizedLinear
            @ModuleInfo(key: "k_proj") var k: QuantizedLinear
            @ModuleInfo(key: "v_proj") var v: QuantizedLinear
            @ModuleInfo(key: "o_proj") var o: QuantizedLinear
            @ModuleInfo(key: "g_proj") var g: Linear
            override init() {
                let dense = Linear(64, 64, bias: false)
                _q.wrappedValue = QuantizedLinear(dense, groupSize: 64, bits: 4, mode: .affine)
                _k.wrappedValue = QuantizedLinear(dense, groupSize: 64, bits: 4, mode: .affine)
                _v.wrappedValue = QuantizedLinear(dense, groupSize: 64, bits: 4, mode: .affine)
                _o.wrappedValue = QuantizedLinear(dense, groupSize: 64, bits: 4, mode: .affine)
                _g.wrappedValue = Linear(64, 64, bias: false)
                super.init()
            }
        }
        final class FakeLayer: Module {
            @ModuleInfo(key: "input_layernorm") var norm = RMSNorm(dimensions: 64)
            @ModuleInfo(key: "self_attn") var attn = FakeAttn()
        }
        final class FakeModel: Module {
            @ModuleInfo(key: "layer") var layer = FakeLayer()
        }
        let model = FakeModel()
        let fused = FusedRMSNormQuantizedLinear.install(in: model)
        #expect(fused == 0)
        #expect(type(of: model.layer.norm) == RMSNorm.self)
    }
}

@Suite("Round 4 audit CLI regression tests (#689)", .serialized)
struct Round4CLIRegressionTests {
    @Test("ComposeRecord.alive returns false for non-positive pid (#689)")
    func composeRecordRejectsNonPositivePID() {
        let zero = ComposeRecord(name: "test", pid: 0, port: 8080, file: "compose.yaml", started: nil)
        let negOne = ComposeRecord(name: "test", pid: -1, port: 8080, file: "compose.yaml", started: nil)
        #expect(!ComposeRecord.alive(zero))
        #expect(!ComposeRecord.alive(negOne))
    }

    @Test("ComposeAdapterSpec rejects dot-only output paths while ComposeFile accepts relative output (#689)")
    func composeAdapterOutputValidation() throws {
        let yamlValid = """
        version: 1
        name: test-sys
        models:
          m:
            source: mlx-community/Qwen2.5-0.5B-Instruct-4bit
        adapters:
          a:
            base: m
            corpus:
              source: ./data.jsonl
            output: ./adapters/my_lora
        pipeline:
          - m
        """
        let spec = try ComposeFile.parse(yamlValid, file: "/tmp/compose.yaml", context: ComposeContext().with(base: "/tmp/compose.yaml"))
        #expect(spec.adapters["a"]?.output == "/tmp/adapters/my_lora")

        let yamlDot = yamlValid.replacingOccurrences(of: "./adapters/my_lora", with: ".")
        #expect(throws: ServeConfigError.self) {
            _ = try ComposeFile.parse(yamlDot, file: "/tmp/compose.yaml", context: ComposeContext().with(base: "/tmp/compose.yaml"))
        }
    }

    @Test("SessionsCommand.List rejects negative --limit (#689)")
    func sessionsListRejectsNegativeLimit() {
        #expect(throws: (any Error).self) {
            _ = try SessionsCommand.List.parse(["--limit", "-1"])
        }
    }
}
