import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("GGUF matvec against MLX affine quantizedMatmul on Qwen3.8-27B FFN shapes (#408 phase 2 gate)", .serialized)
struct GGUFKernelBenchmarkTests {
    static let enabled = ProcessInfo.processInfo.environment["BASALT_E2E_GGUF_KERNEL_BENCH"] == "1"
    static let shapes = ProcessInfo.processInfo.environment["GGUF_TYPES"] == nil ? [(17408, 5120), (5120, 17408)] : [(17408, 5120)]

    static func time(_ layer: Module & UnaryLayer, inputs: Int, iterations: Int = 20, batch: Int = 32) -> Double {
        let xs = (0 ..< batch).map { MLXRandom.normal([1, 1, inputs], key: MLXRandom.key(UInt64($0))).asType(.bfloat16) }
        eval(xs)
        for _ in 0 ..< 3 { eval(xs.map { layer($0) }) }
        let start = ContinuousClock.now
        for _ in 0 ..< iterations { eval(xs.map { layer($0) }) }
        let elapsed = ContinuousClock.now - start
        let calls = Double(iterations * batch)
        return (Double(elapsed.components.seconds) * 1e6 + Double(elapsed.components.attoseconds) / 1e12) / calls
    }

    @Test("Microseconds per matvec and effective weight bandwidth", .enabled(if: enabled, "set BASALT_E2E_GGUF_KERNEL_BENCH=1"))
    func matvec() throws {
        for (outputs, inputs) in Self.shapes {
            for bits in [3, 4] where ProcessInfo.processInfo.environment["GGUF_TYPES"] == nil || bits == 3 {
                let dense = MLXRandom.normal([outputs, inputs], scale: 0.02).asType(.bfloat16)
                let layer = QuantizedLinear(weight: dense, bias: nil, groupSize: 64, bits: bits)
                eval(layer)
                let micros = Self.time(layer, inputs: inputs)
                let bytes = Double(layer.weight.nbytes + layer.scales.nbytes + (layer.biases?.nbytes ?? 0))
                print(String(format: "BENCH %5d×%5d affine%d  %8.1f µs  %6.1f GB/s", outputs, inputs, bits, micros, bytes / micros / 1e3))
            }
            for type in (ProcessInfo.processInfo.environment["GGUF_TYPES"] ?? "iq3S,iq3XXS,iq2XXS,iq2S,iq4XS,q4K").split(separator: ",").compactMap({ GGMLType(name: String($0)) }) {
                let bytes = GGUFKernelTests.blocks(type, count: outputs * inputs / type.block.elements, seed: 1)
                let layer = GGUFLinear(type: type, outputs: outputs, inputs: inputs)
                try layer.update(parameters: ModuleParameters.unflattened(["weight": GGUFKernelTests.packed(bytes, rows: outputs)]), verify: .all)
                eval(layer)
                let micros = Self.time(layer, inputs: inputs)
                print(String(format: "BENCH %5d×%5d %-7@ %8.1f µs  %6.1f GB/s", outputs, inputs, type.name, micros, Double(bytes.count) / micros / 1e3))
            }
        }
    }
}
