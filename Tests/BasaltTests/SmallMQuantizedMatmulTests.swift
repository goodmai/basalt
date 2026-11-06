import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltInference
@testable import BasaltCLI

@Suite("Small-M 4-bit affine simdgroup-matrix kernel against dequantize → matmul and stock quantizedMatmul (#208)")
struct SmallMQuantizedMatmulTests {
    static func layer(inputs: Int, outputs: Int, groupSize: Int, dtype: DType, bias: Bool = false) -> QuantizedLinear {
        let key = MLXRandom.key(UInt64(inputs * 7 + outputs * 3 + groupSize))
        let dense = MLXRandom.normal([outputs, inputs], scale: 0.02, key: key).asType(dtype)
        let offset = bias ? MLXRandom.normal([outputs], key: key).asType(dtype) : nil
        let layer = QuantizedLinear(weight: dense, bias: offset, groupSize: groupSize, bits: 4)
        eval(layer)
        return layer
    }

    static func input(rows: Int, inputs: Int, dtype: DType) -> MLXArray {
        MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(UInt64(rows))).asType(dtype)
    }

    static func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && (a .== b).all().item(Bool.self)
    }

    static func relativeError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        let reference = reference.asType(.float32)
        return (abs(actual.asType(.float32) - reference).max() / abs(reference).max()).item(Float.self)
    }

    static func tolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .bfloat16: 1e-2
        case .float16: 2e-3
        default: 1e-4
        }
    }

    @Test("Rows 4...8 match dequantize → matmul within the activation dtype's tolerance and stock within rounding",
          arguments: [32, 64, 128], [DType.bfloat16, .float16, .float32])
    func matchesReference(groupSize: Int, dtype: DType) {
        let stock = Self.layer(inputs: 1024, outputs: 96, groupSize: groupSize, dtype: dtype, bias: true)
        let kernel = SmallMQuantizedLinear(replacing: stock)
        let dense = dequantized(stock.weight, scales: stock.scales, biases: stock.biases, groupSize: groupSize, bits: 4)
        for rows in SmallMQuantizedMatmul.rows {
            let x = Self.input(rows: rows, inputs: 1024, dtype: dtype)
            #expect(SmallMQuantizedMatmul.handles(x, layer: kernel))
            let actual = kernel(x)
            #expect(actual.shape == [1, rows, 96] && actual.dtype == dtype)
            let reference = matmul(x.asType(.float32), dense.asType(.float32).T) + stock.bias!.asType(.float32)
            let error = Self.relativeError(actual, reference)
            #expect(error < Self.tolerance(dtype), "rows \(rows) error \(error)")
            #expect(Self.relativeError(actual, stock(x)) < Self.tolerance(dtype), "rows \(rows) against stock")
        }
    }

    @Test("Rows outside 4...8, a width that is not a multiple of 64, 8-bit and mixed dtypes stay on the stock path")
    func outsideContractFallsBack() {
        let wide = Self.layer(inputs: 512, outputs: 32, groupSize: 64, dtype: .bfloat16)
        let kernel = SmallMQuantizedLinear(replacing: wide)
        for rows in [1, 2, 3, 9, 16] {
            let x = Self.input(rows: rows, inputs: 512, dtype: .bfloat16)
            #expect(!SmallMQuantizedMatmul.handles(x, layer: kernel))
            #expect(Self.identical(kernel(x), wide(x)))
        }
        #expect(!SmallMQuantizedMatmul.handles(Self.input(rows: 4, inputs: 512, dtype: .float16), layer: kernel))
        #expect(!SmallMQuantizedMatmul.fits(Self.layer(inputs: 96, outputs: 32, groupSize: 32, dtype: .bfloat16)))
        #expect(!SmallMQuantizedMatmul.fits(Self.layer(inputs: 512, outputs: 24, groupSize: 64, dtype: .bfloat16)))
        #expect(!SmallMQuantizedMatmul.fits(QuantizedLinear(weight: MLXRandom.normal([32, 512]), bias: nil, groupSize: 64, bits: 8)))
    }

    @Test("Off unless --experimental-smallm-kernel is given")
    func flag() throws {
        #expect(try ServeCommand.parse([]).experimental.smallMKernel == false)
        #expect(try ServeCommand.parse(["--experimental-smallm-kernel"]).experimental.smallMKernel)
    }

    @Test("A quantized tiny Qwen3 with the kernel installed keeps the stock verify-window logits within bf16 tolerance and the same argmax")
    func tinyModelParity() throws {
        let model = try TinyModel.make(.qwen3, adding: "\"hidden_size\": 512, \"intermediate_size\": 1024, ")
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        let window = MLXArray([3, 141, 59, 26, 5, 35].map(Int32.init)).reshaped(1, -1)
        let stock = model(window, cache: nil)
        eval(stock)
        let installed = SmallMQuantizedLinear.install(in: model)
        #expect(installed == model.leafModules().flattened().filter { $0.1 is SmallMQuantizedLinear }.count)
        #expect(installed == 4 * 7 + 1)
        let actual = model(window, cache: nil)
        let error = Self.relativeError(actual, stock)
        #expect(error < 2e-2, "relative max error \(error)")
        #expect(Self.identical(actual.argMax(axis: -1), stock.argMax(axis: -1)))
    }
}
