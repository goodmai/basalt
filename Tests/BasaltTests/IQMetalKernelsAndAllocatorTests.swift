import CoreAI
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("IQ specialized Metal kernels numerical parity and TinyModel admission (#408)", .serialized)
struct IQMetalKernelParityTests {
    static let iqTypes = IQMetalKernels.supportedTypes
    static let dtypes: [DType] = [.float32, .float16, .bfloat16]

    static func maxAbsError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        abs(actual.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
    }

    @Test("Specialized IQ matvec (M=1) and smallM (M=4) match GGUFDequant CPU reference across all 9 IQ formats",
          arguments: iqTypes, dtypes)
    func matvecAndSmallMParity(type: GGMLType, dtype: DType) throws {
        let (outputs, inputs) = (37, 1_280)
        let blockCount = outputs * inputs / type.block.elements
        let bytes = GGUFKernelTests.blocks(type, count: blockCount, seed: UInt64(type.rawValue) &+ 101, scale: 0.004)
        let packed = GGUFKernelTests.packed(bytes, rows: outputs)
        let cpuWeight = GGUFKernelTests.dense(bytes, type, rows: outputs, columns: inputs)
        let layer = GGUFLinear(type: type, outputs: outputs, inputs: inputs)
        try layer.update(parameters: ModuleParameters.unflattened(["weight": packed]), verify: .all)

        let x1 = (MLXRandom.normal([1, inputs], key: MLXRandom.key(UInt64(type.rawValue))) * 0.125).asType(dtype)
        let ref1 = matmul(x1.asType(.float32), cpuWeight.T)
        let mvOut = IQMetalKernels.matvec(x1, packed, type: type)
        let lin1 = layer(x1)
        eval(mvOut, lin1, ref1)
        #expect(mvOut.shape == [1, outputs] && mvOut.dtype == dtype)
        let relTol: Float = switch dtype {
        case .float32: 1e-5
        case .float16: 1e-3
        default: 8e-3
        }
        #expect(GGUFKernelTests.relativeError(mvOut, ref1) <= relTol, "\(type.name) M=1 dtype=\(dtype)")
        #expect(GGUFKernelTests.relativeError(lin1, ref1) <= relTol, "\(type.name) GGUFLinear M=1 dtype=\(dtype)")
        if dtype == .float32 {
            #expect(Self.maxAbsError(mvOut, ref1) <= 1e-4, "\(type.name) M=1 fp32 abs")
        }

        let x4 = (MLXRandom.normal([4, inputs], key: MLXRandom.key(UInt64(type.rawValue) &+ 17)) * 0.125).asType(dtype)
        let ref4 = matmul(x4.asType(.float32), cpuWeight.T)
        let smOut = IQMetalKernels.smallM(x4, packed, type: type)
        let lin4 = layer(x4.reshaped(1, 4, inputs))
        eval(smOut, lin4, ref4)
        #expect(smOut.shape == [4, outputs] && smOut.dtype == dtype)
        #expect(lin4.shape == [1, 4, outputs] && lin4.dtype == dtype)
        #expect(GGUFKernelTests.relativeError(smOut, ref4) <= relTol, "\(type.name) M=4 dtype=\(dtype)")
        #expect(GGUFKernelTests.relativeError(lin4.reshaped(4, outputs), ref4) <= relTol, "\(type.name) GGUFLinear M=4 dtype=\(dtype)")
        if dtype == .float32 {
            #expect(Self.maxAbsError(smOut, ref4) <= 1e-4, "\(type.name) M=4 fp32 abs")
        }
    }

    @Test("TinyModel qwen3 end-to-end IQ forward pass matches dense MLX.matmul logits and greedy token IDs (#228)")
    func tinyQwen3AdmissionParity() throws {
        let model = try TinyModel.make(.qwen3)
        var entries: [String: GGUFNativeLoader.Entry] = [:]
        var packedWeights: [(String, MLXArray)] = []
        var denseWeights: [(String, MLXArray)] = []
        let leaves = model.leafModules().flattened().filter { $0.1 is Linear || $0.1 is Embedding }
        for (index, (path, module)) in leaves.enumerated() {
            let weight = (module as? Linear)?.weight ?? (module as! Embedding).weight
            let (rows, columns) = (weight.dim(0), weight.dim(1))
            guard columns % 32 == 0 else { continue }
            let bytes = GGUFKernelTests.blocks(.iq4NL, count: rows * columns / 32, seed: UInt64(index + 50), scale: 0.002)
            entries[path] = GGUFNativeLoader.Entry(type: GGMLType.iq4NL.name, dtype: "\(weight.dtype)")
            packedWeights.append((path + ".weight", GGUFKernelTests.packed(bytes, rows: rows)))
            denseWeights.append((path + ".weight", GGUFKernelTests.dense(bytes, .iq4NL, rows: rows, columns: columns).asType(weight.dtype)))
        }
        #expect(entries.count >= 28)
        model.update(parameters: ModuleParameters.unflattened(denseWeights))
        eval(model)
        let prompt = Array(TinyModel.tokens("iq specialized metal decode").prefix(6))
        let promptArray = MLXArray(prompt.map(Int32.init))[.newAxis]
        let refLogits = model(promptArray, cache: nil).asType(.float32)
        let refGreedy = try CacheBlendQualityTests.greedy(prompt, count: 10, model: model)
        eval(refLogits)

        try GGUFNativeLoader.install(entries, in: model)
        try model.update(parameters: ModuleParameters.unflattened(packedWeights), verify: [.noUnusedKeys, .shapeMismatch])
        eval(model)
        let iqLogits = model(promptArray, cache: nil).asType(.float32)
        let iqGreedy = try CacheBlendQualityTests.greedy(prompt, count: 10, model: model)
        eval(iqLogits)

        #expect(Self.maxAbsError(iqLogits, refLogits) <= 1e-2)
        #expect(iqGreedy == refGreedy)
    }

    @Test("256-block IQ formats preserve two-layer MLP logits and greedy argmax against dense matmul",
          arguments: [GGMLType.iq4XS, .iq2XXS, .iq3XXS, .iq1S])
    func twoLayerBlock256Parity(type: GGMLType) throws {
        let (hidden, inter, vocab) = (256, 512, 128)
        let b1 = GGUFKernelTests.blocks(type, count: inter * hidden / 256, seed: 301, scale: 0.003)
        let b2 = GGUFKernelTests.blocks(type, count: vocab * inter / 256, seed: 302, scale: 0.003)
        let l1 = GGUFLinear(type: type, outputs: inter, inputs: hidden)
        let l2 = GGUFLinear(type: type, outputs: vocab, inputs: inter)
        try l1.update(parameters: ModuleParameters.unflattened(["weight": GGUFKernelTests.packed(b1, rows: inter)]), verify: .all)
        try l2.update(parameters: ModuleParameters.unflattened(["weight": GGUFKernelTests.packed(b2, rows: vocab)]), verify: .all)
        let w1 = GGUFKernelTests.dense(b1, type, rows: inter, columns: hidden)
        let w2 = GGUFKernelTests.dense(b2, type, rows: vocab, columns: inter)

        for rows in [1, 4] {
            let x = (MLXRandom.normal([rows, hidden], key: MLXRandom.key(UInt64(rows + 9))) * 0.25).asType(.float16)
            let iqOut = l2(MLXNN.silu(l1(x))).asType(.float32)
            let refH = MLXNN.silu(matmul(x.asType(.float32), w1.T).asType(.float16)).asType(.float32)
            let refOut = matmul(refH, w2.T)
            eval(iqOut, refOut)
            #expect(GGUFKernelTests.relativeError(iqOut, refOut) <= 2e-3)
            #expect(argMax(iqOut, axis: -1).asArray(Int32.self) == argMax(refOut, axis: -1).asArray(Int32.self))
        }
    }

    @Test("BASALT_BENCH=1 micro-benchmark comparing IQMetalKernels vs legacy GGUFKernels on 4096x14336")
    func envGatedMicroBenchmark() {
        guard ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1" else { return }
        let (outputs, inputs) = (14_336, 4_096)
        for type in [GGMLType.iq4XS, .iq4NL, .iq2XXS, .iq1M] {
            let bytes = GGUFKernelTests.blocks(type, count: outputs * inputs / type.block.elements, seed: 77, scale: 0.004)
            let packed = GGUFKernelTests.packed(bytes, rows: outputs)
            eval(packed)
            for m in [1, 4] {
                let xs = (0 ..< 16).map { MLXRandom.normal([m, inputs], key: MLXRandom.key(UInt64(m * 100 + $0))).asType(.float16) }
                eval(xs)
                let iqUs = Self.batchedMicroseconds(warmup: 3, runs: 8, batch: xs.count) {
                    xs.map { m == 1 ? IQMetalKernels.matvec($0, packed, type: type) : IQMetalKernels.smallM($0, packed, type: type) }
                }
                let legacyUs = Self.batchedMicroseconds(warmup: 3, runs: 8, batch: xs.count) {
                    xs.map { GGUFKernels.legacyMatvec($0, packed, type: type) }
                }
                let tileUs = Self.batchedMicroseconds(warmup: 2, runs: 4, batch: xs.count) {
                    xs.map { GGUFKernels.matmul($0, packed, type: type, columns: inputs) }
                }
                let weightBytes = Double(bytes.count + m * inputs * 2 + m * outputs * 2)
                let gbps = weightBytes / (iqUs * 1e-6) / 1e9
                print(String(format: "[IQBench] %@ M=%d iq=%.1fus (%.1f GB/s) legacy=%.1fus tile=%.1fus",
                             type.name, m, iqUs, gbps, legacyUs, tileUs))
                #expect(iqUs > 0 && legacyUs > 0 && tileUs > 0)
            }
        }
    }

    private static func batchedMicroseconds(warmup: Int, runs: Int, batch: Int, _ ops: () -> [MLXArray]) -> Double {
        let clock = ContinuousClock()
        for _ in 0 ..< warmup { eval(ops()) }
        var samples: [Double] = []
        for _ in 0 ..< runs {
            let elapsed = clock.measure { eval(ops()) }
            let us = (Double(elapsed.components.seconds) * 1e6 + Double(elapsed.components.attoseconds) * 1e-12) / Double(batch)
            samples.append(us)
        }
        samples.sort()
        return samples[samples.count / 2]
    }
}

@Suite("RCO Lagrangian sensitivity bit-budget allocator (#408)", .serialized)
struct RCOSensitivityAllocatorTests {
    @Test("IQBlockDescriptor covers all 9 IQ formats and 6 K-quant anchors with exact block geometry")
    func descriptorTableCompleteness() {
        #expect(IQBlockDescriptor.iqFormats.count == 9)
        #expect(IQBlockDescriptor.kQuantAnchors.count == 6)
        #expect(IQBlockDescriptor.all.count == 15)
        #expect(IQBlockDescriptor.iq1S.bitsPerWeight == 1.5625)
        #expect(IQBlockDescriptor.iq1M.bitsPerWeight == 1.75)
        #expect(IQBlockDescriptor.iq2XXS.bitsPerWeight == 2.0625)
        #expect(IQBlockDescriptor.iq4XS.bitsPerWeight == 4.25)
        #expect(IQBlockDescriptor.iq4NL.bitsPerWeight == 4.5)
        #expect(IQBlockDescriptor.q4K.bitsPerWeight == 4.5)
        #expect(IQBlockDescriptor.q8_0.bitsPerWeight == 8.5)
        #expect(IQBlockDescriptor.iqFormats.allSatisfy { $0.isIQ && $0.lutSizeBytes > 0 && $0.hasImportanceMatrixSupport })
        #expect(IQBlockDescriptor.kQuantAnchors.allSatisfy { !$0.isIQ && $0.lutSizeBytes == 0 })
    }

    @Test("RCOSensitivityAllocator enforces monotonicity, <0.05 bpw budget adherence, and sensitive-role floor")
    func monotonicityBudgetAndSensitiveFloor() {
        let layers = Self.syntheticTransformerProfile()
        let allocator = RCOSensitivityAllocator()
        for target in [2.75, 3.50, 4.25] {
            let config = RCOSensitivityAllocator.BudgetConfig(targetBPW: target, minFloorForSensitiveRoles: .iq4XS)
            let plan = allocator.allocate(layers: layers, config: config)
            #expect(plan.allocations.count == layers.count)
            #expect(plan.achievedBPW <= target + 1e-9)
            #expect(abs(plan.achievedBPW - target) < 0.05, "target=\(target) achieved=\(plan.achievedBPW)")
            Self.assertMonotonicity(layers: layers, allocations: plan.allocations)
            for alloc in plan.allocations where alloc.role.requiresSensitiveFloor {
                #expect(alloc.bitsPerWeight >= IQBlockDescriptor.iq4XS.bitsPerWeight)
            }
        }

        let strictQ4Floor = RCOSensitivityAllocator.BudgetConfig(targetBPW: 2.60, minFloorForSensitiveRoles: .q4K)
        let q4Plan = allocator.allocate(layers: layers, config: strictQ4Floor)
        for alloc in q4Plan.allocations where alloc.role.requiresSensitiveFloor {
            #expect(alloc.bitsPerWeight >= IQBlockDescriptor.q4K.bitsPerWeight)
        }
    }

    private static func syntheticTransformerProfile() -> [RCOSensitivityAllocator.LayerSensitivity] {
        var result: [RCOSensitivityAllocator.LayerSensitivity] = [
            .init(name: "model.embed_tokens", weightCount: 4_194_304, sensitivity: 4.5, role: .embedding, columns: 4_096),
            .init(name: "lm_head", weightCount: 4_194_304, sensitivity: 6.2, role: .lmHead, columns: 4_096),
        ]
        for idx in 0 ..< 16 {
            let depthCurve = 1.0 + 2.5 * exp(-Double(idx) / 4.0) + 0.8 * Double(idx == 15 ? 1 : 0)
            result.append(.init(
                name: "layers.\(idx).attn", weightCount: 4_194_304,
                sensitivity: depthCurve * 1.4, role: .attention, columns: 4_096))
            result.append(.init(
                name: "layers.\(idx).moe.shared", weightCount: 2_097_152,
                sensitivity: depthCurve * 1.8, role: .moeShared, columns: 4_096))
            result.append(.init(
                name: "layers.\(idx).moe.routed", weightCount: 16_777_216,
                sensitivity: depthCurve * 0.45 + Double(idx) * 0.01, role: .moeRouted, columns: 4_096))
            result.append(.init(
                name: "layers.\(idx).ffn", weightCount: 8_388_608,
                sensitivity: depthCurve * 0.85 + Double(idx) * 0.015, role: .ffn, columns: 4_096))
        }
        return result
    }

    private static func assertMonotonicity(
        layers: [RCOSensitivityAllocator.LayerSensitivity],
        allocations: [RCOSensitivityAllocator.LayerAllocation]
    ) {
        for i in layers.indices {
            for j in layers.indices where layers[i].role == layers[j].role && layers[i].columns == layers[j].columns {
                if layers[i].sensitivity > layers[j].sensitivity {
                    #expect(allocations[i].bitsPerWeight + 1e-12 >= allocations[j].bitsPerWeight,
                            "\(allocations[i].name) < \(allocations[j].name)")
                }
            }
        }
    }
}

@Suite("UnifiedMemoryBridge zero-copy page-aligned MLXArray and CoreAI NDArray bridge (#490)", .serialized)
struct UnifiedMemoryBridgeTests {
    @Test("Outbound MLXArray Span borrow and inbound 16KB page-aligned UnifiedPageBuffer are zero-copy")
    func zeroCopyOutboundSpanAndInboundPageBuffer() throws {
        let values = (0 ..< 1_024).map { Float($0) * 0.125 - 32.0 }
        let source = MLXArray(values, [16, 64])
        eval(source)
        let expectedBase = try #require(UnifiedMemoryBridge.baseAddress(of: source))
        let (borrowedBase, readBack) = UnifiedMemoryBridge.withContiguousSpan(of: source, as: Float.self) { span in
            let ptr = span.withUnsafeBufferPointer { $0.baseAddress.map(UnsafeRawPointer.init) }
            var out: [Float] = []
            out.reserveCapacity(span.count)
            for idx in 0 ..< span.count { out.append(span[idx]) }
            return (ptr, out)
        }
        #expect(borrowedBase == expectedBase)
        #expect(readBack == values)

        let pageBuffer = try UnifiedPageBuffer<Float>(shape: [8, 128])
        let rawAddr = pageBuffer.rawBaseAddress
        #expect(Int(bitPattern: rawAddr) % UnifiedMemoryBridge.pageSize == 0)
        #expect(pageBuffer.allocatedByteCount >= UnifiedMemoryBridge.pageSize)
        pageBuffer.withMutableSpan { span in
            for idx in 0 ..< span.count { span[idx] = values[idx] * 2.0 }
        }
        let imported = pageBuffer.intoMLXArray()
        eval(imported)
        #expect(imported.shape == [8, 128] && imported.dtype == .float32)
        #expect(UnifiedMemoryBridge.baseAddress(of: imported) == rawAddr)
        #expect(imported.asArray(Float.self) == values.map { $0 * 2.0 })
    }

    @Test("UnifiedMemoryBridge and CoreAIVisionTower planar NDArray bridges preserve exact Float16 values")
    func coreAIPlanarRoundtrip() throws {
        let ffnValues = (0 ..< (5 * 64)).map { Float16(Float($0 % 29) * 0.25 - 3.5) }
        let ffnInput = MLXArray(ffnValues, [1, 5, 64])
        eval(ffnInput)
        let ffnPlanar = UnifiedMemoryBridge.exportPlanarNDArray(
            from: ffnInput.reshaped([5, 64]).transposed(), shape: [1, 64, 1, 5], scalarType: .float16, as: Float16.self
        )
        #expect(ffnPlanar.shape == [1, 64, 1, 5])
        let ffnRestored = UnifiedMemoryBridge.importPlanarNDArray(
            ffnPlanar, rows: 64, columns: 5, as: Float16.self
        ).transposed().reshaped([1, 5, 64])
        eval(ffnRestored)
        #expect(ffnRestored.shape == [1, 5, 64] && ffnRestored.dtype == .float16)
        #expect(ffnRestored.asArray(Float16.self) == ffnValues)

        let visValues = (0 ..< (6 * 32)).map { Float16(Float($0 % 23) * 0.5 - 5.0) }
        let visRows = MLXArray(visValues, [6, 32])
        eval(visRows)
        let visPlanar = try CoreAIVisionTower.planar(visRows, length: 8)
        #expect(visPlanar.shape == [1, 32, 1, 8])
        let visRestored = try CoreAIVisionTower.rows(visPlanar, count: 6)
        eval(visRestored)
        #expect(visRestored.shape == [6, 32] && visRestored.dtype == .float16)
        #expect(visRestored.asArray(Float16.self) == visValues)
    }
}
