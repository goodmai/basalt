import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

struct ExpertTensor: Sendable, Equatable {
    let file: Int
    let offset: Int64
    let expertBytes: Int
    let shape: [Int]
    let dtype: DType
}

struct ExpertLayout: Sendable {
    let path: String
    let numExperts: Int
    let weight: ExpertTensor
    var scales: ExpertTensor?
    var biases: ExpertTensor?
    var bias: ExpertTensor?
    var globalScales: [Float]?
    var quantization: (groupSize: Int, bits: Int, mode: QuantizationMode)?

    var expertBytes: ByteCount {
        ByteCount([weight, scales, biases, bias].compactMap { $0?.expertBytes }.reduce(0, +))
    }

    var totalBytes: ByteCount { expertBytes * Int64(numExperts) }

    var dimensions: (input: Int, output: Int) {
        let columns = weight.shape.last ?? 0
        return (quantization.map { columns * 32 / $0.bits } ?? columns, weight.shape.first ?? 0)
    }
}

struct ExpertArrays {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let bias: MLXArray?
    let globalScale: MLXArray?

    func taking(_ slots: [Int]) -> ExpertArrays {
        let pick = MLXArray(slots.map(Int32.init))
        return ExpertArrays(
            weight: weight.take(pick, axis: 0), scales: scales?.take(pick, axis: 0),
            biases: biases?.take(pick, axis: 0), bias: bias?.take(pick, axis: 0),
            globalScale: globalScale?.take(pick, axis: 0))
    }
}

struct ModuleSlots: ~Copyable {
    var table: ExpertSlotTable
    let buffers: UniqueArray<SlotBuffer>
    let arrays: ExpertArrays?
    let expertGlobalScales: [Float]?
    var slotGlobalScales: [Float]
    private var _positions: UniqueArray<Int>

    init(layout: ExpertLayout, slots: Int) throws(BasaltError) {
        table = ExpertSlotTable(slots: slots)
        var initialPositions = UniqueArray<Int>()
        initialPositions.reserveCapacity(layout.numExperts)
        for _ in 0 ..< layout.numExperts { initialPositions.append(0) }
        _positions = initialPositions
        expertGlobalScales = layout.globalScales
        slotGlobalScales = layout.globalScales != nil && slots > 0 ? [Float](repeating: 1, count: slots) : []
        let tensors = [layout.weight, layout.scales, layout.biases, layout.bias]
        var buffers = UniqueArray<SlotBuffer>()
        buffers.reserveCapacity(tensors.count)
        var made: [MLXArray?] = []
        for tensor in tensors {
            guard let tensor, slots > 0 else {
                made.append(nil)
                continue
            }
            let buffer = try SlotBuffer(slots: slots, tensor: tensor)
            made.append(buffer.array)
            buffers.append(buffer)
        }
        self.buffers = buffers
        arrays = made[0].map { ExpertArrays(weight: $0, scales: made[1], biases: made[2], bias: made[3], globalScale: nil) }
    }

    var positions: UniqueArray<Int> {
        borrow { _positions }
        mutate { &_positions }
    }

    mutating func mapRouted(_ routed: [Int], ids: [Int], slots: [Int]?) -> [UInt32] {
        do {
            var span = _positions.mutableSpan
            if let slots {
                for (slot, id) in zip(slots, ids) { span[id] = slot }
            } else {
                for (rank, id) in ids.enumerated() { span[id] = rank }
            }
        }
        let span = _positions.span
        var mapped: [UInt32] = []
        mapped.reserveCapacity(routed.count)
        for id in routed { mapped.append(UInt32(span[id])) }
        return mapped
    }

    var slotBytes: Int {
        var total = 0
        for index in 0 ..< buffers.count { total += buffers[index].byteCount }
        return total
    }

    mutating func appendRequests(for loads: [ExpertSlotTable.Load], files: [FileHandle], into out: inout [ExpertReader.Request]) -> ByteCount {
        var bytes: Int64 = 0
        for load in loads {
            if let expertGlobalScales, load.slot < slotGlobalScales.count {
                slotGlobalScales[load.slot] = expertGlobalScales[load.expert]
            }
            for index in 0 ..< buffers.count {
                let tensor = buffers[index].tensor
                bytes += Int64(tensor.expertBytes)
                out.append(ExpertReader.Request(
                    descriptor: files[tensor.file].fileDescriptor,
                    offset: tensor.offset + Int64(load.expert * tensor.expertBytes),
                    destination: buffers[index].destination(slot: load.slot),
                    count: tensor.expertBytes))
            }
        }
        return ByteCount(bytes)
    }
}

struct LayerRouter {
    let scores: (MLXArray) -> MLXArray

    static func resolve(
        parent: String, modules: [String: Module], parameters: [String: MLXArray]
    ) -> (router: LayerRouter?, normWeight: MLXArray?) {
        let parts = parent.split(separator: ".")
        guard parts.count >= 3 else { return (nil, nil) }
        let moePath = parts.dropLast().joined(separator: ".")
        let layerPath = parts.dropLast(2).joined(separator: ".")
        let normNames = ["post_attention_layernorm", "ffn_norm", "pre_feedforward_layernorm_2", "norm"]
        let normEntry = normNames.lazy.compactMap { name -> (any UnaryLayer, MLXArray?)? in
            guard let norm = modules["\(layerPath).\(name)"] as? any UnaryLayer else { return nil }
            return (norm, parameters["\(layerPath).\(name).weight"])
        }.first
        if let gate = modules["\(moePath).gate"] as? Linear, let (norm, weight) = normEntry {
            let expertBias = parameters["\(moePath).expert_bias"]
            return (LayerRouter { hidden in
                let logits = gate(norm(hidden))
                guard let expertBias else { return logits }
                return MLX.sigmoid(logits.asType(.float32)) + expertBias.asType(.float32)
            }, weight)
        }
        if let gateWeight = parameters["\(moePath).gate.weight"], let (norm, weight) = normEntry {
            let bias = parameters["\(moePath).gate.e_score_correction_bias"]
            return (LayerRouter { hidden in
                let normed = norm(hidden)
                let logits = matmul(normed.asType(gateWeight.dtype), gateWeight.transposed(1, 0))
                guard let bias else { return logits }
                return MLX.sigmoid(logits.asType(.float32)) + bias.asType(.float32)
            }, weight)
        }
        if let proj = modules["\(layerPath).router.proj"] as? Linear {
            let scale = parameters["\(layerPath).router.scale"]
            return (LayerRouter { hidden in
                let rootSize = pow(Float(hidden.dim(-1)), -0.5)
                let normed = scale.map { MLXFast.rmsNorm(hidden, weight: ($0 * rootSize).asType(hidden.dtype), eps: 1e-6) } ?? hidden
                return proj(normed)
            }, normEntry?.1)
        }
        return (nil, normEntry?.1)
    }
}

final class LayerState {
    let group: Int
    let layer: Int
    let parent: String
    let modules: [Int]
    let entry: Int
    let exit: Int
    var router: LayerRouter?
    var normWeight: MLXArray?
    var gate = ExpertPrefetchGate(topK: 0)
    var lastPredicted: [Int] = []
    var prefillCounts: [Int: Int] = [:]
    var singleToken = false
    var inFlight: ExpertReader.InFlight?

    init(group: Int, parent: String, modules: [Int], layouts: [ExpertLayout]) {
        self.group = group
        self.parent = parent
        self.modules = modules
        let parts = parent.split(separator: ".")
        layer = parts.dropLast(2).last.flatMap { Int($0) } ?? group
        entry = modules.first { layouts[$0].path.hasSuffix(".up_proj") || layouts[$0].path.hasSuffix(".gate_up_proj") } ?? modules[0]
        exit = modules.first { layouts[$0].path.hasSuffix(".down_proj") } ?? modules[modules.count - 1]
    }

    static func layerNumber(of parent: String) -> Int? {
        parent.split(separator: ".").dropLast(2).last.flatMap { Int($0) }
    }

    static func partition(_ layouts: [ExpertLayout]) -> (groups: [LayerState], groupOfModule: [Int]) {
        var buckets: [String: [Int]] = [:]
        for (index, layout) in layouts.enumerated() {
            let parent = layout.path.split(separator: ".").dropLast().joined(separator: ".")
            buckets[parent, default: []].append(index)
        }
        let sorted = buckets.sorted {
            (layerNumber(of: $0.key) ?? .max, $0.key) < (layerNumber(of: $1.key) ?? .max, $1.key)
        }
        var mapping = [Int](repeating: 0, count: layouts.count)
        let groups = sorted.enumerated().map { offset, entry in
            for member in entry.value { mapping[member] = offset }
            return LayerState(group: offset, parent: entry.key, modules: entry.value, layouts: layouts)
        }
        return (groups, mapping)
    }
}

final class ExpertStore {
    let layouts: [ExpertLayout]
    private let files: [FileHandle]
    fileprivate var modules: UniqueArray<ModuleSlots>
    private let groups: [LayerState]
    private let groupOfModule: [Int]
    private var hasPrefill = false
    var warmStart = true
    var nextLayerPrefetch = true
    private(set) var stats = ExpertCacheStats()
    private(set) var routing = ExpertRouting()
    let capacity: ByteCount

    init(files urls: [URL], layouts: [ExpertLayout], capacity: ByteCount) throws {
        files = try urls.map { url in
            let handle = try FileHandle(forReadingFrom: url)
            _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
            return handle
        }
        self.layouts = layouts
        self.capacity = capacity
        let perExpert = layouts.reduce(ByteCount.zero) { $0 + $1.expertBytes }
        let slots = perExpert > .zero ? min(layouts.map(\.numExperts).min() ?? 0, Int(capacity.rawValue / perExpert.rawValue)) : 0
        var builtModules = UniqueArray<ModuleSlots>()
        builtModules.reserveCapacity(layouts.count)
        for layout in layouts { builtModules.append(try ModuleSlots(layout: layout, slots: slots)) }
        modules = builtModules
        (groups, groupOfModule) = LayerState.partition(layouts)
    }

    deinit {
        drainAllInFlight()
    }

    var expertBytes: ByteCount { layouts.reduce(.zero) { $0 + $1.totalBytes } }
    var slotBytes: ByteCount {
        var total: Int64 = 0
        for index in 0 ..< modules.count { total += Int64(modules[index].slotBytes) }
        return ByteCount(total)
    }
    var reserve: ByteCount { capacity + (layouts.map(\.totalBytes).max() ?? .zero) * 3 }
    var unclaimedReserve: ByteCount { max(.zero, reserve - slotBytes) }
    var expertCountsByLayer: [Int: Int] {
        Dictionary(groups.map { ($0.layer, layouts[$0.entry].numExperts) }, uniquingKeysWith: { first, _ in first })
    }

    func attachRouters(in model: Module) {
        let moduleMap = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let parameterMap = Dictionary(model.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        for state in groups {
            let resolved = LayerRouter.resolve(parent: state.parent, modules: moduleMap, parameters: parameterMap)
            (state.router, state.normWeight) = (resolved.router, resolved.normWeight)
        }
    }

    func resetStats() {
        drainAllInFlight()
        stats = ExpertCacheStats()
    }

    func slotArrays(module: Int) -> ExpertArrays? {
        guard let base = modules[module].arrays else { return nil }
        let scales = modules[module].slotGlobalScales
        guard !scales.isEmpty else { return base }
        return ExpertArrays(
            weight: base.weight, scales: base.scales, biases: base.biases, bias: base.bias,
            globalScale: MLXArray(scales, [scales.count]).asType(.float32))
    }

    func step(module: Int, x: MLXArray, indices: MLXArray) -> [Int] {
        let state = groups[groupOfModule[module]]
        guard module == state.entry else {
            state.inFlight.take()?.finish()
            return indices.asType(.int32).asArray(Int32.self).map(Int.init)
        }
        let single = x.size == x.dim(-1)
        state.singleToken = single
        guard single else {
            drainAllInFlight()
            let routed = indices.asType(.int32).asArray(Int32.self).map(Int.init)
            routing.record(module: state.layer, routed)
            for expert in routed { state.prefillCounts[expert, default: 0] += 1 }
            hasPrefill = true
            return routed
        }
        let pred = predictionArray(from: state, x: x, topK: indices.size)
        if let pred { eval(indices, pred) } else { eval(indices) }
        let routed = indices.asType(.int32).asArray(Int32.self).map(Int.init)
        let predicted = pred?.asArray(Int32.self).map(Int.init) ?? []
        if hasPrefill { warmFromPrefill() }
        state.inFlight.take()?.finish()
        routing.record(module: state.layer, routed)
        observePrediction(on: state, actual: routed)
        if !predicted.isEmpty { startPrefetch(group: state.group + 1, predicted: predicted) }
        return routed
    }

    func isSingleTokenExit(module: Int) -> Bool {
        let state = groups[groupOfModule[module]]
        return module == state.exit && state.singleToken
    }

    func slots(module: Int, _ ids: [Int]) -> [Int]? {
        let state = groups[groupOfModule[module]]
        state.inFlight.take()?.finish()
        guard let placement = modules[module].table.place(ids) else { return nil }
        var loadRequests = requests(module: module, placement.loads)
        for sibling in state.modules where sibling != module {
            if let ahead = modules[sibling].table.place(ids, prefetch: true) {
                loadRequests += requests(module: sibling, ahead.loads)
                stats.evictions += ahead.evictions
            }
        }
        ExpertReader.read(loadRequests)
        stats.hits += ids.count - placement.misses
        stats.misses += placement.misses
        stats.evictions += placement.evictions
        stats.prefetchHits += placement.prefetchHits
        return placement.slots
    }

    func warmFromPrefill() {
        hasPrefill = false
        var loadRequests: [ExpertReader.Request] = []
        for state in groups {
            defer { state.prefillCounts.removeAll(keepingCapacity: true) }
            guard warmStart, !state.prefillCounts.isEmpty else { continue }
            let ranked = state.prefillCounts.sorted { ($0.value, -$0.key) > ($1.value, -$1.key) }.map(\.key)
            for member in state.modules {
                let placement = modules[member].table.warm(ranked)
                loadRequests += requests(module: member, placement.loads)
                stats.warmed += placement.loads.count
                stats.evictions += placement.evictions
            }
        }
        ExpertReader.read(loadRequests)
    }

    private func predictionArray(from state: LayerState, x: MLXArray, topK: Int) -> MLXArray? {
        guard nextLayerPrefetch, topK > 0, state.group + 1 < groups.count else { return nil }
        let next = groups[state.group + 1]
        guard next.layer == state.layer + 1, let router = next.router, modules[next.entry].table.capacity >= topK * 2 else { return nil }
        var hidden = x.reshaped(1, x.dim(-1))
        if let weight = state.normWeight {
            hidden = MLX.where(MLX.abs(weight) .> 1e-6, hidden / weight.asType(hidden.dtype), hidden)
        }
        let scores = router.scores(hidden)
        let count = min(topK, scores.dim(-1))
        guard count > 0 else { return nil }
        return FusedMoERouterKernel.supports(scores, k: count) ? FusedMoERouterKernel.route(scores, k: count).indices : MLX.argSort(-scores, axis: -1)[0..., ..<count].asType(.int32)
    }

    private func observePrediction(on state: LayerState, actual: [Int]) {
        guard !state.lastPredicted.isEmpty else { return }
        if state.gate.scores.isEmpty { state.gate = ExpertPrefetchGate(topK: state.lastPredicted.count) }
        state.gate.observe(predicted: state.lastPredicted, actual: Set(actual))
        state.lastPredicted.removeAll(keepingCapacity: true)
    }

    private func startPrefetch(group: Int, predicted: [Int]) {
        let next = groups[group]
        next.inFlight.take()?.finish()
        if next.gate.scores.isEmpty { next.gate = ExpertPrefetchGate(topK: predicted.count) }
        next.lastPredicted = predicted
        let candidates = next.gate.select(predicted)
        guard !candidates.isEmpty else { return }
        var loadRequests: [ExpertReader.Request] = []
        for member in next.modules {
            guard let ahead = modules[member].table.place(candidates, prefetch: true, speculative: true) else { continue }
            loadRequests += requests(module: member, ahead.loads)
            stats.prefetched += ahead.loads.count
            stats.evictions += ahead.evictions
        }
        guard !loadRequests.isEmpty else { return }
        next.inFlight = ExpertReader.InFlight(loadRequests)
    }

    private func drainAllInFlight() {
        for state in groups {
            state.inFlight.take()?.finish()
            state.lastPredicted.removeAll(keepingCapacity: true)
        }
    }

    private func requests(module: Int, _ loads: [ExpertSlotTable.Load]) -> [ExpertReader.Request] {
        var batch: [ExpertReader.Request] = []
        stats.bytesRead += modules[module].appendRequests(for: loads, files: files, into: &batch)
        return batch
    }

    func read(module: Int, _ ids: [Int]) -> ExpertArrays {
        stats.misses += ids.count
        stats.evictions += ids.count
        let layout = layouts[module]
        let tensors = [layout.weight, layout.scales, layout.biases, layout.bias]
        let pointers = tensors.map { tensor in
            tensor.map { UnsafeMutableRawPointer.allocate(byteCount: SlotBuffer.padded(ids.count * $0.expertBytes), alignment: SlotBuffer.page) }
        }
        let loadRequests = zip(tensors, pointers).flatMap { tensor, pointer -> [ExpertReader.Request] in
            guard let tensor, let pointer else { return [] }
            stats.bytesRead += ByteCount(Int64(ids.count * tensor.expertBytes))
            return ids.enumerated().map { index, id in
                ExpertReader.Request(
                    descriptor: files[tensor.file].fileDescriptor, offset: tensor.offset + Int64(id * tensor.expertBytes),
                    destination: pointer + index * tensor.expertBytes, count: tensor.expertBytes)
            }
        }
        ExpertReader.read(loadRequests)
        let parts = zip(tensors, pointers).map { tensor, pointer -> MLXArray? in
            guard let tensor, let pointer else { return nil }
            let flat = MLXArray(rawPointer: pointer, [SlotBuffer.padded(ids.count * tensor.expertBytes)], dtype: .uint8) { pointer.deallocate() }
            return flat[0 ..< ids.count * tensor.expertBytes].view(dtype: tensor.dtype).reshaped([ids.count] + tensor.shape)
        }
        let gs = layout.globalScales.map { table in MLXArray(ids.map { table[$0] }, [ids.count]).asType(.float32) }
        return ExpertArrays(weight: parts[0]!, scales: parts[1], biases: parts[2], bias: parts[3], globalScale: gs)
    }
}

extension ExpertStore {
    func mapRouted(module: Int, _ routed: [Int], ids: [Int], slots: [Int]?) -> [UInt32] {
        modules[module].mapRouted(routed, ids: ids, slots: slots)
    }
}

final class StreamedSwitchLinear: SwitchLinear, UntracedModule {
    let store: ExpertStore
    let module: Int

    init(store: ExpertStore, module: Int) {
        self.store = store
        self.module = module
        let layout = store.layouts[module]
        let dimensions = layout.dimensions
        super.init(
            inputDims: dimensions.input, outputDims: dimensions.output, numExperts: layout.numExperts,
            weight: MLXArray.zeros([1], dtype: layout.weight.dtype))
        freeze()
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        let layout = store.layouts[module]
        let routed = store.step(module: module, x: x, indices: indices)
        let ids = Array(Set(routed)).sorted()
        let sorted = Self.residentKernel(sorted: sortedIndices, assignments: indices.size, experts: layout.numExperts, quantized: layout.quantization != nil)
        let output: MLXArray
        if let slots = store.slots(module: module, ids), let arrays = store.slotArrays(module: module) {
            if sorted, !zip(slots, slots.dropFirst()).allSatisfy(<) {
                let mapped = store.mapRouted(module: module, routed, ids: ids, slots: nil)
                output = project(x, arrays.taking(slots), MLXArray(mapped, indices.shape), sorted: true)
            } else {
                let mapped = store.mapRouted(module: module, routed, ids: ids, slots: slots)
                output = project(x, arrays, MLXArray(mapped, indices.shape), sorted: sorted)
            }
        } else {
            let mapped = store.mapRouted(module: module, routed, ids: ids, slots: nil)
            output = project(x, store.read(module: module, ids), MLXArray(mapped, indices.shape), sorted: sorted)
        }
        if store.isSingleTokenExit(module: module) { asyncEval(output) }
        return output
    }

    static func residentKernel(sorted: Bool, assignments: Int, experts: Int, quantized: Bool) -> Bool {
        quantized ? sorted && assignments / max(1, experts) >= 4 : sorted
    }

    private func project(_ x: MLXArray, _ experts: ExpertArrays, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        let quantization = experts.scales == nil ? nil : store.layouts[module].quantization
        var result =
            if let quantization, let scales = experts.scales {
                MLX.gatherQuantizedMM(
                    x, experts.weight, scales: scales, biases: experts.biases, rhsIndices: indices, transpose: true,
                    groupSize: quantization.groupSize, bits: quantization.bits, mode: quantization.mode,
                    globalScale: experts.globalScale, sortedIndices: sorted)
            } else {
                MLX.gatherMM(x, experts.weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
            }
        if let bias = experts.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return result
    }
}
