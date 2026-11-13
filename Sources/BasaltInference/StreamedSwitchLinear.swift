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

    func taking(_ slots: [Int]) -> ExpertArrays {
        let pick = MLXArray(slots.map(Int32.init))
        return ExpertArrays(weight: weight.take(pick, axis: 0), scales: scales?.take(pick, axis: 0), biases: biases?.take(pick, axis: 0), bias: bias?.take(pick, axis: 0))
    }
}

struct ModuleSlots {
    var table: ExpertSlotTable
    let buffers: [(tensor: ExpertTensor, buffer: SlotBuffer)]
    let arrays: ExpertArrays?

    init(layout: ExpertLayout, slots: Int) throws(BasaltError) {
        table = ExpertSlotTable(slots: slots)
        let tensors = [layout.weight, layout.scales, layout.biases, layout.bias]
        var buffers: [(ExpertTensor, SlotBuffer)] = []
        var made: [MLXArray?] = []
        for tensor in tensors {
            guard let tensor, slots > 0 else {
                made.append(nil)
                continue
            }
            let buffer = try SlotBuffer(slots: slots, tensor: tensor)
            buffers.append((tensor, buffer))
            made.append(buffer.array)
        }
        self.buffers = buffers
        arrays = made[0].map { ExpertArrays(weight: $0, scales: made[1], biases: made[2], bias: made[3]) }
    }
}

final class ExpertStore {
    let layouts: [ExpertLayout]
    private let files: [FileHandle]
    private var modules: [ModuleSlots]
    private let siblings: [[Int]]
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
        modules = try layouts.map { try ModuleSlots(layout: $0, slots: slots) }
        let parents = layouts.map { $0.path.split(separator: ".").dropLast().joined(separator: ".") }
        siblings = parents.map { parent in parents.indices.filter { parents[$0] == parent } }
    }

    var expertBytes: ByteCount { layouts.reduce(.zero) { $0 + $1.totalBytes } }
    var slotBytes: ByteCount { ByteCount(Int64(modules.reduce(0) { $0 + $1.buffers.reduce(0) { $0 + $1.buffer.stride * $1.buffer.array.dim(0) } })) }
    var reserve: ByteCount { capacity + (layouts.map(\.totalBytes).max() ?? .zero) * 3 }
    var unclaimedReserve: ByteCount { max(.zero, reserve - slotBytes) }

    func slotArrays(module: Int) -> ExpertArrays? { modules[module].arrays }

    func slots(module: Int, _ ids: [Int]) -> [Int]? {
        routing.record(module: module, ids)
        guard let placement = modules[module].table.place(ids) else { return nil }
        var requests = requests(module: module, placement.loads)
        for sibling in siblings[module] where sibling != module {
            if let ahead = modules[sibling].table.place(ids, prefetch: true) {
                requests += self.requests(module: sibling, ahead.loads)
                stats.evictions += ahead.evictions
            }
        }
        ExpertReader.read(requests)
        stats.hits += ids.count - placement.misses
        stats.misses += placement.misses
        stats.evictions += placement.evictions
        return placement.slots
    }

    private func requests(module: Int, _ loads: [ExpertSlotTable.Load]) -> [ExpertReader.Request] {
        loads.flatMap { load in
            modules[module].buffers.map { tensor, buffer in
                stats.bytesRead += ByteCount(Int64(tensor.expertBytes))
                return ExpertReader.Request(
                    descriptor: files[tensor.file].fileDescriptor, offset: tensor.offset + Int64(load.expert * tensor.expertBytes),
                    destination: buffer.destination(slot: load.slot), count: tensor.expertBytes)
            }
        }
    }

    func read(module: Int, _ ids: [Int]) -> ExpertArrays {
        stats.misses += ids.count
        stats.evictions += ids.count
        let layout = layouts[module]
        let tensors = [layout.weight, layout.scales, layout.biases, layout.bias]
        let pointers = tensors.map { tensor in
            tensor.map { UnsafeMutableRawPointer.allocate(byteCount: SlotBuffer.padded(ids.count * $0.expertBytes), alignment: SlotBuffer.page) }
        }
        let requests = zip(tensors, pointers).flatMap { tensor, pointer -> [ExpertReader.Request] in
            guard let tensor, let pointer else { return [] }
            stats.bytesRead += ByteCount(Int64(ids.count * tensor.expertBytes))
            return ids.enumerated().map { index, id in
                ExpertReader.Request(
                    descriptor: files[tensor.file].fileDescriptor, offset: tensor.offset + Int64(id * tensor.expertBytes),
                    destination: pointer + index * tensor.expertBytes, count: tensor.expertBytes)
            }
        }
        ExpertReader.read(requests)
        let parts = zip(tensors, pointers).map { tensor, pointer -> MLXArray? in
            guard let tensor, let pointer else { return nil }
            let flat = MLXArray(rawPointer: pointer, [SlotBuffer.padded(ids.count * tensor.expertBytes)], dtype: .uint8) { pointer.deallocate() }
            return flat[0 ..< ids.count * tensor.expertBytes].view(dtype: tensor.dtype).reshaped([ids.count] + tensor.shape)
        }
        return ExpertArrays(weight: parts[0]!, scales: parts[1], biases: parts[2], bias: parts[3])
    }
}

final class StreamedSwitchLinear: SwitchLinear {
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
        let routed = indices.asType(.int32).asArray(Int32.self).map(Int.init)
        let ids = Array(Set(routed)).sorted()
        let sorted = Self.residentKernel(sorted: sortedIndices, assignments: indices.size, experts: layout.numExperts, quantized: layout.quantization != nil)
        var position = [Int](repeating: 0, count: layout.numExperts)
        guard let slots = store.slots(module: module, ids), let arrays = store.slotArrays(module: module) else {
            for (rank, id) in ids.enumerated() { position[id] = rank }
            return project(x, store.read(module: module, ids), MLXArray(routed.map { UInt32(position[$0]) }, indices.shape), sorted: sorted)
        }
        guard !sorted || zip(slots, slots.dropFirst()).allSatisfy(<) else {
            for (rank, id) in ids.enumerated() { position[id] = rank }
            return project(x, arrays.taking(slots), MLXArray(routed.map { UInt32(position[$0]) }, indices.shape), sorted: true)
        }
        for (slot, id) in zip(slots, ids) { position[id] = slot }
        return project(x, arrays, MLXArray(routed.map { UInt32(position[$0]) }, indices.shape), sorted: sorted)
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
                    groupSize: quantization.groupSize, bits: quantization.bits, mode: quantization.mode, sortedIndices: sorted)
            } else {
                MLX.gatherMM(x, experts.weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
            }
        if let bias = experts.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return result
    }
}
