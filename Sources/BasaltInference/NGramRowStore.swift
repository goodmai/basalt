import BasaltCore
import Darwin
import Foundation
import MLX
import MLXNN

public struct NGramQuantization: Sendable, Equatable, Codable {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public static let affine4g32 = NGramQuantization(groupSize: 32, bits: 4, mode: .affine)

    public init(groupSize: Int = 32, bits: Int = 4, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        let rawMode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
        mode = rawMode == "mxfp4" ? .mxfp4 : .affine
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(groupSize, forKey: .groupSize)
        try container.encode(bits, forKey: .bits)
        try container.encode(mode == .mxfp4 ? "mxfp4" : "affine", forKey: .mode)
    }
}

public struct NGramRowStoreConfig: Sendable, Equatable {
    public var tablePrefix: String?
    public var quantization: NGramQuantization?
    public var maxCacheBytes: ByteCount
    public var maxCachedRows: Int?
    public var layerID: Int

    public init(
        tablePrefix: String? = nil,
        quantization: NGramQuantization? = nil,
        maxCacheBytes: ByteCount = ModelFitAnalyzer.defaultNGramRowCacheBytes,
        maxCachedRows: Int? = nil,
        layerID: Int = 2
    ) {
        self.tablePrefix = tablePrefix
        self.quantization = quantization
        self.maxCacheBytes = max(.zero, maxCacheBytes)
        self.maxCachedRows = maxCachedRows
        self.layerID = layerID
    }
}

public struct NGramRowCacheStats: Sendable, Equatable, Codable {
    public var hits = 0
    public var misses = 0
    public var evictions = 0
    public var bytesRead = ByteCount.zero
    public var cachedRows = 0
    public var capacityRows = 0

    public init() {}

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }
}

struct OwnedFileDescriptor: ~Copyable {
    private var rawDescriptor: Int32

    init(opening url: URL, noCache: Bool = true) throws(BasaltError) {
        let resolved = url.resolvingSymlinksInPath().path
        let fd = Darwin.open(resolved, O_RDONLY)
        guard fd >= 0 else {
            throw .inferenceError("ngram row store: cannot open \(url.lastPathComponent): \(String(cString: strerror(errno)))")
        }
        if noCache {
            _ = Darwin.fcntl(fd, F_NOCACHE, 1)
        }
        rawDescriptor = fd
    }

    var descriptor: Int32 { rawDescriptor }

    consuming func close() {
        let fd = rawDescriptor
        rawDescriptor = -1
        if fd >= 0 { _ = Darwin.close(fd) }
    }

    deinit {
        if rawDescriptor >= 0 { _ = Darwin.close(rawDescriptor) }
    }
}

public final class NGramRowStore {
    public let index: NGramTableIndex
    public let layerID: Int
    public private(set) var stats = NGramRowCacheStats()
    private var descriptors: UniqueArray<OwnedFileDescriptor>
    private var cache: NGramRowLRUCache
    private var isClosed = false

    public convenience init(
        directory: URL,
        config: NGramRowStoreConfig = NGramRowStoreConfig()
    ) throws(BasaltError) {
        let urls = ModelFitAnalyzer.safetensorsShards(in: directory)
        guard !urls.isEmpty else {
            throw .inferenceError("ngram row store: no safetensors shards in \(directory.lastPathComponent)")
        }
        try self.init(shards: urls, config: config)
    }

    public init(
        shards urls: [URL],
        config: NGramRowStoreConfig = NGramRowStoreConfig()
    ) throws(BasaltError) {
        guard !urls.isEmpty else {
            throw .inferenceError("ngram row store: shard list is empty")
        }
        var parsedShards: [SafetensorsShard] = []
        parsedShards.reserveCapacity(urls.count)
        for url in urls {
            parsedShards.append(try SafetensorsShard(url: url))
        }
        let tableIndex = try Self.buildIndex(from: parsedShards, config: config)
        var opened = UniqueArray<OwnedFileDescriptor>()
        opened.reserveCapacity(urls.count)
        for url in urls {
            opened.append(try OwnedFileDescriptor(opening: url, noCache: true))
        }
        let maxByBytes = Int(min(Int64(Int.max), config.maxCacheBytes.rawValue / Int64(max(1, tableIndex.rowTotalBytes))))
        let capacity = config.maxCachedRows.map { max(0, min($0, tableIndex.totalRows)) }
            ?? max(0, min(maxByBytes, tableIndex.totalRows))
        self.index = tableIndex
        self.layerID = config.layerID
        self.descriptors = opened
        self.cache = NGramRowLRUCache(capacity: capacity, rowBytes: tableIndex.rowTotalBytes)
        self.stats.capacityRows = capacity
    }

    public func close() {
        guard !isClosed else { return }
        isClosed = true
        while !descriptors.isEmpty {
            let handle = descriptors.removeLast()
            handle.close()
        }
    }

    public func lookup(_ indices: [Int]) throws(BasaltError) -> MLXArray {
        guard !isClosed else {
            throw .inferenceError("ngram row store: store is closed")
        }
        guard !indices.isEmpty else {
            return MLXArray.zeros([0, index.embedDim], dtype: index.outputDType)
        }
        try validateIndices(indices)
        let (uniqueRows, gatherMap) = Self.deduplicate(indices)
        let uniqueOutput = try fetchUnique(uniqueRows, totalRequested: indices.count)
        guard uniqueRows.count < indices.count else { return uniqueOutput }
        return uniqueOutput.take(MLXArray(gatherMap), axis: 0)
    }

    public func lookup(_ indices: MLXArray) throws(BasaltError) -> MLXArray {
        guard !isClosed else {
            throw .inferenceError("ngram row store: store is closed")
        }
        guard !indices.dtype.isFloatingPoint, indices.dtype != .bool, indices.dtype != .complex64 else {
            throw .inferenceError("ngram row store: indices must have integer dtype, got \(indices.dtype)")
        }
        let originalShape = indices.shape
        guard indices.size > 0 else {
            return MLXArray.zeros(originalShape + [index.embedDim], dtype: index.outputDType)
        }
        let flat = indices.flattened().asType(.int64)
        eval(flat)
        let rawIDs = flat.asArray(Int64.self)
        var rowIDs: [Int] = []
        rowIDs.reserveCapacity(rawIDs.count)
        for value in rawIDs {
            guard value >= 0, value < Int64(index.totalRows) else {
                throw .inferenceError("ngram row store: index \(value) out of bounds 0..<\(index.totalRows)")
            }
            rowIDs.append(Int(value))
        }
        let rows = try lookup(rowIDs)
        return rows.reshaped(originalShape + [index.embedDim])
    }

    public func prefetch(_ indices: [Int]) throws(BasaltError) {
        guard !isClosed else {
            throw .inferenceError("ngram row store: store is closed")
        }
        try validateIndices(indices)
        guard !indices.isEmpty, cache.capacity > 0 else { return }
        let (uniqueRows, _) = Self.deduplicate(indices)
        let target = Array(uniqueRows.suffix(cache.capacity))
        var uncached: [Int] = []
        for row in target {
            if cache.touch(row) == nil { uncached.append(row) }
        }
        guard !uncached.isEmpty else { return }
        try loadRowsIntoCacheOnly(uncached)
    }

    public func resetStats() {
        stats = NGramRowCacheStats()
        stats.cachedRows = cache.count
        stats.capacityRows = cache.capacity
    }

    private func validateIndices(_ indices: [Int]) throws(BasaltError) {
        for row in indices {
            guard row >= 0, row < index.totalRows else {
                throw .inferenceError("ngram row store: index \(row) out of bounds 0..<\(index.totalRows)")
            }
        }
    }

    public static func droppedResidentKeys(at urls: [URL]) throws(BasaltError) -> [Int: Set<String>] {
        var shards: [SafetensorsShard] = []
        shards.reserveCapacity(urls.count)
        for url in urls {
            shards.append(try SafetensorsShard(url: url))
        }
        return droppedResidentKeys(in: shards)
    }

    static func droppedResidentKeys(in shards: [SafetensorsShard]) -> [Int: Set<String>] {
        var dropped: [Int: Set<String>] = [:]
        for (index, shard) in shards.enumerated() {
            let matches = Set(shard.entries.keys.filter { ModelFitAnalyzer.isSSDResidentPLEKey($0) })
            if !matches.isEmpty { dropped[index] = matches }
        }
        return dropped
    }

    public static func residentHeaderDroppingPLE(at shardURL: URL) throws(BasaltError) -> Data {
        let shard = try SafetensorsShard(url: shardURL)
        let dropped = Set(shard.entries.keys.filter { ModelFitAnalyzer.isSSDResidentPLEKey($0) })
        return try shard.header(placeholders: [:], dropping: dropped)
    }

    public static func writeResidentShardDroppingPLE(
        from sourceURL: URL,
        to destinationURL: URL
    ) throws(BasaltError) -> Set<String> {
        let shard = try SafetensorsShard(url: sourceURL)
        let dropped = Set(shard.entries.keys.filter { ModelFitAnalyzer.isSSDResidentPLEKey($0) })
        let header = try shard.header(placeholders: [:], dropping: dropped)
        let files = FileManager.default
        do {
            if files.fileExists(atPath: destinationURL.path) {
                try files.removeItem(at: destinationURL)
            }
            try files.copyItem(at: sourceURL.resolvingSymlinksInPath(), to: destinationURL)
            let handle = try FileHandle(forWritingTo: destinationURL)
            defer { try? handle.close() }
            try handle.seek(toOffset: 8)
            try handle.write(contentsOf: header)
        } catch {
            try? files.removeItem(at: destinationURL)
            throw .inferenceError("ngram row store: failed writing resident shard \(destinationURL.lastPathComponent): \(error.localizedDescription)")
        }
        return dropped
    }
}

extension NGramRowStore {
    fileprivate struct BatchLayout {
        let uniqueCount: Int
        let weightBytes: Int
        let scalesOffset: Int
        let scalesBytes: Int
        let biasesOffset: Int
        let biasesBytes: Int
        let totalBytes: Int

        init(uniqueCount: Int, index: NGramTableIndex) throws(BasaltError) {
            let (weightTotal, ov1) = uniqueCount.multipliedReportingOverflow(by: index.rowWeightBytes)
            let (scalesTotal, ov2) = uniqueCount.multipliedReportingOverflow(by: index.rowScalesBytes)
            let (biasesTotal, ov3) = uniqueCount.multipliedReportingOverflow(by: index.rowBiasesBytes)
            let (ws, ov4) = weightTotal.addingReportingOverflow(scalesTotal)
            let (total, ov5) = ws.addingReportingOverflow(biasesTotal)
            guard !ov1, !ov2, !ov3, !ov4, !ov5 else {
                throw .inferenceError("ngram row store: batch byte size overflow for \(uniqueCount) rows")
            }
            self.uniqueCount = uniqueCount
            self.weightBytes = weightTotal
            self.scalesOffset = weightTotal
            self.scalesBytes = scalesTotal
            self.biasesOffset = ws
            self.biasesBytes = biasesTotal
            self.totalBytes = total
        }
    }

    private func fetchUnique(_ uniqueRows: [Int], totalRequested: Int) throws(BasaltError) -> MLXArray {
        let layout = try BatchLayout(uniqueCount: uniqueRows.count, index: index)
        var scratch = [UInt8](repeating: 0, count: layout.totalBytes)
        var missed: [(position: Int, rowID: Int)] = []
        var readBytes: Int64 = 0

        try populateScratch(
            uniqueRows: uniqueRows, layout: layout, scratch: &scratch,
            missed: &missed, readBytes: &readBytes
        )
        insertMissedIntoCache(missed: missed, layout: layout, scratch: scratch)
        stats.misses += missed.count
        stats.hits += totalRequested - missed.count
        stats.bytesRead += ByteCount(readBytes)
        stats.cachedRows = cache.count
        return decodeBatch(scratch: scratch, layout: layout)
    }

    private func loadRowsIntoCacheOnly(_ rowIDs: [Int]) throws(BasaltError) {
        let layout = try BatchLayout(uniqueCount: rowIDs.count, index: index)
        var scratch = [UInt8](repeating: 0, count: layout.totalBytes)
        var resolved: [(position: Int, rowID: Int, location: NGramRowLocation)] = []
        resolved.reserveCapacity(rowIDs.count)
        for (position, rowID) in rowIDs.enumerated() {
            resolved.append((position, rowID, try index.locate(row: rowID)))
        }
        var readBytes: Int64 = 0
        scratch.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            var requests: [ExpertReader.Request] = []
            requests.reserveCapacity(resolved.count * 3)
            for item in resolved {
                readBytes += appendRequests(for: item.location, position: item.position, base: base, layout: layout, into: &requests)
            }
            if !requests.isEmpty { ExpertReader.read(requests) }
        }
        insertMissedIntoCache(missed: resolved.map { ($0.position, $0.rowID) }, layout: layout, scratch: scratch)
        stats.bytesRead += ByteCount(readBytes)
        stats.cachedRows = cache.count
    }

    private func populateScratch(
        uniqueRows: [Int],
        layout: BatchLayout,
        scratch: inout [UInt8],
        missed: inout [(position: Int, rowID: Int)],
        readBytes: inout Int64
    ) throws(BasaltError) {
        var resolvedMisses: [(position: Int, location: NGramRowLocation)] = []
        var cachedHits: [(position: Int, slot: Int)] = []
        for (position, rowID) in uniqueRows.enumerated() {
            if let slot = cache.touch(rowID) {
                cachedHits.append((position, slot))
            } else {
                resolvedMisses.append((position, try index.locate(row: rowID)))
                missed.append((position, rowID))
            }
        }
        scratch.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            fillCachedHits(cachedHits, into: base, layout: layout)
            var requests: [ExpertReader.Request] = []
            requests.reserveCapacity(resolvedMisses.count * 3)
            for miss in resolvedMisses {
                readBytes += appendRequests(for: miss.location, position: miss.position, base: base, layout: layout, into: &requests)
            }
            if !requests.isEmpty { ExpertReader.read(requests) }
        }
    }

    private func appendRequests(
        for location: NGramRowLocation,
        position: Int,
        base: UnsafeMutableRawPointer,
        layout: BatchLayout,
        into requests: inout [ExpertReader.Request]
    ) -> Int64 {
        requests.append(ExpertReader.Request(
            descriptor: descriptors[location.weightFileIndex].descriptor,
            offset: location.weightOffset,
            destination: base + position * index.rowWeightBytes,
            count: index.rowWeightBytes
        ))
        var bytes = Int64(index.rowWeightBytes)
        if let scalesFile = location.scalesFileIndex, let scalesOff = location.scalesOffset, index.rowScalesBytes > 0 {
            requests.append(ExpertReader.Request(
                descriptor: descriptors[scalesFile].descriptor,
                offset: scalesOff,
                destination: base + layout.scalesOffset + position * index.rowScalesBytes,
                count: index.rowScalesBytes
            ))
            bytes += Int64(index.rowScalesBytes)
        }
        if let biasesFile = location.biasesFileIndex, let biasesOff = location.biasesOffset, index.rowBiasesBytes > 0 {
            requests.append(ExpertReader.Request(
                descriptor: descriptors[biasesFile].descriptor,
                offset: biasesOff,
                destination: base + layout.biasesOffset + position * index.rowBiasesBytes,
                count: index.rowBiasesBytes
            ))
            bytes += Int64(index.rowBiasesBytes)
        }
        return bytes
    }
}

extension NGramRowStore {
    private func fillCachedHits(
        _ hits: [(position: Int, slot: Int)],
        into base: UnsafeMutableRawPointer,
        layout: BatchLayout
    ) {
        guard !hits.isEmpty else { return }
        var rowBuffer = [UInt8](repeating: 0, count: index.rowTotalBytes)
        rowBuffer.withUnsafeMutableBytes { tmp in
            guard let tmpBase = tmp.baseAddress else { return }
            for hit in hits {
                cache.copyBytes(fromSlot: hit.slot, into: tmpBase)
                (base + hit.position * index.rowWeightBytes).copyMemory(from: tmpBase, byteCount: index.rowWeightBytes)
                if index.rowScalesBytes > 0 {
                    (base + layout.scalesOffset + hit.position * index.rowScalesBytes)
                        .copyMemory(from: tmpBase + index.rowWeightBytes, byteCount: index.rowScalesBytes)
                }
                if index.rowBiasesBytes > 0 {
                    (base + layout.biasesOffset + hit.position * index.rowBiasesBytes)
                        .copyMemory(from: tmpBase + index.rowWeightBytes + index.rowScalesBytes, byteCount: index.rowBiasesBytes)
                }
            }
        }
    }

    private func insertMissedIntoCache(
        missed: [(position: Int, rowID: Int)],
        layout: BatchLayout,
        scratch: [UInt8]
    ) {
        guard cache.capacity > 0, !missed.isEmpty else { return }
        var packed = [UInt8](repeating: 0, count: index.rowTotalBytes)
        scratch.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for item in missed {
                packed.withUnsafeMutableBytes { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    dstBase.copyMemory(from: base + item.position * index.rowWeightBytes, byteCount: index.rowWeightBytes)
                    if index.rowScalesBytes > 0 {
                        (dstBase + index.rowWeightBytes)
                            .copyMemory(from: base + layout.scalesOffset + item.position * index.rowScalesBytes, byteCount: index.rowScalesBytes)
                    }
                    if index.rowBiasesBytes > 0 {
                        (dstBase + index.rowWeightBytes + index.rowScalesBytes)
                            .copyMemory(from: base + layout.biasesOffset + item.position * index.rowBiasesBytes, byteCount: index.rowBiasesBytes)
                    }
                }
                let evicted = packed.withUnsafeBytes { src -> Bool in
                    guard let srcBase = src.baseAddress else { return false }
                    return cache.store(rowID: item.rowID, from: srcBase)
                }
                if evicted { stats.evictions += 1 }
            }
        }
    }

    private func decodeBatch(scratch: [UInt8], layout: BatchLayout) -> MLXArray {
        let weights = MLXArray(Data(scratch[0 ..< layout.weightBytes]), [layout.uniqueCount, index.weightCols], dtype: index.weightDType)
        guard let quant = index.quantization, let scalesDType = index.scalesDType else { return weights }
        let scales = MLXArray(
            Data(scratch[layout.scalesOffset ..< layout.scalesOffset + layout.scalesBytes]),
            [layout.uniqueCount, index.groupCount], dtype: scalesDType
        )
        let biases: MLXArray? = if let biasesDType = index.biasesDType, layout.biasesBytes > 0 {
            MLXArray(
                Data(scratch[layout.biasesOffset ..< layout.biasesOffset + layout.biasesBytes]),
                [layout.uniqueCount, index.groupCount], dtype: biasesDType
            )
        } else {
            nil
        }
        return MLX.dequantized(weights, scales: scales, biases: biases, groupSize: quant.groupSize, bits: quant.bits, mode: quant.mode)
    }

    private static func deduplicate(_ indices: [Int]) -> (unique: [Int], gather: [Int32]) {
        var unique: [Int] = []
        unique.reserveCapacity(indices.count)
        var positionOf: [Int: Int32] = [:]
        var gather: [Int32] = []
        gather.reserveCapacity(indices.count)
        for row in indices {
            if let pos = positionOf[row] {
                gather.append(pos)
            } else {
                let pos = Int32(unique.count)
                unique.append(row)
                positionOf[row] = pos
                gather.append(pos)
            }
        }
        return (unique, gather)
    }
}
