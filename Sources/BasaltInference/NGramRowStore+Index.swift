import BasaltCore
import Foundation
import MLX

public struct NGramRowLocation: Sendable, Equatable {
    public let logicalShard: Int
    public let localRow: Int
    public let weightFileIndex: Int
    public let weightOffset: Int64
    public let scalesFileIndex: Int?
    public let scalesOffset: Int64?
    public let biasesFileIndex: Int?
    public let biasesOffset: Int64?
}

public struct NGramShardSpan: Sendable, Equatable {
    public let logicalShard: Int
    public let rowStart: Int
    public let rowCount: Int
    public let weightFileIndex: Int
    public let weightOffset: Int64
    public let scalesFileIndex: Int?
    public let scalesOffset: Int64?
    public let biasesFileIndex: Int?
    public let biasesOffset: Int64?
}

public struct NGramTableIndex: Sendable, Equatable {
    public let tablePrefix: String
    public let totalRows: Int
    public let embedDim: Int
    public let weightCols: Int
    public let groupCount: Int
    public let weightDType: DType
    public let scalesDType: DType?
    public let biasesDType: DType?
    public let quantization: NGramQuantization?
    public let rowWeightBytes: Int
    public let rowScalesBytes: Int
    public let rowBiasesBytes: Int
    public let shards: [NGramShardSpan]

    public var rowTotalBytes: Int { rowWeightBytes + rowScalesBytes + rowBiasesBytes }
    public var outputDType: DType { scalesDType ?? weightDType }

    public func locate(row idx: Int) throws(BasaltError) -> NGramRowLocation {
        guard idx >= 0, idx < totalRows else {
            throw .inferenceError("ngram row store: index \(idx) out of bounds 0..<\(totalRows)")
        }
        var low = 0
        var high = shards.count - 1
        while low <= high {
            let mid = low + (high - low) / 2
            let shard = shards[mid]
            if idx < shard.rowStart {
                high = mid - 1
            } else if idx >= shard.rowStart + shard.rowCount {
                low = mid + 1
            } else {
                return try makeLocation(shard: shard, row: idx)
            }
        }
        throw .inferenceError("ngram row store: index \(idx) did not map to any shard")
    }

    private func makeLocation(shard: NGramShardSpan, row idx: Int) throws(BasaltError) -> NGramRowLocation {
        let localRow = idx - shard.rowStart
        let weightOffset = try Self.checkedOffset(base: shard.weightOffset, row: localRow, stride: rowWeightBytes)
        let scalesOffset: Int64? = if let base = shard.scalesOffset {
            try Self.checkedOffset(base: base, row: localRow, stride: rowScalesBytes)
        } else {
            nil
        }
        let biasesOffset: Int64? = if let base = shard.biasesOffset {
            try Self.checkedOffset(base: base, row: localRow, stride: rowBiasesBytes)
        } else {
            nil
        }
        return NGramRowLocation(
            logicalShard: shard.logicalShard,
            localRow: localRow,
            weightFileIndex: shard.weightFileIndex,
            weightOffset: weightOffset,
            scalesFileIndex: shard.scalesFileIndex,
            scalesOffset: scalesOffset,
            biasesFileIndex: shard.biasesFileIndex,
            biasesOffset: biasesOffset
        )
    }

    static func checkedOffset(base: Int64, row: Int, stride: Int) throws(BasaltError) -> Int64 {
        let (delta, mulOverflow) = Int64(row).multipliedReportingOverflow(by: Int64(stride))
        let (offset, addOverflow) = base.addingReportingOverflow(delta)
        guard !mulOverflow, !addOverflow, offset >= 0 else {
            throw .inferenceError("ngram row store: byte offset overflow for row \(row)")
        }
        return offset
    }
}

struct NGramRowLRUCache: Sendable {
    private struct Node: Sendable {
        var rowID: Int
        var prev: Int
        var next: Int
    }

    let capacity: Int
    let rowBytes: Int
    private var slotOf: [Int: Int] = [:]
    private var nodes: [Node] = []
    private var storage: [UInt8] = []
    private var head = -1
    private var tail = -1

    init(capacity: Int, rowBytes: Int) {
        self.capacity = max(0, capacity)
        self.rowBytes = max(1, rowBytes)
    }

    var count: Int { nodes.count }

    mutating func touch(_ rowID: Int) -> Int? {
        guard let slot = slotOf[rowID] else { return nil }
        moveToFront(slot)
        return slot
    }

    func copyBytes(fromSlot slot: Int, into destination: UnsafeMutableRawPointer) {
        let start = slot * rowBytes
        storage.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            destination.copyMemory(from: base + start, byteCount: rowBytes)
        }
    }

    mutating func store(rowID: Int, from source: UnsafeRawPointer) -> Bool {
        guard capacity > 0 else { return false }
        if let existing = slotOf[rowID] {
            writeBytes(from: source, intoSlot: existing)
            moveToFront(existing)
            return false
        }
        if nodes.count < capacity {
            let slot = nodes.count
            nodes.append(Node(rowID: rowID, prev: -1, next: -1))
            storage.append(contentsOf: repeatElement(UInt8(0), count: rowBytes))
            writeBytes(from: source, intoSlot: slot)
            slotOf[rowID] = slot
            attachFront(slot)
            return false
        }
        let slot = tail
        let evicted = nodes[slot].rowID
        slotOf.removeValue(forKey: evicted)
        detach(slot)
        nodes[slot].rowID = rowID
        writeBytes(from: source, intoSlot: slot)
        slotOf[rowID] = slot
        attachFront(slot)
        return true
    }

    private mutating func writeBytes(from source: UnsafeRawPointer, intoSlot slot: Int) {
        let start = slot * rowBytes
        storage.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            (base + start).copyMemory(from: source, byteCount: rowBytes)
        }
    }

    private mutating func moveToFront(_ slot: Int) {
        guard slot != head else { return }
        detach(slot)
        attachFront(slot)
    }

    private mutating func detach(_ slot: Int) {
        let prev = nodes[slot].prev
        let next = nodes[slot].next
        if prev != -1 { nodes[prev].next = next } else { head = next }
        if next != -1 { nodes[next].prev = prev } else { tail = prev }
        nodes[slot].prev = -1
        nodes[slot].next = -1
    }

    private mutating func attachFront(_ slot: Int) {
        nodes[slot].prev = -1
        nodes[slot].next = head
        if head != -1 { nodes[head].prev = slot } else { tail = slot }
        head = slot
    }
}

extension NGramRowStore {
    fileprivate struct RawShardRecord {
        let logicalShard: Int
        let rowCount: Int
        let weightCols: Int
        let weightDType: DType
        let weightFileIndex: Int
        let weightOffset: Int64
        let groupCount: Int
        let scalesDType: DType?
        let scalesFileIndex: Int?
        let scalesOffset: Int64?
        let biasesDType: DType?
        let biasesFileIndex: Int?
        let biasesOffset: Int64?
    }

    static func buildIndex(
        from shards: [SafetensorsShard],
        config: NGramRowStoreConfig
    ) throws(BasaltError) -> NGramTableIndex {
        let prefix = try resolvePrefix(in: shards, explicit: config.tablePrefix)
        let rawRecords = try collectShardRecords(in: shards, prefix: prefix)
        guard !rawRecords.isEmpty else {
            throw .inferenceError("ngram row store: no shards found for prefix '\(prefix)'")
        }
        let sorted = rawRecords.sorted { $0.logicalShard < $1.logicalShard }
        let first = sorted[0]
        let (quant, embedDim) = try resolveQuantization(first: first, override: config.quantization)
        var spans: [NGramShardSpan] = []
        spans.reserveCapacity(sorted.count)
        var runningRows = 0

        for (expectedIdx, rec) in sorted.enumerated() {
            guard rec.logicalShard == expectedIdx else {
                throw .inferenceError("ngram row store: missing logical shard \(expectedIdx) (found \(rec.logicalShard))")
            }
            guard rec.weightCols == first.weightCols,
                  rec.weightDType == first.weightDType,
                  rec.groupCount == first.groupCount,
                  rec.scalesDType == first.scalesDType,
                  rec.biasesDType == first.biasesDType else {
                throw .inferenceError("ngram row store: shard \(rec.logicalShard) schema disagrees with shard 0")
            }
            let (nextRows, rowOverflow) = runningRows.addingReportingOverflow(rec.rowCount)
            guard !rowOverflow else {
                throw .inferenceError("ngram row store: total row count overflow")
            }
            spans.append(NGramShardSpan(
                logicalShard: rec.logicalShard, rowStart: runningRows, rowCount: rec.rowCount,
                weightFileIndex: rec.weightFileIndex, weightOffset: rec.weightOffset,
                scalesFileIndex: rec.scalesFileIndex, scalesOffset: rec.scalesOffset,
                biasesFileIndex: rec.biasesFileIndex, biasesOffset: rec.biasesOffset
            ))
            runningRows = nextRows
        }
        return try assembleTableIndex(prefix: prefix, totalRows: runningRows, embedDim: embedDim, first: first, quant: quant, spans: spans)
    }

    private static func assembleTableIndex(
        prefix: String,
        totalRows: Int,
        embedDim: Int,
        first: RawShardRecord,
        quant: NGramQuantization?,
        spans: [NGramShardSpan]
    ) throws(BasaltError) -> NGramTableIndex {
        let (rowWeight, ovW) = first.weightCols.multipliedReportingOverflow(by: first.weightDType.size)
        let scalesPair = first.scalesDType.map { first.groupCount.multipliedReportingOverflow(by: $0.size) } ?? (partialValue: 0, overflow: false)
        let biasesPair = first.biasesDType.map { first.groupCount.multipliedReportingOverflow(by: $0.size) } ?? (partialValue: 0, overflow: false)
        guard !ovW, !scalesPair.overflow, !biasesPair.overflow else {
            throw .inferenceError("ngram row store: row byte stride overflow")
        }
        return NGramTableIndex(
            tablePrefix: prefix, totalRows: totalRows, embedDim: embedDim,
            weightCols: first.weightCols, groupCount: first.groupCount,
            weightDType: first.weightDType, scalesDType: first.scalesDType, biasesDType: first.biasesDType,
            quantization: quant, rowWeightBytes: rowWeight, rowScalesBytes: scalesPair.partialValue,
            rowBiasesBytes: biasesPair.partialValue, shards: spans
        )
    }

    private static func resolvePrefix(in shards: [SafetensorsShard], explicit: String?) throws(BasaltError) -> String {
        if let explicit, !explicit.isEmpty {
            return explicit.hasSuffix(".") ? String(explicit.dropLast()) : explicit
        }
        let marker = ModelFitAnalyzer.ssdResidentPLEKeyMarker
        for shard in shards {
            for key in shard.entries.keys.sorted() where key.hasSuffix(".weight") && ModelFitAnalyzer.isSSDResidentPLEKey(key) {
                if let range = key.range(of: marker) {
                    return String(key[..<range.upperBound].dropLast())
                }
            }
        }
        throw .inferenceError("ngram row store: no '\(marker)*' weight tensors found in shards")
    }

    private static func collectShardRecords(
        in shards: [SafetensorsShard],
        prefix: String
    ) throws(BasaltError) -> [RawShardRecord] {
        let keyPrefix = prefix + "."
        var tensorFile: [String: Int] = [:]
        for (fileIndex, shard) in shards.enumerated() {
            for key in shard.entries.keys where key.hasPrefix(keyPrefix) {
                guard tensorFile[key] == nil else {
                    throw .inferenceError("ngram row store: duplicate tensor key '\(key)'")
                }
                tensorFile[key] = fileIndex
            }
        }
        var records: [RawShardRecord] = []
        for key in tensorFile.keys.sorted() where key.hasSuffix(".weight") {
            let tail = key.dropFirst(keyPrefix.count).dropLast(".weight".count)
            guard let logicalShard = Int(tail), logicalShard >= 0 else { continue }
            records.append(try parseShardRecord(
                shards: shards, tensorFile: tensorFile, logicalShard: logicalShard, baseKey: "\(keyPrefix)\(logicalShard)"
            ))
        }
        return records
    }
}

extension NGramRowStore {
    private static func parseShardRecord(
        shards: [SafetensorsShard],
        tensorFile: [String: Int],
        logicalShard: Int,
        baseKey: String
    ) throws(BasaltError) -> RawShardRecord {
        let weightKey = "\(baseKey).weight"
        let weightFile = tensorFile[weightKey]!
        let weight = try parseTensor2D(shard: shards[weightFile], key: weightKey)
        let scalesKey = "\(baseKey).scales"
        let scalesFile = tensorFile[scalesKey]
        let scales = if let scalesFile { try parseTensor2D(shard: shards[scalesFile], key: scalesKey) } else { nil as (rows: Int, cols: Int, dtype: DType, offset: Int64)? }
        let biasesKey = "\(baseKey).biases"
        let biasesFile = tensorFile[biasesKey]
        let biases = if let biasesFile { try parseTensor2D(shard: shards[biasesFile], key: biasesKey) } else { nil as (rows: Int, cols: Int, dtype: DType, offset: Int64)? }
        if let scales, scales.rows != weight.rows {
            throw .inferenceError("ngram row store: shard \(logicalShard) scales row count \(scales.rows) != weight rows \(weight.rows)")
        }
        if let biases {
            guard let scales, biases.rows == scales.rows, biases.cols == scales.cols, biases.dtype == scales.dtype else {
                throw .inferenceError("ngram row store: shard \(logicalShard) biases shape/dtype disagree with scales")
            }
        }
        return RawShardRecord(
            logicalShard: logicalShard, rowCount: weight.rows,
            weightCols: weight.cols, weightDType: weight.dtype, weightFileIndex: weightFile, weightOffset: weight.offset,
            groupCount: scales?.cols ?? 0, scalesDType: scales?.dtype, scalesFileIndex: scalesFile, scalesOffset: scales?.offset,
            biasesDType: biases?.dtype, biasesFileIndex: biasesFile, biasesOffset: biases?.offset
        )
    }

    private static func parseTensor2D(
        shard: SafetensorsShard,
        key: String
    ) throws(BasaltError) -> (rows: Int, cols: Int, dtype: DType, offset: Int64) {
        guard let entry = shard.entries[key] as? [String: Any],
              let dtypeName = entry["dtype"] as? String,
              let dtype = parseDType(dtypeName),
              let shape = entry["shape"] as? [Int], shape.count == 2,
              shape[0] > 0, shape[1] > 0,
              let span = shard.span(of: key) else {
            throw .inferenceError("ngram row store: invalid 2D tensor metadata for \(key)")
        }
        let (elements, ov1) = Int64(shape[0]).multipliedReportingOverflow(by: Int64(shape[1]))
        let (expectedBytes, ov2) = elements.multipliedReportingOverflow(by: Int64(dtype.size))
        guard !ov1, !ov2, span.length == expectedBytes else {
            throw .inferenceError("ngram row store: byte length mismatch for \(key)")
        }
        return (shape[0], shape[1], dtype, span.offset)
    }

    private static func parseDType(_ name: String) -> DType? {
        switch name {
        case "U32": .uint32
        case "BF16": .bfloat16
        case "F16": .float16
        case "F32": .float32
        default: nil
        }
    }

    private static func resolveQuantization(
        first: RawShardRecord,
        override: NGramQuantization?
    ) throws(BasaltError) -> (NGramQuantization?, Int) {
        guard first.scalesDType != nil else {
            guard first.weightDType == .bfloat16 || first.weightDType == .float16 || first.weightDType == .float32 else {
                throw .inferenceError("ngram row store: unquantized weight dtype \(first.weightDType) is unsupported")
            }
            return (nil, first.weightCols)
        }
        guard first.weightDType == .uint32 else {
            throw .inferenceError("ngram row store: quantized table weight dtype must be U32")
        }
        let quant = override ?? .affine4g32
        guard quant.groupSize > 0, quant.bits > 0, quant.bits <= 8 else {
            throw .inferenceError("ngram row store: invalid quantization bits=\(quant.bits) groupSize=\(quant.groupSize)")
        }
        let (embedDim, ovDim) = first.groupCount.multipliedReportingOverflow(by: quant.groupSize)
        let (totalBits, ovBits) = embedDim.multipliedReportingOverflow(by: quant.bits)
        guard !ovDim, !ovBits, embedDim > 0, totalBits % 32 == 0, totalBits / 32 == first.weightCols else {
            throw .inferenceError("ngram row store: packed cols \(first.weightCols) disagree with quantization \(quant.bits)-bit g\(quant.groupSize)")
        }
        return (quant, embedDim)
    }
}
