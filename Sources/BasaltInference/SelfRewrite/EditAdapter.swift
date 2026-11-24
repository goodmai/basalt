import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Synchronization
import BasaltCore

public final class EditAdapter: ModelAdapter, Sendable {
    public static let programFile = "edit.json"
    public static let weightsFile = "edit.safetensors"
    static let emptySentinelKey = "__basalt_edit_empty__"

    private struct AppliedState: Sendable {
        let undoEntries: [EditUndoEntry]
        let metrics: [EditOpMetrics]
        let loraScales: [String: Float]
    }

    public let program: EditProgram
    public let tensors: [String: MaterializedArray]
    private let appliedStates = Mutex<[UInt: AppliedState]>([:])

    public init(program: EditProgram, tensors: [String: MLXArray] = [:]) throws {
        var materialized: [String: MaterializedArray] = [:]
        var floatValues: [String: [Float]] = [:]
        for (key, value) in tensors {
            try EditProgramValidator.validateIdentifier(key, field: "safetensors_key")
            let frozen = materialize(value)
            materialized[key] = frozen
            floatValues[key] = frozen.asType(.float32).asArray(Float.self)
        }
        try EditProgramValidator.validate(program, tensorValues: floatValues)
        self.program = program
        self.tensors = materialized
    }

    public init(validatedProgram: EditProgram, materializedTensors: [String: MaterializedArray]) throws {
        var floatValues: [String: [Float]] = [:]
        for (key, value) in materializedTensors {
            try EditProgramValidator.validateIdentifier(key, field: "safetensors_key")
            floatValues[key] = value.asType(.float32).asArray(Float.self)
        }
        try EditProgramValidator.validate(validatedProgram, tensorValues: floatValues)
        self.program = validatedProgram
        self.tensors = materializedTensors
    }

    public static func isEditAdapterDirectory(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appending(component: programFile).path)
    }

    public static func from(directory: URL) throws -> EditAdapter {
        let programURL = directory.appending(component: programFile)
        let weightsURL = directory.appending(component: weightsFile)
        let programData = try Data(contentsOf: programURL)
        let program = try JSONDecoder().decode(EditProgram.self, from: programData)
        let rawArrays = try MLX.loadArrays(url: weightsURL)
        eval(Array(rawArrays.values))
        var loaded: [String: MaterializedArray] = [:]
        for (key, array) in rawArrays where key != emptySentinelKey {
            try EditProgramValidator.validateIdentifier(key, field: "safetensors_key")
            loaded[key] = materialize(array)
        }
        return try EditAdapter(validatedProgram: program, materializedTensors: loaded)
    }

    public func save(to directory: URL) throws {
        try withTaskCancellationShield {
            let files = FileManager.default
            try files.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try files.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let programData = try encoder.encode(program)
            let programURL = directory.appending(component: Self.programFile)
            try programData.write(to: programURL, options: .atomic)
            try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: programURL.path)
            try AdapterCheckpoint.fullSync(programURL)

            let weightsURL = directory.appending(component: Self.weightsFile)
            let stagingURL = directory.appending(component: ".edit-\(UUID().uuidString).safetensors")
            defer { try? files.removeItem(at: stagingURL) }
            var arraysToSave: [String: MLXArray] = [:]
            for (key, value) in tensors {
                arraysToSave[key] = value.reshaped(value.shape)
            }
            if arraysToSave.isEmpty {
                arraysToSave[Self.emptySentinelKey] = MLXArray.zeros([1], dtype: .float32)
            }
            try MLX.save(arrays: arraysToSave, url: stagingURL)
            try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: stagingURL.path)
            try AdapterCheckpoint.fullSync(stagingURL)
            if files.fileExists(atPath: weightsURL.path) {
                _ = try files.replaceItemAt(weightsURL, withItemAt: stagingURL)
            } else {
                try files.moveItem(at: stagingURL, to: weightsURL)
            }
            try AdapterCheckpoint.fullSync(directory)
        }
    }

    public static func setApproval(_ status: EditApprovalStatus, at directory: URL) throws -> EditProgram {
        try withTaskCancellationShield {
            let existing = try EditAdapter.from(directory: directory)
            var updated = existing.program
            updated.approval = status
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(updated)
            let programURL = directory.appending(component: Self.programFile)
            try data.write(to: programURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: programURL.path)
            try AdapterCheckpoint.fullSync(programURL)
            try AdapterCheckpoint.fullSync(directory)
            return updated
        }
    }

    public func load(into model: any LanguageModel) throws {
        let modelKey = UInt(bitPattern: ObjectIdentifier(model))
        let alreadyLoaded = appliedStates.withLock { $0[modelKey] != nil }
        guard !alreadyLoaded else {
            throw BasaltError.invalidRequestStructure(details: "edit adapter is already loaded on this model; unload it first")
        }
        var rawTensors: [String: MLXArray] = [:]
        for (key, value) in tensors {
            rawTensors[key] = value.reshaped(value.shape)
        }
        var context = EditOpExecutor.ExecutionContext()
        let result = try EditOpExecutor.execute(program: program, tensors: rawTensors, on: model, context: &context)
        let state = AppliedState(undoEntries: result.undoEntries, metrics: result.metrics, loraScales: context.loraScales)
        appliedStates.withLock { $0[modelKey] = state }
    }

    public func fuse(with model: any LanguageModel) throws {
        try load(into: model)
        _ = PHMAdapterApplicator.fuse(in: model)
        let modelKey = UInt(bitPattern: ObjectIdentifier(model))
        _ = appliedStates.withLock { $0.removeValue(forKey: modelKey) }
    }

    public func unload(from model: any LanguageModel) {
        let modelKey = UInt(bitPattern: ObjectIdentifier(model))
        guard let state = appliedStates.withLock({ $0.removeValue(forKey: modelKey) }) else { return }
        var scales = state.loraScales
        do {
            try EditOpExecutor.undo(state.undoEntries, in: model, loraScales: &scales)
        } catch {
            BasaltLogger(module: "EditAdapter").error("Undoing the edit adapter failed, so the model may still carry its edits: \(MLXInferenceEngine.briefly(error))")
        }
    }

    public func lastMetrics(for model: any LanguageModel) -> [EditOpMetrics] {
        let modelKey = UInt(bitPattern: ObjectIdentifier(model))
        return appliedStates.withLock { $0[modelKey]?.metrics ?? [] }
    }
}
