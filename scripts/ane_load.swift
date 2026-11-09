import CoreAI
import Foundation

struct Options {
    var model: URL
    var function = "main"
    var unit = ComputeUnitKind.neuralEngine
    var duty = 1.0
    var seconds = 30.0
    var weightBytes = 0.0
    var window = 0.0

    init(_ arguments: [String]) {
        var values: [String: String] = [:]
        var iterator = arguments.makeIterator()
        while let key = iterator.next() { values[key] = iterator.next() }
        guard let path = values["--model"] else {
            FileHandle.standardError.write(Data("usage: ane_load.swift --model <x.aimodel> [--function main] [--unit ane|gpu|cpu] [--duty 0.25] [--seconds 30] [--weight-bytes N] [--window 5]\n".utf8))
            exit(2)
        }
        model = URL(fileURLWithPath: path)
        function = values["--function"] ?? function
        unit = ["gpu": .gpu, "cpu": .cpu][values["--unit"] ?? ""] ?? .neuralEngine
        duty = min(1, max(0.01, Double(values["--duty"] ?? "") ?? duty))
        seconds = Double(values["--seconds"] ?? "") ?? seconds
        weightBytes = Double(values["--weight-bytes"] ?? "") ?? weightBytes
        window = Double(values["--window"] ?? "") ?? window
    }
}

func elapsed(since start: ContinuousClock.Instant) -> Double {
    let duration = ContinuousClock.now - start
    return Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}

func inputs(of function: InferenceFunction) throws -> [String: NDArray] {
    var arrays: [String: NDArray] = [:]
    for name in function.descriptor.inputNames {
        guard case .ndArray(let descriptor)? = function.descriptor.inputDescriptor(of: name), !descriptor.hasDynamicShape else {
            throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "input \(name) is not a static NDArray"])
        }
        arrays[name] = NDArray(shape: descriptor.shape, scalarType: descriptor.scalarType)
    }
    return arrays
}

let options = Options(Array(CommandLine.arguments.dropFirst()))
let specialization = SpecializationOptions(preferredComputeUnitKind: options.unit)
let cold = ContinuousClock.now
try AIModelCache.default.deleteEntries(for: options.model)
_ = try await AIModel.specialize(contentsOf: options.model, options: specialization)
let specializeSeconds = elapsed(since: cold)
let warm = ContinuousClock.now
let model = try await AIModel.specialize(contentsOf: options.model, options: specialization)
let cachedSeconds = elapsed(since: warm)
guard let function = try model.loadFunction(named: options.function) else {
    FileHandle.standardError.write(Data("no function \(options.function)\n".utf8))
    exit(1)
}
let arrays = try inputs(of: function)
_ = try await function.run(inputs: arrays)
var (runs, busy, windowRuns) = (0, 0.0, 0)
let start = ContinuousClock.now
var windowStart = start
while elapsed(since: start) < options.seconds {
    let begin = ContinuousClock.now
    _ = try await function.run(inputs: arrays)
    let took = elapsed(since: begin)
    (runs, busy, windowRuns) = (runs + 1, busy + took, windowRuns + 1)
    if options.window > 0, elapsed(since: windowStart) >= options.window {
        let rate = Double(windowRuns) / elapsed(since: windowStart)
        FileHandle.standardOutput.write(Data(String(format: "window t=%.1f per_s=%.2f weight_gb_s=%.1f\n", Date().timeIntervalSince1970, rate, options.weightBytes * rate / 1e9).utf8))
        (windowStart, windowRuns) = (ContinuousClock.now, 0)
    }
    let ahead = busy / options.duty - elapsed(since: start)
    if options.duty < 1, ahead > 0 { try await Task.sleep(for: .seconds(ahead)) }
}
let wall = elapsed(since: start)
let perSecond = Double(runs) / wall
let report = [
    "model": options.model.lastPathComponent, "unit": "\(options.unit)", "duty": String(format: "%.2f", options.duty),
    "specialize_s": String(format: "%.2f", specializeSeconds), "cache_hit_s": String(format: "%.3f", cachedSeconds),
    "runs": "\(runs)", "per_s": String(format: "%.2f", perSecond), "ms_per_run": String(format: "%.3f", busy / Double(max(runs, 1)) * 1000),
    "busy": String(format: "%.2f", busy / wall), "weight_gb_s": String(format: "%.1f", options.weightBytes * perSecond / 1e9),
]
print(report.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))
