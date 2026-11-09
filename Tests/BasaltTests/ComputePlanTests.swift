import Foundation
import IOKit.ps
import Testing
@testable import BasaltCore
@testable import BasaltCLI

@Suite("Compute plan: placement by the #273 rule, one memory budget, power-degraded mode (#399)")
struct ComputePlanTests {
    static let chat = Workload(name: "qwen3.8-27b", role: "chat", unit: .gpu, kind: .decode, residentBytes: 14_000_000_000)
    static let embed = Workload(name: "embeddinggemma", role: "embed", unit: .ane, kind: .encoder, residentBytes: 300_000_000)
    static let jev = Workload(name: "openjev-0.6b", role: "decide", unit: .gpu, kind: .encoder, residentBytes: 2_400_000_000)

    @Test("ANE encoders run beside decode, GPU encoders are gated, bandwidth-bound work is paced, training is exclusive")
    func placements() {
        let verdict = ComputePlan.admit([Self.chat, Self.embed, Self.jev], budgetBytes: 18_000_000_000)
        #expect(verdict.refusal == nil && !verdict.degraded)
        let byRole = Dictionary(uniqueKeysWithValues: verdict.placements.map { ($0.workload.role, $0) })
        #expect(byRole["embed"]?.concurrentWithDecode == true && byRole["embed"]?.expectedDecodeDrop == 0.0...0.07)
        #expect(byRole["decide"]?.concurrentWithDecode == false && byRole["decide"]?.expectedDecodeDrop == 0.27...0.32)
        let stream = ComputePlan.place(Workload(name: "w", role: "r", unit: .ane, kind: .bandwidthStreaming, residentBytes: 0), degraded: false)
        #expect(stream.maxDuty == 0.1 && !stream.concurrentWithDecode)
        #expect(!ComputePlan.place(Workload(name: "t", role: "trainer", unit: .gpu, kind: .training, residentBytes: 0), degraded: false).concurrentWithDecode)
    }

    @Test("A set past the working set is refused with each model's share in the message")
    func memoryRefusal() {
        let verdict = ComputePlan.admit([Self.chat, Self.jev, Workload(name: "glm-ocr", role: "ocr", unit: .gpu, kind: .encoder, residentBytes: 2_000_000_000)], budgetBytes: 18_000_000_000)
        #expect(verdict.refusal?.contains("18.4 GB resident against a 18.0 GB working set") == true)
        #expect(verdict.refusal?.contains("openjev-0.6b 2.4 GB") == true)
    }

    @Test("On battery, or on a small adapter that is not charging a low battery, sidecars stop running beside decode")
    func powerDegraded() {
        #expect(PowerState(onAC: false, charging: false, batteryPercent: 80, adapterWatts: nil).degraded)
        #expect(PowerState(onAC: true, charging: false, batteryPercent: 6, adapterWatts: 20).degraded)
        #expect(!PowerState(onAC: true, charging: true, batteryPercent: 10, adapterWatts: 68).degraded)
        #expect(!PowerState.desktop.degraded)
        let verdict = ComputePlan.admit([Self.chat, Self.embed], budgetBytes: 0, power: PowerState(onAC: false, charging: false, batteryPercent: 50, adapterWatts: nil))
        #expect(verdict.degraded && verdict.placements.allSatisfy { !$0.concurrentWithDecode })
    }

    @Test("Power sources parse from the IOKit dictionaries")
    func powerParsing() {
        let state = PowerSource.state(
            battery: [kIOPSPowerSourceStateKey: kIOPSACPowerValue, kIOPSIsChargingKey: true, kIOPSCurrentCapacityKey: 41],
            adapter: [kIOPSPowerAdapterWattsKey: 68])
        #expect(state == PowerState(onAC: true, charging: true, batteryPercent: 41, adapterWatts: 68))
    }

    @Test("Weight bytes follow Hugging Face cache symlinks and count only safetensors and aimodel files")
    func weightBytes() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "plan-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let blobs = root.appending(path: "blobs"), snapshot = root.appending(path: "snapshot")
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data(count: 1000).write(to: blobs.appending(path: "a"))
        try Data(count: 50).write(to: snapshot.appending(path: "config.json"))
        try FileManager.default.createSymbolicLink(at: snapshot.appending(path: "model.safetensors"), withDestinationURL: blobs.appending(path: "a"))
        #expect(ServeCommand.weightBytes(snapshot.path) == 1000)
    }
}
