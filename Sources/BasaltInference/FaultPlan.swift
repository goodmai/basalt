struct FaultPlan: Sendable {
    enum Point: CaseIterable, Sendable {
        case tokenize, restoreFromDisk, snapshotBoundary, anchorOffsets, launch, settle, keep, load, unload
    }

    enum Fault: Error, CaseIterable, Sendable {
        case cancellation, modelError, metalTimeout
    }

    var point: Point?
    var fault = Fault.modelError

    var endsStream: Bool { point == .settle || point == .keep }

    func check(_ here: Point) throws {
        guard point == here else { return }
        throw fault == .cancellation ? CancellationError() : fault as any Error
    }
}

extension MLXInferenceEngine {
    func inject(_ plan: FaultPlan) {
        faults = plan
    }

    func checkInvariants(idle: Bool = false) -> [String] {
        var problems = readPromptCaches { $0.checkInvariants(leases: idle ? 0 : nil) }
        if !launches.keys.allSatisfy({ leases[$0] != nil }) {
            problems.append("a launch outlived its lease")
        }
        if idle, !leases.isEmpty || !launches.isEmpty {
            problems.append("\(leases.count) leases and \(launches.count) launches held while idle")
        }
        return problems
    }
}
