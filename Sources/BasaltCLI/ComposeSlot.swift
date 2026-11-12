import Foundation
import BasaltCore
import BasaltCloud
import BasaltServer

actor ComposeSlot {
    let spec: ComposeSpec
    nonisolated let registry = PipelineRegistry([:])
    nonisolated let cloudGate: CloudGate
    nonisolated let events = ComposeEvents()
    let key: @Sendable (ComposeSpec.Model?) -> String
    private(set) var host: ComposeHost?

    init(spec: ComposeSpec, cloudGate: CloudGate, key: @escaping @Sendable (ComposeSpec.Model?) -> String) {
        (self.spec, self.cloudGate, self.key) = (spec, cloudGate, key)
    }

    func started(_ host: ComposeHost) { self.host = host }

    func status() async -> ComposeStatus {
        if let host { return await host.status() }
        let models = spec.models.sorted { $0.key < $1.key }.map { name, model in
            ComposeModelStatus(name: name, source: ComposeHost.describe(model.source), keep: ComposeHost.keep(model), state: model.isCloud ? ComposeHost.cloudState(nil, key: key(model)) : .loading)
        }
        return ComposeStatus(name: spec.name, pipelines: spec.pipelineNames, models: models, spentToday: await cloudGate.spent(CloudGateCore.total))
    }

    func apply(_ request: ComposeApply) async throws(BasaltError) -> ComposeDiff {
        guard let host else { throw .serverBusy(retryAfterSeconds: 5) }
        return try await host.apply(request)
    }

    func preview(_ request: ComposeApply) async throws(BasaltError) -> ComposeDiff {
        guard let host else { throw .serverBusy(retryAfterSeconds: 5) }
        return try await host.preview(request)
    }

    nonisolated var control: ComposeControl {
        var control = ComposeControl(status: { await self.status() }, apply: { request throws(BasaltError) in try await self.apply(request) })
        (control.preview, control.events) = ({ request throws(BasaltError) in try await self.preview(request) }, events)
        return control
    }
}
