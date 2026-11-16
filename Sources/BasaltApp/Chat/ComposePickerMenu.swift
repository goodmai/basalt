import BasaltAppCore
import BasaltCore
import SwiftUI
import UniformTypeIdentifiers

public struct ComposePickerMenu: View {
    let model: AppModel
    @State private var choosingCompose = false

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        Menu {
            if let compose = model.monitor.state.compose {
                Section("Compose system \(compose.name)") {
                    ForEach(compose.models, id: \.name) { entry in
                        Text(verbatim: "\(entry.name) · \(entry.state)")
                    }
                }
            }
            if !model.library.state.models.isEmpty {
                Section("Models") {
                    ForEach(model.library.state.models) { item in
                        Button {
                            Task { try? await model.hotSwitchModel(item.id) }
                        } label: {
                            if item.id == model.model {
                                Label(item.id, systemImage: "checkmark")
                            } else {
                                Text(verbatim: item.id)
                            }
                        }
                    }
                }
            }
            Section("Pipelines") {
                ForEach(PipelinePreset.allCases, id: \.rawValue) { preset in
                    Button(preset.rawValue) {
                        Task { await model.switchEngine(["serve", "--model", model.configuredModel, "--pipeline-preset", preset.rawValue]) }
                    }
                }
                Button("Choose Compose File…") { choosingCompose = true }
            }
        } label: {
            Label(model.monitor.state.compose?.name ?? model.model, systemImage: "square.stack.3d.up")
        }
        .fileImporter(isPresented: $choosingCompose, allowedContentTypes: [.yaml, .data]) { result in
            guard let url = try? result.get() else { return }
            Task { _ = try? await model.applyComposeFile(url) }
        }
    }
}
