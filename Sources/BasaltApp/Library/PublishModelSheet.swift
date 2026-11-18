import BasaltAppCore
import SwiftUI

public struct PublishModelSheet: View {
    let store: LibraryStore
    @Environment(\.glassSurfacePolicy) private var policy

    public init(store: LibraryStore) {
        self.store = store
    }

    public var body: some View {
        if let draft = store.state.publishDraft {
            VStack(alignment: .leading, spacing: 16) {
                header(draft)
                Divider()
                formBody(draft)
                Spacer(minLength: 4)
                Divider()
                footer(draft)
            }
            .padding(20)
            .frame(width: 480, height: 340)
        }
    }

    private func header(_ draft: PublishDraft) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "icloud.and.arrow.up.fill")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text("Publish to Hugging Face")
                    .font(.headline)
                HStack(spacing: 8) {
                    Text(draft.sourceId)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Text(ByteCountFormatter.string(fromByteCount: draft.sourceBytes, countStyle: .file))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
    }

    private func formBody(_ draft: PublishDraft) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Target Repository ID")
                    .font(.subheadline.weight(.semibold))
                TextField(
                    "owner/model-name",
                    text: Binding(
                        get: { draft.repoId },
                        set: { val in
                            var next = draft
                            next.repoId = val
                            store.send(.updatePublishDraft(next))
                        }
                    )
                )
                .textFieldStyle(.roundedBorder)
            }

            Toggle(
                "Private Repository",
                isOn: Binding(
                    get: { draft.isPrivate },
                    set: { val in
                        var next = draft
                        next.isPrivate = val
                        store.send(.updatePublishDraft(next))
                    }
                )
            )

            VStack(alignment: .leading, spacing: 4) {
                Text("Commit Message")
                    .font(.subheadline.weight(.semibold))
                TextField(
                    "Upload MLX checkpoint via Basalt",
                    text: Binding(
                        get: { draft.commitMessage },
                        set: { val in
                            var next = draft
                            next.commitMessage = val
                            store.send(.updatePublishDraft(next))
                        }
                    )
                )
                .textFieldStyle(.roundedBorder)
            }
        }
    }

    private func footer(_ draft: PublishDraft) -> some View {
        HStack {
            Button("Cancel") {
                store.send(.closePublish)
            }
            Spacer()
            Button("Publish Model") {
                store.send(.startPublish)
            }
            .adaptiveGlassButton(prominent: true)
            .disabled(!draft.isValid)
        }
    }
}
