import BasaltAppCore
import BasaltCore
import SwiftUI

extension ConvertModelSheet {
    func adapterSection(_ draft: ConversionDraft) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Merge LoRA Adapter (Optional)")
                .font(.subheadline.weight(.semibold))
            TextField("org/adapter-repo", text: Binding(
                get: { draft.adapterRepoId },
                set: { value in
                    var next = draft
                    next.adapterRepoId = value
                    store.send(.updateConvertDraft(next))
                }
            ))
            .textFieldStyle(.plain)
            .font(.callout.monospaced())
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 8))
            Text("Fuses a PEFT or mlx-lm LoRA repository into the converted weights.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 12))
    }

    nonisolated static func isValidOptionalAdapter(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        let normalized = trimmed.hasPrefix(ServedAdapterSource.repositoryPrefix)
            ? String(trimmed.dropFirst(ServedAdapterSource.repositoryPrefix.count))
            : trimmed
        return LibraryReducer.isRepoID(normalized)
    }
}
