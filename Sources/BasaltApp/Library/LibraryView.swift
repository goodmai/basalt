import BasaltAppCore
import SwiftUI

public struct LibraryView: View {
    let store: LibraryStore
    @State private var repo = ""
    @State private var deleting: String?

    public init(store: LibraryStore) { self.store = store }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                TextField("Hugging Face repo, such as mlx-community/gemma-4-12B-it-qat-4bit", text: $repo).onSubmit(download)
                Button("Download", action: download).disabled(!LibraryReducer.isRepoID(repo))
            }
            .padding(12)
            if let problem = store.state.problem {
                Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.secondary).padding(.horizontal, 12).padding(.bottom, 8)
            }
            List {
                if !store.state.downloads.isEmpty {
                    Section("Downloading") {
                        ForEach(store.state.downloads.keys.sorted(), id: \.self) { id in DownloadRow(id: id, download: store.state.downloads[id] ?? Download(), store: store) }
                    }
                }
                Section("On this Mac") {
                    ForEach(store.state.models) { model in ModelRow(model: model, fit: store.state.fits[model.id], store: store) { deleting = model.id } }
                }
            }
        }
        .frame(minWidth: 560, minHeight: 360)
        .confirmationDialog("Delete \(deleting ?? "")?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            Button("Delete", role: .destructive) { deleting.map { store.send(.delete($0)) } }
        } message: {
            Text("Its files are removed from the Hugging Face cache. It can be downloaded again.")
        }
        .task { store.send(.appeared) }
    }

    func download() {
        store.send(.download(repo.trimmingCharacters(in: .whitespaces)))
        repo = ""
    }
}

struct ModelRow: View {
    let model: LibraryModel
    let fit: ModelFit?
    let store: LibraryStore
    let delete: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.id).lineLimit(1)
                Text(ByteCountFormatter.string(fromByteCount: model.sizeBytes, countStyle: .file)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            FitBadge(fit: fit)
            if model.isLoaded {
                Text("Loaded").foregroundStyle(.green)
            } else {
                Button(store.state.loading == model.id ? "Loading…" : "Load") { store.send(.load(model.id)) }.disabled(!store.state.canLoad(model.id))
                Button("Delete", systemImage: "trash", role: .destructive, action: delete).labelStyle(.iconOnly)
            }
        }
    }
}

struct DownloadRow: View {
    let id: String
    let download: Download
    let store: LibraryStore

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(id).lineLimit(1)
                Spacer()
                if download.error != nil { Button("Retry") { store.send(.download(id)) } }
                Button("Cancel") { store.send(.cancelDownload(id)) }
            }
            ProgressView(value: download.fraction)
            Text(download.error ?? "\(ByteCountFormatter.string(fromByteCount: download.done, countStyle: .file)) of \(ByteCountFormatter.string(fromByteCount: download.total, countStyle: .file))")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct FitBadge: View {
    let fit: ModelFit?

    var body: some View {
        if let fit {
            Text(fit.fits ? "fits" : "does not fit")
                .font(.caption.weight(.medium))
                .padding(.horizontal, 8).padding(.vertical, 2)
                .background((fit.fits ? Color.green : Color.red).opacity(0.18), in: .capsule)
                .accessibilityLabel(fit.fits ? Text("Fits on this Mac") : Text("Does not fit on this Mac"))
                .help(fit.warning ?? "\(ByteCountFormatter.string(fromByteCount: fit.requiredBytes, countStyle: .memory)) needed, \(ByteCountFormatter.string(fromByteCount: fit.availableBytes, countStyle: .memory)) available")
        } else {
            ProgressView().controlSize(.small).accessibilityLabel("Checking whether it fits")
        }
    }
}
