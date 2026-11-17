import BasaltAppCore
import BasaltCore
import Foundation

extension AppModel {
    public var visibleSummaries: [ConversationSummary] {
        let trimmed = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return summaries }
        return summaries.filter { summary in
            summary.title.localizedStandardContains(trimmed) || (matchedConversationIDs?.contains(summary.id) == true)
        }
    }

    public func updateSearch(_ query: String) async {
        searchQuery = query
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            matchedConversationIDs = nil
            return
        }
        let ids = (try? await repository.search(trimmed)) ?? []
        guard searchQuery == query else { return }
        matchedConversationIDs = Set(ids)
    }

    public func switchEngine(_ arguments: [String]) async {
        if let index = arguments.firstIndex(of: "--model"), arguments.indices.contains(index + 1) {
            configuredModel = arguments[index + 1]
        }
        let current = await supervisor.launch
        let isReady = if case .ready = engine.phase { true } else { false }
        switch HotSwitchPlan.plan(from: current.arguments, to: arguments, isReady: isReady) {
        case .noop:
            await supervisor.start()
        case .hotLoadModel(let id):
            if (try? await hotSwitchModel(id)) != nil { return }
            await supervisor.relaunch(HelperLaunch(executable: current.executable, arguments: arguments, environment: current.environment))
        case .hotApplyCompose(let path):
            if let bundled = ComposeProfileCatalog.bundledYAML[path] {
                if (try? await applyComposeYAML(bundled, filePath: path)) == true {
                    composeProfiles.send(.syncActive(path))
                    return
                }
            } else if (try? await applyComposeFile(URL(fileURLWithPath: path))) == true {
                composeProfiles.send(.syncActive(path))
                return
            }
            composeProfiles.send(.syncActive(path))
            await supervisor.relaunch(HelperLaunch(executable: current.executable, arguments: arguments, environment: current.environment))
        case .relaunch:
            await supervisor.relaunch(HelperLaunch(executable: current.executable, arguments: arguments, environment: current.environment))
        }
    }

    @discardableResult
    public func hotSwitchModel(_ id: String, client override: EngineClient? = nil) async throws -> String {
        guard LibraryReducer.isRepoID(id) else {
            throw EngineError.malformed("not a repo id: \(id)")
        }
        let resolved: EngineClient? = if let override {
            override
        } else {
            await supervisor.endpoint.map { EngineClient.http($0) }
        }
        guard let client = resolved else { throw EngineError.notRunning }
        let loaded = try await client.loadModel(id)
        configuredModel = loaded
        await supervisor.updateLaunchArguments(["serve", "--model", loaded])
        UserDefaults.standard.set(loaded, forKey: EngineChoice.modelKey)
        library.send(.appeared)
        return loaded
    }

    @discardableResult
    public func applyComposeFile(_ url: URL, client override: EngineClient? = nil) async throws -> Bool {
        let yaml = try String(contentsOf: url, encoding: .utf8)
        return try await applyComposeYAML(yaml, base: url.deletingLastPathComponent().path, filePath: url.path, client: override)
    }

    @discardableResult
    public func applyComposeYAML(
        _ yaml: String,
        base: String? = nil,
        filePath: String? = nil,
        client override: EngineClient? = nil
    ) async throws -> Bool {
        let resolved: EngineClient? = if let override {
            override
        } else {
            await supervisor.endpoint.map { EngineClient.http($0) }
        }
        if let client = resolved {
            do {
                let diff = try await client.applyCompose(ComposeApplyCall(yaml: yaml, base: base))
                if let refusal = diff.refusal {
                    throw EngineError.refused(status: 409, message: refusal)
                }
                if diff.restart == nil {
                    if let filePath {
                        await supervisor.updateLaunchArguments(["serve", "--compose", filePath])
                        UserDefaults.standard.set(filePath, forKey: EngineChoice.composeKey)
                    }
                    return true
                }
            } catch EngineError.refused(404, _) {
            }
        }
        guard let filePath else { return false }
        let current = await supervisor.launch
        UserDefaults.standard.set(filePath, forKey: EngineChoice.composeKey)
        await supervisor.relaunch(HelperLaunch(executable: current.executable, arguments: ["serve", "--compose", filePath], environment: current.environment))
        return false
    }

    func submitVoiceTranscript(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if selection == nil { newConversation() }
        guard let id = selection, let store = chats[id] else { return }
        store.send(.draftChanged(trimmed))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
    }

    func handleAnswered(_ snapshot: ConversationSnapshot) {
        answered(snapshot)
        if let last = snapshot.messages.last, last.role == .assistant {
            voice.store.send(.assistantAnswered(last.id, last.text))
        }
    }

    func watchComposeEvents(for state: EngineState) {
        composeEventsTask?.cancel()
        composeEventsTask = nil
        guard case .ready = state.phase else { return }
        let supervisor = supervisor
        composeEventsTask = Task { [weak self] in
            guard let endpoint = await supervisor.endpoint else { return }
            let stream = EngineClient.http(endpoint).composeEvents()
            do {
                for try await event in stream {
                    await MainActor.run { self?.latestComposeEvent = event }
                }
            } catch {}
        }
    }
}
