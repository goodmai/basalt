import BasaltCore
import Foundation
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app phase 5: the update check against GitHub releases, and a Russian table that covers every UI string (#434 phase 5)")
struct AppShipTests {
    static let release = #"{"tag_name":"v0.11.0","html_url":"https://github.com/goodmai/basalt/releases/tag/v0.11.0","draft":false,"prerelease":false}"#

    @Test("Versions compare numerically; drafts, pre-releases, older tags and links off github.com are ignored")
    func versions() {
        #expect(UpdateCheck.isNewer("v0.11.0", than: "0.10.0") && UpdateCheck.isNewer("0.10.1", than: "0.10") && !UpdateCheck.isNewer("v0.10.0", than: "0.10.0"))
        #expect(!UpdateCheck.isNewer("0.9.9", than: "0.10.0") && UpdateCheck.isNewer("1.0.0-beta", than: "0.10.0"))
        #expect(UpdateCheck.parse(Data(Self.release.utf8), current: "0.10.0") == AvailableUpdate(version: "0.11.0", url: URL(string: "https://github.com/goodmai/basalt/releases/tag/v0.11.0")!))
        for changed in [Self.release.replacingOccurrences(of: "\"draft\":false", with: "\"draft\":true"),
                        Self.release.replacingOccurrences(of: "\"prerelease\":false", with: "\"prerelease\":true"),
                        Self.release.replacingOccurrences(of: "https://github.com", with: "https://evil.example")] {
            #expect(UpdateCheck.parse(Data(changed.utf8), current: "0.10.0") == nil)
        }
        #expect(UpdateCheck.parse(Data(Self.release.utf8), current: "0.11.0") == nil && UpdateCheck.parse(Data("[]".utf8), current: "0") == nil)
    }

    @MainActor
    @Test("The app checks at most once a day, never when turned off, and shows what it found")
    func schedule() async throws {
        let suite = "basalt-update-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = try AppModel(launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: []), model: "org/model", storeURL: nil, drafts: DraftStore(load: { _ in "" }, save: { _, _ in }))
        let found = AvailableUpdate(version: "0.11.0", url: URL(string: "https://github.com/goodmai/basalt/releases/tag/v0.11.0")!)
        let now = Date()
        defaults.set(false, forKey: UpdateCheck.enabledKey)
        await model.checkForUpdate(defaults: defaults, now: now) { found }
        #expect(model.update == nil)
        defaults.set(true, forKey: UpdateCheck.enabledKey)
        await model.checkForUpdate(defaults: defaults, now: now) { found }
        #expect(model.update == found && defaults.object(forKey: UpdateCheck.lastKey) as? Date == now)
        await model.checkForUpdate(defaults: defaults, now: now.addingTimeInterval(3600)) { nil }
        #expect(model.update == found && !UpdateCheck.due(last: now, now: now.addingTimeInterval(3600)) && UpdateCheck.due(last: now, now: now.addingTimeInterval(90_000)))
    }

    @Test("Every static string in the app's views has a Russian translation")
    func russian() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let table = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        let untranslated: Set<String> = ["Basalt", "GPU", "OPENROUTER_API_KEY"]
        let call = #/(?:Text|Button|Label|Toggle|TextField|Section|Tab|LabeledContent|Picker|SecureField|Window|WindowGroup|ContentUnavailableView|Link|DisclosureGroup|String\(localized:|accessibilityLabel\(Text|confirmationDialog)\(\s*"([^"\\]+)"/#
        var missing: [String] = []
        for folder in ["Sources/BasaltApp", "Sources/BasaltMac"] {
            let files = FileManager.default.enumerator(atPath: root.appending(path: folder).path)?.compactMap { $0 as? String }.filter { $0.hasSuffix(".swift") } ?? []
            for file in files {
                let source = try String(contentsOf: root.appending(path: "\(folder)/\(file)"), encoding: .utf8)
                for match in source.matches(of: call) where !untranslated.contains(String(match.1)) && table[String(match.1)] == nil { missing.append("\(file): \(match.1)") }
            }
        }
        #expect(missing.isEmpty, "untranslated: \(missing)")
        #expect(table.values.allSatisfy { !$0.isEmpty } && table.count >= 90)
    }
}
