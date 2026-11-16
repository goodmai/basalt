import Foundation

public enum AppSettings {
    public static let modelKey = "model"
    public static let quickAskKey = "quickAsk"
    public static let keepEngineOnQuitKey = "keepEngineOnQuit"

    public static func keepEngineOnQuit(in defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: keepEngineOnQuitKey)
    }

    public static func setKeepEngineOnQuit(_ enabled: Bool, in defaults: UserDefaults = .standard) {
        defaults.set(enabled, forKey: keepEngineOnQuitKey)
    }
}
