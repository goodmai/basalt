import Foundation
import MLX

public enum MetallibOverride {
    public static let environmentKey = "BASALT_METALLIB"

    @discardableResult
    public static func apply(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        guard let path = environment[environmentKey], !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return nil }
        let url = URL(fileURLWithPath: path)
        GPU.metallib = url
        return url
    }
}
