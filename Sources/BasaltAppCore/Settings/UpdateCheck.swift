import BasaltCore
import Foundation

public struct AvailableUpdate: Sendable, Equatable {
    public var version: String
    public var url: URL
}

public enum UpdateCheck {
    public static let enabledKey = "checkUpdates"
    public static let lastKey = "lastUpdateCheck"
    public static let interval: TimeInterval = 86_400
    public static let releases = URL(string: "https://api.github.com/repos/goodmai/basalt/releases/latest")!

    public static func components(_ version: String) -> [Int] {
        version.trimmingCharacters(in: CharacterSet(charactersIn: "vV ")).split(separator: "-").first.map { $0.split(separator: ".").map { Int($0) ?? 0 } } ?? []
    }

    public static func isNewer(_ candidate: String, than current: String) -> Bool {
        let (a, b) = (components(candidate), components(current))
        for index in 0 ..< max(a.count, b.count) {
            let (x, y) = (index < a.count ? a[index] : 0, index < b.count ? b[index] : 0)
            if x != y { return x > y }
        }
        return false
    }

    public static func due(last: Date?, now: Date) -> Bool { last.map { now.timeIntervalSince($0) >= interval } ?? true }

    public static func parse(_ data: Data, current: String) -> AvailableUpdate? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let tag = json["tag_name"] as? String,
              json["draft"] as? Bool != true, json["prerelease"] as? Bool != true,
              let link = (json["html_url"] as? String).flatMap(URL.init(string:)), link.host == "github.com", isNewer(tag, than: current) else { return nil }
        return AvailableUpdate(version: String(tag.trimmingCharacters(in: CharacterSet(charactersIn: "vV"))), url: link)
    }

    public static func latest(current: String = HealthResponse.version, session: URLSession = .engine) async -> AvailableUpdate? {
        var request = URLRequest(url: releases)
        (request.timeoutInterval, request.cachePolicy) = (10, .reloadIgnoringLocalCacheData)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        guard let (data, response) = try? await session.data(for: request), (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return parse(data, current: current)
    }
}
