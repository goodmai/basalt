import Foundation

public struct StepLease: Sendable {
    public let enter: @Sendable () async throws(BasaltError) -> Int
    public let leave: @Sendable (Int) async -> Void

    public init(enter: @escaping @Sendable () async throws(BasaltError) -> Int, leave: @escaping @Sendable (Int) async -> Void) {
        self.enter = enter
        self.leave = leave
    }
}
