import Foundation

public enum EngineError: Error, Sendable, Equatable {
    case notRunning
    case starting
    case refused(status: Int, message: String)
    case malformed(String)
    case transport(String)
    case cancelled
}
