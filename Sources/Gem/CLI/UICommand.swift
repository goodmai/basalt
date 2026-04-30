import Foundation
import ArgumentParser

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
public struct UICommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "ui",
        abstract: "Launch the OpenAPI REST UI client."
    )

    @Option(name: .shortAndLong, help: "REST API base URL")
    var url: String = "http://127.0.0.1:8080"

    public init() {}

    public mutating func run() async throws {
        let logger = GemLogger(module: "UICommand")
        logger.info("Launching REST UI client connecting to \(url)...")
        let baseURL = url
        
        await MainActor.run {
            launchRESTUI(baseURL: baseURL)
        }
    }
}
