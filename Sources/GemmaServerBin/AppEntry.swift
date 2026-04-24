import ArgumentParser
import GemmaServerCore

@main
@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct AppEntry: AsyncParsableCommand {
    static let configuration = GemmaServerCLI.configuration
}
