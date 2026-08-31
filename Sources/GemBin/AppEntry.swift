import ArgumentParser
import GemCore
import Foundation

@main
@available(macOS 14.0, *)
struct AppEntry {
    static func main() {
        // Execute the CLI asynchronously with proper run loop management
        Task {
            await GemCLI.main()
            Darwin.exit(0)
        }

        // Keep the main thread alive and process events, avoiding deadlocks
        // with MainActor or Hummingbird while the CLI runs.
        RunLoop.main.run()
    }
}