import ArgumentParser
import BasaltCLI
import BasaltInference
import Foundation

@main
@available(macOS 14.0, *)
struct AppEntry {
    static func main() {
        MetallibOverride.apply()
        Task {
            await BasaltCLI.main(BasaltCLI.configuredArguments())
            Darwin.exit(0)
        }

        // Keep the main thread alive and process events, avoiding deadlocks
        // with MainActor or Hummingbird while the CLI runs.
        RunLoop.main.run()
    }
}