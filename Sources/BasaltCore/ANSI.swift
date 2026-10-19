import Foundation

// MARK: — ANSI helpers

package func bold(_ s: String)   -> String { "\u{1B}[1m\(s)\u{1B}[0m" }
package func dim(_ s: String)    -> String { "\u{1B}[2m\(s)\u{1B}[0m" }
package func green(_ s: String)  -> String { "\u{1B}[32m\(s)\u{1B}[0m" }
package func red(_ s: String)    -> String { "\u{1B}[31m\(s)\u{1B}[0m" }
package func yellow(_ s: String) -> String { "\u{1B}[33m\(s)\u{1B}[0m" }
package func blue(_ s: String)   -> String { "\u{1B}[34m\(s)\u{1B}[0m" }
package func cyan(_ s: String)   -> String { "\u{1B}[36m\(s)\u{1B}[0m" }

package func padded(_ s: String, _ width: Int) -> String {
    s.count >= width ? String(s.prefix(width)) : s + String(repeating: " ", count: width - s.count)
}

package func formatBytes(_ bytes: Int64) -> String {
    switch bytes {
    case 0..<1_024:                    return "\(bytes) B"
    case 0..<1_048_576:                return String(format: "%.1f KB", Double(bytes) / 1_024)
    case 0..<1_073_741_824:            return String(format: "%.1f MB", Double(bytes) / 1_048_576)
    default:                           return String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
    }
}
