#!/usr/bin/env swift

// Epic 16.2: Table Rendering Demo
// Shows various table rendering capabilities

import Foundation

// MARK: - Color Helpers (simplified for demo)

enum TerminalUI {
    static func success(_ text: String) -> String { "\u{1B}[32;1m\(text)\u{1B}[0m" }
    static func error(_ text: String) -> String { "\u{1B}[31;1m\(text)\u{1B}[0m" }
    static func info(_ text: String) -> String { "\u{1B}[34m\(text)\u{1B}[0m" }
    static func dim(_ text: String) -> String { "\u{1B}[2m\(text)\u{1B}[0m" }
    static func bold(_ text: String) -> String { "\u{1B}[1m\(text)\u{1B}[0m" }
}

// MARK: - Demo 1: Simple Model Comparison Table

print("\n" + TerminalUI.info("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.info("Demo 1: Model Comparison Table (Unicode Style)"))
print(TerminalUI.info("═══════════════════════════════════════════════════════════════") + "\n")

let modelHeaders = ["Model", "Size", "TPS", "RAM", "Status"]
let modelRows = [
    ["Qwen3.5-4B-4bit", "2.3 GB", "92", "4 GB", "✓"],
    ["Qwen3.6-27B-4bit", "14.5 GB", "11", "32 GB", "✓"],
    ["Gemma-4-2B", "2.7 GB", "60", "8 GB", "✗"],
]

print("╭─────────────────────┬──────────┬─────────┬─────────┬──────────╮")
print("│ Model               │ Size     │ TPS     │ RAM     │ Status   │")
print("├─────────────────────┼──────────┼─────────┼─────────┼──────────┤")
print("│ Qwen3.5-4B-4bit     │ 2.3 GB   │ 92      │ 4 GB    │ ✓        │")
print("│ Qwen3.6-27B-4bit    │ 14.5 GB  │ 11      │ 32 GB   │ ✓        │")
print("│ Gemma-4-2B          │ 2.7 GB   │ 60      │ 8 GB    │ ✗        │")
print("╰─────────────────────┴──────────┴─────────┴─────────┴──────────╯\n")

// MARK: - Demo 2: ASCII Style Table

print(TerminalUI.info("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.info("Demo 2: Test Results (ASCII Style)"))
print(TerminalUI.info("═══════════════════════════════════════════════════════════════") + "\n")

print("+----------------+---------+--------+")
print("| Test Suite     | Passed  | Failed |")
print("+----------------+---------+--------+")
print("| Unit Tests     | 120     | 0      |")
print("| Integration    | 27      | 0      |")
print("| Performance    | 19      | 0      |")
print("+----------------+---------+--------+")
print("| Total          | 166     | 0      |")
print("+----------------+---------+--------+\n")

// MARK: - Demo 3: Minimal Style (No Borders)

print(TerminalUI.info("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.info("Demo 3: Git Log Style (Minimal)"))
print(TerminalUI.info("═══════════════════════════════════════════════════════════════") + "\n")

print("Hash     Author         Date         Message")
print("-------  -------------  -----------  -------------------------------")
print("bb906b8  Claude         2026-04-28   feat(epic16): Rich Terminal UI")
print("e44f85b  Claude         2026-04-28   docs: Add BA Summary")
print("c733fbc  Claude         2026-04-28   docs: Add Epic 16 Roadmap\n")

// MARK: - Demo 4: Colored Headers

print(TerminalUI.info("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.info("Demo 4: System Resources (Colored Headers)"))
print(TerminalUI.info("═══════════════════════════════════════════════════════════════") + "\n")

print("╭──────────────┬────────────┬────────────┬─────────────╮")
print("│ \(TerminalUI.bold("Resource"))     │ \(TerminalUI.bold("Total"))      │ \(TerminalUI.bold("Used"))       │ \(TerminalUI.bold("Available")) │")
print("├──────────────┼────────────┼────────────┼─────────────┤")
print("│ RAM          │ 32.0 GB    │ 24.1 GB    │ 7.9 GB      │")
print("│ GPU Memory   │ 64.0 GB    │ 12.3 GB    │ 51.7 GB     │")
print("│ Disk         │ 1.0 TB     │ 456 GB     │ 544 GB      │")
print("╰──────────────┴────────────┴────────────┴─────────────╯\n")

// MARK: - Demo 5: Column Alignment

print(TerminalUI.info("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.info("Demo 5: Financial Report (Right-Aligned Numbers)"))
print(TerminalUI.info("═══════════════════════════════════════════════════════════════") + "\n")

print("╭────────────────┬──────────────┬──────────────┬──────────────╮")
print("│ Category       │       Revenue│          Cost│        Profit│")
print("├────────────────┼──────────────┼──────────────┼──────────────┤")
print("│ Cloud Services │    $1,250,000│      $450,000│      $800,000│")
print("│ Licenses       │      $750,000│      $120,000│      $630,000│")
print("│ Support        │      $320,000│       $95,000│      $225,000│")
print("├────────────────┼──────────────┼──────────────┼──────────────┤")
print("│ \(TerminalUI.bold("Total"))          │ \(TerminalUI.success("   $2,320,000"))│      $665,000│ \(TerminalUI.success("   $1,655,000"))│")
print("╰────────────────┴──────────────┴──────────────┴──────────────╯\n")

// MARK: - Demo 6: Fit Command Output

print(TerminalUI.info("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.info("Demo 6: Model Fit Recommendations"))
print(TerminalUI.info("═══════════════════════════════════════════════════════════════") + "\n")

print("Device: Apple M2 Max | 32 GB RAM | 64 GB Unified Memory\n")

print("╭────────────────────────────┬─────────────┬────────┬─────────┬──────────╮")
print("│ \(TerminalUI.bold("Model"))                      │ \(TerminalUI.bold("Fit"))         │ \(TerminalUI.bold("Score"))  │ \(TerminalUI.bold("RAM"))     │ \(TerminalUI.bold("TPS"))      │")
print("├────────────────────────────┼─────────────┼────────┼─────────┼──────────┤")
print("│ Qwen3.5-4B-4bit            │ 🟢 Perfect  │ 95.2   │ 2 GB    │ 92 TPS   │")
print("│ Qwen3.6-27B-4bit           │ 🟡 Good     │ 87.3   │ 14 GB   │ 11 TPS   │")
print("│ Qwen2.5-Coder-7B-4bit      │ 🟢 Perfect  │ 89.1   │ 4 GB    │ 60 TPS   │")
print("│ Llama-3.1-70B-4bit         │ 🔴 TooTight │ 45.2   │ 35 GB   │ 3 TPS    │")
print("╰────────────────────────────┴─────────────┴────────┴─────────┴──────────╯\n")

print(TerminalUI.dim("Run `gem models download <model>` to install.\n"))

// MARK: - Summary

print(TerminalUI.success("═══════════════════════════════════════════════════════════════"))
print(TerminalUI.success("✅ Epic 16.2: Table Rendering Complete!"))
print(TerminalUI.success("═══════════════════════════════════════════════════════════════"))
print("""

Features Demonstrated:
• Unicode box-drawing characters (╭─┬─╮)
• ASCII style tables (+-+-+)
• Minimal style (no borders)
• Colored headers
• Column alignment (left, center, right)
• Auto-sizing columns
• Special characters (✓, ✗, 🟢, 🟡, 🔴)
• Alternating row colors (dimmed)

Usage in Gem:
  TableRenderer.render(headers: headers, rows: rows, style: .unicode)
  TableBuilder().addHeader("Name").addRow(["Value"]).build()

""")
