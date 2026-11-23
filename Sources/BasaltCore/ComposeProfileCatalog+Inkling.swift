import Foundation

extension ComposeProfileCatalog {
    static let inklingAgents: [V22ComposeSpecialist] = [
        inklingAgent(tier: .gb24, id: "24gb-agent-inkling", pack: "mlx-community/Inkling-2-Mini-oQ3e-mtp", peakGB: 16.5, context: "32K"),
        inklingAgent(tier: .gb36, id: "36gb-agent-inkling", pack: "mlx-community/Inkling-2-Mini-oQ4e-mtp", peakGB: 20.8, context: "64K"),
        inklingAgent(tier: .gb48, id: "48gb-agent-inkling", pack: "mlx-community/Inkling-2-Mini-oQ4e-mtp", peakGB: 23.4, context: "128K"),
    ]

    private static func inklingAgent(tier: ComposeMemoryTier, id: String, pack: String, peakGB: Double, context: String) -> V22ComposeSpecialist {
        V22ComposeSpecialist(
            id: id,
            tier: tier,
            archetype: .codingAgent,
            primaryModel: pack,
            estimatedPeakGB: peakGB,
            badges: ["Inkling 2", "Tool calls", "DSpark MTP", context],
            yaml: """
                version: "2.2"
                name: \(id)
                memory: auto
                mcp:
                  enabled: true
                  rest: true
                models:
                  agent:
                    source: \(pack)
                pipeline:
                  answer:
                    model: agent
                """
        )
    }
}
