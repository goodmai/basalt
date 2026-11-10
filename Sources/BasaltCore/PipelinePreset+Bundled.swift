import Foundation

extension PipelinePreset {
    public static let gemma = PipelineMember(.chat, "mlx-community/gemma-4-12B-it-qat-4bit", bytes: 10_987_772_430)
    public static let qwen = PipelineMember(.chat, "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly", bytes: 11_771_374_457)
    public static let paddleOCR = PipelineMember(.ocr, "mlx-community/PaddleOCR-VL-1.5-8bit", bytes: 1_088_866_628)
    public static let nemotronTranscribe = PipelineMember(.audio, "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX", bytes: 14_900_664_231)
    public static let selfDecisions = PipelineMember(.decide, "self")

    static let bundled = (
        documentQA: PipelineProfile(members: [gemma, paddleOCR, selfDecisions], largerChat: qwen, abstainBelow: 0.75, minMargin: 0.1, questions: [:]),
        voiceAgent: PipelineProfile(members: [gemma, nemotronTranscribe], largerChat: qwen, abstainBelow: nil, minMargin: nil, questions: [:]),
        supportTriage: PipelineProfile(members: [gemma, selfDecisions], largerChat: qwen, abstainBelow: 0.75, minMargin: 0.1, questions: triageQuestions)
    )

    public static let triageQuestions: [String: SystemOneQuestion] = [
        "department": SystemOneQuestion(type: .choice, instructions: "Which department should handle this ticket?", criteria: .named([
            "account": "Account: login, password, profile settings.",
            "billing": "Billing: invoices, charges, refunds of payments.",
            "returns": "Returns: sending items back and exchanges.",
            "sales": "Sales: prices, quotes and new purchases.",
            "shipping": "Shipping: delivery status, tracking, couriers.",
            "technical": "Technical support: the app or device does not work.",
        ])),
        "late": SystemOneQuestion(type: .noul, instructions: "The order was delivered later than promised.", criteria: .named([
            "false": "The order was delivered on time or early.", "true": "The order was delivered later than promised.",
        ])),
        "severity": SystemOneQuestion(type: .score, instructions: "How severe is the incident?", criteria: .levels([
            "No impact: cosmetic or informational.", "Minor: one user affected, workaround exists.",
            "Major: many users affected or a key feature down.", "Critical: outage or data loss.",
        ])),
    ]
}
