import Foundation

extension ModelDatabase {
    static let translateGemmaModels: [ModelDefinition] = [
        ModelDefinition(
            id: ComposeProfileCatalog.translateGemma12BPack,
            name: "TranslateGemma 4 12B Pilot (PV-MACaQ 4-bit MXFP4)",
            family: .gemma,
            task: .chat,
            modality: .multimodal,
            spec: ModelDefinition.Spec(
                ramMB: 6700,
                contextWindow: 262144,
                quantization: "PV-MACaQ-4bit-MXFP4",
                quality: 0.91,
                totalParameters: 12.0,
                activeParameters: 12.0
            )
        ),
        ModelDefinition(
            id: ComposeProfileCatalog.translateGemma26BPack,
            name: "TranslateGemma 4 26B A4B v2 (PV-MACaQ 4-bit MXFP4)",
            family: .gemma,
            task: .chat,
            modality: .multimodal,
            spec: ModelDefinition.Spec(
                ramMB: 14400,
                contextWindow: 262144,
                quantization: "PV-MACaQ-4bit-MXFP4",
                quality: 0.94,
                totalParameters: 26.0,
                activeParameters: 4.0
            )
        ),
    ]
}
