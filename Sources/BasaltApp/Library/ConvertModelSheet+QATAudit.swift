import BasaltAppCore
import BasaltCore
import SwiftUI

extension ConvertModelSheet {
    func qatAuditSection(_ draft: ConversionDraft) -> some View {
        let supported = draft.resolvedTarget.supportsPVTune
        return VStack(alignment: .leading, spacing: 8) {
            Toggle(
                "QAT Fake-Quant Refinement (Domain STE)",
                isOn: Binding(
                    get: { draft.qatRefine },
                    set: { value in
                        var next = draft
                        next.qatRefine = value
                        store.send(.updateConvertDraft(next))
                    }
                )
            )
            .disabled(!supported)
            Toggle(
                "Post-Quantization Quality & Fairness Audit",
                isOn: Binding(
                    get: { draft.runPostQuantAudit },
                    set: { value in
                        var next = draft
                        next.runPostQuantAudit = value
                        store.send(.updateConvertDraft(next))
                    }
                )
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 12))
    }
}
