import BasaltAppCore
import BasaltCore
import SwiftUI

public struct ConvertModelSheet: View {
    let store: LibraryStore
    @Environment(\.glassSurfacePolicy) private var policy

    public init(store: LibraryStore) {
        self.store = store
    }

    public var body: some View {
        if let draft = store.state.convertDraft {
            VStack(alignment: .leading, spacing: 16) {
                header(draft)
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        targetSection(draft)
                        planCard(draft)
                        pvTuneSection(draft)
                        qatAuditSection(draft)
                        outputSection(draft)
                    }
                    .padding(.vertical, 2)
                }
                Divider()
                footer(draft)
            }
            .padding(20)
            .frame(width: 560, height: 670)
        }
    }

    private func header(_ draft: ConversionDraft) -> some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "arrow.triangle.2.circlepath.circle.fill")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 3) {
                Text("Convert & PV-Tune Model")
                    .font(.headline)
                HStack(spacing: 8) {
                    Text(draft.sourceId)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    if let srcFmt = draft.plan?.sourceFormat {
                        Text(srcFmt)
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.16), in: .capsule)
                    }
                    Text(ByteCountFormatter.string(fromByteCount: draft.sourceBytes, countStyle: .file))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
    }

    private func targetSection(_ draft: ConversionDraft) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Target Quantization")
                .font(.subheadline.weight(.semibold))
            HStack {
                Text("Format Family")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Menu {
                    ForEach(ConversionTargetFamily.allCases) { fam in
                        Button(LocalizedStringKey(fam.title)) {
                            var next = draft
                            next.family = fam
                            next.refreshSuggestedOutputId()
                            store.send(.updateConvertDraft(next))
                        }
                    }
                } label: {
                    menuPillLabel(LocalizedStringKey(draft.family.title))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
            targetParameters(draft)
        }
        .padding(12)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 12))
    }
}

extension ConvertModelSheet {
    @ViewBuilder
    fileprivate func targetParameters(_ draft: ConversionDraft) -> some View {
        switch draft.family {
        case .macaq:
            Toggle("Use OCP MXFP4 Microscaling (E2M1 + E8M0)", isOn: binding(draft, \.useMXFP4InMACaQ, refreshOutput: true))
            if !draft.useMXFP4InMACaQ {
                bitAndGroupPickers(draft, bitOptions: [2, 3, 4])
            }
            hadamardBlockPicker(draft)
        case .macaqMoE:
            VStack(alignment: .leading, spacing: 4) {
                Text("Target Budget: \(String(format: "%.2f", draft.moeBudgetBPW)) bpw")
                    .font(.caption)
                Slider(value: binding(draft, \.moeBudgetBPW, refreshOutput: true), in: 2.0 ... 4.0, step: 0.1)
            }
            hadamardBlockPicker(draft)
        case .flatQuant:
            bitAndGroupPickers(draft, bitOptions: [2, 3, 4, 8])
        case .lirQuant:
            bitAndGroupPickers(draft, bitOptions: [2, 3, 4, 8])
            lirSubspacePicker(draft)
        case .affine:
            bitAndGroupPickers(draft, bitOptions: [2, 3, 4, 5, 6, 8])
            recipePicker(draft)
        case .nf4:
            Picker("Group Size", selection: binding(draft, \.groupSize, refreshOutput: true)) {
                ForEach([32, 64, 128], id: \.self) { g in Text("\(g)").tag(g) }
            }
            recipePicker(draft)
        case .mxfp4:
            recipePicker(draft)
        case .qtip:
            Picker("Trellis Rate", selection: binding(draft, \.qtipRate, refreshOutput: true)) {
                Text("2.0 bpw").tag(2.0)
                Text("2.5 bpw").tag(2.5)
                Text("3.0 bpw").tag(3.0)
            }
            .pickerStyle(.segmented)
        case .qtipBudget, .gsqRCO:
            VStack(alignment: .leading, spacing: 4) {
                Text("Target Budget: \(String(format: "%.2f", draft.budgetBPW)) bpw")
                    .font(.caption)
                Slider(value: binding(draft, \.budgetBPW, refreshOutput: true), in: 2.0 ... 4.0, step: 0.25)
            }
        case .dense:
            Picker("Precision", selection: binding(draft, \.denseFP16, refreshOutput: true)) {
                Text("BF16").tag(false)
                Text("FP16").tag(true)
            }
            .pickerStyle(.segmented)
        }
    }

    private func hadamardBlockPicker(_ draft: ConversionDraft) -> some View {
        Picker("Hadamard Block", selection: binding(draft, \.hadamardBlock)) {
            Text("32").tag(32)
            Text("64").tag(64)
        }
        .pickerStyle(.segmented)
    }

    private func lirSubspacePicker(_ draft: ConversionDraft) -> some View {
        Picker("Vulnerable Subspace", selection: binding(draft, \.lirSubspaceDim, refreshOutput: true)) {
            Text("64").tag(64)
            Text("128").tag(128)
            Text("256").tag(256)
        }
        .pickerStyle(.segmented)
    }

    private func bitAndGroupPickers(_ draft: ConversionDraft, bitOptions: [Int]) -> some View {
        HStack(spacing: 16) {
            Picker("Bit-width", selection: binding(draft, \.bits, refreshOutput: true)) {
                ForEach(bitOptions, id: \.self) { b in Text("\(b)-bit").tag(b) }
            }
            .pickerStyle(.segmented)
            Picker("Group Size", selection: binding(draft, \.groupSize, refreshOutput: true)) {
                ForEach([32, 64, 128], id: \.self) { g in Text("\(g)").tag(g) }
            }
            .pickerStyle(.segmented)
        }
    }

    private func recipePicker(_ draft: ConversionDraft) -> some View {
        Picker("Normalization Recipe", selection: binding(draft, \.recipe)) {
            Text("SINQ").tag("sinq")
            Text("A-SINQ").tag("a-sinq")
            Text("MoE-RW").tag("moe-rw")
            Text("RTN").tag("rtn")
        }
        .pickerStyle(.segmented)
    }
}

extension ConvertModelSheet {
    fileprivate func planCard(_ draft: ConversionDraft) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "memorychip")
                .foregroundStyle(.secondary)
            if let plan = draft.plan {
                VStack(alignment: .leading, spacing: 3) {
                    let beforeStr = ByteCountFormatter.string(fromByteCount: plan.sourceBytes, countStyle: .file)
                    let afterStr = ByteCountFormatter.string(fromByteCount: plan.estimatedOutputBytes, countStyle: .file)
                    Text("Estimated Size: \(beforeStr) → \(afterStr)")
                        .font(.caption.weight(.semibold))
                    let peakStr = ByteCountFormatter.string(fromByteCount: plan.peakConversionRAMBytes, countStyle: .memory)
                    Text("Peak Conversion RAM: \(peakStr)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(plan.fitsConversionOnThisMac ? "Fits RAM" : "High RAM")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background((plan.fitsConversionOnThisMac ? Color.green : Color.orange).opacity(0.18), in: .capsule)
            } else {
                Text("Estimating output footprint…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding(10)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 10))
    }

    private func pvTuneSection(_ draft: ConversionDraft) -> some View {
        let supported = draft.resolvedTarget.supportsPVTune
        return VStack(alignment: .leading, spacing: 8) {
            Toggle("PV-Tuning (Teacher KL Code & Scale Refinement)", isOn: binding(draft, \.pvTune, refreshOutput: true))
                .disabled(!supported)
            if supported && draft.pvTune {
                Picker("Preset", selection: binding(draft, \.pvPreset)) {
                    Text("Fast").tag(PVTunePreset.fast)
                    Text("Balanced").tag(PVTunePreset.balanced)
                    Text("Thorough").tag(PVTunePreset.thorough)
                }
                .pickerStyle(.segmented)
                HStack {
                    Text("Teacher Model")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Menu {
                        Button("Source model (default)") {
                            var next = draft; next.pvTeacher = ""; store.send(.updateConvertDraft(next))
                        }
                        ForEach(store.state.models) { m in
                            Button(m.id) {
                                var next = draft; next.pvTeacher = m.id; store.send(.updateConvertDraft(next))
                            }
                        }
                    } label: {
                        menuPillLabel(draft.pvTeacher.isEmpty ? "Source model (default)" : LocalizedStringKey(draft.pvTeacher))
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                }
            } else if !supported {
                Text("PV-Tuning applies to quantized targets (MACaQ, Affine, MXFP4, NF4, QTIP).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 12))
    }

    fileprivate func menuPillLabel(_ title: LocalizedStringKey) -> some View {
        HStack(spacing: 5) {
            Text(title)
                .font(.callout.weight(.medium))
                .lineLimit(1)
            Image(systemName: "chevron.up.chevron.down")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.secondary.opacity(0.14), in: .rect(cornerRadius: 7))
    }

    private func outputSection(_ draft: ConversionDraft) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Output Repository ID")
                .font(.subheadline.weight(.semibold))
            TextField("local/model-name", text: binding(draft, \.outputId))
                .textFieldStyle(.plain)
                .font(.callout.monospaced())
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 8))
        }
    }

    private func footer(_ draft: ConversionDraft) -> some View {
        let valid = LibraryReducer.isRepoID(draft.outputId.trimmingCharacters(in: .whitespacesAndNewlines))
        return HStack {
            Button("Cancel") { store.send(.closeConvert) }
                .adaptiveGlassButton()
            Spacer()
            Button(action: { store.send(.startConversion) }) {
                Text("Start Conversion")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .background((valid ? Color.accentColor : Color.secondary.opacity(0.4)), in: .capsule)
            }
            .buttonStyle(.plain)
            .disabled(!valid)
        }
    }

    fileprivate func binding<V: Equatable>(
        _ draft: ConversionDraft,
        _ keyPath: WritableKeyPath<ConversionDraft, V>,
        refreshOutput: Bool = false
    ) -> Binding<V> {
        Binding(
            get: { draft[keyPath: keyPath] },
            set: { newValue in
                var next = draft
                next[keyPath: keyPath] = newValue
                if refreshOutput { next.refreshSuggestedOutputId() }
                store.send(.updateConvertDraft(next))
            })
    }
}
