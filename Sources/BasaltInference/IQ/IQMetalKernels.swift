import BasaltCore
import Foundation
import MLX

package enum IQMetalKernels {
    package static let supportedTypes: [GGMLType] = [
        .iq1S, .iq1M, .iq2XXS, .iq2XS, .iq2S, .iq3XXS, .iq3S, .iq4NL, .iq4XS,
    ]

    package static let smallMRows = 2 ... 8
    private static let simdgroups = IQMetalShaders.simdgroups
    private static let rowsPerSimdgroup = 4

    private struct Entry: Sendable {
        let matvec: MLXFast.MLXFastKernel
        let smallM: MLXFast.MLXFastKernel
    }

    private static let kernels: [GGMLType: Entry] = Dictionary(
        uniqueKeysWithValues: supportedTypes.map { type in
            let header = IQMetalShaders.header(for: type)
            let mv = MLXFast.metalKernel(
                name: "iq_matvec_\(type.name)",
                inputNames: ["x", "w"],
                outputNames: ["out"],
                source: IQMetalShaders.matvecSource(for: type),
                header: header
            )
            let sm = MLXFast.metalKernel(
                name: "iq_smallm_\(type.name)",
                inputNames: ["x", "w"],
                outputNames: ["out"],
                source: IQMetalShaders.smallMSource(for: type),
                header: header
            )
            return (type, Entry(matvec: mv, smallM: sm))
        }
    )

    package static func supports(_ type: GGMLType) -> Bool {
        kernels[type] != nil
    }

    package static func matvec(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        let vector = x.ndim == 1 ? x.reshaped(1, x.dim(0)) : x
        precondition(vector.dim(0) == 1, "IQMetalKernels.matvec requires M == 1, got \(vector.dim(0))")
        guard let entry = kernels[type] else {
            return GGUFKernels.legacyMatvec(vector, packed, type: type)
        }
        let columns = vector.dim(1)
        let outputs = packed.dim(0)
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        let template: [(String, any KernelTemplateArg)] = [
            ("K", columns),
            ("BE", type.block.elements),
            ("BB", type.block.bytes),
            ("T", vector.dtype),
        ]
        let out = entry.matvec(
            [vector, packed],
            template: template,
            grid: (32, simdgroups * groups, 1),
            threadGroup: (32, simdgroups, 1),
            outputShapes: [[1, outputs]],
            outputDTypes: [vector.dtype]
        )[0]
        return x.ndim == 1 ? out.reshaped(outputs) : out
    }

    package static func smallM(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        let rows = x.dim(0)
        if rows == 1 { return matvec(x, packed, type: type) }
        guard let entry = kernels[type], smallMRows.contains(rows) else {
            return GGUFKernels.matmul(x, packed, type: type, columns: x.dim(1))
        }
        let columns = x.dim(1)
        let outputs = packed.dim(0)
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        let template: [(String, any KernelTemplateArg)] = [
            ("K", columns),
            ("BE", type.block.elements),
            ("BB", type.block.bytes),
            ("M", rows),
            ("T", x.dtype),
        ]
        return entry.smallM(
            [x, packed],
            template: template,
            grid: (32, simdgroups * groups, 1),
            threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows, outputs]],
            outputDTypes: [x.dtype]
        )[0]
    }
}
