import ArgumentParser
import BasaltCore

struct AdapterTrainingOptions: ParsableArguments {
    @Option(name: .long, help: "Adapter method: lora, phm or oft")
    var adapterMethod: String?

    @Option(name: .long, help: "OFT diagonal block size (default 16; must divide adapted input feature dimensions)")
    var oftBlockSize: Int = 16

    @Option(name: .long, help: "OFT orthogonal parameterization: cayley or matrixExp")
    var oftParameterization: String = OrthogonalParameterization.cayley.rawValue

    @Flag(name: .long, help: "Share one orthogonal block across all blocks within each adapted projection")
    var oftBlockShared = false

    @Option(name: .long, help: "PHM hypercomplex algebra dimension (default 8)")
    var phmDim: Int = 8

    @Flag(name: .long, help: "Give each adapted PHM projection an independent hypercomplex basis instead of sharing")
    var phmIndependentBasis = false

    func resolveMethod(fallbackRank: Int) throws -> AdapterMethod? {
        let impliedOFT = adapterMethod == nil && (oftBlockShared || oftBlockSize != 16 || oftParameterization != OrthogonalParameterization.cayley.rawValue)
        let impliedPHM = adapterMethod == nil && (phmIndependentBasis || phmDim != 8)
        guard let raw = adapterMethod ?? (impliedOFT ? "oft" : (impliedPHM ? "phm" : nil)) else {
            return nil
        }
        switch raw {
        case "lora":
            let method = AdapterMethod.lora(rank: fallbackRank, scale: 10.0)
            try validate(method)
            return method
        case "phm":
            let method = AdapterMethod.phm(
                hypercomplexDim: phmDim,
                factorRank: fallbackRank,
                sharedBasis: !phmIndependentBasis
            )
            try validate(method)
            return method
        case "oft":
            guard let param = OrthogonalParameterization(rawValue: oftParameterization) else {
                throw ValidationError("--oft-parameterization must be cayley or matrixExp")
            }
            let method = AdapterMethod.oft(
                blockSize: oftBlockSize,
                parameterization: param,
                blockShared: oftBlockShared
            )
            try validate(method)
            return method
        default:
            throw ValidationError("--adapter-method must be lora, phm or oft")
        }
    }

    private func validate(_ method: AdapterMethod) throws {
        do {
            try method.validate(prefix: "adapter")
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }
}
