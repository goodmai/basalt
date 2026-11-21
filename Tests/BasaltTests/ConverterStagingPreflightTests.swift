import Testing
@testable import BasaltInference

@Suite("Converter preflight: a quantized source is sized by the dense copy it decodes to")
struct ConverterStagingPreflightTests {
    static let gptOSS = DetectedSourceCheckpoint(
        format: "mxfp4", effectiveBPW: 4.25, totalBytes: 10_410_000_000, requiresCanonicalDecode: true)

    @Test("A staged source is sized as its dense BF16 copy: 4.25 bpw MXFP4 grows 3.8x, 8-bit affine 2x, BF16 stays the same")
    func denseCopy() {
        #expect(UniversalModelConverter.denseCopyBytes(Self.gptOSS) == 39_190_588_235)
        let affine8 = DetectedSourceCheckpoint(format: "affine:8:64", effectiveBPW: 8, totalBytes: 1_000, requiresCanonicalDecode: false)
        #expect(UniversalModelConverter.denseCopyBytes(affine8) == 2_000)
        let dense = DetectedSourceCheckpoint(format: "dense:bf16", effectiveBPW: 16, totalBytes: 1_000, requiresCanonicalDecode: false)
        #expect(UniversalModelConverter.denseCopyBytes(dense) == 1_000)
    }

    @Test("gpt-oss-20b is refused on a 24 GB Mac before any work, for disk first and then memory; a 64 GB Mac with room passes")
    func refusals() throws {
        let working = UniversalModelConverter.denseCopyBytes(Self.gptOSS)
        let gb: Int64 = 1 << 30
        let disk = try #require(UniversalModelConverter.stagingProblem(workingBytes: working, freeDiskBytes: 26 * gb, totalRAMBytes: 24 * gb))
        #expect(disk.contains("temporary volume"))
        let memory = try #require(UniversalModelConverter.stagingProblem(workingBytes: working, freeDiskBytes: 60 * gb, totalRAMBytes: 24 * gb))
        #expect(memory.contains("resident"))
        #expect(UniversalModelConverter.stagingProblem(workingBytes: working, freeDiskBytes: 60 * gb, totalRAMBytes: 64 * gb) == nil)
    }
}
