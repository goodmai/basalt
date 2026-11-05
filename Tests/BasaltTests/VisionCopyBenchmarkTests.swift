import Testing
import Foundation
import Darwin
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
import MLX
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Image copy cost on a real vision model: TTFT and peak memory for 1, 4 and 8 images at the request limits (#345)", .serialized)
struct VisionCopyBenchmarkTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", env["BASALT_VISION_BENCH"] == "1" else { return nil }
        return env["BASALT_E2E_VISION_MODEL_PATH"]
    }

    static let runs = 3

    struct Sample {
        let ttft: Duration
        let mlxPeak: Int
        let footprintPeak: UInt64
        let promptTokens: Int
    }

    static func noisePNG(bytes budget: Int) throws -> Data {
        let height = Int((Double(budget) * 0.95 / 3 * 9 / 16).squareRoot())
        let width = height * 16 / 9
        let sRGB = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: sRGB, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        arc4random_buf(context.data, width * height * 4)
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(context.makeImage()), nil)
        #expect(CGImageDestinationFinalize(destination))
        try #require(data.length <= budget, "a \(width)x\(height) noise PNG is \(data.length) bytes, over its budget of \(budget)")
        return data as Data
    }

    static func body(images count: Int) throws -> (body: ByteBuffer, imageBytes: Int) {
        let images = try (0..<count).map { _ in try noisePNG(bytes: min(ImageInput.maxBytes, ImageInput.maxTotalBytes / count)) }
        let parts: [[String: Any]] = images.map { ["type": "image_url", "image_url": ["url": "data:image/png;base64,\($0.base64EncodedString())"]] }
        let message: [String: Any] = ["role": "user", "content": [["type": "text", "text": "Name these images in one word."]] + parts]
        let json = try JSONSerialization.data(
            withJSONObject: ["messages": [message], "max_tokens": 1, "temperature": 0], options: .withoutEscapingSlashes)
        return (ByteBuffer(bytes: json), images.reduce(0) { $0 + $1.count })
    }

    static func resetFootprintPeak() {
        typealias Reset = @convention(c) (pid_t) -> Int32
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "proc_reset_footprint_interval") else { return }
        _ = unsafeBitCast(symbol, to: Reset.self)(getpid())
    }

    static func footprintPeak() -> UInt64 {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return result == 0 ? info.ri_interval_max_phys_footprint : 0
    }

    static func measure(_ client: some TestClientProtocol, _ body: ByteBuffer) async throws -> Sample {
        Memory.clearCache()
        Memory.peakMemory = 0
        resetFootprintPeak()
        let started = ContinuousClock.now
        let text = try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: body) {
            String(buffer: $0.body)
        }
        let ttft = ContinuousClock.now - started
        let reply = try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any], "\(text.prefix(500))")
        let prompt = try #require((reply["usage"] as? [String: Any])?["prompt_tokens"] as? Int, "\(text.prefix(500))")
        return Sample(ttft: ttft, mlxPeak: Memory.peakMemory, footprintPeak: footprintPeak(), promptTokens: prompt)
    }

    static func summary(_ count: Int, imageBytes: Int, _ samples: [Sample]) -> String {
        let ms = samples.map { $0.ttft / .milliseconds(1) }.sorted()
        let mlx = samples.map(\.mlxPeak).sorted()[samples.count / 2]
        let footprint = samples.map(\.footprintPeak).sorted()[samples.count / 2]
        return String(
            format: "images %ld bytes %ld prompt %ld ttft median %.0f ms runs %@ mlx peak median %ld MB footprint peak median %llu MB",
            count, imageBytes, samples[0].promptTokens, ms[ms.count / 2], ms.map { String(format: "%.0f", $0) }.joined(separator: "/"),
            mlx >> 20, footprint >> 20)
    }

    @Test("1, 4 and 8 noise PNGs at the 20 MB / 32 MB limits through OpenAI data: URIs: median TTFT, MLX peak and process footprint peak",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1, BASALT_VISION_BENCH=1 and BASALT_E2E_VISION_MODEL_PATH (Gemma 4 26B-A4B), release build"))
    func imageCopyCost() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("vision-copy", modelPath: path)
        defer { log.releaseGPU() }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), telemetry: TelemetryTracker())
        try await log.step("load") { try await orchestrator.loadModel(path: path) }
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: path)).makeRouter())
        try await app.test(.router) { client in
            _ = try await log.step("warm-up") { try await Self.measure(client, try Self.body(images: 1).body) }
            for count in [1, 4, 8] {
                var samples: [Sample] = []
                var imageBytes = 0
                for _ in 0..<Self.runs {
                    let request = try Self.body(images: count)
                    imageBytes = request.imageBytes
                    samples.append(try await log.step("images-\(count)") { try await Self.measure(client, request.body) })
                }
                #expect(samples.allSatisfy { $0.promptTokens > count * 64 }, "every image reached the vision tower")
                log.record("result", "\(Self.summary(count, imageBytes: imageBytes, samples)) \(log.host())")
            }
        }
        Memory.clearCache()
    }
}
