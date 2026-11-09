import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Chat decode beside continuous OCR: PaddleOCR-VL's vision tower on the GPU against the Neural Engine, in one run (#224, #400, #273)", .serialized)
struct VisionTowerCoRunTests {
    struct Setup {
        let chat: String
        let ocr: String
        let tower: String
        let page: URL
        let basalt: URL
    }

    static var setup: Setup? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", let chat = env["BASALT_E2E_CORUN_CHAT_MODEL_PATH"], let ocr = env["BASALT_E2E_PADDLE_MODEL_PATH"],
              let tower = env["BASALT_E2E_ANE_TOWER"], let page = env["BASALT_E2E_CORUN_PAGE"], let basalt = env["BASALT_E2E_CORUN_BASALT"] else { return nil }
        return Setup(chat: chat, ocr: ocr, tower: tower, page: URL(fileURLWithPath: page), basalt: URL(fileURLWithPath: basalt))
    }

    static let (tokens, runs, port) = (256, 3, 18_517)

    static func serve(_ setup: Setup, tower: Bool) async throws -> Process {
        let process = Process()
        process.executableURL = setup.basalt
        process.arguments = ["serve", "--model-path", setup.ocr, "-p", "\(port)"] + (tower ? ["--experimental-ane-vision-tower", setup.tower] : [])
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        let deadline = ContinuousClock.now + .seconds(1500)
        while ContinuousClock.now < deadline, process.isRunning {
            try Task.checkCancellation()
            if let (_, response) = try? await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/models")!),
               (response as? HTTPURLResponse)?.statusCode == 200 { return process }
            try await Task.sleep(for: .seconds(1))
        }
        process.terminate()
        throw BasaltError.inferenceError("the OCR server did not come up on port \(port)")
    }

    static func ocrLoop(_ page: Data) -> Task<(pages: Int, seconds: Double), any Error> {
        Task {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/ocr")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: ["file": page.base64EncodedString()])
            request.timeoutInterval = 600
            let start = ContinuousClock.now
            var pages = 0
            while !Task.isCancelled {
                guard let (_, response) = try? await URLSession.shared.data(for: request) else { break }
                guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw BasaltError.inferenceError("OCR answered \(String(describing: response))") }
                pages += 1
            }
            return (pages, (ContinuousClock.now - start) / .seconds(1))
        }
    }

    static func decodeRate(_ container: ModelContainer) async throws -> Double {
        var rates: [Double] = []
        for _ in 0 ..< runs {
            try Task.checkCancellation()
            rates.append(try await container.perform { context in
                let input = try await context.processor.prepare(input: UserInput(prompt: "Write a long story about a lighthouse keeper."))
                var iterator = try TokenIterator(input: input, model: context.model, processor: nil, sampler: ArgMaxSampler(), maxTokens: tokens + 1)
                _ = iterator.next()
                let start = ContinuousClock.now
                var count = 0
                while iterator.next() != nil { count += 1 }
                return Double(count) / ((ContinuousClock.now - start) / .seconds(1))
            })
        }
        return rates.sorted()[runs / 2]
    }

    @Test("Gemma 4 decode tok/s solo, beside continuous OCR with the GPU tower and beside continuous OCR with the ANE tower",
          .enabled(if: setup != nil, "set BASALT_E2E=1, BASALT_E2E_CORUN_CHAT_MODEL_PATH, BASALT_E2E_PADDLE_MODEL_PATH, BASALT_E2E_ANE_TOWER, BASALT_E2E_CORUN_PAGE and BASALT_E2E_CORUN_BASALT (release Basalt binary)"),
          .timeLimit(.minutes(60)))
    func coRun() async throws {
        let setup = try #require(Self.setup)
        let page = try Data(contentsOf: setup.page)
        let log = try await ModelRunLog("ane-vision-corun", modelPath: setup.chat)
        defer { log.releaseGPU() }
        await MLXInferenceEngine.registerAliases()
        let container = try await log.step("load") { try await loadModelContainer(from: URL(fileURLWithPath: setup.chat), using: DefaultTokenizerLoader()) }
        _ = try await Self.decodeRate(container)
        let solo = try await log.step("solo") { try await Self.decodeRate(container) }
        log.record("result", String(format: "solo decode %.2f tok/s %@", solo, log.host()))
        for tower in [false, true] {
            let server = try await log.step(tower ? "serve-ane" : "serve-gpu") { try await Self.serve(setup, tower: tower) }
            defer { server.terminate(); server.waitUntilExit() }
            let loop = Self.ocrLoop(page)
            let rate = try await Self.decodeRate(container)
            loop.cancel()
            let ocr = try await loop.value
            log.record("result", String(
                format: "OCR tower %@: decode %.2f tok/s (%+.1f%% against solo), OCR %d pages in %.1f s (%.3f pages/s) %@",
                tower ? "ANE" : "GPU", rate, 100 * (rate - solo) / solo, ocr.pages, ocr.seconds, Double(ocr.pages) / ocr.seconds, log.host()))
        }
        Memory.clearCache()
    }
}
