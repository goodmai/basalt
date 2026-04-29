import Foundation
import Testing
@testable import GemCore

class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        return true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        return request
    }

    override func startLoading() {
        guard let handler = MockURLProtocol.requestHandler else {
            fatalError("Handler is unavailable.")
        }
        
        do {
            let (response, data) = try handler(request)
            // Send the response and data to the client
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            
            // To simulate download progress, we send the data in chunks
            let chunkSize = 1024
            var offset = 0
            while offset < data.count {
                let end = min(offset + chunkSize, data.count)
                let chunk = data[offset..<end]
                client?.urlProtocol(self, didLoad: chunk)
                offset += chunkSize
            }
            
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

@Suite("HuggingFaceHub Download Tests")
struct HuggingFaceHubTests {

    @Test("Test download model files via mocked URLSession")
    func testDownloadModel() async throws {
        // Create an ephemeral session configuration and inject our MockURLProtocol
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]

        let hub = HuggingFaceHub(configuration: config)

        let repoId = "test-community/test-model"
        
        // Define our mocked responses
        MockURLProtocol.requestHandler = { request in
            let urlString = request.url?.absoluteString ?? ""
            
            // Mock for API model details
            if urlString.contains("/api/models/\(repoId)") {
                let json = """
                {
                    "id": "\(repoId)",
                    "siblings": [
                        {"rfilename": "config.json", "size": 10},
                        {"rfilename": "model.safetensors", "size": 20},
                        {"rfilename": "ignore.gguf", "size": 100}
                    ]
                }
                """
                let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
                return (response, json.data(using: .utf8)!)
            }
            
            // Mock for HEAD requests to get content length
            if request.httpMethod == "HEAD" {
                let response = HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Length": "15"] // fake size
                )!
                return (response, Data())
            }
            
            // Mock for actual file downloads
            if urlString.contains("/resolve/main/") {
                let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Length": "15"])!
                return (response, Data(repeating: 0x42, count: 15))
            }
            
            fatalError("Unexpected request: \(urlString)")
        }

        // Clear cache if exists for isolation
        let destDir = ModelCache.cacheDir(for: repoId)
        if FileManager.default.fileExists(atPath: destDir.path) {
            try? FileManager.default.removeItem(at: destDir)
        }

        // Run the download
        actor Counter {
            var value = 0
            func increment() { value += 1 }
        }
        let counter = Counter()
        
        let resultPath = try await hub.download(repoId: repoId, token: nil) { filename, downloaded, total in
            Task { await counter.increment() }
        }
        
        let finalCount = await counter.value
        #expect(resultPath.path.hasSuffix("models--test-community--test-model/snapshots/main"))
        #expect(finalCount > 0, "Progress callback should be called")

        // Verify that the files were created correctly (config.json and model.safetensors)
        let configPath = destDir.appendingPathComponent("config.json")
        let safetensorsPath = destDir.appendingPathComponent("model.safetensors")
        let ignorePath = destDir.appendingPathComponent("ignore.gguf")

        #expect(FileManager.default.fileExists(atPath: configPath.path))
        #expect(FileManager.default.fileExists(atPath: safetensorsPath.path))
        #expect(!FileManager.default.fileExists(atPath: ignorePath.path), ".gguf files should be ignored")
        
        // Clean up
        try? FileManager.default.removeItem(at: destDir)
    }
}
