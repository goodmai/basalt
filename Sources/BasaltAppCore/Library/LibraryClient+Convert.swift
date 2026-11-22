import BasaltCore
import Foundation

extension LibraryClient {
    public typealias ConversionProgressHandler = @Sendable (ModelConversionProgress) -> Void

    static func livePlanConversion(
        _ source: String,
        target: QuantizationTarget,
        budgetGB: Double?,
        pvTune: Bool,
        endpoint: EngineEndpoint,
        session: URLSession
    ) async throws -> ModelConversionPlan {
        var components = URLComponents(url: endpoint.base.appending(path: "v1/models/convert/plan"), resolvingAgainstBaseURL: false)
        var items = [
            URLQueryItem(name: "source", value: source),
            URLQueryItem(name: "target", value: target.specString),
            URLQueryItem(name: "pv_tune", value: pvTune ? "true" : "false"),
        ]
        if let budgetGB {
            items.append(URLQueryItem(name: "budget_gb", value: String(budgetGB)))
        }
        components?.queryItems = items
        var request = endpoint.request("v1/models/convert/plan")
        request.url = components?.url
        let (data, response) = try await session.data(for: request)
        try EngineClient.check(response, data)
        return try JSONDecoder().decode(ModelConversionPlan.self, from: data)
    }

    static func liveConvertModel(
        _ req: ModelConversionRequest,
        endpoint: EngineEndpoint,
        session: URLSession,
        onProgress: @escaping ConversionProgressHandler
    ) async throws -> ModelConversionResult {
        var wire = req
        wire.stream = true
        let body = try JSONEncoder().encode(wire)
        var request = endpoint.request("v1/models/convert", method: "POST", body: body)
        request.timeoutInterval = 7 * 24 * 3600
        let (bytes, response) = try await session.bytes(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            var errData = Data()
            for try await byte in bytes.prefix(4096) { errData.append(byte) }
            try EngineClient.check(response, errData)
        }
        return try await relayConversionSSE(bytes, onProgress: onProgress)
    }

    static func relayConversionSSE(
        _ bytes: URLSession.AsyncBytes,
        onProgress: @escaping ConversionProgressHandler
    ) async throws -> ModelConversionResult {
        let decoder = JSONDecoder()
        var parser = SSEParser()
        var pending: [UInt8] = []
        for try await byte in bytes {
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                if let result = try handleConversionEvent(event, decoder: decoder, onProgress: onProgress) {
                    return result
                }
            }
            pending.removeAll(keepingCapacity: true)
        }
        if !pending.isEmpty {
            pending.append(UInt8(ascii: "\n"))
        }
        pending.append(UInt8(ascii: "\n"))
        for event in parser.feed(pending) {
            if let result = try handleConversionEvent(event, decoder: decoder, onProgress: onProgress) {
                return result
            }
        }
        throw EngineError.transport("model conversion stream closed before completion")
    }

    private static func handleConversionEvent(
        _ event: SSEEvent,
        decoder: JSONDecoder,
        onProgress: @escaping ConversionProgressHandler
    ) throws -> ModelConversionResult? {
        let data = Data(event.data.utf8)
        switch event.name {
        case "progress":
            if let prog = try? decoder.decode(ModelConversionProgress.self, from: data) {
                onProgress(prog)
            }
            return nil
        case "done":
            return try decoder.decode(ModelConversionResult.self, from: data)
        case "error":
            if let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                let msg = (json["message"] as? String)
                    ?? ((json["error"] as? [String: Any])?["message"] as? String)
                    ?? "Conversion failed"
                throw EngineError.refused(status: 422, message: msg)
            }
            throw EngineError.refused(status: 422, message: event.data)
        default:
            return nil
        }
    }
}
