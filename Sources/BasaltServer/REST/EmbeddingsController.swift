import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct EmbeddingsController: Sendable {
    let engine: (any EmbeddingEngine)?

    struct EmbeddingRequest: Decodable {
        let input: [EmbeddingInput]
        let encodingFormat: String?
        let dimensions: Int?

        enum CodingKeys: String, CodingKey {
            case input, dimensions
            case encodingFormat = "encoding_format"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let text = try? container.decode(String.self, forKey: .input) {
                input = [.text(text)]
            } else if let texts = try? container.decode([String].self, forKey: .input) {
                input = texts.map(EmbeddingInput.text)
            } else if let ids = try? container.decode([Int].self, forKey: .input) {
                input = [.tokens(ids)]
            } else {
                input = try container.decode([[Int]].self, forKey: .input).map(EmbeddingInput.tokens)
            }
            encodingFormat = try container.decodeIfPresent(String.self, forKey: .encodingFormat)
            dimensions = try container.decodeIfPresent(Int.self, forKey: .dimensions)
        }
    }

    struct RerankRequest: Decodable {
        struct Document: Codable { let text: String }

        let query: String
        let documents: [String]
        let topN: Int?
        let returnDocuments: Bool?

        enum CodingKeys: String, CodingKey {
            case query, documents
            case topN = "top_n"
            case returnDocuments = "return_documents"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            query = try container.decode(String.self, forKey: .query)
            if let texts = try? container.decode([String].self, forKey: .documents) {
                documents = texts
            } else {
                documents = try container.decode([Document].self, forKey: .documents).map(\.text)
            }
            topN = try container.decodeIfPresent(Int.self, forKey: .topN)
            returnDocuments = try container.decodeIfPresent(Bool.self, forKey: .returnDocuments)
        }
    }

    enum Vector: Encodable {
        case float([Float])
        case base64(String)

        func encode(to encoder: any Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .float(let values): try container.encode(values)
            case .base64(let text): try container.encode(text)
            }
        }
    }

    struct EmbeddingList: Encodable {
        struct Item: Encodable {
            let object = "embedding"
            let index: Int
            let embedding: Vector
        }

        struct Usage: Encodable {
            let promptTokens: Int
            let totalTokens: Int

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case totalTokens = "total_tokens"
            }
        }

        let object = "list"
        let data: [Item]
        let model: String
        let usage: Usage
    }

    struct RerankList: Encodable {
        struct Result: Encodable {
            let index: Int
            let relevanceScore: Double
            let document: RerankRequest.Document?

            enum CodingKeys: String, CodingKey {
                case index, document
                case relevanceScore = "relevance_score"
            }
        }

        let model: String
        let results: [Result]
    }

    @Sendable
    func embeddings(request: Request, context: BasicRequestContext) async throws -> Response {
        guard let engine, let model = engine.models.first(where: { $0.task == .embedding }) else {
            return failure(.notFound, "No embedding model is loaded; start basalt serve with --embedding-model <repo|dir>.")
        }
        let data = Data(try await request.body.collect(upTo: ChatPipeline.bodyLimit).readableBytesView)
        guard let dto = try? JSONDecoder().decode(EmbeddingRequest.self, from: data) else {
            return failure(.badRequest, "\"input\" must be a string, an array of strings or an array of token arrays.")
        }
        guard !dto.input.isEmpty, dto.input.allSatisfy(\.hasContent) else {
            return failure(.badRequest, "\"input\" must not be empty and must not contain empty strings or token arrays.")
        }
        guard dto.input.count <= EmbeddingLimits.maxInputs else {
            return failure(.contentTooLarge, "At most \(EmbeddingLimits.maxInputs) inputs per request, got \(dto.input.count).")
        }
        guard dto.encodingFormat == nil || dto.encodingFormat == "float" || dto.encodingFormat == "base64" else {
            return failure(.badRequest, "\"encoding_format\" must be \"float\" or \"base64\".")
        }
        if let dimensions = dto.dimensions {
            guard model.truncatesDimensions else {
                return failure(.badRequest, "\(model.id) was not trained with Matryoshka representation learning, so it does not take \"dimensions\".")
            }
            guard dimensions > 0 else { return failure(.badRequest, "\"dimensions\" must be positive.") }
        }
        do {
            let result = try await engine.embed(dto.input, dimensions: dto.dimensions)
            let base64 = dto.encodingFormat == "base64"
            let items = result.vectors.enumerated().map { index, vector in
                EmbeddingList.Item(index: index, embedding: base64 ? .base64(Self.base64(vector)) : .float(vector))
            }
            return try json(EmbeddingList(
                data: items, model: model.id,
                usage: .init(promptTokens: result.promptTokens, totalTokens: result.promptTokens)))
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/embeddings")
        }
    }

    @Sendable
    func rerank(request: Request, context: BasicRequestContext) async throws -> Response {
        guard let engine, let model = engine.models.first(where: { $0.task == .rerank }) else {
            return failure(.notFound, "No rerank model is loaded; start basalt serve with --rerank-model <repo|dir>.")
        }
        let data = Data(try await request.body.collect(upTo: ChatPipeline.bodyLimit).readableBytesView)
        guard let dto = try? JSONDecoder().decode(RerankRequest.self, from: data) else {
            return failure(.badRequest, "A rerank request needs \"query\" and \"documents\" (strings or {\"text\": ...} objects).")
        }
        guard !dto.query.isEmpty, !dto.documents.isEmpty else {
            return failure(.badRequest, "\"query\" and \"documents\" must not be empty.")
        }
        guard dto.documents.count <= EmbeddingLimits.maxDocuments else {
            return failure(.contentTooLarge, "At most \(EmbeddingLimits.maxDocuments) documents per request, got \(dto.documents.count).")
        }
        if let topN = dto.topN, topN < 1 { return failure(.badRequest, "\"top_n\" must be at least 1.") }
        do {
            let scores = try await engine.rerank(query: dto.query, documents: dto.documents)
            guard scores.count == dto.documents.count else {
                throw BasaltError.inferenceError("the reranker returned \(scores.count) scores for \(dto.documents.count) documents")
            }
            let ranked = scores.indices.sorted { scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1) }
            let results = ranked.prefix(dto.topN ?? ranked.count).map { index in
                RerankList.Result(
                    index: index, relevanceScore: scores[index],
                    document: dto.returnDocuments == true ? .init(text: dto.documents[index]) : nil)
            }
            return try json(RerankList(model: model.id, results: results))
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/rerank")
        }
    }

    static func base64(_ vector: [Float]) -> String {
        var bytes = Data(capacity: vector.count * 4)
        for value in vector {
            for shift in stride(from: 0, to: 32, by: 8) {
                bytes.append(UInt8(truncatingIfNeeded: value.bitPattern >> shift))
            }
        }
        return bytes.base64EncodedString()
    }

    private func json(_ body: some Encodable) throws -> Response {
        var buffer = ByteBuffer()
        buffer.writeBytes(try JSONEncoder().encode(body))
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }

    private func failure(_ status: HTTPResponse.Status, _ message: String) -> Response {
        ProviderErrorResponse.response(status: status, message: message, shape: .openAI)
    }
}

private extension EmbeddingInput {
    var hasContent: Bool {
        switch self {
        case .text(let text): !text.isEmpty
        case .tokens(let ids): !ids.isEmpty
        }
    }
}
