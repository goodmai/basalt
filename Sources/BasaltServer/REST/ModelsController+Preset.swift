import Foundation
import Hummingbird
import BasaltCore

extension ModelsController {
    func previewPreset(_ name: String, memory: String) async -> Response {
        guard let preset = PipelinePreset(rawValue: name) else {
            return ProviderErrorResponse.response(status: .badRequest, message: "pipeline_preset must be one of \(PipelinePreset.names)", shape: .openAI)
        }
        guard let memory = PipelineMemory(rawValue: memory) else {
            return ProviderErrorResponse.response(status: .badRequest, message: "pipeline_memory must be resident, sequential or auto", shape: .openAI)
        }
        let plan = preset.plan(budgetBytes: await SystemProfiler().detectResources().gpuMemory, power: PowerSource.current(), memory: memory)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var buffer = ByteBuffer()
        buffer.writeBytes((try? encoder.encode(plan)) ?? Data())
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }
}
