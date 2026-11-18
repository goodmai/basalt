import Foundation
import BasaltCore
import Yams

extension ComposeFile.Walk {
    mutating func imagine(_ node: Node?, into spec: inout ComposeSpec) {
        let path = "\(prefix).imagine"
        guard let node else { return }
        if let single = node.scalar?.string, node.mapping == nil, node.sequence == nil {
            spec.imagine = ComposeSpec.Imagine(chain: [single])
            return
        }
        if node.sequence != nil {
            spec.imagine = ComposeSpec.Imagine(chain: list(node, path))
            return
        }
        guard let field = fields(node, path: path, section: "imagine") else { return }
        let chain: [String]
        if let single = field["chain"]?.scalar?.string,
           field["chain"]?.mapping == nil,
           field["chain"]?.sequence == nil {
            chain = [single]
        } else {
            chain = list(field["chain"], "\(path).chain")
        }
        var stage = ComposeSpec.Imagine(
            chain: chain,
            upsampleVia: text(field["upsample_via"], "\(path).upsample_via"),
            shareEncoder: text(field["share_encoder"], "\(path).share_encoder"),
            steps: integer(field["steps"], "\(path).steps"),
            size: text(field["size"], "\(path).size")
        )
        stage.decoderVia = text(field["decoder_via"], "\(path).decoder_via")
        stage.teacacheThreshold = number(field["teacache_threshold"], "\(path).teacache_threshold")
        stage.previewEverySteps = integer(field["preview_every_steps"], "\(path).preview_every_steps")
        stage.candidates = integer(field["candidates"], "\(path).candidates")
        stage.judgeVia = text(field["judge_via"], "\(path).judge_via")
        imagineEnums(field, path: path, into: &stage)
        spec.imagine = stage
    }

    private mutating func imagineEnums(_ field: [String: Node], path: String, into stage: inout ComposeSpec.Imagine) {
        if let raw = text(field["format"], "\(path).format") {
            stage.format = ImageOutputFormat(rawValue: raw)
            if stage.format == nil {
                let choices = ImageOutputFormat.allCases.map(\.rawValue).joined(separator: ", ")
                issue("\(path).format", field["format"]!, "must be one of \(choices)")
            }
        }
        if let raw = text(field["on_gpu_busy"], "\(path).on_gpu_busy") {
            stage.onGPUBusy = ImageOnGPUBusy(rawValue: raw)
            if stage.onGPUBusy == nil {
                let choices = ImageOnGPUBusy.allCases.map(\.rawValue).joined(separator: ", ")
                issue("\(path).on_gpu_busy", field["on_gpu_busy"]!, "must be one of \(choices)")
            }
        }
    }
}
