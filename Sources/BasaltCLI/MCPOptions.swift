import ArgumentParser
import BasaltCore
import Foundation

struct MCPOptions: ParsableArguments {
    @Flag(name: .customLong("mcp"), help: "Start MCP stdio transport in addition to REST")
    var mcp = false

    @Flag(name: .customLong("rest"), help: "Start REST only, suppress MCP stdio")
    var rest = false

    @Option(
        name: .customLong("image-dir"),
        help: "With --mcp, a directory the generate tool may read image paths from, after resolving symlinks; repeatable (default: none, every path is refused)"
    )
    var imageDirs: [String] = []

    @Flag(
        name: .customLong("research-tools"),
        help: "With --mcp, also list and accept the experiment tools (intervene, weights_delta, overlays_list, overlay_undo, experiment_compare, adapter_apply, gate_check, tensor_pack, tensor_unpack, train_experts, steer_search, steer_commit, introspect): in-memory overlays with undo and a TTL (an adapter that passed its quality gate has none); steering and ablation apply to research calls only, while a weight delta, adapter or pack changes the live model, marks every response as modified and turns off prompt-cache reuse until it is undone; model files on disk are never written (train_experts saves its adapter to a new directory) (default: off; the read-only X-ray tools are always on)"
    )
    var researchTools = false

    @Option(
        name: .customLong("mcp-enable"),
        help: "With --mcp, enable an MCP tool or group (serving, gemma, media, harness, xray, surgery, research, all); repeatable or comma-separated"
    )
    var enable: [String] = []

    @Option(
        name: .customLong("mcp-disable"),
        help: "With --mcp, disable an MCP tool or group (serving, gemma, media, harness, xray, surgery, research, all); overrides --mcp-enable and --research-tools; repeatable or comma-separated"
    )
    var disable: [String] = []

    func validate() throws {
        do {
            _ = try toolFilter()
        } catch {
            throw ValidationError(error.description)
        }
    }

    func toolFilter() throws(MCPToolFilter.InvalidSelectorError) -> MCPToolFilter {
        try MCPToolFilter(enable: enable, disable: disable)
    }
}
