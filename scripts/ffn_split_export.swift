#!/usr/bin/env swift
import Foundation

let resolvedBefore = "2026-11-10T00:00:00Z"
let pins = ["coreai-torch==0.4.3", "coreai-core==1.0.0b3", "coreai-opt==0.4.0", "ml-dtypes==0.6.0", "torch==2.15.0", "safetensors==0.8.0", "numpy==2.4.6"]

let usage = """
    Usage:
      swift scripts/ffn_split_export.swift <MLX checkpoint dir> <out dir> <layer> <share> <tokens> <fp16|int4|int8|lut4>

    Exports the ANE share of one dense FFN layer for the prefill split of #274: the first <share> of the
    intermediate columns of gate_proj and up_proj and the matching columns of down_proj, as a Core AI
    model with one function len<tokens> that maps x (1, hidden, 1, tokens) to the partial down sum y
    of the same shape. The weights are the checkpoint's own affine 4- or 8-bit weights dequantized;
    int4 requantizes them with coreai-opt as asymmetric 4-bit with a minimum and a scale per block of 32
    inputs, the form of MLX's affine grid (the w4 preset is symmetric per output channel, 23% off); int8
    is symmetric per output channel; lut4 is coreai-opt's k-means palette, 16 values per 16 output
    channels; fp16 keeps them as they are. Writes <out>/ffn-<layer>-<share>-<fp16|int4>.aimodel, a metadata JSON and a reference input with
    its float32 output from the dequantized weights beside it. Runs in a
    throwaway uv environment with the versions pinned above (checked with the Sonatype guide), every
    transitive package resolved as of \(resolvedBefore).
    """

let exporter = #"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import load_file
from coreai_torch import TorchConverter, get_decomp_table

source, out, layer, share, tokens, weights = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
config = json.loads((source / "config.json").read_text())
text = config.get("text_config", config)
hidden, intermediate, act = text["hidden_size"], text["intermediate_size"], text.get("hidden_activation") or text.get("hidden_act")
index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
prefix = next(k[: -len(".gate_proj.weight")] for k in index if k.endswith(f"layers.{layer}.mlp.gate_proj.weight"))
quant = config.get("quantization", {})
shards = {}

def tensor(key):
    shard = index[key]
    if shard not in shards:
        shards[shard] = load_file(str(source / shard))
    return shards[shard][key]

def dense(name):
    key = f"{prefix}.{name}"
    q = quant.get(key, quant)
    bits, group = q["bits"], q["group_size"]
    packed = tensor(key + ".weight").view(np.uint32)
    per = 32 // bits
    codes = (packed[..., None] >> (np.arange(per, dtype=np.uint32) * bits)) & ((1 << bits) - 1)
    codes = codes.reshape(packed.shape[0], -1).astype(np.float32)
    scales = tensor(key + ".scales").astype(np.float32)
    biases = tensor(key + ".biases").astype(np.float32)
    rows, cols = codes.shape
    return (codes.reshape(rows, cols // group, group) * scales[..., None] + biases[..., None]).reshape(rows, cols)

columns = max(64, int(round(intermediate * share / 64)) * 64)
gate, up, down = dense("gate_proj")[:columns], dense("up_proj")[:columns], dense("down_proj")[:, :columns]

def conv(w):
    module = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
    module.weight.data = torch.from_numpy(np.ascontiguousarray(w).reshape(w.shape[0], w.shape[1], 1, 1))
    return module

class Share(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate, self.up, self.down = conv(gate), conv(up), conv(down)

    def forward(self, x):
        g = self.gate(x)
        g = nn.functional.gelu(g, approximate="tanh") if "gelu" in act else nn.functional.silu(g)
        return self.down(g * self.up(x))

torch.manual_seed(0)
example = (torch.randn(1, hidden, 1, tokens, dtype=torch.float16) * 0.5,)
with torch.no_grad():
    reference = Share().float().eval()(example[0].float())
model = Share().to(torch.float16).eval()
if weights in ("int4", "int8"):
    import coreai_opt as opt
    from coreai_opt.quantization import QuantizationSpec, Quantizer, QuantizerConfig
    from coreai_opt.quantization.config import ModuleQuantizerConfig
    from coreai_opt.quantization.spec import PerBlockGranularity, QuantizationFormulation, QuantizationScheme
    from coreai_opt.quantization.spec import PerChannelGranularity
    spec = QuantizationSpec(dtype=torch.uint4, qscheme=QuantizationScheme.ASYMMETRIC, qformulation=QuantizationFormulation.MINVAL,
                            granularity=PerBlockGranularity(axis=1, block_size=32)) if weights == "int4" else QuantizationSpec(
                            dtype=torch.int8, granularity=PerChannelGranularity(axis=0))
    config = QuantizerConfig(global_config=ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec}))
    quantizer = Quantizer(model, config)
    quantizer.prepare(example)
    model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
if weights == "lut4":
    import coreai_opt as opt
    from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
    palettizer = KMeansPalettizer(model, KMeansPalettizerConfig.presets.w4())
    palettizer.prepare(example)
    model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
with torch.no_grad():
    program = torch.export.export(model, args=example, strict=False).run_decompositions(get_decomp_table())
    converter = TorchConverter().add_exported_program(program, entrypoint_name=f"len{tokens}", input_names=["x"], output_names=["y"])
out.mkdir(parents=True, exist_ok=True)
name = f"ffn-{layer}-{share:g}-{weights}"
shutil.rmtree(out / f"{name}.aimodel", ignore_errors=True)
program = converter.to_coreai()
program.save_asset(out / f"{name}.aimodel")
from safetensors.numpy import save_file
save_file({"x": example[0].numpy(), "y": reference.numpy()}, str(out / f"{name}.reference.safetensors"))
(out / f"{name}.json").write_text(json.dumps({"layer": layer, "hidden": hidden, "intermediate": intermediate, "columns": columns, "tokens": tokens,
                                             "weights": weights, "activation": act, "prefix": prefix}, indent=1))
print("wrote", out / f"{name}.aimodel", "columns", columns)
"""#

let arguments = CommandLine.arguments
guard arguments.count == 7, Int(arguments[3]) != nil, let share = Double(arguments[4]), share > 0, share < 1,
      Int(arguments[5]).map({ $0 > 0 }) ?? false, ["fp16", "int4", "int8", "lut4"].contains(arguments[6])
else {
    print(usage)
    exit(2)
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
process.arguments = ["uv", "run", "--quiet", "--no-project", "--python", "3.12", "--exclude-newer", resolvedBefore] + pins.flatMap { ["--with", $0] }
    + ["python", "-I", "-c", exporter] + Array(arguments[1...6])
try process.run()
process.waitUntilExit()
exit(process.terminationStatus)
