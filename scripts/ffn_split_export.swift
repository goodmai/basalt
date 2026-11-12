#!/usr/bin/env swift
import Foundation

let resolvedBefore = "2026-11-10T00:00:00Z"
let pins = ["coreai-torch==0.4.3", "coreai-core==1.0.0b3", "coreai-opt==0.4.0", "ml-dtypes==0.6.0", "torch==2.15.0", "safetensors==0.8.0", "numpy==2.4.6"]

let usage = """
    Usage:
      swift scripts/ffn_split_export.swift <MLX checkpoint dir> <out dir> <layers> <share> <tokens> <fp16|int4|int8|lut4>

    Exports the ANE share of dense FFN layers for the prefill split of #274. <layers> is one index, a
    range such as 0-47, or all (every layer with a dense MLP). For each layer: the first <share> of the
    intermediate columns of gate_proj and up_proj and the matching columns of down_proj, as a Core AI
    model with one function len<n> per length in <tokens> (comma-separated) that maps x (1, hidden, 1, n)
    to the partial down sum y of the same shape.

    The weights are the checkpoint's own affine 4- or 8-bit weights dequantized:
    - int8: symmetric per output channel, 1.3% off the GPU block at 2.5x its speed on Gemma 4 12B;
    - int4: asymmetric with a minimum and a scale per block of 32 inputs (the w4 preset is symmetric per
      output channel and lands 23% off);
    - lut4: coreai-opt's k-means palette, 16 values per 16 output channels;
    - fp16: as they are.

    Writes <out>/ffn-<layer>-<share>-<weights>.aimodel with a metadata JSON per layer, a float32
    reference input and output for the first layer, and <out>/manifest.json listing the layers. Runs in a
    throwaway uv environment with the versions pinned above (checked with the Sonatype guide), every
    transitive package resolved as of \(resolvedBefore).
    """

let exporter = #"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import load_file, save_file
from coreai_torch import TorchConverter, get_decomp_table

source, out, spec_layers, share, weights = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[6]
lengths = [int(n) for n in sys.argv[5].split(",")]
config = json.loads((source / "config.json").read_text())
text = config.get("text_config", config)
hidden, intermediate, act = text["hidden_size"], text["intermediate_size"], text.get("hidden_activation") or text.get("hidden_act")
index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
quant = config.get("quantization", {})
dense_layers = sorted(int(k.split("layers.")[1].split(".")[0]) for k in index if k.endswith(".mlp.gate_proj.weight"))
if spec_layers == "all":
    layers = dense_layers
elif "-" in spec_layers:
    first, last = map(int, spec_layers.split("-"))
    layers = [n for n in dense_layers if first <= n <= last]
else:
    layers = [int(spec_layers)]
columns = max(64, int(round(intermediate * share / 64)) * 64)
shards = {}

def tensor(key):
    shard = index[key]
    if shard not in shards:
        shards.clear()
        shards[shard] = load_file(str(source / shard))
    return shards[shard][key]

def dense(prefix, name):
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

def conv(w):
    module = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
    module.weight.data = torch.from_numpy(np.ascontiguousarray(w).reshape(w.shape[0], w.shape[1], 1, 1))
    return module

class Share(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down = conv(gate), conv(up), conv(down)

    def forward(self, x):
        g = self.gate(x)
        g = nn.functional.gelu(g, approximate="tanh") if "gelu" in act else nn.functional.silu(g)
        return self.down(g * self.up(x))

def compressed(model, example):
    import coreai_opt as opt
    if weights == "lut4":
        from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
        palettizer = KMeansPalettizer(model, KMeansPalettizerConfig.presets.w4())
        palettizer.prepare(example)
        return palettizer.finalize(backend=opt.ExportBackend.CoreAI)
    from coreai_opt.quantization import QuantizationSpec, Quantizer, QuantizerConfig
    from coreai_opt.quantization.config import ModuleQuantizerConfig
    from coreai_opt.quantization.spec import PerBlockGranularity, PerChannelGranularity, QuantizationFormulation, QuantizationScheme
    spec = QuantizationSpec(dtype=torch.uint4, qscheme=QuantizationScheme.ASYMMETRIC, qformulation=QuantizationFormulation.MINVAL,
                            granularity=PerBlockGranularity(axis=1, block_size=32)) if weights == "int4" else QuantizationSpec(
                            dtype=torch.int8, granularity=PerChannelGranularity(axis=0))
    config = QuantizerConfig(global_config=ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec}))
    quantizer = Quantizer(model, config)
    quantizer.prepare(example)
    return quantizer.finalize(backend=opt.ExportBackend.CoreAI)

out.mkdir(parents=True, exist_ok=True)
entries = {}
for position, layer in enumerate(layers):
    prefix = next(k[: -len(".gate_proj.weight")] for k in index if k.endswith(f"layers.{layer}.mlp.gate_proj.weight"))
    gate, up, down = dense(prefix, "gate_proj")[:columns], dense(prefix, "up_proj")[:columns], dense(prefix, "down_proj")[:, :columns]
    torch.manual_seed(0)
    example = (torch.randn(1, hidden, 1, lengths[0], dtype=torch.float16) * 0.5,)
    model = Share(gate, up, down).to(torch.float16).eval()
    if weights != "fp16":
        model = compressed(model, example)
    converter = TorchConverter()
    with torch.no_grad():
        for n in lengths:
            args = (torch.randn(1, hidden, 1, n, dtype=torch.float16),)
            program = torch.export.export(model, args=args, strict=False).run_decompositions(get_decomp_table())
            converter = converter.add_exported_program(program, entrypoint_name=f"len{n}", input_names=["x"], output_names=["y"])
    name = f"ffn-{layer}-{share:g}-{weights}"
    shutil.rmtree(out / f"{name}.aimodel", ignore_errors=True)
    converter.to_coreai().save_asset(out / f"{name}.aimodel")
    if position == 0:
        with torch.no_grad():
            reference = Share(gate, up, down).float().eval()(example[0].float())
        save_file({"x": example[0].numpy(), "y": reference.numpy()}, str(out / f"{name}.reference.safetensors"))
    (out / f"{name}.json").write_text(json.dumps({"layer": layer, "hidden": hidden, "intermediate": intermediate, "columns": columns,
                                                 "tokens": lengths[0], "weights": weights, "activation": act, "prefix": prefix}, indent=1))
    entries[str(layer)] = f"{name}.aimodel"
    print("wrote", out / f"{name}.aimodel", flush=True)
manifest = {"ffn_split": {"hidden": hidden, "intermediate": intermediate, "columns": columns, "share": share, "weights": weights,
                          "activation": act, "lengths": lengths, "layers": entries}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("wrote", out / "manifest.json", "layers", len(entries), "columns", columns)
"""#

let arguments = CommandLine.arguments
let layerSpec = arguments.count == 7 ? arguments[3] : ""
let layersValid = layerSpec == "all" || Int(layerSpec) != nil
    || layerSpec.split(separator: "-").compactMap { Int($0) }.count == 2 && layerSpec.split(separator: "-").count == 2
guard arguments.count == 7, layersValid, let share = Double(arguments[4]), share > 0, share < 1,
      arguments[5].split(separator: ",").allSatisfy({ Int($0).map { $0 > 0 } ?? false }), ["fp16", "int4", "int8", "lut4"].contains(arguments[6])
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
