#!/usr/bin/env swift
import Foundation

let resolvedBefore = "2026-11-10T00:00:00Z"
let pins = ["coreai-torch==0.4.3", "coreai-core==1.0.0b3", "torch==2.15.0", "safetensors==0.8.0", "numpy==2.4.6"]

let usage = """
    Usage:
      swift scripts/paddle_tower_export.swift <PaddleOCR-VL snapshot dir> <out dir> [buckets, default 1024,2048,5120]

    Exports PaddleOCR-VL's vision tower (NaViT encoder and projector) as a Core AI bundle for
    serve --experimental-ane-vision-tower (#224, #400): <out>/tower.aimodel with one function per
    bucket (n<tokens>) and <out>/metadata.json. A page uses the smallest bucket that holds its patch
    grid. Runs in a throwaway uv environment with the versions pinned above (checked with the
    Sonatype guide), every transitive package resolved as of \(resolvedBefore).

    The graph is laid out for the Neural Engine as in Apple's ml-ane-transformers: activations are
    (B, C, 1, S), projections are 1x1 convolutions, and attention is one einsum per head over keys
    transposed once per layer. Writing attention as matmuls over per-head transposes doubled the
    tower's time on the ANE and spilled the score matrix to DRAM. Weights are the checkpoint's own
    8-bit affine weights dequantized to float16, the values the MLX tower multiplies. The first load
    in each process specializes every bucket; the 5120 bucket takes about 20 minutes on an M4 Pro.
    """

let exporter = #"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch import TorchConverter, get_decomp_table
from safetensors import safe_open

MASK_NEG = -30000.0
snapshot, out = Path(sys.argv[1]), Path(sys.argv[2])
buckets = [int(v) for v in sys.argv[3].split(",")]
cfg = json.loads((snapshot / "config.json").read_text())
vc, quant = cfg["vision_config"], cfg.get("quantization", {})
D, HEADS, FF, LAYERS, P = vc["hidden_size"], vc["num_attention_heads"], vc["intermediate_size"], vc["num_hidden_layers"], vc["patch_size"]
TEXT = cfg["hidden_size"]
raw = {}
with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
    for key in f.keys():
        if key.startswith("visual."):
            t = f.get_tensor(key)
            raw[key[len("visual."):]] = t.float().numpy() if t.is_floating_point() else t.view(torch.int32).numpy().view(np.uint32)


def weight(name):
    w = raw[name + ".weight"]
    if name + ".scales" not in raw:
        return w.astype(np.float32)
    bits, group = quant.get("bits", 8), quant.get("group_size", 64)
    q = np.unpackbits(w.view(np.uint8), bitorder="little").reshape(w.shape[0], -1, bits)
    q = (q * (1 << np.arange(bits))).sum(-1).astype(np.float32)
    q = q.reshape(w.shape[0], -1, group) * raw[name + ".scales"][..., None] + raw[name + ".biases"][..., None]
    return q.reshape(w.shape[0], -1).astype(np.float32)


def conv(cin, cout, name=None):
    module = nn.Conv2d(cin, cout, 1, bias=True)
    if name is not None:
        w = weight(name)
        module.weight.data = torch.from_numpy(w.reshape(w.shape[0], -1, 1, 1))
        module.bias.data = torch.from_numpy(raw[name + ".bias"].astype(np.float32))
    return module


class LayerNormBC1S(nn.Module):
    def __init__(self, name, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(raw[name + ".weight"].astype(np.float32)).reshape(1, -1, 1, 1))
        self.bias = nn.Parameter(torch.from_numpy(raw[name + ".bias"].astype(np.float32)).reshape(1, -1, 1, 1))
        self.register_buffer("eps_sqrt", torch.tensor(eps ** 0.5), persistent=False)

    def forward(self, x):
        m = torch.clamp(x.abs().amax(1, keepdim=True), min=1e-4)
        c = x / m - (x / m).mean(1, keepdim=True)
        e = (self.eps_sqrt / m) * (self.eps_sqrt / m)
        return c * torch.rsqrt((c * c).mean(1, keepdim=True) + e) * self.weight + self.bias


def rope(x, cos, sin):
    b, c, _, length = x.shape
    d = c // HEADS
    x = x.reshape(b, HEADS, d, length)
    rotated = torch.cat((-x[:, :, d // 2:], x[:, :, :d // 2]), dim=2)
    return (x * cos + rotated * sin).reshape(b, c, 1, length)


def attention(q, k, v, bias):
    d = q.shape[1] // HEADS
    keys = k.transpose(1, 3).split(d, dim=3)
    heads = [torch.einsum("bkhq,bchk->bchq", (torch.einsum("bchq,bkhc->bkhq", qh, kh) + bias).softmax(dim=1), vh)
             for qh, kh, vh in zip((q * d ** -0.5).split(d, dim=1), keys, v.split(d, dim=1))]
    return torch.cat(heads, 1)


class Layer(nn.Module):
    def __init__(self, i):
        super().__init__()
        p = "layers.%d." % i
        self.norm1, self.norm2 = LayerNormBC1S(p + "layer_norm1", 1e-6), LayerNormBC1S(p + "layer_norm2", 1e-6)
        self.qkv, self.out = conv(D, 3 * D, p + "self_attn.qkv"), conv(D, D, p + "self_attn.out_proj")
        self.fc1, self.fc2 = conv(D, FF, p + "mlp.fc1"), conv(FF, D, p + "mlp.fc2")

    def forward(self, x, cos, sin, bias):
        q, k, v = self.qkv(self.norm1(x)).split(D, dim=1)
        x = x + self.out(attention(rope(q, cos, sin), rope(k, cos, sin), v, bias))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = conv(3 * P * P, D, "embeddings.patch_embedding")
        self.layers = nn.ModuleList(Layer(i) for i in range(LAYERS))
        self.post, self.pre = LayerNormBC1S("post_layernorm", vc["layer_norm_eps"]), LayerNormBC1S("projector.pre_norm", 1e-6)
        self.linear1, self.linear2 = conv(4 * D, 4 * D, "projector.linear_1"), conv(4 * D, TEXT, "projector.linear_2")

    def forward(self, patches, pos, cos, sin, bias):
        x = self.patch(patches) + pos
        for layer in self.layers:
            x = layer(x, cos, sin, bias)
        x = self.pre(self.post(x))
        n = x.shape[3]
        merged = x.reshape(D, n // 4, 4).permute(2, 0, 1).reshape(1, 4 * D, 1, n // 4)
        return self.linear2(F.gelu(self.linear1(merged)))


tower = Tower().to(torch.float16).eval()
out.mkdir(parents=True, exist_ok=True)
converter = TorchConverter()
with torch.no_grad():
    for n in buckets:
        h = torch.float16
        args = (torch.randn(1, 3 * P * P, 1, n, dtype=h), torch.randn(1, D, 1, n, dtype=h), torch.ones(1, 1, D // HEADS, n, dtype=h),
                torch.zeros(1, 1, D // HEADS, n, dtype=h), torch.zeros(1, n, 1, 1, dtype=h))
        program = torch.export.export(tower, args=args, strict=False).run_decompositions(get_decomp_table())
        converter = converter.add_exported_program(
            program, entrypoint_name="n%d" % n, input_names=["patches", "pos", "cos", "sin", "bias"], output_names=["features"])
        print("traced n%d" % n, flush=True)
shutil.rmtree(out / "tower.aimodel", ignore_errors=True)
converter.to_coreai().save_asset(out / "tower.aimodel")
manifest = {"paddle_vision_tower": {"asset": "tower", "masked": MASK_NEG, "patch": P, "merge": vc["spatial_merge_size"], "hidden": D,
                                    "text_hidden": TEXT, "head_dim": D // HEADS, "functions": [{"function": "n%d" % n, "tokens": n} for n in buckets]}}
(out / "metadata.json").write_text(json.dumps(manifest, indent=1))
print("wrote", out)
"""#

let arguments = CommandLine.arguments
guard arguments.count == 3 || arguments.count == 4 else {
    print(usage)
    exit(2)
}
let buckets = arguments.count == 4 ? arguments[3] : "1024,2048,5120"
guard buckets.split(separator: ",").allSatisfy({ Int($0).map { $0 > 0 && $0 % 4 == 0 } ?? false }) else {
    print("buckets must be positive multiples of 4, such as 1024,2048,5120")
    exit(2)
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
process.arguments = ["uv", "run", "--quiet", "--no-project", "--python", "3.12", "--exclude-newer", resolvedBefore] + pins.flatMap { ["--with", $0] }
    + ["python", "-I", "-c", exporter, arguments[1], arguments[2], buckets]
try process.run()
process.waitUntilExit()
exit(process.terminationStatus)
