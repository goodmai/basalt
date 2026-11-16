#!/usr/bin/env swift
import Foundation

let resolvedBefore = "2026-11-10T00:00:00Z"
let pins = ["coreai-torch==0.4.3", "coreai-core==1.0.0b3", "torch==2.15.0", "safetensors==0.8.0", "numpy==2.4.6"]
let defaultShapes = "1x256,4x256,1x512,4x512,1x1024,4x1024"

let usage = """
    Usage:
      swift scripts/laya_decision_export.swift <Laya/ModernBERT checkpoint dir> <out dir> [shapes, default \(defaultShapes)]
      swift scripts/laya_decision_export.swift --emit-sample-manifest <out file> [shapes, default \(defaultShapes)]

    Exports a Laya / ModernBERT typed-decision checkpoint as a Core AI decision bundle for
    serve --decision-model <out dir> (#234):
      - <out>/decision.aimodel (static BC1S entrypoints b<batch>_n<length> on the Neural Engine)
      - <out>/token_table.aimodel (CPU embedding table entrypoint main: input_ids -> inputs_embeds)
      - <out>/tokenizer/tokenizer.json
      - <out>/laya_decision.json

    Runs in a throwaway uv environment with the versions pinned above, every transitive package
    resolved as of \(resolvedBefore).
    """

struct ShapeBucket {
    let batch: Int
    let length: Int

    var name: String { "b\(batch)_n\(length)" }
}

func parseShapes(_ spec: String) -> [ShapeBucket]? {
    var buckets: [ShapeBucket] = []
    for raw in spec.split(separator: ",", omittingEmptySubsequences: false) {
        let parts = raw.split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2, let batch = Int(parts[0]), let length = Int(parts[1]), batch > 0, length >= 4 else {
            return nil
        }
        buckets.append(ShapeBucket(batch: batch, length: length))
    }
    return buckets.isEmpty ? nil : buckets
}

func sampleManifest(_ buckets: [ShapeBucket]) -> [String: Any] {
    let maxLen = buckets.map(\.length).max() ?? 1024
    let headMaxLen = min(192, maxLen - 1)
    let functions: [[String: Any]] = buckets.map { ["function": $0.name, "batch": $0.batch, "length": $0.length] }
    let body: [String: Any] = [
        "model_id": "basalt-community/Laya-Decision-395M-ANE",
        "max_len": maxLen,
        "head_max_len": headMaxLen,
        "max_option_tokens": 48,
        "tokens": ["cls": 50281, "sep": 50282, "mask": 50284, "pad": 50283],
        "masked": -30000.0,
        "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
        "temperature_by_options": [
            "choice:2": 1.9063563346862793,
            "choice:3-5": 1.7601518630981445,
            "choice:6-10": 1.0000158548355103,
            "choice:11+": 0.10058280825614929,
            "noul:2": 1.983399510383606,
            "score:3-5": 1.2514300346374512,
        ],
        "functions": functions,
    ]
    return ["laya_decision": body]
}

func writeSampleManifest(to path: String, buckets: [ShapeBucket]) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let data = try JSONSerialization.data(withJSONObject: sampleManifest(buckets), options: [.prettyPrinted, .sortedKeys])
    try (data + Data("\n".utf8)).write(to: url, options: .atomic)
}

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
shapes = [(int(p.split("x")[0]), int(p.split("x")[1])) for p in sys.argv[3].split(",")]
enc_cfg_path = snapshot / "encoder" / "config.json"
ecfg = json.loads((enc_cfg_path if enc_cfg_path.exists() else snapshot / "config.json").read_text())
rl_path = snapshot / "rl_agent_config.json"
rl_cfg = json.loads(rl_path.read_text()) if rl_path.exists() else {}
tok_src = snapshot / "tokenizer" / "tokenizer.json"
if not tok_src.exists():
    tok_src = snapshot / "tokenizer.json"
tok_json = json.loads(tok_src.read_text())
added = {t["content"]: int(t["id"]) for t in tok_json.get("added_tokens", []) if "content" in t and "id" in t}
cls_id = int(ecfg.get("cls_token_id", added.get("[CLS]", 50281)))
sep_id = int(ecfg.get("sep_token_id", added.get("[SEP]", 50282)))
pad_id = int(ecfg.get("pad_token_id", added.get("[PAD]", 50283)))
mask_id = int(ecfg.get("mask_token_id", added.get("[MASK]", 50284)))

raw = {}
index_path = snapshot / "model.safetensors.index.json"
shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists() else ["model.safetensors"]
for shard in shards:
    with safe_open(str(snapshot / shard), framework="pt") as f:
        for key in f.keys():
            raw[key] = f.get_tensor(key).float().numpy()

ep = "encoder." if "encoder.final_norm.weight" in raw else ("model." if "model.final_norm.weight" in raw else "")
D = int(ecfg["hidden_size"])
HEADS = int(ecfg["num_attention_heads"])
FF = int(ecfg["intermediate_size"])
LAYERS = int(ecfg["num_hidden_layers"])
NORM_EPS = float(ecfg.get("norm_eps", ecfg.get("layer_norm_eps", 1e-5)))
LOCAL_HALF = int(ecfg.get("local_attention", 128)) // 2
every_n = int(ecfg.get("global_attn_every_n_layers", 3))
layer_types = ecfg.get("layer_types") or ["full_attention" if i % every_n == 0 else "sliding_attention" for i in range(LAYERS)]
rope_cfg = ecfg.get("rope_parameters") or {}
full_theta = float(rope_cfg.get("full_attention", {}).get("rope_theta", ecfg.get("global_rope_theta", 160000.0)))
slide_theta = float(rope_cfg.get("sliding_attention", {}).get("rope_theta", ecfg.get("local_rope_theta", 10000.0)))
head_layers = int(rl_cfg.get("head_layers", sum(1 for k in raw if k.startswith("head.layers.") and k.endswith(".norm1.weight"))))
head_nhead = max(1, D // 64)


def conv(cin, cout, name):
    w_key = name + ".weight" if name + ".weight" in raw else name + "_weight"
    b_key = name + ".bias" if name + ".bias" in raw else (name + "_bias" if name + "_bias" in raw else None)
    module = nn.Conv2d(cin, cout, 1, bias=b_key is not None)
    module.weight.data = torch.from_numpy(raw[w_key].astype(np.float32).reshape(cout, cin, 1, 1))
    if b_key is not None:
        module.bias.data = torch.from_numpy(raw[b_key].astype(np.float32))
    return module


class LayerNormBC1S(nn.Module):
    def __init__(self, name, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(raw[name + ".weight"].astype(np.float32)).reshape(1, -1, 1, 1))
        b = raw[name + ".bias"].astype(np.float32) if name + ".bias" in raw else None
        self.bias = nn.Parameter(torch.from_numpy(b).reshape(1, -1, 1, 1)) if b is not None else None
        self.register_buffer("eps_sqrt", torch.tensor(eps ** 0.5), persistent=False)

    def forward(self, x):
        m = torch.clamp(x.abs().amax(1, keepdim=True), min=1e-4)
        c = x / m - (x / m).mean(1, keepdim=True)
        e = (self.eps_sqrt / m) * (self.eps_sqrt / m)
        y = c * torch.rsqrt((c * c).mean(1, keepdim=True) + e) * self.weight
        return y + self.bias if self.bias is not None else y


def rope_tables(theta, length):
    d = D // HEADS
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    freqs = np.outer(np.arange(length, dtype=np.float32), inv)
    emb = np.concatenate([freqs, freqs], axis=-1).T.reshape(1, 1, d, length)
    return torch.from_numpy(np.cos(emb)), torch.from_numpy(np.sin(emb))


def sliding_band(length):
    pos = np.arange(length)[:, None]
    mask = np.where(np.abs(pos - pos.T) <= LOCAL_HALF, 0.0, MASK_NEG).astype(np.float32)
    return torch.from_numpy(mask.reshape(1, length, 1, length))


def apply_rope(x, cos, sin):
    b, c, _, n = x.shape
    d = c // HEADS
    x = x.reshape(b, HEADS, d, n)
    rot = torch.cat((-x[:, :, d // 2:], x[:, :, :d // 2]), dim=2)
    return (x * cos + rot * sin).reshape(b, c, 1, n)


def attention(q, k, v, attn_bias, n_heads):
    d = q.shape[1] // n_heads
    keys = k.transpose(1, 3).split(d, dim=3)
    heads = [torch.einsum("bkhq,bchk->bchq", (torch.einsum("bchq,bkhc->bkhq", qh, kh) + attn_bias).softmax(dim=1), vh)
             for qh, kh, vh in zip((q * d ** -0.5).split(d, dim=1), keys, v.split(d, dim=1))]
    return torch.cat(heads, 1)


class ModernBertLayer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        p = "%slayers.%d." % (ep, idx)
        self.sliding = layer_types[idx] == "sliding_attention"
        self.attn_norm = nn.Identity() if idx == 0 else LayerNormBC1S(p + "attn_norm", NORM_EPS)
        self.wqkv = conv(D, 3 * D, p + "attn.Wqkv")
        self.wo = conv(D, D, p + "attn.Wo")
        self.mlp_norm = LayerNormBC1S(p + "mlp_norm", NORM_EPS)
        self.wi = conv(D, 2 * FF, p + "mlp.Wi")
        self.mwo = conv(FF, D, p + "mlp.Wo")

    def forward(self, x, ctx):
        cos, sin = (ctx[2], ctx[3]) if self.sliding else (ctx[0], ctx[1])
        attn_bias = ctx[4] if self.sliding else ctx[5]
        q, k, v = self.wqkv(self.attn_norm(x)).split(D, dim=1)
        x = x + self.wo(attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, attn_bias, HEADS))
        u, g = self.wi(self.mlp_norm(x)).split(FF, dim=1)
        return x + self.mwo(F.gelu(u) * g)


class DecisionHeadLayer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        p = "head.layers.%d." % idx
        self.norm1 = LayerNormBC1S(p + "norm1", 1e-5)
        self.qkv = conv(D, 3 * D, p + "self_attn.in_proj")
        self.out = conv(D, D, p + "self_attn.out_proj")
        self.norm2 = LayerNormBC1S(p + "norm2", 1e-5)
        self.fc1 = conv(D, 4 * D, p + "linear1")
        self.fc2 = conv(4 * D, D, p + "linear2")

    def forward(self, x, bias):
        q, k, v = self.qkv(self.norm1(x)).split(D, dim=1)
        x = x + self.out(attention(q, k, v, bias, head_nhead))
        return x + self.fc2(F.relu(self.fc1(self.norm2(x))))


class DecisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_norm = LayerNormBC1S(ep + "embeddings.norm", NORM_EPS)
        self.layers = nn.ModuleList(ModernBertLayer(i) for i in range(LAYERS))
        self.final_norm = LayerNormBC1S(ep + "final_norm", NORM_EPS)
        self.type_proj = nn.Conv2d(3, D, 1, bias=False)
        self.type_proj.weight.data = torch.from_numpy(raw["type_emb.weight"].astype(np.float32).T.reshape(D, 3, 1, 1))
        self.head = nn.ModuleList(DecisionHeadLayer(i) for i in range(head_layers))
        self.s0 = LayerNormBC1S("scorer.0", 1e-5)
        self.s1 = conv(D, D, "scorer.1")
        self.s3 = conv(D, 1, "scorer.3")

    def forward(self, x, bias, qtype, tables):
        x = self.emb_norm(x)
        ctx = (tables[0], tables[1], tables[2], tables[3], torch.minimum(bias, tables[4]), bias)
        for layer in self.layers:
            x = layer(x, ctx)
        x = self.final_norm(x) + self.type_proj(qtype.reshape(qtype.shape[0], 3, 1, 1))
        for layer in self.head:
            x = layer(x, bias)
        return self.s3(F.gelu(self.s1(self.s0(x))))


class BoundDecision(nn.Module):
    def __init__(self, backbone, batch, length):
        super().__init__()
        self.backbone = backbone
        self.batch = batch
        self.length = length
        f_cos, f_sin = rope_tables(full_theta, length)
        s_cos, s_sin = rope_tables(slide_theta, length)
        self.register_buffer("f_cos", f_cos.to(torch.float16), persistent=False)
        self.register_buffer("f_sin", f_sin.to(torch.float16), persistent=False)
        self.register_buffer("s_cos", s_cos.to(torch.float16), persistent=False)
        self.register_buffer("s_sin", s_sin.to(torch.float16), persistent=False)
        self.register_buffer("band", sliding_band(length).to(torch.float16), persistent=False)

    def forward(self, inputs_embeds, bias, qtype):
        x = inputs_embeds.reshape(self.batch, self.length, D).permute(0, 2, 1).reshape(self.batch, D, 1, self.length)
        scores = self.backbone(x, bias, qtype, (self.f_cos, self.f_sin, self.s_cos, self.s_sin, self.band))
        return scores.reshape(self.batch, self.length)


class TokenTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(raw[ep + "embeddings.tok_embeddings.weight"]).to(torch.float16), requires_grad=False)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)


out.mkdir(parents=True, exist_ok=True)
tok_out = out / "tokenizer"
tok_out.mkdir(parents=True, exist_ok=True)
shutil.copy2(tok_src, tok_out / "tokenizer.json")
tok_cfg_src = tok_src.parent / "tokenizer_config.json"
if tok_cfg_src.exists():
    shutil.copy2(tok_cfg_src, tok_out / "tokenizer_config.json")

table = TokenTable().eval()
max_flat = max(8192, max(b * n for b, n in shapes))
with torch.no_grad():
    table_ep = torch.export.export(
        table, args=(torch.zeros(1, 128, dtype=torch.int32),),
        dynamic_shapes={"input_ids": {1: torch.export.Dim("n", min=1, max=max_flat)}}, strict=False
    ).run_decompositions(get_decomp_table())
shutil.rmtree(out / "token_table.aimodel", ignore_errors=True)
TorchConverter().add_exported_program(
    table_ep, entrypoint_name="main", input_names=["input_ids"], output_names=["inputs_embeds"]
).to_coreai().save_asset(out / "token_table.aimodel")

backbone = DecisionBackbone().to(torch.float16).eval()
converter = TorchConverter()
with torch.no_grad():
    for b, n in shapes:
        bound = BoundDecision(backbone, b, n).to(torch.float16).eval()
        args = (
            torch.zeros(1, b * n, D, dtype=torch.float16),
            torch.zeros(b, n, 1, 1, dtype=torch.float16),
            torch.zeros(b, 3, dtype=torch.float16),
        )
        program = torch.export.export(bound, args=args, strict=False).run_decompositions(get_decomp_table())
        converter = converter.add_exported_program(
            program, entrypoint_name="b%d_n%d" % (b, n), input_names=["inputs_embeds", "bias", "qtype"], output_names=["scores"])
        print("traced b%d_n%d" % (b, n), flush=True)
shutil.rmtree(out / "decision.aimodel", ignore_errors=True)
converter.to_coreai().save_asset(out / "decision.aimodel")

max_bucket = max(n for _, n in shapes)
max_len = min(int(rl_cfg.get("max_len", max_bucket)), max_bucket)
head_max_len = min(int(rl_cfg.get("head_max_len", 192)), max_len - 1)
raw_temps = rl_cfg.get("temperature") or (raw["temperature"].tolist() if "temperature" in raw else [1.0, 1.0, 1.0])
temps = [float(v) for v in raw_temps]
by_opts = {k: float(v) for k, v in rl_cfg.get("temperature_by_options", {}).items() if float(v) > 0}
model_id = rl_cfg.get("model_id") or ("convaiinnovations/laya" if rl_path.exists() else snapshot.name)
manifest = {
    "laya_decision": {
        "model_id": model_id,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "max_option_tokens": int(rl_cfg.get("max_option_tokens", 48)),
        "tokens": {"cls": cls_id, "sep": sep_id, "mask": mask_id, "pad": pad_id},
        "masked": MASK_NEG,
        "temperature": temps,
        "temperature_by_options": by_opts,
        "functions": [{"function": "b%d_n%d" % (b, n), "batch": b, "length": n} for b, n in shapes],
    }
}
(out / "laya_decision.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("wrote", out)
"""#

let arguments = CommandLine.arguments
if arguments.count >= 2 && (arguments[1] == "--help" || arguments[1] == "-h") {
    print(usage)
    exit(0)
}
if arguments.count >= 2 && arguments[1] == "--emit-sample-manifest" {
    guard arguments.count == 3 || arguments.count == 4 else {
        print(usage)
        exit(2)
    }
    let spec = arguments.count == 4 ? arguments[3] : defaultShapes
    guard let buckets = parseShapes(spec) else {
        print("shapes must be comma-separated <batch>x<length> pairs with batch >= 1 and length >= 4")
        exit(2)
    }
    try writeSampleManifest(to: arguments[2], buckets: buckets)
    exit(0)
}
guard arguments.count == 3 || arguments.count == 4 else {
    print(usage)
    exit(2)
}
let shapeSpec = arguments.count == 4 ? arguments[3] : defaultShapes
guard parseShapes(shapeSpec) != nil else {
    print("shapes must be comma-separated <batch>x<length> pairs with batch >= 1 and length >= 4")
    exit(2)
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
process.arguments = ["uv", "run", "--quiet", "--no-project", "--python", "3.12", "--exclude-newer", resolvedBefore]
    + pins.flatMap { ["--with", $0] }
    + ["python", "-I", "-c", exporter, arguments[1], arguments[2], shapeSpec]
try process.run()
process.waitUntilExit()
exit(process.terminationStatus)
