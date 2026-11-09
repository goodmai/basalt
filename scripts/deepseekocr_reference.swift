#!/usr/bin/env swift
import Foundation

let resolvedBefore = "2026-11-09T00:00:00Z"

let usage = """
    Usage:
      swift scripts/deepseekocr_reference.swift golden
      swift scripts/deepseekocr_reference.swift pages <model-dir> <pages-dir> <out.json> <page-name>...

    Python mlx-vlm references for the DeepSeek-OCR port (#400 phase 2), run in a throwaway uv
    environment with the versions pinned below (checked with the Sonatype guide) and every
    transitive package resolved as of \(resolvedBefore). The model loads with trust_remote_code=False,
    so Python files shipped in a model directory never run.

    mlx-vlm 0.6.17's DeepEncoder departs from the official deepencoder.py in four places, which
    this script patches back before running it: CLIP's MLP uses quick_gelu and its layer norms eps
    1e-5; CLIP's position table is resized with antialiased bicubic over its spatial axes (stock
    mlx-vlm resizes channels and rows, an NHWC/NCHW mix-up); SAM's relative positions are resized
    with PyTorch's half-pixel linear interpolation. Base mode at 1024 only meets the first two.

      golden  on the tiny fixture's config with closed-form weights: decoder logits for a text
              prompt, and image features plus last logits from mlx-vlm get_input_embeddings for a
              128 view (padded windows), a 192 view (interpolated SAM and CLIP positions and
              relative positions) and a 2x1 grid of 64 tiles beside a 128 view; also the Gundam
              tile grid and the ImageOps.pad box for several page sizes; rewrites
              Tests/BasaltTests/Fixtures/DeepSeekOCRGolden.swift
      pages   greedy tokens with the document prompt for <pages-dir>/<name>.png: 64 tokens per
              non-tiled mode on the page pre-sized to the mode (<name>-<mode>.png, written here:
              stretched for tiny and small, padded for base and large), and 256 tokens in gundam
              mode on the unsized page; read by DeepSeekOCRRealModelTests through
              BASALT_E2E_DEEPSEEK_REFERENCE; also writes <out>-stock-gundam.json, the 1024-token
              gundam text of stock mlx-vlm, for the quality comparison
    """

let pins = ["mlx-vlm==0.6.17", "mlx==0.32.4", "transformers==5.22.2", "pillow==13.0.0", "sentencepiece==0.2.3", "starlette==1.10.0"]
let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()

let official = """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_vlm.models.deepseekocr.sam as sam_module
    import mlx_vlm.models.deepseekocr.vision as vision_module
    from mlx_vlm.models.kernels import bicubic_interpolate

    def rel_pos_linear(q_size, k_size, rel_pos):
        length = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != length:
            scale = rel_pos.shape[0] / length
            source = mx.maximum((mx.arange(length, dtype=mx.float32) + 0.5) * scale - 0.5, 0)
            low = mx.minimum(mx.floor(source).astype(mx.int32), rel_pos.shape[0] - 1)
            high = mx.minimum(low + 1, rel_pos.shape[0] - 1)
            weight = (source - low.astype(mx.float32))[:, None]
            table = rel_pos.astype(mx.float32)
            rel_pos = (table[low] * (1 - weight) + table[high] * weight).astype(rel_pos.dtype)
        coords = mx.arange(q_size)[:, None] - mx.arange(k_size)[None, :] + (k_size - 1)
        return rel_pos[coords]

    class QuickGELU(nn.Module):
        def __call__(self, x):
            return x * mx.sigmoid(1.702 * x)

    def make_official(model):
        for layer in model.vision_model.transformer.layers:
            layer.mlp.activation_fn = QuickGELU()
            layer.layer_norm1.eps = 1e-5
            layer.layer_norm2.eps = 1e-5

    def install_official():
        sam_module.get_rel_pos = rel_pos_linear
        vision_module.interpolate = lambda x, size: bicubic_interpolate(x, size=size, antialias=True)

    """

let golden = official + """
    import json, math, sys
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from PIL import Image, ImageOps
    from mlx_vlm.models.deepseekocr.config import ModelConfig, TextConfig, VisionConfig, ProjectorConfig
    from mlx_vlm.models.deepseekocr.deepseekocr import Model
    from mlx_vlm.models.deepseekocr.sam import SAMEncoder
    from mlx_vlm.models.deepseekocr.processing_deepseekocr import dynamic_preprocess

    text = TextConfig(model_type="deepseek_v2", vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, n_shared_experts=1, n_routed_experts=4,
                      num_experts_per_tok=2, first_k_dense_replace=1, q_lora_rank=None, qk_rope_head_dim=0, qk_nope_head_dim=0,
                      v_head_dim=0, rms_norm_eps=1e-6, rope_theta=10000.0)
    vision = VisionConfig(model_type="vision", layers=2, hidden_size=64, intermediate_size=256, num_attention_heads=4,
                          patch_size=14, image_size=224, layer_norm_eps=1e-6)
    config = ModelConfig(text_config=text, vision_config=vision, projector_config=ProjectorConfig(input_dim=128, n_embed=64),
                         model_type="deepseekocr")
    model = Model(config)
    install_official()
    make_official(model)
    model.sam_model = SAMEncoder(img_size=128, patch_size=16, embed_dim=32, depth=2, num_heads=2, window_size=3,
                                 global_attn_indexes=(1,), final_out_chans=64)
    params = sorted(tree_flatten(model.parameters()), key=lambda kv: kv[0])
    def value(key, t, n):
        shift = 1.0 if "norm" in key and key.endswith(".weight") else 0.0
        return [0.1 * math.sin(0.7 * i + 0.3 * t) + shift for i in range(n)]
    model.update(tree_unflatten([(k, mx.array(value(k, t, v.size), dtype=mx.float32).reshape(v.shape)) for t, (k, v) in enumerate(params)]))

    def pixels(count, side, phase):
        return mx.array([math.sin(0.05 * i + phase) for i in range(count * side * side * 3)], dtype=mx.float32).reshape(count, side, side, 3)

    def case(side, tiles):
        columns, rows, tile = tiles
        q, t = side // 64, tile // 64
        tokens = (q + 1) * q + 1 + ((t * columns + 1) * (t * rows) if columns * rows > 1 else 0)
        ids = mx.array([[1, 2] + [250] * tokens + [3, 4]])
        crops = pixels(columns * rows, tile, 1.0) if columns * rows > 1 else mx.zeros((1, 8, 8, 3))
        values = [crops.transpose(0, 3, 1, 2), pixels(1, side, 0.0).transpose(0, 3, 1, 2)]
        embeds = model.get_input_embeddings(ids, values, mx.array([[columns, rows]]), ids == 250).inputs_embeds
        logits = model.language_model(ids, inputs_embeds=embeds).logits
        return {"side": side, "tiles": list(tiles), "features": embeds[0, 2:2 + tokens].reshape(-1).tolist(), "logits": logits[0, -1].tolist()}

    decoder = model.language_model(mx.array([[5, 9, 13, 17, 21, 25, 29, 33]])).logits[0]
    sizes = [(1700, 2200), (2200, 1700), (1240, 1754), (800, 600), (641, 2000), (3000, 500), (1000, 1000), (2000, 2000), (600, 600)]
    grids = [list(dynamic_preprocess(Image.new("RGB", s))[1]) if max(s) > 640 else [1, 1] for s in sizes]
    pads = [list(ImageOps.pad(Image.new("RGB", s, (255, 255, 255)), (1024, 1024), color=(0, 0, 0)).getbbox()) for s in sizes]
    json.dump({"decoder": decoder.reshape(-1).tolist(), "cases": [case(128, (1, 1, 0)), case(192, (1, 1, 0)), case(128, (2, 1, 64))],
               "sizes": [list(s) for s in sizes], "grids": grids, "pads": pads}, sys.stdout)
    """

let pages = official + """
    import json, sys
    import mlx.core as mx
    from PIL import Image, ImageOps
    from mlx_vlm import load
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.prompt_utils import apply_chat_template

    model_path, pages_dir, out_path, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    model, processor = load(model_path, trust_remote_code=False)
    stock = (sam_module.get_rel_pos, vision_module.interpolate, [layer.mlp.activation_fn for layer in model.vision_model.transformer.layers])
    def use_stock():
        sam_module.get_rel_pos, vision_module.interpolate = stock[0], stock[1]
        for layer, activation in zip(model.vision_model.transformer.layers, stock[2]):
            layer.mlp.activation_fn = activation
            layer.layer_norm1.eps = layer.layer_norm2.eps = 1e-6
    install_official()
    make_official(model)
    prompt = apply_chat_template(processor, model.config, "<|grounding|>Convert the document to markdown.", num_images=1)
    eos = processor.tokenizer.eos_token_id
    def greedy(path, count, side, tile, cropping):
        inputs = processor(text=prompt, images=[Image.open(path).convert("RGB")], base_size=side, image_size=tile, cropping=cropping)
        cache = [KVCache() for _ in model.language_model.model.layers]
        logits = model(inputs["input_ids"], pixel_values=inputs["images"], cache=cache, images_spatial_crop=inputs["images_spatial_crop"],
                       images_seq_mask=inputs["images_seq_mask"]).logits
        tokens = []
        while len(tokens) < count:
            token = int(mx.argmax(logits[0, -1]).item())
            tokens.append(token)
            if token == eos:
                break
            logits = model(mx.array([[token]]), cache=cache).logits
        return {"prompt": prompt, "prompt_ids": [int(t) for t in inputs["input_ids"][0].tolist()], "tokens": tokens,
                "text": processor.tokenizer.decode(tokens), "max_tokens": count}
    results = {}
    for name in names:
        page = Image.open(f"{pages_dir}/{name}.png").convert("RGB")
        for mode, side in [("tiny", 512), ("small", 640), ("base", 1024), ("large", 1280)]:
            sized = page.resize((side, side), Image.BICUBIC) if side <= 640 else ImageOps.pad(page, (side, side), color=(127, 127, 127))
            path = f"{pages_dir}/{name}-{mode}.png"
            sized.save(path)
            results[f"{name}-{mode}"] = dict(greedy(path, 64, side, side, False), image=f"{name}-{mode}.png", resolution=mode)
            print(name, mode, flush=True, file=sys.stderr)
        results[f"{name}-gundam"] = dict(greedy(f"{pages_dir}/{name}.png", 256, 1024, 640, True), image=f"{name}.png", resolution="gundam")
        print(name, "gundam", flush=True, file=sys.stderr)
    use_stock()
    stock_results = {name: greedy(f"{pages_dir}/{name}.png", 1024, 1024, 640, True)["text"] for name in names}
    json.dump(results, open(out_path, "w"))
    json.dump(stock_results, open(out_path.replace(".json", "-stock-gundam.json"), "w"))
    """

func python(_ code: String, _ arguments: [String]) throws -> Data {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["uv", "run", "--quiet", "--no-project", "--python", "3.12", "--exclude-newer", resolvedBefore] + pins.flatMap { ["--with", $0] }
        + ["python", "-I", "-c", code] + arguments
    let output = Pipe()
    process.standardOutput = output
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw NSError(domain: "uv", code: Int(process.terminationStatus)) }
    return data
}

func floats(_ values: [Double]) -> String {
    stride(from: 0, to: values.count, by: 8).map { start in
        "            " + values[start ..< min(start + 8, values.count)].map { String(format: "%.9g", $0) }.joined(separator: ", ") + ","
    }.joined(separator: "\n")
}

func goldenSource(_ json: [String: Any]) throws -> String {
    guard let decoder = json["decoder"] as? [Double], let cases = json["cases"] as? [[String: Any]], let sizes = json["sizes"] as? [[Int]],
          let grids = json["grids"] as? [[Int]], let pads = json["pads"] as? [[Int]] else {
        throw NSError(domain: "golden", code: 1)
    }
    let rendered = try cases.enumerated().map { index, item -> String in
        guard let side = item["side"] as? Int, let tiles = item["tiles"] as? [Int], let features = item["features"] as? [Double],
              let logits = item["logits"] as? [Double] else { throw NSError(domain: "golden", code: 2) }
        return """
            let deepSeekOCRGoldenCase\(index) = DeepSeekOCRGolden.Case(side: \(side), tiles: \(tiles), features: [
            \(floats(features))
            ], logits: [
            \(floats(logits))
            ])

            """
    }
    return """
        struct DeepSeekOCRGolden {
            struct Case {
                let side: Int
                let tiles: [Int]
                let features: [Float]
                let logits: [Float]
            }

            static let sizes = \(sizes)
            static let grids = \(grids)
            static let pads = \(pads)
            static var decoder: [Float] { deepSeekOCRGoldenDecoder }
            static var cases: [Case] { [\(cases.indices.map { "deepSeekOCRGoldenCase\($0)" }.joined(separator: ", "))] }
        }

        let deepSeekOCRGoldenDecoder: [Float] = [
        \(floats(decoder))
        ]

        \(rendered.joined(separator: "\n"))
        """
}

let arguments = Array(CommandLine.arguments.dropFirst())
switch arguments.first {
case "golden":
    let json = try JSONSerialization.jsonObject(with: try python(golden, [])) as? [String: Any] ?? [:]
    let target = repo.appending(path: "Tests/BasaltTests/Fixtures/DeepSeekOCRGolden.swift")
    try goldenSource(json).write(to: target, atomically: true, encoding: .utf8)
    print("wrote \(target.path)")
case "pages" where arguments.count >= 5:
    _ = try python(pages, Array(arguments.dropFirst()))
    print("wrote \(arguments[3])")
default:
    print(usage)
    exit(arguments.first == nil ? 0 : 1)
}
