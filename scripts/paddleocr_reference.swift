#!/usr/bin/env swift
import Foundation

let resolvedBefore = "2026-11-09T00:00:00Z"

let usage = """
    Usage:
      swift scripts/paddleocr_reference.swift golden
      swift scripts/paddleocr_reference.swift pages <model-dir> <pages-dir> <out.json> <page-name>...

    Python mlx-vlm references for the PaddleOCR-VL port (#400), run in a throwaway uv environment
    with the versions pinned below (checked with the Sonatype guide) and every transitive package
    resolved as of \(resolvedBefore). The model loads with trust_remote_code=False, so Python files
    shipped in a model directory never run.

      golden  the mlx-vlm VisionModel output and get_rope_index on the tiny fixture's vision tower,
              with closed-form weights and pixels on two non-square grids (4x6, 2x4); rewrites
              Tests/BasaltTests/Fixtures/PaddleOCRVLGolden.swift
      pages   greedy tokens with the OCR: prompt for <pages-dir>/<name>.png: 64 tokens on the page
              pre-resized with PIL bicubic (<name>-sized.png, written here), and 256 tokens on the
              unsized page through the mlx-vlm processor (<name>-unsized); read by
              PaddleOCRVLRealModelTests through BASALT_E2E_PADDLE_REFERENCE
    """

let pins = ["mlx-vlm==0.6.17", "mlx==0.32.4", "transformers==5.22.2", "pillow==13.0.0", "sentencepiece==0.2.3", "starlette==1.10.0"]
let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()

let golden = """
    import json, math, sys
    from types import SimpleNamespace
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.models.paddleocr_vl.config import VisionConfig
    from mlx_vlm.models.paddleocr_vl.vision import VisionModel, PaddleOCRProjector
    from mlx_vlm.models.paddleocr_vl.language import LanguageModel

    config = VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_channels=3,
                          image_size=56, patch_size=14, layer_norm_eps=1e-6, spatial_merge_size=2)
    vision = VisionModel(config)
    vision.projector = PaddleOCRProjector(32, 64, 2)
    params = sorted(tree_flatten(vision.parameters()), key=lambda kv: kv[0])
    def value(key, t, n):
        shift = 1.0 if "norm" in key and key.endswith(".weight") else 0.0
        return [0.1 * math.sin(0.7 * i + 0.3 * t) + shift for i in range(n)]
    vision.update(tree_unflatten([(k, mx.array(value(k, t, v.size), dtype=mx.float32).reshape(v.shape)) for t, (k, v) in enumerate(params)]))
    count = 32 * 14 * 14 * 3
    pixels = mx.array([math.sin(0.05 * i) for i in range(count)], dtype=mx.float32).reshape(32, 14, 14, 3).transpose(0, 3, 1, 2)[None]
    features = vision(pixels, mx.array([[1, 4, 6], [1, 2, 4]]))
    tokens = [7, 101] + [250] * 6 + [102, 8, 101] + [250] * 2 + [102, 9]
    owner = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=250,
                                                   video_token_id=251, vision_start_token_id=101))
    positions, delta = LanguageModel.get_rope_index(owner, mx.array([tokens]), mx.array([[1, 4, 6], [1, 2, 4]]))
    json.dump({"shape": list(features.shape), "features": features.reshape(-1).tolist(), "tokens": tokens,
               "positions": positions[:, 0, :].tolist(), "delta": int(delta.item())}, sys.stdout)
    """

let pages = """
    import json, sys
    from PIL import Image
    from mlx_vlm import load
    from mlx_vlm.generate.dispatch import stream_generate
    from mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl import smart_resize

    model_path, pages_dir, out_path, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    model, processor = load(model_path, trust_remote_code=False)
    ip = processor.image_processor
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    def greedy(path, count):
        inputs = processor(images=[Image.open(path).convert("RGB")], text=[prompt])
        tokens = [int(r.token) for r in stream_generate(model, processor, prompt, image=[path], max_tokens=count, temperature=0.0) if r.token is not None]
        return {"prompt_ids": [int(t) for t in inputs["input_ids"][0]], "tokens": tokens[:count], "text": processor.tokenizer.decode(tokens[:count]),
                "max_tokens": count}
    results = {}
    for name in names:
        page = f"{pages_dir}/{name}.png"
        image = Image.open(page).convert("RGB")
        h, w = smart_resize(image.height, image.width, factor=ip.patch_size * ip.merge_size, min_pixels=ip.min_pixels, max_pixels=ip.max_pixels)
        sized = f"{pages_dir}/{name}-sized.png"
        image.resize((w, h), Image.BICUBIC).save(sized)
        results[name] = dict(greedy(sized, 64), image=f"{name}-sized.png")
        results[f"{name}-unsized"] = dict(greedy(page, 256), image=f"{name}.png")
        print(name, w, h, flush=True, file=sys.stderr)
    json.dump(results, open(out_path, "w"))
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

func goldenSource(_ json: [String: Any]) throws -> String {
    guard let features = json["features"] as? [Double], let positions = json["positions"] as? [[Int]],
          let tokens = json["tokens"] as? [Int], let delta = json["delta"] as? Int, let shape = json["shape"] as? [Int] else {
        throw NSError(domain: "golden", code: 1)
    }
    let rows = stride(from: 0, to: features.count, by: 8).map { start in
        "        " + features[start ..< min(start + 8, features.count)].map { String(format: "%.9g", $0) }.joined(separator: ", ") + ","
    }
    return """
        enum PaddleOCRVLGolden {
            static let featureShape = \(shape)
            static let tokens = \(tokens)
            static let positions: [[Int32]] = \(positions)
            static let delta = \(delta)
            static let features: [Float] = [
        \(rows.joined(separator: "\n"))
            ]
        }

        """
}

let arguments = Array(CommandLine.arguments.dropFirst())
switch arguments.first {
case "golden":
    let json = try JSONSerialization.jsonObject(with: try python(golden, [])) as? [String: Any] ?? [:]
    let target = repo.appending(path: "Tests/BasaltTests/Fixtures/PaddleOCRVLGolden.swift")
    try goldenSource(json).write(to: target, atomically: true, encoding: .utf8)
    print("wrote \(target.path)")
case "pages" where arguments.count >= 5:
    _ = try python(pages, Array(arguments.dropFirst()))
    print("wrote \(arguments[3])")
default:
    print(usage)
    exit(arguments.first == nil ? 0 : 1)
}
