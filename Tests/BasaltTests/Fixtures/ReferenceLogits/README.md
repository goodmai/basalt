# Reference logits (#324)

Each directory holds a tiny random checkpoint and the logits the Python reference implementation, mlx-lm, computes for it. `ReferenceLogitsTests` loads the checkpoint through `loadModelContainer`, the same model factory path `basalt serve` uses, runs the same 16 tokens through a fresh cache and compares the first and last position logits.

| Directory | `model_type` | What it exercises |
|---|---|---|
| `qwen3` | `qwen3` | dense attention, q/k norm |
| `qwen3_moe` | `qwen3_moe` | softmax top-k router, SwitchGLU |
| `qwen3_5` | `qwen3_5` + `text_config` | gated delta net + full attention, output gate, partial RoPE (0.25), `language_model.` prefix |
| `qwen3_5_moe` | `qwen3_5_moe` + `text_config` | the above with shared-expert MoE |
| `gemma4` | `gemma4_text` | sliding window 8 over 16 tokens, per-layer inputs, KV-shared layers, proportional RoPE on full layers, logit softcap |
| `gemma4_moe` | `gemma4_text` | the 26B-A4B layout: `attention_k_eq_v`, one global KV head, dense MLP plus experts |
| `lfm2` | `lfm2` | short conv + attention, auto-adjusted FFN width |
| `lfm2_moe` | `lfm2_moe` | the LFM2.5-8B-A1B layout: dense first layer, expert bias |
| `nemotron_h` | `nemotron_h` | Mamba-2 + attention + relu² MLP (`M*M-`) |
| `nemotron_h_moe` | `nemotron_h` | sigmoid grouped router with correction bias, shared expert (`ME*E`) |

Shapes follow `TinyModel` in `../TinyModels.swift` (4 layers, hidden 64, 4 heads, 2 KV heads, vocab 256); the RoPE, gating and MoE settings are copied from the checkpoints in the model-runs skill. Weights are float32, so both sides compute in float32.

## Tolerance

`1e-4` on the maximum absolute logit difference. Logits are of order 2 to 5. Measured with mlx 0.32.4 on the stored checkpoints:

| Source of difference | Max abs difference |
|---|---|
| rounding to 6 decimals in `reference.json` | 5e-7 |
| the same mlx-lm model on Metal instead of CPU | 1e-6 to 9e-6 |
| `rms_norm_eps` ×10 (smallest: `gemma4_moe`, last position) | 5.5e-4 |
| RoPE base ×2 (smallest: `qwen3_5`, 4 rotary dims at base 1e7) | 1.6e-4 |
| Gemma 4 sliding window 8 → 9 | 0.87 |
| Gemma 4 full-attention partial rotary 0.25 → 0.5 | 0.28 |
| Nemotron-H `dt` clipped to [0.001, 0.1] | 0.20 |

Float16 weights would raise the noise floor to about 1e-2, which would hide the eps and RoPE base cases, so the fixtures stay float32. The first position has no RoPE dependence, so RoPE and window bugs show up only in `last`.

## Regenerating

mlx-lm runs ephemerally and is not a dependency of basalt. mlx-lm 0.31.1 has no `gemma4`, so the two Gemma 4 directories come from 0.32.2; `reference.json` records the version that wrote it. The generator reads the committed `config.json`, seeds with the sum of the directory name's character codes, adds `0.05 · N(0, 1)` to every parameter of mlx-lm's default initialisation (norm weights included, so a missing `1 + w` offset shows), saves float32 safetensors, reloads them with mlx-lm's loader and runs on CPU. Two runs produce identical bytes.

```
uvx --from mlx-lm==0.31.1 python -I generate.py Tests/BasaltTests/Fixtures/ReferenceLogits qwen3 qwen3_moe qwen3_5 qwen3_5_moe lfm2 lfm2_moe nemotron_h nemotron_h_moe
uvx --from mlx-lm==0.32.2 python -I generate.py Tests/BasaltTests/Fixtures/ReferenceLogits gemma4 gemma4_moe
```

`generate.py`, kept out of the repository because basalt's scripts are Swift:

```python
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import _get_classes, load_model

TOKENS = [2, 17, 42, 101, 7, 230, 64, 13, 199, 88, 5, 150, 33, 250, 71, 120]

mx.set_default_device(mx.cpu)
for family in sys.argv[2:]:
    directory = Path(sys.argv[1]) / family
    config = json.loads((directory / "config.json").read_text())
    mx.random.seed(sum(map(ord, family)))
    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    weights = {k: (v + 0.05 * mx.random.normal(v.shape)).astype(mx.float32) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(directory / "model.safetensors"), weights, metadata={"format": "mlx"})
    loaded, _ = load_model(directory, strict=True)
    logits = loaded(mx.array([TOKENS]), cache=make_prompt_cache(loaded)).astype(mx.float32)[0]
    reference = {
        "generator": f"mlx-lm {mlx_lm.__version__}, mlx {mx.__version__}",
        "tokens": TOKENS,
        "first": [round(x, 6) for x in logits[0].tolist()],
        "last": [round(x, 6) for x in logits[-1].tolist()],
    }
    (directory / "reference.json").write_text(json.dumps(reference) + "\n")
```

## Not covered

- Prism (`prism_hadamard_qwen35`): Hadamard-rotated quantized weights with no mlx-lm counterpart.
- Llama and Gemma 3: out of scope for basalt's runs.
- `nemotron_h` sets `time_step_limit` to `[0, 1e9]` explicitly, like Nemotron-Nano-9B-v2. Nemotron-3-Nano-4B ships only `time_step_min`/`time_step_max`, which mlx-lm turns into a `dt` clip and basalt ignores (see #324).
