# Features and benchmarks

## Features at a glance

Every speed-up is opt-in and keeps greedy output identical unless the row says otherwise. Numbers: M4 Pro 24 GB on power, release build.

| Feature | Flag | What it does | Measured |
|---|---|---|---|
| Prompt lookup (speculative decoding) | `serve --speculative-lookup` | drafts up to 16 tokens copied from the prompt or output, verified in one forward | 1.84–2.66× on code edits (Bonsai 2 27B), 3.45–3.62× (Gemma 4 12B); ~1.0× on prose |
| MTP head drafting | `serve --mtp [--mtp-block N]` | drafts with the model's multi-token-prediction head (`mtp.safetensors`, e.g. MTPLX packs); exact at any temperature. Rounds start at one drafted token and grow up to N−1 (default 4) after four fully accepted rounds; a rejection drops back to one | 1.30× on code, 1.04× on prose, 1.26× on reasoning (Bonsai 2 27B, T=0.7); a follow-up turn reuses the prompt cache like plain decoding does: TTFT 12.5 s → 0.75 s on a 1,149-token turn (Bonsai 2 27B) |
| Ternary Bonsai 2 | (automatic) | loads Hadamard-rotated ternary checkpoints (`prism_hadamard_qwen35`) | 27B model in 7.4 GB |
| Embeddings and rerank | `serve --embedding-model`, `--rerank-model` | OpenAI `/v1/embeddings`, Cohere/Jina `/v1/rerank` next to the chat model | bge-small 229–299 texts/s, Qwen3-Embedding-0.6B 20 texts/s (batch 32) |
| Prompt cache on disk | `serve --prompt-cache-disk-mb N` | a resumed session or restarted server restores the longest matching prefix instead of prefilling it | — |
| KV cache quantization and zones | `serve --kv-bits 4\|8\|auto\|k8v4`, `--kv-zones` | fits longer contexts in the same memory | 16k prompt peak 1.03 GB vs 3.11 GB (Qwen3-4B, 4-bit zones) |
| Concurrent requests | `serve --max-concurrent N` | each running request reserves its KV room; one that does not fit gets 503 `Retry-After` instead of an out-of-memory | — |
| DRY and XTC samplers (experimental) | `serve --experimental-samplers` | `dry_*` / `xtc_*` request fields | not an anti-loop tool: on a 2-bit model DRY hides loops behind corrupted facts (#177) |

## Benchmarks

Two harnesses, both pointed at a running server, both reporting numbers you can
reproduce on your own machine.

**The ladder to the moon** — `benchmarks/ladder/run.py`. Five rungs, each
strictly harder than the one below it, ending in a full Earth-to-Moon mission.
Nothing is graded on how the answer reads: the model's own code is executed
against inputs the prompt never showed it, so a plausible-looking wrong answer
still fails.

| Rung | Task | What it actually tests |
|---|---|---|
| 1 | Biquadratic, real roots | closed form plus the edge cases: none, repeated, zero |
| 2 | Biquadratic, complex roots | branch handling — all four roots, with multiplicity |
| 3 | Fourier coefficients | numeric integration, checked on sawtooth, square wave, cosine |
| 4 | Pump head | units (mm vs m, L/s vs m³/s) + Colebrook, which has no closed form |
| 5 | **Earth → Moon mission** | exhaust velocity from `sqrt(2ηQ)`, Tsiolkovsky per stage, four-body propagation, lunar braking, soft landing — every constant is in the prompt, nothing recallable from a textbook |

The top rung is scored on six invariants rather than one answer, so a model that
gets the energetics right but never propagates the trajectory still shows a
number. Its prompt lives in `benchmarks/ladder/lunar_prompt.md`; the reference
values in `lunar_task.md` are never shown to the model.

`--warmup` prepends three cheap smoke tests (algebra, Fibonacci, translation)
that any instruct model should clear. They are not part of the ladder score.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, official pass@2 rules.
The corpus is not vendored; fetch it into `benchmarks/arc-agi/dataset/` first
(see `benchmarks/arc-agi/ARC_AGI.md`).

### Results

Measured on an M-series Mac with 24 GB unified memory. `thinking` is the chat
template's `<think>` block: on by default for reasoning models, disabled with
`--reasoning-effort none`.

| Model | Warm-up | Ladder | Top rung | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking off) | 3/3 | 1/4 ² | not run | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking on) | 2/3 | 0/4 ² | not run | not run | ~43 |

Ornith 9B rung by rung, thinking off:

| Rung | Result | Detail |
|---|---|---|
| 1 · biquadratic real | ✅ | roots correct, including the repeated and empty cases |
| 2 · biquadratic complex | ❌ | `x⁴ + 1` returns 2 roots instead of 4 |
| 3 · Fourier | ❌ | `a0 = 10.86` for `f(x) = x`, want 0 — the integration is wrong, not the formula |
| 4 · pump head | ❌ | `v = 0.0298` m/s, want 1.79 — never converted mm to m |
| 5 · lunar mission | — | not run |

Warm-up, thinking off: algebra ✅ (163 tok, 4 s), Fibonacci ✅ (757 tok, 16 s),
translation ✅ (303 tok, 7 s — German idiomatic, French paraphrased rather than
translated).

² Rungs 1 and 2 were measured before the ladder was split into separate
prompts: one prompt asked for both functions, the real half was correct and the
complex half was not. Rung 5 has not been run against this model yet.

¹ The ARC-AGI column is a partial run: three training tasks (`007bbfb7`,
`00d62c1b`, `017c7c7b`), none solved at pass@2, 330 s total. It was stopped
before the remaining two, so treat it as a smoke test rather than a score.

The validators are themselves checked, without a model or a GPU:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

With thinking **on**, all three hard tasks and the translation ended in
`finish_reason: length` — the budget went into the `<think>` block and no answer
was ever emitted. That is a budget failure, not a capability ceiling, which is
why the table above reports the two modes separately.

### Reproduce

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Adding a model to the table is one run of each — PRs with new rows are welcome.
