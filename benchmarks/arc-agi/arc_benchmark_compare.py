#!/usr/bin/env python3
"""
ARC-AGI сравнительный бенчмарк: Ollama (GGUF Metal) vs MLX (safetensors 4bit).

Использование:
  python3 arc_benchmark_compare.py --engine ollama   # Ollama port 11434
  python3 arc_benchmark_compare.py --engine mlx      # MLX server port 8080
  python3 arc_benchmark_compare.py --tasks 1 2 3 4 5 # выбрать задачи
"""

import json, urllib.request, re, time, sys, os, argparse, glob
from pathlib import Path

# ─── Конфигурация движков ────────────────────────────────────────────────────
# `gemm serve` ignores the request's model field once a model is loaded, so the
# engine only needs to name the port the server is listening on.
ENGINES = {
    "ollama": {
        "url":   "http://127.0.0.1:11434/api/chat",
        "model": "gemm",
        "type":  "ollama",     # Ollama chat API format
        "label": "Ollama Metal (GGUF)",
    },
    "mlx": {
        "url":   "http://127.0.0.1:8080/v1/chat/completions",
        "model": "gemm",
        "type":  "openai",     # OpenAI-compatible (mlx_lm.server)
        "label": "Native Swift MLX gemm (port 8080)",
    },
    "swift_mlx": {
        "url":   "http://127.0.0.1:8081/v1/chat/completions",
        "model": "gemm",
        "type":  "openai",     # Custom Native Swift MLX server (gemm)
        "label": "Native Swift MLX gemm (port 8081)",
    },
}

CHUNK_TIMEOUT  = 90    # per-read timeout (streaming idle)
ATTEMPT_LIMIT  = 600   # 10 min wall-clock per attempt
MAX_ATTEMPTS   = 10
TASK_DIR       = str(Path(__file__).resolve().parent / "compare-tasks")

# ─── HTTP helpers ────────────────────────────────────────────────────────────

def _stream_ollama(url, model, msgs):
    """Ollama streaming: thinking + content fields."""
    data = json.dumps({
        "model": model, "messages": msgs,
        "stream": True, "options": {"temperature": 0.1}
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    thinking, content = "", ""
    deadline = time.time() + ATTEMPT_LIMIT
    with urllib.request.urlopen(req, timeout=CHUNK_TIMEOUT) as resp:
        for raw in resp:
            if time.time() > deadline:
                print("    [WALL-CLOCK TIMEOUT]", flush=True)
                break
            line = raw.decode().strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            msg = chunk.get("message", {})
            thinking += msg.get("thinking", "") or ""
            content  += msg.get("content",  "") or ""
            if chunk.get("done"):
                ec = chunk.get("eval_count", "?")
                print(f"    [done tokens={ec} think={len(thinking)} content={len(content)}]", flush=True)
                break
    return content if content.strip() else thinking

def _stream_openai(url, model, msgs):
    """OpenAI-compatible streaming (mlx_lm.server).
    Gemma 4 thinking model: reasoning tokens → delta.reasoning,
    actual answer → delta.content (appears after reasoning is complete).
    """
    data = json.dumps({
        "model": model, "messages": msgs,
        "stream": True, "temperature": 0.7,
        "max_tokens": 8192   # thinking budget (~4000-6000) + content (~400)
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    reasoning = ""
    content   = ""
    deadline  = time.time() + ATTEMPT_LIMIT
    with urllib.request.urlopen(req, timeout=CHUNK_TIMEOUT) as resp:
        for raw in resp:
            if time.time() > deadline:
                print("    [WALL-CLOCK TIMEOUT]", flush=True)
                break
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                if line == "data: [DONE]":
                    break
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            reasoning += delta.get("reasoning", "") or ""
            content   += delta.get("content",   "") or ""
    print(f"    [done reasoning={len(reasoning)} content={len(content)}]", flush=True)
    # Return content if present (actual answer), else fall back to reasoning
    return content if content.strip() else reasoning

def query(engine_cfg, msgs):
    if engine_cfg["type"] == "ollama":
        return _stream_ollama(engine_cfg["url"], engine_cfg["model"], msgs)
    else:
        return _stream_openai(engine_cfg["url"], engine_cfg["model"], msgs)

# ─── Grid parsing ────────────────────────────────────────────────────────────

def parse_grid(text):
    text = re.sub(r"```[a-z]*", "", text)
    for m in reversed(list(re.finditer(r"\[\s*\[[\s\S]*?\]\s*\]", text))):
        try:
            g = json.loads(m.group(0))
            if isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) for r in g):
                return g
        except Exception:
            continue
    return None

# ─── Prompt builder ──────────────────────────────────────────────────────────

def make_prompt(task, n_examples=2, suffix=""):
    train = task["train"][-n_examples:]
    test_input = task["test"][0]["input"]
    p = "ARC-AGI puzzle. Learn the transformation rule from examples, then apply it.\n\n"
    for i, ex in enumerate(train, 1):
        p += f"Example {i}: IN={json.dumps(ex['input'])} OUT={json.dumps(ex['output'])}\n\n"
    p += f"Test input: {json.dumps(test_input)}\n\nOutput ONLY the 2D JSON array. No explanation."
    return p + suffix

# ─── Single task benchmark ───────────────────────────────────────────────────

def run_task(task_file, engine_cfg, max_attempts=MAX_ATTEMPTS):
    with open(task_file) as f:
        task = json.load(f)
    expected = task["test"][0]["output"]
    test_in  = task["test"][0]["input"]
    name     = os.path.basename(task_file)
    rows, cols = len(test_in), len(test_in[0])

    print(f"\n  TASK: {name}  grid={rows}×{cols}", flush=True)
    times = []

    for attempt in range(1, max_attempts + 1):
        suffix = "\n\nOutput ONLY raw JSON [[...],[...]]." if attempt > 1 else ""
        msgs = [{"role": "user", "content": make_prompt(task, n_examples=2, suffix=suffix)}]
        print(f"    Attempt {attempt}/{max_attempts} ...", flush=True)
        t0 = time.time()
        try:
            response = query(engine_cfg, msgs)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            times.append(time.time() - t0)
            continue
        elapsed = time.time() - t0
        times.append(elapsed)

        predicted = parse_grid(response)
        print(f"    [{elapsed:.1f}s] parsed={predicted is not None}", flush=True)

        if predicted == expected:
            print(f"    ✅ PASSED (attempt {attempt})", flush=True)
            return {"passed": True, "attempts": attempt, "times": times}

        if predicted:
            if len(predicted) == len(expected):
                diffs = [i for i in range(len(expected)) if predicted[i] != expected[i]]
                print(f"    ✗ {len(diffs)} rows wrong. Row {diffs[0]}: got {predicted[diffs[0]]}", flush=True)
                print(f"    {'':5} expected:  {expected[diffs[0]]}", flush=True)
            else:
                print(f"    ✗ size {len(predicted)}×{len(predicted[0])}", flush=True)
        else:
            print(f"    ✗ parse failed", flush=True)

    print(f"    ❌ FAILED", flush=True)
    return {"passed": False, "attempts": max_attempts, "times": times}

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ollama", "mlx", "swift_mlx"], default="swift_mlx")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1,2,3,4,5])
    args = parser.parse_args()

    engine = ENGINES[args.engine]
    task_files = sorted([
        f"{TASK_DIR}/task_{i}.json" for i in args.tasks
        if os.path.exists(f"{TASK_DIR}/task_{i}.json")
    ])

    print(f"\n{'='*60}", flush=True)
    print(f"ARC-AGI BENCHMARK", flush=True)
    print(f"Engine : {engine['label']}", flush=True)
    print(f"Tasks  : {[os.path.basename(f) for f in task_files]}", flush=True)
    print(f"Timeout: {ATTEMPT_LIMIT}s/attempt, {MAX_ATTEMPTS} attempts max", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {}
    total_t0 = time.time()
    for path in task_files:
        name = os.path.basename(path)
        results[name] = run_task(path, engine)

    total = time.time() - total_t0
    passed = sum(1 for r in results.values() if r["passed"])

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS [{engine['label']}]", flush=True)
    print(f"{'Task':15} {'Result':12} {'Attempts':10} {'Avg time/att':14}", flush=True)
    print(f"{'-'*55}", flush=True)
    for name, r in results.items():
        st  = "✅ PASSED" if r["passed"] else "❌ FAILED"
        att = r["attempts"]
        avg = sum(r["times"]) / len(r["times"]) if r["times"] else 0
        print(f"{name:15} {st:12} {att}/10       {avg:.0f}s avg", flush=True)

    pct = passed / len(task_files) * 100 if task_files else 0
    print(f"\nScore  : {passed}/{len(task_files)} ({pct:.0f}%)", flush=True)
    print(f"Runtime: {total:.0f}s ({total/60:.1f} min)", flush=True)

    # Save result to JSON for comparison
    out_file = f"{TASK_DIR}/../arc_result_{args.engine}.json"
    with open(out_file, "w") as f:
        json.dump({
            "engine": engine["label"],
            "score": f"{passed}/{len(task_files)}",
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "times"} for k, v in results.items()}
        }, f, indent=2)
    print(f"\nSaved: {out_file}", flush=True)

if __name__ == "__main__":
    main()
