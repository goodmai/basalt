#!/usr/bin/env python3
"""
ARC-AGI Official Rules Benchmark Script for Colibri / MLX.

Official ARC-AGI Rules Implemented:
1. Benchmark set: ARC-AGI-1 training (400 tasks) & evaluation (400 tasks).
2. Evaluation Metric: Pass@1 and Pass@2 accuracy (max 2 attempts per task).
3. Grid Parsing: Strict 2D JSON array extraction with regex fallback.
4. Reasoning Channel Support: Captures Gemma 4 <|channel|>thought tokens and final grid output.
5. Resume & Checkpointing: Saves state after every task to allow seamless resuming.
"""

import json
import urllib.request
import re
import time
import sys
import os
import argparse
import glob
from pathlib import Path

from arc_sandbox import DEFAULT_TIMEOUT, run_blocks, sandbox_available

# ─── Default Configurations ──────────────────────────────────────────────────

# `basalt serve` ignores the request's model field once a model is loaded, so the
# engine only needs to name the port the server is listening on.
DEFAULT_ENGINES = {
    "swift_mlx": {
        "url": "http://127.0.0.1:8081/v1/chat/completions",
        "model": "basalt",
        "type": "openai",
        "label": "Native Swift MLX basalt (port 8081)",
    },
    "mlx": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "model": "basalt",
        "type": "openai",
        "label": "Native Swift MLX basalt (port 8080)",
    },
    "ollama": {
        "url": "http://127.0.0.1:11434/api/chat",
        "model": "basalt",
        "type": "ollama",
        "label": "Ollama Server",
    },
}

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
CHUNK_TIMEOUT = 1800   # per-read timeout (seconds, 30 min)
ATTEMPT_TIMEOUT = 1800 # max wall-clock time per attempt (30 min)
REQUEST_OPTIONS = {}
DEFAULT_MAX_TOKENS = 16000

# ─── Grid Extractor ──────────────────────────────────────────────────────────

def parse_grid(text: str):
    """
    Extracts 2D JSON grid from model output text or reasoning output.
    """
    if not text:
        return None
    
    # Strip markdown code blocks
    cleaned = re.sub(r"```[a-zA-Z]*", "", text).strip()
    
    # Find all 2D array patterns [[...],[...]]
    matches = list(re.finditer(r"\[\s*\[[\s\S]*?\]\s*\]", cleaned))
    for m in reversed(matches):
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(row, list) for row in parsed):
                    return parsed
        except Exception:
            continue
    return None

def execute_python_hybrid(text: str, test_input: list[list[int]], timeout: float = DEFAULT_TIMEOUT,
                          train_pairs: list[dict] = ()):
    if not text:
        return None, "empty_text", 0
    code_blocks = re.findall(r"```python\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```[a-zA-Z]*\s*\n(.*def\s+\w+.*)\n\s*```", text, re.DOTALL)
    if code_blocks:
        return run_blocks(list(reversed(code_blocks)), list(train_pairs), test_input, timeout=timeout)
    direct_grid = parse_grid(text)
    if direct_grid:
        return direct_grid, "json_parse_fallback", 0
    return None, "failed", 0

# ─── Prompt Builder ──────────────────────────────────────────────────────────

def build_arc_prompt(task_data: dict, attempt: int = 1, mode: str = "hybrid") -> str:
    """
    Builds ARC-AGI prompt for standard Direct mode or Hybrid (Program-of-Thought) mode.
    """
    if mode == "direct":
        prompt = "ARC-AGI puzzle. Learn the transformation rule from input/output pairs, then apply it to the test input.\n\n"
        for i, example in enumerate(task_data.get("train", []), 1):
            prompt += f"Example {i}:\nIN = {json.dumps(example['input'])}\nOUT = {json.dumps(example['output'])}\n\n"
        prompt += test_input_section(task_data)
        if attempt == 1:
            prompt += "Output ONLY the 2D JSON array for the test output (e.g. [[1,2],[3,4]]). Do not include extra text."
        else:
            prompt += "Output ONLY the final 2D JSON array starting with '[[' and ending with ']]'. Make sure dimensions and colors are exact."
        return prompt

    # Hybrid Mode (Typed OOP DSL Program-of-Thought)
    prompt = ("ARC-AGI puzzle. Each example maps an input grid IN to an output grid OUT by one rule.\n\n"
              "Optional helper DSL, available to your code but not required:\n"
              "- Color (IntEnum: BLACK=0, BLUE=1, RED=2, GREEN=3, YELLOW=4, GREY=5, MAGENTA=6, ORANGE=7, CYAN=8, MAROON=9)\n"
              "- Point(r: int, c: int)\n"
              "- GridObject: .pixels, .bounding_box (min_r, min_c, max_r, max_c), .move(dr, dc), .change_color(new_color)\n"
              "- Grid: .extract_objects(connectivity=4), .filter_by_color(color), .height, .width, .to_matrix()\n\n")
              
    for i, example in enumerate(task_data.get("train", []), 1):
        prompt += f"Example {i}:\nIN = {json.dumps(example['input'])}\nOUT = {json.dumps(example['output'])}\n\n"
        
    prompt += test_input_section(task_data)
    prompt += ("INSTRUCTIONS:\n"
               "1. Identify the rule that maps every example IN to its OUT.\n"
               "2. Write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` that applies it; "
               "plain lists, the DSL above and numpy as `np` all work.\n"
               "3. Wrap the code in one ```python ... ``` block. It is run on every example IN and must "
               "reproduce its OUT before it is applied to TEST_IN.\n")
    return prompt


def test_input_section(task_data: dict) -> str:
    return f"Test input (apply the rule to this):\nTEST_IN = {json.dumps(task_data['test'][0]['input'])}\n\n"

# ─── API Stream Query ────────────────────────────────────────────────────────

def query_engine(engine_cfg: dict, prompt: str, max_tokens: int = 8192, temperature: float | None = None):
    """
    Sends request to LLM server and streams thinking + answer tokens.
    """
    if engine_cfg["type"] == "openai":
        return _stream_openai(engine_cfg["url"], engine_cfg["model"], prompt, max_tokens, temperature)
    elif engine_cfg["type"] == "ollama":
        return _stream_ollama(engine_cfg["url"], engine_cfg["model"], prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown engine type: {engine_cfg['type']}")

def openai_payload(model: str, prompt: str, max_tokens: int, temperature: float | None, options: dict) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    payload.update({key: value for key, value in options.items() if value is not None})
    return payload


def _stream_openai(url: str, model: str, prompt: str, max_tokens: int, temperature: float | None):
    payload = openai_payload(model, prompt, max_tokens, temperature, REQUEST_OPTIONS)
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    reasoning = ""
    content = ""
    meta = {}
    deadline = time.time() + ATTEMPT_TIMEOUT
    
    with urllib.request.urlopen(req, timeout=CHUNK_TIMEOUT) as resp:
        for raw in resp:
            if time.time() > deadline:
                print("    [ATTEMPT TIMEOUT REACHED]")
                break
            line = raw.decode('utf-8').strip()
            if not line or line == "data: [DONE]":
                if line == "data: [DONE]":
                    break
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                chunk = json.loads(line)
                meta.update(stream_meta(chunk))
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                r = delta.get("reasoning", "") or ""
                c = delta.get("content", "") or ""
                reasoning += r
                content += c
            except Exception:
                continue
                
    return reasoning, content, meta


def stream_meta(chunk: dict) -> dict:
    meta = {}
    finish = (chunk.get("choices") or [{}])[0].get("finish_reason")
    if finish:
        meta["finish_reason"] = finish
    usage = chunk.get("usage") or {}
    if "completion_tokens" in usage:
        meta["completion_tokens"] = usage["completion_tokens"]
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning_tokens is not None:
        meta["reasoning_tokens"] = reasoning_tokens
    return meta

def _stream_ollama(url: str, model: str, prompt: str, max_tokens: int, temperature: float | None):
    options = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": options
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    reasoning = ""
    content = ""
    deadline = time.time() + ATTEMPT_TIMEOUT
    
    with urllib.request.urlopen(req, timeout=CHUNK_TIMEOUT) as resp:
        for raw in resp:
            if time.time() > deadline:
                print("    [ATTEMPT TIMEOUT REACHED]")
                break
            line = raw.decode('utf-8').strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                reasoning += msg.get("thinking", "") or ""
                content += msg.get("content", "") or ""
            except Exception:
                continue
                
    return reasoning, content, {}

# ─── Task Execution ──────────────────────────────────────────────────────────

def evaluate_task(task_path: Path, engine_cfg: dict, max_attempts: int = 2, max_tokens: int = 16000, temperature: float | None = None, mode: str = "hybrid", exec_timeout: float = DEFAULT_TIMEOUT):
    task_id = task_path.stem
    with open(task_path, 'r', encoding='utf-8') as f:
        task_data = json.load(f)
        
    expected_grid = task_data["test"][0]["output"]
    test_in_grid = task_data["test"][0]["input"]
    in_shape = f"{len(test_in_grid)}x{len(test_in_grid[0])}"
    out_shape = f"{len(expected_grid)}x{len(expected_grid[0])}"
    
    print(f"\n▶ Task: {task_id} (Input: {in_shape} -> Target Output: {out_shape}) [Mode: {mode.upper()}]")
    
    task_report = {
        "task_id": task_id,
        "input_shape": in_shape,
        "output_shape": out_shape,
        "mode": mode,
        "passed": False,
        "passed_attempt": None,
        "attempts": []
    }
    
    for attempt in range(1, max_attempts + 1):
        print(f"  Attempt {attempt}/{max_attempts}...", end="", flush=True)
        prompt = build_arc_prompt(task_data, attempt=attempt, mode=mode)
        
        t0 = time.time()
        try:
            reasoning, content, meta = query_engine(engine_cfg, prompt, max_tokens=max_tokens, temperature=temperature)
            elapsed = time.time() - t0
            
            full_text = (reasoning + "\n" + content).strip()
            
            train_pairs_passed = None
            if mode == "hybrid":
                predicted_grid, exec_status, train_pairs_passed = execute_python_hybrid(
                    full_text, test_in_grid, timeout=exec_timeout, train_pairs=task_data.get("train", []))
            else:
                text_to_parse = content.strip() if content.strip() else reasoning.strip()
                predicted_grid = parse_grid(text_to_parse)
                exec_status = "direct_json"
            
            is_correct = (predicted_grid == expected_grid)
            
            attempt_info = {
                "attempt": attempt,
                "elapsed_sec": round(elapsed, 2),
                "reasoning_len": len(reasoning),
                "content_len": len(content),
                "exec_status": exec_status,
                "train_pairs_passed": train_pairs_passed,
                "train_pairs": len(task_data.get("train", [])),
                "answered": bool(content.strip()),
                **meta,
                "parsed": (predicted_grid is not None),
                "correct": is_correct,
                "prediction": predicted_grid
            }
            task_report["attempts"].append(attempt_info)
            
            stats = (f"{elapsed:.1f}s, reasoning: {len(reasoning)} chars / {meta.get('reasoning_tokens')} tokens, "
                     f"train {train_pairs_passed}/{len(task_data.get('train', []))}, finish {meta.get('finish_reason')}")
            if is_correct:
                print(f" ✅ CORRECT [{exec_status}] ({stats})")
                task_report["passed"] = True
                task_report["passed_attempt"] = attempt
                break
            else:
                if predicted_grid is None:
                    print(f" ❌ PARSE/EXEC FAILED [{exec_status}] ({stats})")
                else:
                    p_rows = len(predicted_grid)
                    p_cols = len(predicted_grid[0]) if p_rows > 0 else 0
                    print(f" ❌ INCORRECT GRID {p_rows}x{p_cols} [{exec_status}] ({stats})")
                    
        except Exception as e:
            elapsed = time.time() - t0
            print(f" ❌ ERROR: {e} ({elapsed:.1f}s)")
            task_report["attempts"].append({
                "attempt": attempt,
                "elapsed_sec": round(elapsed, 2),
                "error": str(e),
                "correct": False
            })
            
    return task_report

# ─── Main Benchmarking Loop ──────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ARC-AGI Official Rules Benchmark Runner")
    parser.add_argument("--mode", choices=["hybrid", "direct"], default="hybrid" if sandbox_available() else "direct",
                        help="Benchmark mode: hybrid (Program-of-Thought, generated Python runs in a sandboxed child process) "
                             "or direct (2D JSON grid). Default: hybrid where sandbox-exec exists (macOS), else direct")
    parser.add_argument("--exec-timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"Hybrid mode: wall-clock deadline in seconds for one code block on one grid; the sandbox run is "
                             f"capped at it x grids x min(blocks, 8) (default: {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--engine-url", type=str, default=None,
                        help="Override the engine's URL, e.g. http://127.0.0.1:8098/v1/chat/completions")
    parser.add_argument("--split", choices=["training", "evaluation", "all"], default="training",
                        help="ARC-AGI split to run (default: training)")
    parser.add_argument("--engine", choices=list(DEFAULT_ENGINES.keys()), default="swift_mlx",
                        help="Backend engine to benchmark (default: swift_mlx)")
    parser.add_argument("--attempts", type=int, default=2,
                        help="Max attempts per task according to official ARC rules (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to evaluate")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index for task chunking")
    parser.add_argument("--tasks", nargs="+", type=str, default=None,
                        help="Specific task IDs or file paths to evaluate")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max generation tokens budget (default: 16000; with --thinking-budget: "
                             "thinking budget + --answer-tokens unless this is larger)")
    parser.add_argument("--answer-tokens", type=int, default=4000,
                        help="With --thinking-budget: tokens left for the answer after it (default: 4000)")
    parser.add_argument("--reasoning-effort", default=None,
                        help="OpenAI reasoning_effort sent to the engine, e.g. none to switch thinking off")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="thinking_budget sent to the engine: reasoning tokens before the answer is forced")
    parser.add_argument("--top-k", type=int, default=None,
                        help="top_k sent to the engine (default: the model's generation_config)")
    parser.add_argument("--presence-penalty", type=float, default=None,
                        help="presence_penalty sent to the engine")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: the model's generation_config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume benchmark from output JSON file if it exists")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path for results")

    args = parser.parse_args(argv)
    args.max_tokens = effective_max_tokens(args.max_tokens, args.thinking_budget, args.answer_tokens)
    return args


def effective_max_tokens(max_tokens: int | None, thinking_budget: int | None, answer_tokens: int) -> int:
    if thinking_budget is None:
        return max_tokens or DEFAULT_MAX_TOKENS
    return max(max_tokens or 0, thinking_budget + answer_tokens)


def main():
    args = parse_args()
    REQUEST_OPTIONS.update(request_options(args))

    # Determine tasks list
    task_files = []
    if args.tasks:
        for t in args.tasks:
            p = Path(t)
            if p.exists():
                task_files.append(p)
            else:
                # search in dataset root
                found = list(DATASET_ROOT.rglob(f"*{t}*.json"))
                if found:
                    task_files.extend(found)
                else:
                    print(f"Warning: task '{t}' not found.")
    else:
        if args.split in ["training", "all"]:
            task_files.extend(sorted((DATASET_ROOT / "training").glob("*.json")))
        if args.split in ["evaluation", "all"]:
            task_files.extend(sorted((DATASET_ROOT / "evaluation").glob("*.json")))

    if not task_files:
        print(f"Error: No tasks found in dataset path {DATASET_ROOT}")
        sys.exit(1)

    # Slice tasks
    start_idx = args.start_index
    end_idx = start_idx + args.limit if args.limit else len(task_files)
    task_files = task_files[start_idx:end_idx]

    output_file = args.output or f"arc_benchmark_results_{args.engine}_{args.split}_{args.mode}.json"
    report_file = output_file.replace(".json", ".md")

    # Load existing results if resuming
    completed_results = {}
    if args.resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                completed_results = saved_data.get("tasks", {})
                print(f"Resuming from {output_file}: {len(completed_results)} tasks already completed.")
        except Exception as e:
            print(f"Warning: Failed to load existing results from {output_file}: {e}")

    engine_cfg = dict(DEFAULT_ENGINES[args.engine])
    if args.engine_url:
        engine_cfg["url"] = args.engine_url
        engine_cfg["label"] = f"{engine_cfg['label'].split(' (')[0]} ({args.engine_url})"
    if args.mode == "hybrid" and not sandbox_available():
        print("WARNING: hybrid mode without sandbox-exec: generated code runs with rlimits only and can "
              "read your files and reach the network. Use --mode direct on untrusted models.", file=sys.stderr)

    print("=" * 70)
    print("ARC-AGI Official Evaluation Benchmark")
    print(f"Engine      : {engine_cfg['label']}")
    print(f"Split       : {args.split} ({len(task_files)} tasks selected)")
    print(f"Max Attempts: {args.attempts} (Pass@1 & Pass@2)")
    print(f"Max Tokens  : {args.max_tokens}")
    print(f"Temperature : {args.temperature if args.temperature is not None else 'model default'}")
    print(f"Options     : {json.dumps({k: v for k, v in REQUEST_OPTIONS.items() if v is not None})}")
    print(f"Output File : {output_file}")
    print("=" * 70)

    results = completed_results
    start_time = time.time()

    for idx, task_path in enumerate(task_files, 1):
        task_id = task_path.stem
        if task_id in results:
            print(f"Skipping [{idx}/{len(task_files)}] {task_id} (already completed)")
            continue

        print(f"\nProgress: [{idx}/{len(task_files)}]")
        res = evaluate_task(
            task_path,
            engine_cfg,
            max_attempts=args.attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            mode=args.mode,
            exec_timeout=args.exec_timeout
        )
        results[task_id] = res

        # Save progress after every task
        _save_checkpoint(output_file, report_file, engine_cfg, results, start_time)

    # Print Final Summary Table
    _print_final_summary(results, engine_cfg, time.time() - start_time)

def request_options(args) -> dict:
    return {
        "reasoning_effort": args.reasoning_effort,
        "thinking_budget": args.thinking_budget,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
    }


def _save_checkpoint(output_file: str, report_file: str, engine_cfg: dict, results: dict, start_time: float):
    total = len(results)
    pass_at_1 = sum(1 for r in results.values() if r.get("passed_attempt") == 1)
    pass_at_2 = sum(1 for r in results.values() if r.get("passed"))
    
    acc_pass1 = (pass_at_1 / total * 100) if total > 0 else 0.0
    acc_pass2 = (pass_at_2 / total * 100) if total > 0 else 0.0
    
    summary = {
        "engine": engine_cfg["label"],
        "total_tasks_evaluated": total,
        "pass_at_1": pass_at_1,
        "pass_at_1_pct": round(acc_pass1, 2),
        "pass_at_2": pass_at_2,
        "pass_at_2_pct": round(acc_pass2, 2),
        "elapsed_seconds": round(time.time() - start_time, 1),
        "tasks": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    # Generate Markdown Summary
    md_content = f"# ARC-AGI Benchmark Report\n\n"
    md_content += f"- **Engine**: {engine_cfg['label']}\n"
    md_content += f"- **Total Tasks**: {total}\n"
    md_content += f"- **Pass@1 Accuracy**: {pass_at_1}/{total} ({acc_pass1:.2f}%)\n"
    md_content += f"- **Pass@2 Accuracy (Official)**: {pass_at_2}/{total} ({acc_pass2:.2f}%)\n"
    md_content += f"- **Elapsed Time**: {summary['elapsed_seconds']} sec\n\n"
    md_content += "## Task Breakdown\n\n"
    md_content += "| Task ID | Input Shape | Target Shape | Pass@1 | Pass@2 | Time (s) |\n"
    md_content += "|---|---|---|---|---|---|\n"
    
    for task_id, tdata in results.items():
        p1 = "✅" if tdata.get("passed_attempt") == 1 else "❌"
        p2 = "✅" if tdata.get("passed") else "❌"
        tot_time = sum(a.get("elapsed_sec", 0) for a in tdata.get("attempts", []))
        md_content += f"| `{task_id}` | {tdata.get('input_shape')} | {tdata.get('output_shape')} | {p1} | {p2} | {tot_time:.1f} |\n"
        
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

def _print_final_summary(results: dict, engine_cfg: dict, total_elapsed: float):
    total = len(results)
    if total == 0:
        print("No tasks were evaluated.")
        return

    pass_at_1 = sum(1 for r in results.values() if r.get("passed_attempt") == 1)
    pass_at_2 = sum(1 for r in results.values() if r.get("passed"))
    acc_p1 = pass_at_1 / total * 100
    acc_p2 = pass_at_2 / total * 100

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETED: {engine_cfg['label']}")
    print("-" * 70)
    print(f"Total Tasks Evaluated : {total}")
    print(f"Pass@1 Accuracy       : {pass_at_1}/{total} ({acc_p1:.2f}%)")
    print(f"Pass@2 Accuracy (ARC) : {pass_at_2}/{total} ({acc_p2:.2f}%)")
    print(f"Total Elapsed Time    : {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()
