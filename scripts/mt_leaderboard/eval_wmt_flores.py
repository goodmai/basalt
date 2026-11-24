#!/usr/bin/env python3
"""Official WMT24++ and FLORES+ evaluation & HF model-index bundle builder (#734).

Supports deterministic greedy decoding (temperature=0) against an OpenAI-compatible
`basalt serve` endpoint (or offline `--dry-run` / `--predictions` re-scoring),
scoring with `sacrebleu` (exact signature), `Unbabel/wmt22-comet-da` (and optional
`Unbabel/XCOMET-XL`), and `google/metricx-24-hybrid-large-v2p6` / `xxl-v2p6`.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

DEFAULT_WMT24_PAIRS = [
    "en-ja_JP",
    "en-ko_KR",
    "en-id_ID",
    "en-ru_RU",
    "en-zh_CN",
    "en-de_DE",
    "en-es_MX",
    "en-fr_FR",
    "en-ar_EG",
]

FLORES_LANG_MAP = {
    "en": "eng_Latn",
    "ja_JP": "jpn_Jpan",
    "ko_KR": "kor_Hang",
    "id_ID": "ind_Latn",
    "ru_RU": "rus_Cyrl",
    "zh_CN": "zho_Hans",
    "de_DE": "deu_Latn",
    "es_MX": "spa_Latn",
    "fr_FR": "fra_Latn",
    "ar_EG": "arb_Arab",
}


def yaml_scalar(value: str) -> str:
    cleaned = value.replace("\r", " ").replace("\n", " ").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{cleaned}"'


def sanitize_wmt_line(text: str) -> str:
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


def git_head_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def load_segments(dataset: str, pairs: list[str], split: str, limit: int | None, jsonl_path: str | None) -> list[dict[str, Any]]:
    if jsonl_path:
        rows = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows[:limit] if limit else rows

    from datasets import load_dataset  # type: ignore

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        if dataset == "wmt24pp":
            ds = load_dataset("google/wmt24pp", pair, split=split)
            for idx, item in enumerate(ds):
                if limit and idx >= limit:
                    break
                if item.get("is_bad_source"):
                    continue
                rows.append({
                    "id": f"wmt24pp-{pair}-{idx}",
                    "lang_pair": pair,
                    "dataset": "google/wmt24pp",
                    "source": item["source"],
                    "reference": item["target"],
                })
        elif dataset == "flores_plus":
            src_code, tgt_code = pair.split("-", 1)
            src_ds = load_dataset("openlanguagedata/flores_plus", FLORES_LANG_MAP[src_code], split=split)
            tgt_ds = load_dataset("openlanguagedata/flores_plus", FLORES_LANG_MAP[tgt_code], split=split)
            for idx, (s_item, t_item) in enumerate(zip(src_ds, tgt_ds)):
                if limit and idx >= limit:
                    break
                rows.append({
                    "id": f"flores-{pair}-{idx}",
                    "lang_pair": pair,
                    "dataset": "openlanguagedata/flores_plus",
                    "source": s_item["text"],
                    "reference": t_item["text"],
                })
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    return rows


def translate_openai_greedy(endpoint: str, model: str, source: str, pair: str, max_tokens: int) -> str:
    target = pair.split("-", 1)[-1]
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": f"Translate the user text into {target}. Output only the translation."},
            {"role": "user", "content": source},
        ],
    }
    url = endpoint.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def score_sacrebleu_chrf(hyps: list[str], refs: list[str]) -> tuple[float, str, list[float]]:
    try:
        import sacrebleu  # type: ignore

        metric = sacrebleu.metrics.CHRF(word_order=2)
        corpus = metric.corpus_score(hyps, [refs])
        seg_scores = [metric.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
        return float(corpus.score), str(metric.get_signature()), seg_scores
    except ImportError:
        seg_scores = [100.0 if h.strip() == r.strip() else 0.0 for h, r in zip(hyps, refs)]
        mean = sum(seg_scores) / max(1, len(seg_scores))
        return mean, "fallback:exact-match(sacrebleu-not-installed)", seg_scores


def score_comet(rows: list[dict[str, Any]], model_name: str, batch_size: int) -> tuple[float, list[float]]:
    from comet import download_model, load_from_checkpoint  # type: ignore

    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows]
    out = model.predict(data, batch_size=batch_size, gpus=1 if _has_cuda() else 0)
    return float(out.system_score), [float(x) for x in out.scores]


def score_metricx24(rows: list[dict[str, Any]], model_name: str, qe: bool) -> tuple[float, list[float]]:
    import torch  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    scores: list[float] = []
    with torch.no_grad():
        for r in rows:
            text = (
                f"source: {r['source']} candidate: {r['hypothesis']}"
                if qe
                else f"source: {r['source']} candidate: {r['hypothesis']} reference: {r['reference']}"
            )
            encoded = tokenizer(text, max_length=1536, truncation=True, return_tensors="pt").to(device)
            decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
            logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], decoder_input_ids=decoder_input_ids).logits
            val = float(torch.clamp(logits[0, 0, 250089], 0.0, 25.0).item())
            scores.append(val)
    return sum(scores) / max(1, len(scores)), scores


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def render_hf_model_card(summary: dict[str, Any]) -> str:
    dataset_label = "WMT24++" if "wmt24" in summary["dataset"] else ("FLORES+" if "flores" in summary["dataset"] else summary["dataset"])
    langs = sorted({"en"} | {p.split("-", 1)[-1].split("_")[0].lower() for p in summary["pairs"]})
    lines = [
        "---",
        f"license: {yaml_scalar(summary.get('license', 'gemma'))}",
    ]
    if summary.get("base_model"):
        lines.append(f"base_model: {yaml_scalar(summary['base_model'])}")
    lines += [
        "pipeline_tag: translation",
        "language:",
    ]
    for lang in langs:
        lines.append(f"  - {yaml_scalar(lang)}")
    lines += [
        "tags:",
        "  - translation",
        "  - basalt",
        "  - mlx",
        "model-index:",
        f"  - name: {yaml_scalar(summary['model_name'])}",
        "    results:",
    ]
    for pair, metrics in sorted(summary["pairs"].items()):
        lines += [
            "      - task:",
            "          type: translation",
            "          name: Machine Translation",
            "        dataset:",
            f"          name: {yaml_scalar(dataset_label)}",
            f"          type: {yaml_scalar(summary['dataset'])}",
            f"          config: {yaml_scalar(pair)}",
            f"          split: {yaml_scalar(summary['split'])}",
            "        metrics:",
        ]
        for key, label, mtype in [
            ("chrf_pp", "chrF++", "chrf"),
            ("comet", "COMET", "comet"),
            ("xcomet", "XCOMET-XL", "xcomet"),
            ("metricx24", "MetricX-24", "metricx24"),
        ]:
            val = metrics.get(key)
            if val is not None and math.isfinite(val):
                lines += [
                    f"          - name: {yaml_scalar(label)}",
                    f"            type: {yaml_scalar(mtype)}",
                    f"            value: {val:.4f}",
                    "            verified: false",
                ]
    lines += [
        "---",
        "",
        f"# {summary['model_name']}",
        "",
        "## Evaluation Provenance",
        f"- Dataset: `{summary['dataset']}` (`split={summary['split']}`)",
        "- Decoding: greedy (`temperature=0.0`)",
        f"- sacrebleu signature: `{summary.get('sacrebleu_signature', '')}`",
        f"- Basalt commit: `{summary.get('git_commit', 'unknown')}`",
    ]
    if summary.get("predictions_repo"):
        lines.append(f"- Raw predictions dataset: `{summary['predictions_repo']}`")
    lines += [
        "",
        "## Per-Pair Results",
        "| Pair | Segments | chrF++ | COMET | MetricX-24 |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, m in sorted(summary["pairs"].items()):
        chrf = f"{m['chrf_pp']:.2f}" if m.get("chrf_pp") is not None else "—"
        comet = f"{m['comet']:.4f}" if m.get("comet") is not None else "—"
        mx = f"{m['metricx24']:.2f}" if m.get("metricx24") is not None else "—"
        lines.append(f"| `{pair}` | {m['segments']} | {chrf} | {comet} | {mx} |")
    return "\n".join(lines) + "\n"


def validate_model_card(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        return ["missing YAML frontmatter delimiters"]
    end = lines.index("---", 1)
    fm = lines[1:end]
    errors: list[str] = []
    for token in ["model-index:", "  - name:", "    results:", "          type: translation", "        dataset:", "        metrics:"]:
        if not any(line.startswith(token) for line in fm):
            errors.append(f"missing required field {token.strip()}")
    for line in fm:
        stripped = line.strip()
        if stripped.startswith("value:"):
            raw = stripped.split(":", 1)[1].strip()
            try:
                if not math.isfinite(float(raw)):
                    errors.append(f"non-finite value {raw}")
            except ValueError:
                errors.append(f"invalid numeric value {raw}")
    return errors


def write_bundle(output_dir: pathlib.Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wmt_dir = output_dir / "wmt"
    wmt_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["lang_pair"]].append(row)
    for pair, items in sorted(by_pair.items()):
        clean_lines = [sanitize_wmt_line(it["hypothesis"]) for it in items]
        (wmt_dir / f"{pair}.txt").write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

    (output_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card = render_hf_model_card(summary)
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    return validate_model_card(card)


def build_dry_run_rows(pairs: list[str], dataset_type: str) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        rows.append({
            "id": f"dryrun-{pair}-0",
            "lang_pair": pair,
            "dataset": dataset_type,
            "source": f"Run `basalt serve` to benchmark {pair}.",
            "reference": f"Reference translation for {pair} with `basalt serve`.",
            "hypothesis": f"Reference translation for {pair} with `basalt serve`.",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official WMT24++ / FLORES+ MT leaderboard evaluator (#734)")
    parser.add_argument("--model", required=True, help="Model name or HF repo id (e.g. Bosalt/TranslateGemma-4-12B-Pilot)")
    parser.add_argument("--base-model", default=None, help="Optional base model id for model-card provenance")
    parser.add_argument("--dataset", choices=["wmt24pp", "flores_plus", "jsonl"], default="wmt24pp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pairs", default=",".join(DEFAULT_WMT24_PAIRS))
    parser.add_argument("--jsonl", default=None, help="Input JSONL segments for --dataset jsonl")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scorers", default="sacrebleu", help="Comma-separated: sacrebleu,comet,xcomet,metricx24,metricx24_qe")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--xcomet-model", default="Unbabel/XCOMET-XL")
    parser.add_argument("--metricx-model", default="google/metricx-24-hybrid-large-v2p6")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--predictions-repo", default="goodmai/translategemma4-eval-results")
    parser.add_argument("--output", required=True, help="Output bundle directory")
    parser.add_argument("--dry-run", action="store_true", help="Run offline schema/bundle dry run without network or GPU")
    args = parser.parse_args(argv)

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    dataset_type = {"wmt24pp": "google/wmt24pp", "flores_plus": "openlanguagedata/flores_plus", "jsonl": "custom/jsonl"}[args.dataset]

    if args.dry_run:
        rows = build_dry_run_rows(pairs, dataset_type)
    else:
        rows = load_segments(args.dataset, pairs, args.split, args.limit, args.jsonl)
        for row in rows:
            if "hypothesis" not in row:
                row["hypothesis"] = translate_openai_greedy(args.endpoint, args.model, row["source"], row["lang_pair"], args.max_tokens)

    scorers = {s.strip() for s in args.scorers.split(",") if s.strip()}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["lang_pair"]].append(row)

    pair_summaries: dict[str, dict[str, Any]] = {}
    sacrebleu_sig = ""
    for pair, items in sorted(by_pair.items()):
        entry: dict[str, Any] = {"segments": len(items)}
        if "sacrebleu" in scorers:
            chrf, sig, segs = score_sacrebleu_chrf([i["hypothesis"] for i in items], [i["reference"] for i in items])
            sacrebleu_sig = sig
            entry["chrf_pp"] = chrf
            for item, val in zip(items, segs):
                item["chrf_pp"] = val
        if not args.dry_run and "comet" in scorers:
            sys_score, segs = score_comet(items, args.comet_model, args.batch_size)
            entry["comet"] = sys_score
            for item, val in zip(items, segs):
                item["comet"] = val
        if not args.dry_run and "xcomet" in scorers:
            sys_score, segs = score_comet(items, args.xcomet_model, args.batch_size)
            entry["xcomet"] = sys_score
            for item, val in zip(items, segs):
                item["xcomet"] = val
        if not args.dry_run and ("metricx24" in scorers or "metricx24_qe" in scorers):
            qe = "metricx24_qe" in scorers and "metricx24" not in scorers
            sys_score, segs = score_metricx24(items, args.metricx_model, qe=qe)
            entry["metricx24"] = sys_score
            for item, val in zip(items, segs):
                item["metricx24"] = val
        pair_summaries[pair] = entry

    summary = {
        "model_name": args.model,
        "base_model": args.base_model,
        "dataset": dataset_type,
        "split": args.split,
        "temperature": 0.0,
        "sacrebleu_signature": sacrebleu_sig,
        "predictions_repo": args.predictions_repo,
        "git_commit": git_head_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": pair_summaries,
    }
    errors = write_bundle(pathlib.Path(args.output), rows, summary)
    if errors:
        print("Validation errors:", errors, file=sys.stderr)
        return 1
    print(f"Wrote MT leaderboard bundle to {args.output} ({len(rows)} segments across {len(pair_summaries)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
