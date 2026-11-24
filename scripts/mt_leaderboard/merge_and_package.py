#!/usr/bin/env python3
"""Merge PEFT LoRA adapters into standalone BF16 safetensors + MLX siblings & HF publish (#734).

Preserves tokenizer files, chat_template.jinja, and writes basalt_merge_provenance.json
with base_model, adapter_path, training_config, and commit SHA. Supports --dry-run for
offline verification without GPU or Hugging Face writes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

PRESERVED_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
    "generation_config.json",
]


def git_head_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def load_optional_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_provenance(
    base_model: str,
    adapter_path: str,
    mlx_quant: str | None,
    hf_bf16_repo: str | None,
    hf_mlx_repo: str | None,
    hf_lora_repo: str | None,
    adapter_config: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "training_config": adapter_config or {},
        "mlx_quant": mlx_quant,
        "targets": {
            "lora_repo": hf_lora_repo,
            "merged_bf16_repo": hf_bf16_repo,
            "mlx_repo": hf_mlx_repo,
        },
        "git_commit": git_head_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def copy_preserved_files(source_dir: pathlib.Path, dest_dir: pathlib.Path) -> list[str]:
    copied: list[str] = []
    for name in PRESERVED_TOKENIZER_FILES:
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)
            copied.append(name)
    return copied


def merge_peft_to_bf16(base_model: str, adapter_path: str, out_bf16: pathlib.Path) -> None:
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_bf16.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(out_bf16, safe_serialization=True)
    tokenizer.save_pretrained(out_bf16)


def convert_mlx_sibling(bf16_dir: pathlib.Path, mlx_dir: pathlib.Path, quant: str, dry_run: bool) -> list[str]:
    cmd = ["basalt", "models", "convert", str(bf16_dir), "--output", str(mlx_dir), "--quant", quant]
    if not dry_run:
        subprocess.run(cmd, check=True)
    return cmd


def publish_hf_directory(local_dir: pathlib.Path, repo_id: str, commit_message: str, dry_run: bool) -> list[str]:
    cmd = ["basalt", "models", "publish", str(local_dir), "--repo", repo_id, "--message", commit_message]
    if not dry_run:
        subprocess.run(cmd, check=True)
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA -> BF16 + MLX packaging & HF upload (#734)")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="Local LoRA directory or HF repo id")
    parser.add_argument("--output-dir", required=True, help="Working output directory for merged BF16 & MLX siblings")
    parser.add_argument("--mlx-quant", default="4bit", help="Optional MLX target format (4bit, nvfp4, macaq, or none)")
    parser.add_argument("--hf-lora-repo", default=None)
    parser.add_argument("--hf-bf16-repo", default=None)
    parser.add_argument("--hf-mlx-repo", default=None)
    parser.add_argument("--eval-bundle-dir", default=None, help="Optional eval bundle dir containing README.md to attach")
    parser.add_argument("--dry-run", action="store_true", help="Emit packaging plan and provenance without merging or uploading")
    args = parser.parse_args(argv)

    out_root = pathlib.Path(args.output_dir)
    bf16_dir = out_root / "merged-bf16"
    mlx_dir = out_root / f"mlx-{args.mlx_quant}"
    bf16_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = pathlib.Path(args.adapter)
    adapter_cfg = load_optional_json(adapter_dir / "adapter_config.json") if adapter_dir.is_dir() else None
    preserved: list[str] = []

    if not args.dry_run:
        merge_peft_to_bf16(args.base_model, args.adapter, bf16_dir)
    if adapter_dir.is_dir():
        preserved = copy_preserved_files(adapter_dir, bf16_dir)

    if args.eval_bundle_dir:
        card = pathlib.Path(args.eval_bundle_dir) / "README.md"
        if card.is_file():
            shutil.copy2(card, bf16_dir / "README.md")

    quant = None if args.mlx_quant.lower() == "none" else args.mlx_quant
    provenance = build_provenance(
        base_model=args.base_model,
        adapter_path=args.adapter,
        mlx_quant=quant,
        hf_bf16_repo=args.hf_bf16_repo,
        hf_mlx_repo=args.hf_mlx_repo,
        hf_lora_repo=args.hf_lora_repo,
        adapter_config=adapter_cfg,
    )
    provenance["preserved_files"] = preserved
    commands: list[list[str]] = []
    if quant:
        commands.append(convert_mlx_sibling(bf16_dir, mlx_dir, quant, dry_run=args.dry_run))
    if args.hf_lora_repo and adapter_dir.is_dir():
        commands.append(publish_hf_directory(adapter_dir, args.hf_lora_repo, "Publish PEFT LoRA adapter (#734)", dry_run=args.dry_run))
    if args.hf_bf16_repo:
        commands.append(publish_hf_directory(bf16_dir, args.hf_bf16_repo, "Publish merged BF16 checkpoint (#734)", dry_run=args.dry_run))
    if args.hf_mlx_repo and quant:
        commands.append(publish_hf_directory(mlx_dir, args.hf_mlx_repo, f"Publish MLX {quant} checkpoint (#734)", dry_run=args.dry_run))

    provenance["planned_commands"] = commands
    (bf16_dir / "basalt_merge_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared MT packaging plan at {bf16_dir} (dry_run={args.dry_run}, commands={len(commands)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
