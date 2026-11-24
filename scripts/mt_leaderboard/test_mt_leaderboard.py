#!/usr/bin/env python3
"""Offline unit tests for scripts/mt_leaderboard (#734)."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_wmt_flores  # noqa: E402
import merge_and_package  # noqa: E402


class MTLeaderboardScriptsTest(unittest.TestCase):
    def test_eval_dry_run_builds_valid_model_index_and_wmt_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "bundle"
            rc = eval_wmt_flores.main([
                "--model", "Bosalt/TranslateGemma-4-12B-Pilot",
                "--base-model", "google/gemma-4-12b-it",
                "--dataset", "wmt24pp",
                "--pairs", "en-ja_JP,en-ru_RU",
                "--output", str(out),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            readme = (out / "README.md").read_text(encoding="utf-8")
            self.assertEqual(eval_wmt_flores.validate_model_card(readme), [])
            self.assertIn("model-index:", readme)
            self.assertIn("google/wmt24pp", readme)
            ja_lines = (out / "wmt" / "en-ja_JP.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(ja_lines), 1)
            preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(preds), 2)
            self.assertEqual(preds[0]["lang_pair"], "en-ja_JP")

    def test_wmt_line_sanitization_strips_embedded_newlines(self) -> None:
        self.assertEqual(eval_wmt_flores.sanitize_wmt_line("a\nb\r\nc"), "a b c")

    def test_merge_and_package_dry_run_preserves_templates_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            adapter_dir = tmp_path / "lora"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8")
            (adapter_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            eval_dir = tmp_path / "eval"
            eval_dir.mkdir()
            (eval_dir / "README.md").write_text("---\nmodel-index: []\n---\n# Card\n", encoding="utf-8")

            out_dir = tmp_path / "pkg"
            rc = merge_and_package.main([
                "--base-model", "google/gemma-4-26b-a4b-it",
                "--adapter", str(adapter_dir),
                "--output-dir", str(out_dir),
                "--mlx-quant", "nvfp4",
                "--hf-bf16-repo", "Bosalt/TranslateGemma-4-26B-A4B-v2",
                "--hf-mlx-repo", "Bosalt/TranslateGemma-4-26B-A4B-v2-MLX-NVFP4",
                "--eval-bundle-dir", str(eval_dir),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            bf16 = out_dir / "merged-bf16"
            self.assertTrue((bf16 / "chat_template.jinja").is_file())
            self.assertTrue((bf16 / "README.md").is_file())
            prov = json.loads((bf16 / "basalt_merge_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(prov["base_model"], "google/gemma-4-26b-a4b-it")
            self.assertEqual(prov["training_config"]["r"], 16)
            self.assertEqual(prov["mlx_quant"], "nvfp4")
            self.assertEqual(len(prov["planned_commands"]), 3)


if __name__ == "__main__":
    unittest.main()
