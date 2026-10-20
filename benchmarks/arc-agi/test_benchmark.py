import unittest
from unittest import mock

import arc_agi_benchmark
import arc_sandbox
from arc_agi_benchmark import build_arc_prompt, execute_python_hybrid, openai_payload, request_options, stream_meta
from arc_sandbox import sandbox_available

TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
        {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]},
    ],
    "test": [{"input": [[0, 1], [2, 3]], "output": [[2, 0], [3, 1]]}],
}
ROTATE = "```python\ndef transform(grid):\n    return [list(row) for row in zip(*grid[::-1])]\n```"
RAISES = "```python\ndef transform(grid):\n    raise ValueError(grid)\n```"
WRONG = "```python\ndef transform(grid):\n    return [[9]]\n```"
HANG = "```python\ndef transform(grid):\n    while True:\n        pass\n```"
BIG_ON_TEST = ("```python\ndef transform(grid):\n    rows = [list(row) for row in zip(*grid[::-1])]\n"
               "    return [[1] * 3000 for _ in range(3000)] if rows == [[2, 0], [3, 1]] else rows\n```")
IGNORES_ALARM = ("```python\nimport signal\ndef transform(grid):\n    signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
                 "    while True:\n        pass\n```")
C_LOOP = "```python\ndef transform(grid):\n    return [[sum(range(10 ** 13)) % 10]]\n```"
REWRITES_EXPECTED = ("```python\nimport gc\nfor item in gc.get_objects():\n"
                     "    if isinstance(item, dict) and 'output' in item:\n        item['output'] = [[9]]\n"
                     "def transform(grid):\n    return [[9]]\n```")
FORGES_RESULT_LINE = ("```python\nimport json\nreal_dumps = json.dumps\n"
                      "json.dumps = lambda outcome: real_dumps({**outcome, 'grid': [[9]], 'passed': 2})\n"
                      "def transform(grid):\n    return [[9]]\n```")
DEEP_RESULT_LINE = ("```python\nimport os\ndef transform(grid):\n"
                    "    os.write(1, b'\\n' + b'[' * 1_000_000 + b'\\n')\n    os._exit(0)\n```")
LONG_RESULT_LINE = ("```python\nimport os\ndef transform(grid):\n"
                    "    os.write(1, b'\\n[' + b'[],' * 5_000_000 + b'[]]\\n')\n    os._exit(0)\n```")
FILLS_THE_DISK = ("```python\ndef transform(grid):\n    for index in range(40000):\n"
                  "        open(f'f{index}', 'wb').close()\n    while True:\n        pass\n```")
FULL_SIZE_GRID = "```python\ndef transform(grid):\n    return [[9] * 30 for _ in range(30)]\n```"
FIRST_PAIR_ONLY = ("```python\ndef transform(grid):\n    if grid[0][0] == 1:\n        return [[3, 1], [4, 2]]\n"
                   "    return [[2, 0], [3, 1]]\n```")


class PromptTest(unittest.TestCase):
    def test_the_test_input_has_its_own_label(self):
        for mode in ("hybrid", "direct"):
            prompt = build_arc_prompt(TASK, mode=mode)
            self.assertEqual(prompt.count("\nIN = "), len(TASK["train"]), mode)
            self.assertIn("Test input (apply the rule to this):\nTEST_IN = [[0, 1], [2, 3]]", prompt, mode)

    def test_hybrid_instructions_are_neutral(self):
        prompt = build_arc_prompt(TASK, mode="hybrid")
        self.assertIn("Identify the rule that maps every example IN to its OUT", prompt)
        self.assertIn("not required", prompt)
        self.assertNotIn("object abstractions", prompt)
        self.assertNotIn("expert", prompt)


class PayloadTest(unittest.TestCase):
    def test_unset_options_and_temperature_are_left_to_the_model(self):
        args = arc_agi_benchmark.parse_args([])
        payload = openai_payload("m", "p", 8000, args.temperature, request_options(args))
        self.assertNotIn("temperature", payload)
        for key in ("reasoning_effort", "thinking_budget", "top_k", "presence_penalty"):
            self.assertNotIn(key, payload)
        self.assertEqual(payload["stream_options"], {"include_usage": True})

    def test_flags_reach_the_request(self):
        args = arc_agi_benchmark.parse_args([
            "--reasoning-effort", "low", "--thinking-budget", "5000", "--top-k", "20",
            "--presence-penalty", "1.5", "--temperature", "0.6",
        ])
        payload = openai_payload("m", "p", 8000, args.temperature, request_options(args))
        self.assertEqual(
            {k: payload[k] for k in ("reasoning_effort", "thinking_budget", "top_k", "presence_penalty", "temperature")},
            {"reasoning_effort": "low", "thinking_budget": 5000, "top_k": 20, "presence_penalty": 1.5,
             "temperature": 0.6})

    def test_usage_and_finish_reason_are_read_from_the_stream(self):
        self.assertEqual(stream_meta({"choices": [{"delta": {}, "finish_reason": "length"}]}), {"finish_reason": "length"})
        usage = {"choices": [], "usage": {"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 700}}}
        self.assertEqual(stream_meta(usage), {"completion_tokens": 900, "reasoning_tokens": 700})

    def test_max_tokens_makes_room_for_the_answer_after_the_thinking_budget(self):
        def max_tokens(*argv):
            return arc_agi_benchmark.parse_args(list(argv)).max_tokens
        self.assertEqual(max_tokens(), 16000)
        self.assertEqual(max_tokens("--max-tokens", "6000"), 6000)
        self.assertEqual(max_tokens("--thinking-budget", "5000"), 9000)
        self.assertEqual(max_tokens("--thinking-budget", "5000", "--answer-tokens", "1000"), 6000)
        self.assertEqual(max_tokens("--thinking-budget", "5000", "--max-tokens", "6000"), 9000)
        self.assertEqual(max_tokens("--thinking-budget", "5000", "--max-tokens", "20000"), 20000)


    def test_answer_tokens_below_one_is_rejected(self):
        for value in ("0", "-5"):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                arc_agi_benchmark.parse_args(["--thinking-budget", "5000", "--answer-tokens", value])
        self.assertEqual(arc_agi_benchmark.parse_args(["--answer-tokens", "1"]).answer_tokens, 1)


@unittest.skipUnless(sandbox_available(), "sandbox-exec is not available on this platform")
class TrainVerificationTest(unittest.TestCase):
    def test_a_transform_that_reproduces_every_example_is_applied_to_the_test_input(self):
        self.assertEqual(execute_python_hybrid(ROTATE, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_transform_that_misses_an_example_fails_without_the_json_fallback(self):
        text = FIRST_PAIR_ONLY + "\nFinal answer: [[2, 0], [3, 1]]"
        self.assertEqual(execute_python_hybrid(text, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         (None, "train_mismatch", 1))

    def test_an_earlier_block_that_reproduces_the_examples_wins_over_a_later_one_that_does_not(self):
        text = ROTATE + "\n" + FIRST_PAIR_ONLY
        self.assertEqual(execute_python_hybrid(text, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_without_a_matching_block_the_best_train_score_is_reported(self):
        text = FIRST_PAIR_ONLY + "\n" + WRONG
        self.assertEqual(execute_python_hybrid(text, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         (None, "train_mismatch", 1))

    def test_failing_code_does_not_fall_back_to_a_json_grid_in_the_text(self):
        text = RAISES + "\nFinal answer: [[2, 0], [3, 1]]"
        self.assertEqual(execute_python_hybrid(text, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         (None, "exec_failed", 0))

    def test_each_block_gets_its_own_deadline_for_every_grid_plus_startup(self):
        with mock.patch.object(arc_sandbox, "_supervise", wraps=arc_sandbox._supervise) as supervise:
            execute_python_hybrid(ROTATE + "\n" + WRONG, TASK["test"][0]["input"], timeout=1.5,
                                  train_pairs=TASK["train"])
        budget = 1.5 * 3 + arc_sandbox.STARTUP_ALLOWANCE
        self.assertEqual([call.args[1] for call in supervise.call_args_list], [budget, budget])

    def test_a_block_that_ignores_the_alarm_is_killed_alone_and_an_older_block_still_answers(self):
        self.assertEqual(execute_python_hybrid(ROTATE + "\n" + IGNORES_ALARM, TASK["test"][0]["input"], timeout=1,
                                               train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_block_stuck_in_a_c_loop_is_killed_alone_and_an_older_block_still_answers(self):
        self.assertEqual(execute_python_hybrid(ROTATE + "\n" + C_LOOP, TASK["test"][0]["input"], timeout=1,
                                               train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_block_cannot_tamper_its_way_into_matching_the_examples(self):
        for block in (REWRITES_EXPECTED, FORGES_RESULT_LINE):
            self.assertEqual(execute_python_hybrid(block, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                             (None, "train_mismatch", 0), block)

    def test_a_result_line_too_deep_to_parse_fails_alone_and_an_older_block_still_answers(self):
        self.assertEqual(execute_python_hybrid(ROTATE + "\n" + DEEP_RESULT_LINE, TASK["test"][0]["input"],
                                               train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_parse_error_of_any_kind_fails_only_its_block(self):
        with mock.patch.object(arc_sandbox.json, "loads", side_effect=RecursionError("deep")):
            self.assertEqual(arc_sandbox._parse_outputs('{"outputs": [[[1]]], "timed_out": false}', 1), ([], False))

    def test_a_limit_outranks_a_timeout_when_no_block_answers(self):
        self.assertEqual(execute_python_hybrid(FILLS_THE_DISK + "\n" + HANG, TASK["test"][0]["input"], timeout=1,
                                               train_pairs=TASK["train"])[1],
                         "exec_disk_limit")

    def test_honest_full_size_grids_fit_within_the_result_line_bound(self):
        pairs = [{"input": [[0] * 30 for _ in range(30)], "output": [[9] * 30 for _ in range(30)]}] * 10
        self.assertEqual(execute_python_hybrid(FULL_SIZE_GRID, [[0] * 30 for _ in range(30)], train_pairs=pairs),
                         ([[9] * 30 for _ in range(30)], "python_oop_exec_success", 10))

    def test_a_result_line_too_long_for_its_grids_is_rejected_unparsed_and_an_older_block_still_answers(self):
        with mock.patch.object(arc_sandbox.json, "loads", wraps=arc_sandbox.json.loads) as loads:
            outcome = execute_python_hybrid(ROTATE + "\n" + LONG_RESULT_LINE, TASK["test"][0]["input"],
                                            train_pairs=TASK["train"])
        self.assertEqual(outcome, ([[2, 0], [3, 1]], "python_oop_exec_success", 2))
        limit = 3 * arc_sandbox.RESULT_BYTES_PER_GRID + arc_sandbox.RESULT_LINE_OVERHEAD_BYTES
        self.assertTrue(all(len(call.args[0]) <= limit for call in loads.call_args_list))

    def test_a_block_over_the_disk_limit_is_killed_alone_and_an_older_block_still_answers(self):
        self.assertEqual(execute_python_hybrid(ROTATE + "\n" + FILLS_THE_DISK, TASK["test"][0]["input"],
                                               train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_hanging_block_times_out_alone_and_the_next_block_still_answers(self):
        self.assertEqual(execute_python_hybrid(ROTATE + "\n" + HANG, TASK["test"][0]["input"], timeout=1,
                                               train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_test_grid_beyond_the_arc_bound_is_no_answer_and_the_next_block_still_answers(self):
        self.assertEqual(execute_python_hybrid(ROTATE + "\n" + BIG_ON_TEST, TASK["test"][0]["input"],
                                               train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_only_the_last_written_blocks_are_tried_and_the_time_limit_is_capped_by_them(self):
        text = "\n".join([ROTATE] + [WRONG] * arc_sandbox.MAX_BLOCKS)
        with mock.patch.object(arc_sandbox, "_supervise", wraps=arc_sandbox._supervise) as supervise:
            outcome = execute_python_hybrid(text, TASK["test"][0]["input"], timeout=1.5, train_pairs=TASK["train"])
        self.assertEqual(outcome, (None, "train_mismatch", 0))
        budget = 1.5 * 3 + arc_sandbox.STARTUP_ALLOWANCE
        self.assertEqual([call.args[1] for call in supervise.call_args_list], [budget] * arc_sandbox.MAX_BLOCKS)


if __name__ == "__main__":
    unittest.main()
