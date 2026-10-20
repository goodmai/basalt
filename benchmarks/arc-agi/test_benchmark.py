import unittest

import arc_agi_benchmark
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


@unittest.skipUnless(sandbox_available(), "sandbox-exec is not available on this platform")
class TrainVerificationTest(unittest.TestCase):
    def test_a_transform_that_reproduces_every_example_is_applied_to_the_test_input(self):
        self.assertEqual(execute_python_hybrid(ROTATE, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         ([[2, 0], [3, 1]], "python_oop_exec_success", 2))

    def test_a_transform_that_misses_an_example_fails_without_the_json_fallback(self):
        text = FIRST_PAIR_ONLY + "\nFinal answer: [[2, 0], [3, 1]]"
        self.assertEqual(execute_python_hybrid(text, TASK["test"][0]["input"], train_pairs=TASK["train"]),
                         (None, "train_mismatch", 1))


if __name__ == "__main__":
    unittest.main()
