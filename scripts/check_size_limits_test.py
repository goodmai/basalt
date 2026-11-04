#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import unittest

CHECKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_size_limits.py")


def branchy(name: str, branches: int, params: int = 1) -> str:
    args = ", ".join(f"p{i}: Int" for i in range(params))
    body = "".join(f"    if p0 == {i} {{ return {i} }}\n" for i in range(branches))
    return f"func {name}({args}) -> Int {{\n{body}    return -1\n}}\n"


def long_function(name: str, lines: int) -> str:
    body = "".join(f"    total += {i}\n" for i in range(lines))
    return f"func {name}() -> Int {{\n    var total = 0\n{body}    return total\n}}\n"


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


class CheckerTest(unittest.TestCase):
    def repo(self, before, after):
        d = tempfile.mkdtemp()
        run(["git", "init", "-q", "-b", "main"], d)
        for key, value in [("user.name", "t"), ("user.email", "t@t")]:
            run(["git", "config", key, value], d)
        for files, message in ((before, "base"), (after, "pr")):
            for name, body in files.items():
                os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
                with open(os.path.join(d, name), "w") as f:
                    f.write(body)
            run(["git", "add", "-A"], d)
            run(["git", "commit", "-qm", message, "--allow-empty"], d)
            if message == "base":
                run(["git", "checkout", "-qb", "pr"], d)
        return subprocess.run([sys.executable, CHECKER, "main"], cwd=d, capture_output=True, text=True)

    def test_new_code_within_the_limits_passes(self):
        result = self.repo({}, {"Sources/A.swift": branchy("small", 5) + long_function("short", 20)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_new_function_over_each_limit_fails(self):
        result = self.repo({}, {"Sources/A.swift": branchy("complex", 20) + long_function("long", 80)
                                + branchy("wide", 1, params=8)})
        self.assertEqual(result.returncode, 1)
        self.assertIn("complex p0 : Int: CCN 21 > 15", result.stdout)
        self.assertIn(": long: lines of code", result.stdout)
        self.assertIn("parameters 8 > 6", result.stdout)

    def test_existing_function_over_a_limit_may_shrink_but_not_grow(self):
        before = {"Sources/A.swift": branchy("legacy", 20)}
        self.assertEqual(self.repo(before, {"Sources/A.swift": branchy("legacy", 18)}).returncode, 0)
        grown = self.repo(before, {"Sources/A.swift": branchy("legacy", 22)})
        self.assertEqual(grown.returncode, 1)
        self.assertIn("CCN 23 > 15, was 21", grown.stdout)

    def test_untouched_function_over_a_limit_does_not_fail_an_edit_elsewhere(self):
        before = {"Sources/A.swift": branchy("legacy", 20) + branchy("other", 2)}
        result = self.repo(before, {"Sources/A.swift": branchy("legacy", 20) + branchy("other", 3)})
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_long_source_file_may_not_grow_but_a_test_file_may(self):
        legacy = "".join(f"let v{i} = {i}\n" for i in range(700))
        before = {"Sources/Big.swift": legacy, "Tests/BigTests.swift": legacy}
        grown = self.repo(before, {"Sources/Big.swift": legacy + "let extra = 1\n",
                                   "Tests/BigTests.swift": legacy + "let extra = 1\n"})
        self.assertEqual(grown.returncode, 1)
        self.assertIn("Sources/Big.swift: 701 lines > 600, was 700", grown.stdout)
        self.assertNotIn("Tests/BigTests.swift", grown.stdout)
        self.assertEqual(self.repo(before, {"Sources/Big.swift": legacy[: -len("let v699 = 699\n")]}).returncode, 0)


if __name__ == "__main__":
    unittest.main()
