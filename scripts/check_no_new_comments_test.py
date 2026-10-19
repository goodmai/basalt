#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import unittest

CHECKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_no_new_comments.py")

BEFORE = "let kept = 1 // an old comment the PR does not touch\n"

CLEAN = '''let url = "https://example.com/a//b"
let s = "a // b /* c */"
let i = "\\(f("//"))"
let m = """
    https://x.y // inside a multi-line string
    /* still a string */
    """
let raw = #"a "quoted" // raw"#
let raw2 = #"\\"# + "x // y"
let deeper = ##"x "# // y"##
let closure = "\\(xs.map { "\\($0) // no" }.joined(separator: ")"))"
let r = #/x//y/#
let bare = /a\\/b/
let cls = /[/]x/
let q = a / b
let product = a
    * b
'''

COMMENTED = [
    "let t = 1 // trailing",
    "// full line",
    "/// doc",
    "/*",
    "   body line without a star",
    "*/",
    "case .x:// after a colon",
    "let n = 1 /* inline */ + 2",
    "/* outer /* nested */ still comment */",
    "let q = a / b // after a division",
]


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


class CheckerTest(unittest.TestCase):
    def repo(self, files, config=()):
        d = tempfile.mkdtemp()
        run(["git", "init", "-q", "-b", "main"], d)
        for key, value in [("user.name", "t"), ("user.email", "t@t"), *config]:
            run(["git", "config", key, value], d)
        with open(os.path.join(d, "Old.swift"), "w") as f:
            f.write(BEFORE)
        run(["git", "add", "-A"], d)
        run(["git", "commit", "-qm", "base"], d)
        run(["git", "checkout", "-qb", "pr"], d)
        for name, body in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(body)
        run(["git", "add", "-A"], d)
        run(["git", "commit", "-qm", "pr"], d)
        return subprocess.run([sys.executable, CHECKER, "main"], cwd=d, capture_output=True, text=True)

    def test_code_that_only_looks_like_comments_passes(self):
        result = self.repo({"Clean.swift": CLEAN, "Package.swift": "// swift-tools-version: 6.0\nlet p = 1\n"})
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_every_comment_form_is_reported(self):
        result = self.repo({"Bad.swift": "\n".join(COMMENTED) + "\n"})
        self.assertEqual(result.returncode, 1)
        for number in range(1, len(COMMENTED) + 1):
            self.assertIn(f"Bad.swift:{number}:", result.stdout)

    def test_a_quote_inside_a_bare_regex_cannot_hide_the_next_line(self):
        result = self.repo({"R.swift": 'let r = /"/\n// real comment\nlet s = "x"\n'})
        self.assertEqual(result.returncode, 1)
        self.assertIn("R.swift:2:", result.stdout)

    def test_crlf_and_directive_only_in_package_swift(self):
        result = self.repo({"Crlf.swift": "let a = 1\r\n// c\r\n", "Other.swift": "// swift-tools-version: 6.0\n"})
        self.assertIn("Crlf.swift:2:", result.stdout)
        self.assertIn("Other.swift:1:", result.stdout)

    def test_a_renamed_file_counts_as_added(self):
        result = self.repo({"Old.swift": "", "Renamed.swift": BEFORE})
        self.assertEqual(result.returncode, 1)
        self.assertIn("Renamed.swift:1:", result.stdout)

    def test_untouched_old_comments_are_not_reported(self):
        result = self.repo({"New.swift": "let x = 1\n"})
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_non_ascii_path_and_noprefix_config_cannot_hide_a_comment(self):
        result = self.repo({"Ünï code.swift": "// hidden?\n"}, config=[("diff.noprefix", "true"), ("color.diff", "always")])
        self.assertEqual(result.returncode, 1)
        self.assertIn("Ünï code.swift:1:", result.stdout)


if __name__ == "__main__":
    unittest.main()
