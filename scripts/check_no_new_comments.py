#!/usr/bin/env python3
import re
import subprocess
import sys

ALLOWED = re.compile(r"^\s*//\s*swift-tools-version:")
FULL_LINE = re.compile(r"^\s*(//|/\*|\*/|\*\s|\*$)")


def trailing_comment(line: str) -> bool:
    in_string = False
    i = 0
    while i < len(line) - 1:
        c = line[i]
        if c == "\\" and in_string:
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        elif not in_string and line.startswith("//", i) and (i == 0 or line[i - 1] != ":"):
            return True
        elif not in_string and line.startswith("/*", i):
            return True
        i += 1
    return False


def added_lines(base: str):
    diff = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD", "--", "*.swift"],
        check=True, capture_output=True, text=True,
    ).stdout
    path, number = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else None
        elif raw.startswith("@@"):
            number = int(re.search(r"\+(\d+)", raw).group(1))
        elif raw.startswith("+") and path:
            yield path, number, raw[1:]
            number += 1


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    found = [
        (path, n, text)
        for path, n, text in added_lines(base)
        if not ALLOWED.match(text) and (FULL_LINE.match(text) or trailing_comment(text))
    ]
    for path, n, text in found:
        print(f"{path}:{n}: {text.strip()}")
    if found:
        print(f"\n{len(found)} added comment line(s). Move the explanation to the commit message, "
              "the PR or a test name (see CONTRIBUTING.md).")
        return 1
    print("No comments added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
