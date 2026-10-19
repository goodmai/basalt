#!/usr/bin/env python3
import re
import subprocess
import sys

DIRECTIVE = re.compile(r"^\s*//\s*swift-tools-version:")
HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout


def changed_swift_files(base: str) -> list[str]:
    out = git("diff", "--name-only", "--diff-filter=d", "-z", f"{base}...HEAD", "--", "*.swift")
    return [p for p in out.split("\0") if p]


def added_line_numbers(base: str, path: str) -> set[int]:
    out = git("diff", "--no-color", "--no-ext-diff", "--unified=0", f"{base}...HEAD", "--", path)
    lines: set[int] = set()
    for row in out.splitlines():
        match = HUNK.match(row)
        if match:
            start, count = int(match.group(1)), int(match.group(2) or "1")
            lines.update(range(start, start + count))
    return lines


def comment_lines(source: str) -> set[int]:
    found: set[int] = set()
    line, i, n = 1, 0, len(source)
    stack: list[tuple] = [("code",)]

    def at(s: str) -> bool:
        return source.startswith(s, i)

    while i < n:
        state = stack[-1]
        c = source[i]
        if c == "\n":
            if state[0] == "line_comment":
                stack.pop()
            line += 1
            i += 1
            continue
        kind = state[0]
        if kind == "line_comment":
            found.add(line)
            i += 1
        elif kind == "block_comment":
            found.add(line)
            if at("/*"):
                stack.append(("block_comment",))
                i += 2
            elif at("*/"):
                stack.pop()
                i += 2
            else:
                i += 1
        elif kind == "string":
            _, hashes, multiline = state
            closer = ('"""' if multiline else '"') + "#" * hashes
            escape = "\\" + "#" * hashes
            if at(escape + "("):
                stack.append(("interp", 0))
                i += len(escape) + 1
            elif at(escape):
                i += len(escape) + 1
            elif at(closer):
                stack.pop()
                i += len(closer)
            else:
                i += 1
        elif kind == "regex":
            _, hashes = state
            closer = "/" + "#" * hashes
            if c == "\\":
                i += 2
            elif at(closer):
                stack.pop()
                i += len(closer)
            else:
                i += 1
        else:
            if kind == "interp" and c == ")":
                if state[1] == 0:
                    stack.pop()
                else:
                    stack[-1] = ("interp", state[1] - 1)
                i += 1
            elif kind == "interp" and c == "(":
                stack[-1] = ("interp", state[1] + 1)
                i += 1
            elif at("//"):
                found.add(line)
                stack.append(("line_comment",))
                i += 2
            elif at("/*"):
                found.add(line)
                stack.append(("block_comment",))
                i += 2
            else:
                hashes = len(source[i:]) - len(source[i:].lstrip("#"))
                rest = source[i + hashes:]
                if rest.startswith('"""'):
                    stack.append(("string", hashes, True))
                    i += hashes + 3
                elif rest.startswith('"'):
                    stack.append(("string", hashes, False))
                    i += hashes + 1
                elif hashes and rest.startswith("/"):
                    stack.append(("regex", hashes))
                    i += hashes + 1
                else:
                    i += max(hashes, 1)
    return found


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    hits = []
    for path in changed_swift_files(base):
        source = git("show", f"HEAD:{path}")
        rows = source.split("\n")
        added = added_line_numbers(base, path)
        for number in sorted(comment_lines(source) & added):
            text = rows[number - 1] if number <= len(rows) else ""
            if not DIRECTIVE.match(text):
                hits.append((path, number, text.strip()))
    for path, number, text in hits:
        print(f"{path}:{number}: {text}")
    if hits:
        print(f"\n{len(hits)} added comment line(s). Move the explanation to the commit message, "
              "the PR or a test name (see CONTRIBUTING.md).")
        return 1
    print("No comments added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
