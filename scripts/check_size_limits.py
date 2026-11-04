#!/usr/bin/env python3
import csv
import io
import os
import shlex
import subprocess
import sys
import tempfile

MAX_CCN = 15
MAX_NLOC = 60
MAX_PARAMS = 6
MAX_FILE_LINES = 600
LIZARD = shlex.split(os.environ.get("LIZARD", "uvx --quiet lizard==1.24.0"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout


def changed_swift_files(base: str) -> list[str]:
    out = git("diff", "--name-only", "--diff-filter=d", "-z", f"{base}...HEAD", "--", "*.swift")
    return [p for p in out.split("\0") if p]


def source_at(ref: str, path: str) -> str | None:
    try:
        return git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def functions(source: str | None) -> dict[str, tuple[tuple[int, int, int], int]]:
    if source is None:
        return {}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "File.swift")
        with open(path, "w") as f:
            f.write(source)
        out = subprocess.run([*LIZARD, "-l", "swift", "--csv", path], check=True, capture_output=True, text=True).stdout
    found: dict[str, tuple[tuple[int, int, int], int]] = {}
    for row in csv.reader(io.StringIO(out)):
        metrics, name, line = (int(row[1]), int(row[0]), int(row[3])), row[8], int(row[9])
        found[name] = max(found.get(name, ((0, 0, 0), 0)), (metrics, line))
    return found


def violations(path: str, base: str | None, head: str) -> list[str]:
    hits = []
    before, after = functions(base), functions(head)
    for name, ((ccn, nloc, params), line) in sorted(after.items()):
        was_ccn, was_nloc, was_params = before.get(name, ((0, 0, 0), 0))[0]
        for metric, value, was, limit in (("CCN", ccn, was_ccn, MAX_CCN), ("lines of code", nloc, was_nloc, MAX_NLOC),
                                          ("parameters", params, was_params, MAX_PARAMS)):
            if value > limit and value > was:
                grown = f", was {was}" if was > limit else ""
                hits.append(f"{path}:{line}: {name}: {metric} {value} > {limit}{grown}")
    lines, was_lines = head.count("\n"), (base or "").count("\n")
    if not path.startswith("Tests/") and lines > MAX_FILE_LINES and lines > was_lines:
        grown = f", was {was_lines}" if was_lines > MAX_FILE_LINES else ""
        hits.append(f"{path}: {lines} lines > {MAX_FILE_LINES}{grown}")
    return hits


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    hits = []
    for path in changed_swift_files(base):
        hits += violations(path, source_at(base, path), source_at("HEAD", path) or "")
    for hit in hits:
        print(hit)
    if hits:
        print(f"\n{len(hits)} limit(s) exceeded or grown. Split along an existing seam "
              "(see \"Size and complexity\" in CONTRIBUTING.md).")
        return 1
    print("No size or complexity limit exceeded or grown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
