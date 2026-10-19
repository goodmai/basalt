#!/usr/bin/env python3
import json
import os
import stat
import subprocess
import tempfile
import unittest

AUDIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "security_audit.sh")

STUB_CURL = """#!/bin/sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-d" ]; then
        printf '%s\\n' "$2" >> "$OSV_QUERIES"
        case "$2" in
            *deadbeef*) printf '{"vulns":[{"id":"OSV-TEST-1","summary":"stub"}]}' ;;
            *) printf '{}' ;;
        esac
        exit 0
    fi
    shift
done
exit 1
"""


def pin(identity, **state):
    return {"identity": identity, "kind": "remoteSourceControl",
            "location": f"https://github.com/example/{identity}.git", "state": state}


class OSVScanTest(unittest.TestCase):
    def scan(self, pins):
        d = tempfile.mkdtemp()
        curl = os.path.join(d, "curl")
        with open(curl, "w") as f:
            f.write(STUB_CURL)
        os.chmod(curl, os.stat(curl).st_mode | stat.S_IEXEC)
        resolved = os.path.join(d, "Package.resolved")
        with open(resolved, "w") as f:
            json.dump({"pins": pins, "version": 2}, f)
        queries = os.path.join(d, "queries")
        env = dict(os.environ, PATH=d + os.pathsep + os.environ["PATH"], OSV_QUERIES=queries)
        result = subprocess.run(["bash", AUDIT, "--osv-only", resolved], capture_output=True, text=True, env=env)
        sent = [json.loads(line) for line in open(queries)] if os.path.exists(queries) else []
        return result, sent

    def test_a_revision_only_pin_is_queried_by_its_commit(self):
        result, sent = self.scan([pin("lib", revision="0123abcd", version=None), pin("nio", revision="ffff", version="2.1.0")])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn({"commit": "0123abcd"}, sent)
        self.assertIn({"package": {"name": "github.com/example/nio", "ecosystem": "SwiftURL"}, "version": "2.1.0"}, sent)
        self.assertIn("ok          lib @ commit 0123abcd", result.stdout)

    def test_a_vulnerable_commit_fails_the_scan(self):
        result, _ = self.scan([pin("lib", revision="deadbeef")])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("VULNERABLE  lib @ commit deadbeef", result.stdout)

    def test_a_pin_without_version_or_revision_is_unchecked(self):
        result, _ = self.scan([pin("lib"), pin("nio", version="2.1.0")])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("UNCHECKED   lib", result.stdout)


if __name__ == "__main__":
    unittest.main()
