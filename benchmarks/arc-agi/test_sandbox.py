import errno
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import arc_agi_benchmark
from arc_agi_benchmark import execute_python_hybrid
from arc_sandbox import run_transform, sandbox_available

BLOCKED = [[0]]
ESCAPED = [[1]]
SANDBOX_REFUSALS = (errno.EPERM, errno.EACCES, errno.EAGAIN)
MEMORY_SOFT_CAP_BYTES = 3 << 29
UNLISTED_EXTENSION = "_uuid"
CANARY = "arc-sandbox-canary-5d1c"
PROCESS_SYSCTL_PROBE = f"""
def transform(grid):
    import os
    try:
        import ctypes
    except ImportError:
        return [[0]]
    libc = ctypes.CDLL(None, use_errno=True)
    def read(*mib):
        names = (ctypes.c_int * len(mib))(*mib)
        size = ctypes.c_size_t(1 << 20)
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(names, len(mib), buffer, ctypes.byref(size), None, 0) != 0:
            return b""
        return buffer.raw[:size.value]
    parent_args = read(1, 49, os.getppid())
    process_table = read(1, 14, 0)
    return [[1, int({CANARY!r}.encode() in parent_args), len(parent_args), len(process_table)]]
"""


def attempt(body: str) -> str:
    indented = "\n".join("        " + line for line in body.strip().splitlines())
    return (
        "import errno\n"
        "def transform(grid):\n"
        "    try:\n"
        f"{indented}\n"
        "    except OSError as error:\n"
        f"        if error.errno in {SANDBOX_REFUSALS!r}:\n"
        "            return [[0]]\n"
        "        raise\n"
        "    return [[1]]\n"
    )


class ModeDefaultTest(unittest.TestCase):
    def test_default_mode_is_direct_without_sandbox_exec(self):
        with mock.patch.object(arc_agi_benchmark, "sandbox_available", return_value=False):
            self.assertEqual(arc_agi_benchmark.parse_args([]).mode, "direct")

    def test_default_mode_is_hybrid_with_sandbox_exec(self):
        with mock.patch.object(arc_agi_benchmark, "sandbox_available", return_value=True):
            self.assertEqual(arc_agi_benchmark.parse_args([]).mode, "hybrid")


@unittest.skipUnless(sandbox_available(), "sandbox-exec is not available on this platform")
class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.outside = Path(tempfile.mkdtemp(prefix="arc-outside-")).resolve()
        self.watched = [self.outside, Path.cwd(), Path(__file__).resolve().parent, Path.home(), Path("/tmp")]

    def tearDown(self):
        for path in self.outside.iterdir():
            path.unlink()
        self.outside.rmdir()

    def run_code(self, code: str):
        return run_transform([code], [[5, 6], [7, 8]])

    def assert_blocked(self, body: str):
        grid, status = self.run_code(attempt(body))
        self.assertEqual((grid, status), (BLOCKED, "python_oop_exec_success"))

    def assert_no_pwned(self):
        for directory in self.watched:
            self.assertFalse((directory / "PWNED").exists(), directory)

    def test_a_broken_probe_is_not_counted_as_blocked(self):
        self.assertEqual(self.run_code(attempt("undefined_name")), (None, "exec_failed"))
        self.assertEqual(self.run_code(attempt("raise FileNotFoundError(errno.ENOENT, 'x')")), (None, "exec_failed"))

    def test_reading_etc_passwd_fails(self):
        self.assert_blocked("open('/etc/passwd').read()")

    def test_reading_files_outside_the_allow_list_fails(self):
        self.assert_blocked(f"open({str(Path(__file__).resolve())!r}).read()")

    def test_listing_the_benchmark_directory_fails(self):
        self.assert_blocked(f"import os\nos.listdir({str(Path(__file__).resolve().parent)!r})")

    def test_os_system_cannot_create_files(self):
        self.assert_blocked(f"""
import os
if os.system('touch PWNED; touch {self.outside}/PWNED') != 0:
    raise PermissionError(errno.EPERM, 'os.system could not start a shell')
""")
        self.assert_no_pwned()

    def test_subprocess_is_blocked(self):
        self.assert_blocked(f"import subprocess\nsubprocess.run(['touch', '{self.outside}/PWNED'], check=True)")
        self.assert_no_pwned()

    def test_writing_outside_temp_dir_fails(self):
        self.assert_blocked(f"open('{self.outside}/PWNED', 'w').write('x')")
        self.assert_no_pwned()

    def test_writing_inside_temp_dir_works(self):
        grid, _ = self.run_code(attempt("open('scratch.txt', 'w').write('x')"))
        self.assertEqual(grid, ESCAPED)

    def test_socket_connect_fails(self):
        self.assert_blocked("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)")

    def test_environment_is_scrubbed(self):
        os.environ["ARC_SANDBOX_FAKE_API_KEY"] = "secret"
        try:
            grid, _ = self.run_code("import os\ndef transform(grid):\n"
                                    "    return [[sum(name in os.environ for name in ('ARC_SANDBOX_FAKE_API_KEY', 'HOME', 'PATH'))]]")
        finally:
            del os.environ["ARC_SANDBOX_FAKE_API_KEY"]
        self.assertEqual(grid, [[0]])

    def test_child_cannot_read_parent_environment_through_sysctl(self):
        runner = ("import json, sys\nfrom arc_sandbox import run_transform\n"
                  "print(json.dumps(run_transform([sys.stdin.read()], [[1]])))")
        parent = subprocess.run([sys.executable, "-c", runner], input=PROCESS_SYSCTL_PROBE, capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parent,
                                env={**os.environ, "ARC_SANDBOX_FAKE_API_KEY": CANARY}, check=True)
        self.assertNotIn(CANARY, parent.stdout)
        self.assertEqual(json.loads(parent.stdout), [BLOCKED, "python_oop_exec_success"])

    def test_extension_module_outside_the_allowlist_cannot_be_loaded(self):
        if importlib.util.find_spec(UNLISTED_EXTENSION) is None:
            self.skipTest(f"{UNLISTED_EXTENSION} is not built into this Python")
        code = ("def transform(grid):\n    try:\n"
                f"        import {UNLISTED_EXTENSION}\n"
                "    except ImportError as error:\n"
                "        return [[0]] if 'sandbox' in str(error) else [[2]]\n"
                "    return [[1]]")
        self.assertEqual(self.run_code(code), (BLOCKED, "python_oop_exec_success"))

    def test_infinite_loop_is_killed_and_recorded_as_failed(self):
        started = time.monotonic()
        text = "```python\ndef transform(grid):\n    while True:\n        pass\n```"
        grid, status = execute_python_hybrid(text, [[1]], timeout=2)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual((grid, status), (None, "exec_timeout"))

    def test_memory_hog_is_killed_near_the_limit(self):
        code = "def transform(grid):\n    b = bytearray(4 << 30)\n    b[::4096] = b'x' * len(b[::4096])\n    return [[1]]"
        self.assertEqual(self.run_code(code), (None, "exec_memory_limit"))
        self.assertLess(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, MEMORY_SOFT_CAP_BYTES)

    def test_filling_the_temp_dir_with_many_files_is_killed(self):
        code = ("def transform(grid):\n    chunk = b'x' * (1 << 20)\n    index = 0\n    while True:\n"
                "        with open(f'f{index}', 'wb') as out:\n            out.write(chunk)\n        index += 1")
        self.assertEqual(self.run_code(code), (None, "exec_disk_limit"))

    def test_creating_many_empty_files_is_killed(self):
        code = ("def transform(grid):\n    index = 0\n    while True:\n"
                "        open(f'f{index}', 'wb').close()\n        index += 1")
        self.assertEqual(self.run_code(code), (None, "exec_disk_limit"))

    def test_rotation_returns_the_right_grid(self):
        text = "```python\ndef transform(grid):\n    return [list(row) for row in zip(*grid[::-1])]\n```"
        self.assertEqual(execute_python_hybrid(text, [[1, 2], [3, 4]]), ([[3, 1], [4, 2]], "python_oop_exec_success"))

    def test_dsl_helpers_are_available(self):
        code = ("def transform(grid: Grid) -> Grid:\n"
                "    objects = grid.extract_objects()\n"
                "    return [[len(objects), int(Color.RED)]]")
        self.assertEqual(self.run_code(code), ([[4, 2]], "python_oop_exec_success"))

    def test_numpy_is_available_when_installed(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy is not installed")
        code = "def transform(grid):\n    return np.rot90(np.array(grid.to_matrix())).tolist()"
        self.assertEqual(self.run_code(code), ([[6, 8], [5, 7]], "python_oop_exec_success"))


if __name__ == "__main__":
    unittest.main()
