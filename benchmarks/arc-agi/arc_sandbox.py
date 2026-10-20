import functools
import json
import os
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
CHILD_SCRIPT = os.path.realpath(__file__)
HELPERS_MODULE = os.path.join(HERE, "arc_helpers.py")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
PYTHON = os.path.realpath(sys.executable)
DEFAULT_TIMEOUT = 10.0
MEMORY_LIMIT_BYTES = 1 << 30
OUTPUT_LIMIT_BYTES = 16 << 20
DISK_LIMIT_BYTES = 64 << 20
DISK_ENTRY_MIN_BYTES = 4096
OPEN_FILES_LIMIT = 64
WATCH_INTERVAL = 0.01
RUSAGE_INFO_V0 = 0
RUSAGE_INFO_V0_WORDS = 12
RUSAGE_PHYS_FOOTPRINT_WORD = 9
UNAME_SYSCTLS = ("kern.ostype", "kern.hostname", "kern.osrelease", "kern.version", "hw.machine")
NATIVE_LIBRARY_PATTERN = "[.](so|dylib)$"
ALLOWED_EXTENSIONS = ("_posixsubprocess", "_random", "_struct", "binascii", "math", "resource", "select", "zlib")
NUMPY_TEST_EXTENSIONS_PATTERN = "/(_multiarray_tests|_rational_tests|_umath_tests|_operand_flag_tests|_struct_ufunc_tests|_simd)[.][^/]*$"
EXTENSION_DIR = os.path.join(os.path.realpath(sysconfig.get_path("platstdlib")), "lib-dynload")
EXTENSION_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
FUNCTION_NAMES = ("transform", "solve", "transform_grid")
MAX_BLOCKS = 8
STARTUP_ALLOWANCE = 3.0
MAX_GRID_SIDE = 30
RESULT_BYTES_PER_GRID = 3 << 10
RESULT_LINE_OVERHEAD_BYTES = 64
NUMPY_IMAGES_PROBE = """
import ctypes, json, os, sys
sys.path.append(sys.argv[1])
import numpy
dyld = ctypes.CDLL(None)
dyld._dyld_get_image_name.restype = ctypes.c_char_p
names = (dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count()))
print(json.dumps(sorted({os.path.realpath(name.decode()) for name in names if name})))
"""

def sandbox_available() -> bool:
    return sys.platform == "darwin" and os.access(SANDBOX_EXEC, os.X_OK)


@functools.lru_cache(maxsize=None)
def _numpy_paths():
    try:
        import numpy
    except ImportError:
        return None, []
    package = os.path.dirname(os.path.realpath(numpy.__file__))
    if sys.platform != "darwin":
        return package, []
    site = os.path.dirname(package)
    try:
        images = json.loads(subprocess.run([PYTHON, "-I", "-S", "-c", NUMPY_IMAGES_PROBE, site],
                                           capture_output=True, env={}, check=True, timeout=60).stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        print(f"warning: could not list numpy's libraries ({error}); the sandbox allows numpy without them",
              file=sys.stderr)
        images = []
    skipped = ("/System/", "/usr/lib/", os.path.realpath(sys.base_prefix) + "/", site + "/")
    return package, [p for p in images if not p.startswith(skipped)]


def _ancestors(path: str):
    parent = os.path.dirname(path)
    while parent != path:
        yield parent
        path, parent = parent, os.path.dirname(parent)


def _quote(path: str) -> str:
    return json.dumps(path)


def sandbox_profile(work_dir: str, numpy_package, images) -> str:
    python_root = os.path.realpath(sys.base_prefix)
    trees = [python_root, work_dir, "/usr/lib", "/System"]
    files = ["/", CHILD_SCRIPT, HELPERS_MODULE, "/dev/null", "/dev/urandom"]
    if numpy_package:
        trees.append(numpy_package)
        files += images
    ancestors = sorted({a for p in trees + files for a in _ancestors(p)})
    native_roots = [python_root] + ([os.path.dirname(numpy_package)] if numpy_package else [])
    link_dirs = sorted({os.path.dirname(p) for p in images}) + native_roots
    if "/Cellar/" in python_root:
        link_dirs.append(python_root.split("/Cellar/")[0] + "/opt")
    readable = [f"(subpath {_quote(p)})" for p in trees] + [f"(literal {_quote(p)})" for p in files]
    allowed_native = [f"(literal {_quote(os.path.join(EXTENSION_DIR, name + EXTENSION_SUFFIX))})" for name in ALLOWED_EXTENSIONS]
    if numpy_package:
        allowed_native.append(f"(subpath {_quote(numpy_package)})")
    executables = [PYTHON, os.path.join(python_root, "Resources/Python.app/Contents/MacOS/Python")]
    return "\n".join([
        "(version 1)",
        "(deny default)",
        "(allow process-exec " + " ".join(f"(literal {_quote(p)})" for p in executables) + ")",
        "(allow file-read-metadata " + " ".join(f"(literal {_quote(p)})" for p in ancestors)
        + "".join(f" (subpath {_quote(p)})" for p in link_dirs) + ")",
        "(allow file-read* " + " ".join(readable) + ")",
        f"(allow file-write* (subpath {_quote(work_dir)}) (literal \"/dev/null\"))",
        "(deny file-read* " + " ".join(f"(require-all (subpath {_quote(p)}) (regex #{_quote(NATIVE_LIBRARY_PATTERN)}))"
                                       for p in native_roots) + ")",
        "(allow file-read* " + " ".join(allowed_native) + ")",
        *([f"(deny file-read* (require-all (subpath {_quote(numpy_package)}) "
           f"(regex #{_quote(NUMPY_TEST_EXTENSIONS_PATTERN)})))"] if numpy_package else []),
        "(allow sysctl-read " + " ".join(f"(sysctl-name {_quote(n)})" for n in UNAME_SYSCTLS) + ")",
        "(deny network*)",
    ])


def run_transform(code_blocks, grid, timeout: float = DEFAULT_TIMEOUT):
    grid, status, _ = run_blocks(code_blocks, [], grid, timeout=timeout)
    return grid, status


def run_blocks(code_blocks, train_pairs, test_input, timeout: float = DEFAULT_TIMEOUT):
    numpy_package, images = _numpy_paths()
    blocks = list(code_blocks[:MAX_BLOCKS])
    inputs = [pair["input"] for pair in train_pairs] + [test_input]
    block_budget = timeout * len(inputs) + STARTUP_ALLOWANCE
    total_deadline = time.monotonic() + block_budget * len(blocks)
    best, answered, timed_out = 0, False, False
    for code in blocks:
        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            return None, "exec_timeout", best
        outputs, block_timed_out, limit = _run_child(code, inputs, timeout, min(block_budget, remaining),
                                                     numpy_package, images)
        if limit:
            return None, limit, best
        train_outputs = outputs[:len(train_pairs)]
        passed = sum(output == pair["output"] for output, pair in zip(train_outputs, train_pairs))
        if passed == len(train_pairs) and len(outputs) == len(inputs) and outputs[-1] is not None:
            return outputs[-1], "python_oop_exec_success", passed
        best = max(best, passed)
        answered = answered or any(output is not None for output in train_outputs)
        timed_out = timed_out or block_timed_out
    if answered:
        return None, "train_mismatch", best
    return None, "exec_timeout" if timed_out else "exec_failed", best


def _run_child(code, inputs, timeout, wall_seconds, numpy_package, images):
    request = {"code": code, "inputs": inputs, "timeout": timeout, "numpy": numpy_package,
               "cpu_seconds": int(wall_seconds) + 1}
    with tempfile.TemporaryDirectory(prefix="arc-sandbox-") as raw_dir:
        work_dir = os.path.realpath(raw_dir)
        command = [PYTHON, "-I", "-S", CHILD_SCRIPT]
        if sandbox_available():
            command = [SANDBOX_EXEC, "-p", sandbox_profile(work_dir, numpy_package, images)] + command
        output_path = os.path.join(work_dir, "stdout.json")
        with open(output_path, "w+b") as output:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                                       stderr=subprocess.DEVNULL, cwd=work_dir, env={})
            try:
                process.stdin.write(json.dumps(request).encode())
                process.stdin.close()
            except BrokenPipeError:
                pass
            stopped = _supervise(process, wall_seconds, work_dir)
            if stopped == "exec_timeout":
                return [], True, None
            if stopped:
                return [], False, stopped
            line = _last_line(output, len(inputs) * RESULT_BYTES_PER_GRID + RESULT_LINE_OVERHEAD_BYTES)
    return (*_parse_outputs(line, len(inputs)), None)


def _last_line(output, max_length: int):
    size = output.seek(0, os.SEEK_END)
    output.seek(max(0, size - max_length - 2))
    _, newline, line = output.read().rstrip(b"\n").rpartition(b"\n")
    return line if newline else None


def _parse_outputs(line, count):
    if line is None:
        return [], False
    try:
        outcome = json.loads(line)
        outputs, timed_out = outcome["outputs"], outcome["timed_out"]
    except Exception:
        return [], False
    if not isinstance(outputs, list) or len(outputs) > count:
        return [], False
    if not all(output is None or _is_grid(output) for output in outputs):
        return [], False
    return outputs, timed_out is True


def _footprint_bytes(pid: int) -> int:
    if sys.platform != "darwin":
        try:
            with open(f"/proc/{pid}/statm") as statm:
                return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return 0
    import ctypes
    usage = (ctypes.c_uint64 * RUSAGE_INFO_V0_WORDS)()
    if ctypes.CDLL(None).proc_pid_rusage(pid, RUSAGE_INFO_V0, usage) != 0:
        return 0
    return usage[RUSAGE_PHYS_FOOTPRINT_WORD]


def _disk_bytes(work_dir: str) -> int:
    total = 0
    for root, dirs, files in os.walk(work_dir):
        for name in dirs + files:
            try:
                size = os.lstat(os.path.join(root, name)).st_size
            except OSError:
                size = 0
            total += max(size, DISK_ENTRY_MIN_BYTES)
    return total


def _supervise(process, timeout: float, work_dir: str):
    deadline = time.monotonic() + timeout
    while True:
        try:
            process.wait(timeout=WATCH_INTERVAL)
            return None
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            reason = "exec_timeout"
        elif _footprint_bytes(process.pid) > MEMORY_LIMIT_BYTES:
            reason = "exec_memory_limit"
        elif _disk_bytes(work_dir) > DISK_LIMIT_BYTES:
            reason = "exec_disk_limit"
        else:
            continue
        process.kill()
        process.wait()
        return reason


def _is_grid(value) -> bool:
    return (isinstance(value, list) and 0 < len(value) <= MAX_GRID_SIDE
            and all(isinstance(row, list) and 0 < len(row) <= MAX_GRID_SIDE and len(row) == len(value[0])
                    and all(type(cell) is int and 0 <= cell <= 9 for cell in row) for row in value))


def _apply_limits(cpu_seconds: int):
    import resource
    limits = [
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_AS", MEMORY_LIMIT_BYTES),
        ("RLIMIT_DATA", MEMORY_LIMIT_BYTES),
        ("RLIMIT_FSIZE", OUTPUT_LIMIT_BYTES),
        ("RLIMIT_NOFILE", OPEN_FILES_LIMIT),
        ("RLIMIT_NPROC", 0),
        ("RLIMIT_CORE", 0),
    ]
    applied = []
    for name, value in limits:
        if not hasattr(resource, name):
            continue
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
            applied.append(name)
        except (ValueError, OSError):
            continue
    return applied


def _pick_function(namespace, base_names):
    for name in FUNCTION_NAMES:
        if callable(namespace.get(name)):
            return namespace[name]
    for name, item in namespace.items():
        if name not in base_names and callable(item) and getattr(item, "__name__", "") != "<lambda>":
            return item
    return None


def _to_matrix(result, grid_type):
    if isinstance(result, grid_type) or hasattr(result, "to_matrix"):
        result = result.to_matrix()
    elif hasattr(result, "tolist"):
        result = result.tolist()
    if not (isinstance(result, list) and len(result) <= MAX_GRID_SIDE
            and all(isinstance(row, list) and len(row) <= MAX_GRID_SIDE for row in result)):
        return None
    matrix = [[int(cell) for cell in row] for row in result]
    return matrix if _is_grid(matrix) else None


def _run_block(code, grid, base):
    namespace = dict(base)
    exec(code, namespace)
    fn = _pick_function(namespace, set(base) | {"__builtins__"})
    if fn is None:
        return None
    grid_type = base["Grid"]
    try:
        result = fn(grid_type([list(row) for row in grid]))
    except Exception:
        result = fn([list(row) for row in grid])
    return _to_matrix(result, grid_type)


class _BlockTimeout(BaseException):
    pass


def _raise_block_timeout(signum, frame):
    raise _BlockTimeout


def _run_block_or_none(code, grid, base, seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds, seconds)
    try:
        return _run_block(code, grid, base)
    except _BlockTimeout:
        raise
    except BaseException:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _outputs_for_inputs(request, base):
    outputs = []
    try:
        for grid in request["inputs"]:
            outputs.append(_run_block_or_none(request["code"], grid, base, request["timeout"]))
        return outputs, False
    except _BlockTimeout:
        signal.setitimer(signal.ITIMER_REAL, 0)
        return outputs, True


def _import_by_path(name: str, path: str, package_dir=None):
    import importlib.util
    locations = [package_dir] if package_dir else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _child_main():
    request = json.loads(sys.stdin.read())
    arc_helpers = _import_by_path("arc_helpers", HELPERS_MODULE)
    base = dict(arc_helpers.DSL_NAMES)
    if request["numpy"]:
        try:
            base["np"] = _import_by_path("numpy", os.path.join(request["numpy"], "__init__.py"), request["numpy"])
        except ImportError:
            sys.modules.pop("numpy", None)
    _apply_limits(request["cpu_seconds"])
    signal.signal(signal.SIGALRM, _raise_block_timeout)
    outputs, timed_out = _outputs_for_inputs(request, base)
    os.write(1, b"\n" + json.dumps({"outputs": outputs, "timed_out": timed_out}).encode() + b"\n")
    os._exit(0)


if __name__ == "__main__":
    _child_main()
