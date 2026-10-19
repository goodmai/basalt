import json
import os
import subprocess
import sys
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
OPEN_FILES_LIMIT = 64
WATCH_INTERVAL = 0.1
UNAME_SYSCTLS = ("kern.ostype", "kern.hostname", "kern.osrelease", "kern.version", "hw.machine")
RAW_SYSCALL_MODULES = r"/lib-dynload/(_ctypes|_test|_xxtest)[^/]*$"
FUNCTION_NAMES = ("transform", "solve", "transform_grid")

def sandbox_available() -> bool:
    return sys.platform == "darwin" and os.access(SANDBOX_EXEC, os.X_OK)


def _loaded_images():
    import ctypes
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    names = (dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count()))
    return sorted({os.path.realpath(name.decode()) for name in names if name})


def _numpy_paths():
    try:
        import numpy
    except ImportError:
        return None, []
    package = os.path.dirname(os.path.realpath(numpy.__file__))
    images = _loaded_images() if sys.platform == "darwin" else []
    return package, [p for p in images if not p.startswith(("/System/", "/usr/lib/"))]


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
    files = ["/", CHILD_SCRIPT, HELPERS_MODULE, HERE, "/dev/null", "/dev/urandom"]
    if numpy_package:
        trees.append(numpy_package)
        files += [os.path.dirname(numpy_package)] + images
    ancestors = sorted({a for p in trees + files for a in _ancestors(p)})
    link_dirs = sorted({os.path.dirname(p) for p in images})
    if "/Cellar/" in python_root:
        link_dirs.append(python_root.split("/Cellar/")[0] + "/opt")
    readable = [f"(subpath {_quote(p)})" for p in trees] + [f"(literal {_quote(p)})" for p in files]
    executables = [PYTHON, os.path.join(python_root, "Resources/Python.app/Contents/MacOS/Python")]
    return "\n".join([
        "(version 1)",
        "(deny default)",
        "(allow process-exec " + " ".join(f"(literal {_quote(p)})" for p in executables) + ")",
        "(allow file-read-metadata " + " ".join(f"(literal {_quote(p)})" for p in ancestors)
        + "".join(f" (subpath {_quote(p)})" for p in link_dirs) + ")",
        "(allow file-read* " + " ".join(readable) + ")",
        f"(allow file-write* (subpath {_quote(work_dir)}) (literal \"/dev/null\"))",
        f"(deny file-read* (regex #{_quote(RAW_SYSCALL_MODULES)}))",
        "(allow sysctl-read " + " ".join(f"(sysctl-name {_quote(n)})" for n in UNAME_SYSCTLS) + ")",
        "(deny network*)",
    ])


def run_transform(code_blocks, grid, timeout: float = DEFAULT_TIMEOUT):
    numpy_package, images = _numpy_paths()
    site = [os.path.dirname(numpy_package)] if numpy_package else []
    request = {"blocks": code_blocks, "grid": grid, "paths": [HERE] + site,
               "cpu_seconds": int(timeout) + 1}
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
            stopped = _supervise(process, timeout)
            if stopped:
                return None, stopped
            output.seek(0)
            lines = output.read().splitlines()
    try:
        result = json.loads(lines[-1])["grid"]
    except (IndexError, ValueError, KeyError, TypeError):
        return None, "exec_failed"
    if _is_grid(result):
        return result, "python_oop_exec_success"
    return None, "exec_failed"


def _rss_bytes(pid: int) -> int:
    probe = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(probe.stdout.strip() or 0) * 1024


def _supervise(process, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        try:
            process.wait(timeout=WATCH_INTERVAL)
            return None
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            reason = "exec_timeout"
        elif _rss_bytes(process.pid) > MEMORY_LIMIT_BYTES:
            reason = "exec_memory_limit"
        else:
            continue
        process.kill()
        process.wait()
        return reason


def _is_grid(value) -> bool:
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(row, list) and all(type(cell) is int for cell in row) for row in value))


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
    if isinstance(result, list) and result and all(isinstance(row, list) for row in result):
        return [[int(cell) for cell in row] for row in result]
    return None


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


def _child_main():
    request = json.loads(sys.stdin.read())
    sys.path.extend(request["paths"])
    import arc_helpers
    base = dict(arc_helpers.DSL_NAMES)
    try:
        import numpy
        base["np"] = numpy
    except ImportError:
        pass
    _apply_limits(request["cpu_seconds"])
    result = None
    for code in request["blocks"]:
        try:
            result = _run_block(code, request["grid"], base)
        except BaseException:
            result = None
        if result is not None:
            break
    os.write(1, b"\n" + json.dumps({"grid": result}).encode() + b"\n")
    os._exit(0)


if __name__ == "__main__":
    _child_main()
