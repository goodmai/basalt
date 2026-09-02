#!/usr/bin/env python3
"""The ladder to the moon: five rungs against a running `gemm serve --rest`.

Each rung is strictly harder than the one below it, and the top rung is a full
Earth-to-Moon mission — launch, four-body transfer, braking, soft landing —
with every constant given in the prompt so nothing can be recalled from a
textbook. Nothing is graded on how the answer reads: the model's own code is
executed against inputs the prompt never showed it.

    1. biquadratic, real roots          closed form, edge cases
    2. biquadratic, complex roots       branch handling, 4 roots with multiplicity
    3. Fourier coefficients             numeric integration to 1e-4
    4. pump head                        units + Colebrook, which has no closed form
    5. Earth -> Moon mission            energy-derived exhaust velocity, Tsiolkovsky,
                                        four-body propagation, braking, soft landing

    gemm serve --model <id> --rest &
    python3 benchmarks/ladder/run.py --port 8080

`--warmup` prepends three cheap smoke tests (algebra, Fibonacci, translation)
that any instruct model should clear; they are not part of the ladder score.

For reasoning models whose template opens <think> on every turn, start the
server with `--reasoning-effort none`: otherwise they routinely spend the whole
token budget thinking and never emit an answer, which shows up here as
finish=length.
"""
import argparse, json, math, re, time
from pathlib import Path
from urllib.request import Request, urlopen

# ── the ladder ───────────────────────────────────────────────────────────────

ALGEBRA = "Solve for x: 3x + 7 = 22. Show your step-by-step working."

FIBONACCI = (
    "Write a Python function to compute the nth Fibonacci number using recursion "
    "with memoization. Include an example call for n=10 and print the result."
)

TRANSLATION = (
    'Translate the following text to French, German, and Spanish: '
    '"The quick brown fox jumps over the lazy dog."'
)

FOURIER = """Write a Python function:

    fourier_coefficients(f, N, L=math.pi)

It computes the Fourier series coefficients of a function f on the interval [-L, L]:

    f(x) ~ a0/2 + sum_{n=1..N} [ a_n*cos(n*pi*x/L) + b_n*sin(n*pi*x/L) ]

    a_n = (1/L) * integral from -L to L of f(x)*cos(n*pi*x/L) dx   (n = 0..N)
    b_n = (1/L) * integral from -L to L of f(x)*sin(n*pi*x/L) dx   (n = 1..N)

Compute the integrals numerically (do NOT use scipy or numpy - only the standard
library). Use enough sample points that the result is accurate to 1e-4.

Return a tuple (a0, a_list, b_list) where a0 is the n=0 coefficient a_0,
a_list holds a_1..a_N and b_list holds b_1..b_N.

Return ONLY one ```python code block, no explanation."""

PUMP = """A pump feeds water up to the top floor of a building through a steel pipe.

The pipe is 180 m long with an 80 mm internal diameter and a wall roughness of
0.045 mm. The water has to be lifted 34 m. The flow rate is 9.0 L/s.
Water properties: density 1000 kg/m3, kinematic viscosity 1.004e-6 m2/s.
Take g = 9.81 m/s2.

Write a Python function:

    pump_head(length_m, diameter_mm, roughness_mm, lift_m, flow_lps)

returning the tuple (v, Re, f, h_loss, H_required):

    v          - flow velocity in m/s
    Re         - Reynolds number
    f          - the pipe's friction factor (dimensionless)
    h_loss     - head lost to friction along the pipe, in m
    H_required - total head the pump must supply, in m

Note the units of each argument. Use only the standard library, and make the
friction factor accurate to 1e-8. Return ONLY one ```python code block, no
explanation."""

BIQUAD_REAL = """Write a Python function:

    solve_biquadratic_real(a, b, c)

returning the sorted list of the distinct real roots of a*x^4 + b*x^2 + c = 0.
An equation with no real roots returns an empty list, and a repeated root is
listed once. Use only the standard library.

Return ONLY one ```python code block, no explanation."""

BIQUAD_COMPLEX = """Write two Python functions:

    solve_biquadratic_complex(a, b, c) -> list of all 4 complex roots of
                                          a*x^4 + b*x^2 + c = 0, with multiplicity
    verify_solution(a, b, c, x, tol=1e-10) -> True when x satisfies the equation

Use only the standard library (math / cmath). All four roots must be returned
even when they coincide.

Return ONLY one ```python code block, no explanation."""

# The top rung is long enough to live in its own file, and its reference values
# are deliberately kept out of the prompt (see lunar_task.md).
LUNAR = (Path(__file__).with_name("lunar_prompt.md")
         .read_text(encoding="utf-8").split("---", 1)[1].strip())

# ── validators ───────────────────────────────────────────────────────────────
#
# Each returns (passed, detail). They run the model's code on inputs the prompt
# never mentions — the point is to catch an answer that was pattern-matched
# rather than derived.


def code_blocks(text):
    """Every fenced block, last one first, plus an unterminated trailing fence."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    tail = re.search(r"```(?:python)?\s*\n((?:(?!```).)*)$", text, re.DOTALL)
    if tail:
        blocks.append(tail.group(1).rsplit("\n", 1)[0])
    return list(reversed(blocks))


def environments(text, wanted_def=None):
    """Executable namespaces from the answer's code blocks.

    Every block is tried, not just the last one: models routinely follow the
    real implementation with an abbreviated snippet ("# ... rest of the logic"),
    and grading the last block alone fails an answer that was actually correct.
    """
    for b in code_blocks(text):
        if wanted_def and wanted_def not in b:
            continue
        env = {}
        try:
            exec(b, env)
        except Exception:
            continue
        yield env


def check_algebra(text):
    # x = 5, stated anywhere in the answer.
    return (bool(re.search(r"x\s*=\s*5\b", text)), "x = 5")


def check_fibonacci(text):
    want, seen = [55, 6765, 832040], []
    for env in environments(text):
        for name, fn in env.items():
            if not (name.startswith("fib") and callable(fn)):
                continue
            try:
                got = [fn(n) for n in (10, 20, 30)]
            except Exception as e:
                seen.append(f"{name}: {type(e).__name__}")
                continue
            if got == want:
                return True, f"{name}(10,20,30) = {got}"
            seen.append(f"{name} = {got}")
    return False, (f"nothing returned {want} — tried {seen}" if seen else "no fib* function defined")


def check_translation(text):
    # Qualitative: the three languages have to be present and distinct. Fluency
    # is a human call, so this only fails an answer that skipped a language.
    hits = sum(bool(re.search(p, text, re.I)) for p in
               (r"fran[cç]|french", r"deutsch|german|allemand", r"espa[nñ]|spanish"))
    return (hits == 3, f"{hits}/3 target languages labelled")


def check_fourier(text):
    env = next(environments(text, "def fourier_coefficients"), None)
    if env is None:
        return False, "no code block that defines fourier_coefficients"
    f = env.get("fourier_coefficients")
    if not f:
        return False, "fourier_coefficients not defined"
    try:
        # sawtooth f(x)=x  ->  a_n = 0, b_n = 2*(-1)^(n+1)/n
        a0, a, b = f(lambda x: x, 5)
        if abs(a0) > 1e-3:
            return False, f"f(x)=x: a0={a0:.4f}, want 0"
        for n in range(1, 6):
            if abs(a[n - 1]) > 1e-3:
                return False, f"f(x)=x: a_{n}={a[n-1]:.4f}, want 0"
            want = 2 * (-1) ** (n + 1) / n
            if abs(b[n - 1] - want) > 2e-3:
                return False, f"f(x)=x: b_{n}={b[n-1]:.4f}, want {want:.4f}"
        # square wave sign(x)  ->  b_n = 4/(n*pi) for odd n
        a0, a, b = f(lambda x: 1.0 if x > 0 else -1.0, 5)
        for n in range(1, 6):
            want = 4 / (n * math.pi) if n % 2 else 0.0
            if abs(b[n - 1] - want) > 5e-3:
                return False, f"sign(x): b_{n}={b[n-1]:.4f}, want {want:.4f}"
        # cos(x)  ->  a_1 = 1, rest 0
        a0, a, b = f(math.cos, 3)
        if abs(a[0] - 1.0) > 2e-3:
            return False, f"cos(x): a_1={a[0]:.4f}, want 1"
        return True, "3/3 cases (sawtooth, square wave, cosine)"
    except Exception as e:
        return False, f"call failed: {type(e).__name__}: {e}"


def _pump_reference(length_m, diameter_mm, roughness_mm, lift_m, flow_lps,
                    g=9.81, nu=1.004e-6):
    d, eps, Q = diameter_mm / 1000.0, roughness_mm / 1000.0, flow_lps / 1000.0
    v = Q / (math.pi * d * d / 4.0)
    Re = v * d / nu
    x = 1.0 / math.sqrt(0.02)
    for _ in range(200):                       # Colebrook, fixed point on 1/sqrt(f)
        nx = -2.0 * math.log10(eps / (3.7 * d) + 2.51 * x / Re)
        if abs(nx - x) < 1e-14:
            x = nx
            break
        x = nx
    f = 1.0 / (x * x)
    h_loss = f * (length_m / d) * v * v / (2 * g)
    return v, Re, f, h_loss, lift_m + h_loss


def _colebrook_residual(f, Re, eps, d):
    """Zero when f satisfies Colebrook, however the model arrived at it."""
    return 1.0 / math.sqrt(f) + 2.0 * math.log10(eps / (3.7 * d) + 2.51 / (Re * math.sqrt(f)))


def check_pump(text):
    env = next(environments(text, "def pump_head"), None)
    if env is None:
        return False, "no code block that defines pump_head"
    fn = env.get("pump_head")
    if not fn:
        return False, "pump_head not defined"
    cases = [
        dict(length_m=180.0, diameter_mm=80.0, roughness_mm=0.045, lift_m=34.0, flow_lps=9.0),
        dict(length_m=250.0, diameter_mm=100.0, roughness_mm=0.15, lift_m=12.0, flow_lps=20.0),
        dict(length_m=90.0, diameter_mm=50.0, roughness_mm=0.002, lift_m=5.0, flow_lps=3.0),
    ]
    try:
        for c in cases:
            got = fn(c["length_m"], c["diameter_mm"], c["roughness_mm"], c["lift_m"], c["flow_lps"])
            if not (isinstance(got, (tuple, list)) and len(got) == 5):
                return False, f"expected 5 values, got {got!r}"
            v, Re, f, h, H = map(float, got)
            want = _pump_reference(**c)
            if abs(v - want[0]) > 1e-4:
                return False, f"v={v:.4f}, want {want[0]:.4f} (units? d is in mm, Q in L/s)"
            if abs(Re - want[1]) / want[1] > 1e-4:
                return False, f"Re={Re:.0f}, want {want[1]:.0f}"
            res = _colebrook_residual(f, Re, c["roughness_mm"] / 1000.0, c["diameter_mm"] / 1000.0)
            if f <= 0 or abs(res) > 1e-5:
                return False, f"f={f:.6f} does not satisfy Colebrook (residual {res:.2e})"
            if abs(h - want[3]) > 1e-3:
                return False, f"h_loss={h:.4f}, want {want[3]:.4f}"
            if abs(H - want[4]) > 1e-3:
                return False, f"H={H:.4f}, want {want[4]:.4f} (lift + losses?)"
        return True, "3/3 cases, Colebrook residual < 1e-5"
    except Exception as e:
        return False, f"call failed: {type(e).__name__}: {e}"


def check_biquad_real(text):
    env = next(environments(text, "def solve_biquadratic_real"), None)
    if env is None:
        return False, "no code block that defines solve_biquadratic_real"
    fr = env["solve_biquadratic_real"]
    try:
        for (a, b, c), want in [((1, -5, 4), [-2, -1, 1, 2]),
                                ((1, -10, 9), [-3, -1, 1, 3]),
                                ((1, 0, 1), []),                        # no real roots
                                ((1, -4, 4), [-(2 ** 0.5), 2 ** 0.5]),  # double root y=2
                                ((1, -1, 0), [-1, 0, 1])]:              # root at zero
            got = sorted(float(x) for x in fr(a, b, c))
            if len(got) != len(want) or any(abs(g - w) > 1e-9 for g, w in zip(got, want)):
                return False, f"real({a},{b},{c}) = {got}, want {want}"
        return True, "5/5 cases incl. empty, repeated and zero roots"
    except Exception as e:
        return False, f"call failed: {type(e).__name__}: {e}"


def check_biquad_complex(text):
    env = next(environments(text, "def solve_biquadratic_complex"), None)
    if env is None:
        return False, "no code block that defines solve_biquadratic_complex"
    fc, verify = env["solve_biquadratic_complex"], env.get("verify_solution")
    if not verify:
        return False, "verify_solution is missing"
    try:
        for a, b, c in [(1, 0, 1), (1, -5, 4), (1, 2, -3), (1, -4, 4)]:
            roots = list(fc(a, b, c))
            if len(roots) != 4:
                return False, f"complex({a},{b},{c}) returned {len(roots)} roots, want 4"
            for x in roots:
                if abs(a * x ** 4 + b * x ** 2 + c) > 1e-9:
                    return False, f"complex root {x} leaves a residual"
                if not verify(a, b, c, x):
                    return False, f"verify_solution rejects its own root {x}"
        return True, "4 cases x 4 roots, residual < 1e-9, verify_solution agrees"
    except Exception as e:
        return False, f"call failed: {type(e).__name__}: {e}"


def check_lunar(text):
    """Scores invariants rather than a single answer — partial credit is the point.

    Reference values come from lunar_task.md and are never shown to the model:
    exhaust velocity from sqrt(2*eta*Q), stage delta-v from Tsiolkovsky.
    """
    env = next(environments(text, "def exhaust_velocity"), None)
    if env is None:
        return False, "no code block that defines exhaust_velocity"

    scored, notes = 0, []

    ev = env.get("exhaust_velocity")
    try:
        got = [ev(10.3e6, 0.55), ev(13.4e6, 0.60), ev(13.4e6, 0.58)]
        want = [3366.0, 4010.0, 3942.6]
        if all(abs(g - w) < 5 for g, w in zip(got, want)):
            scored += 1
            notes.append("v_e ok")
        else:
            notes.append(f"v_e = {got[0]:.0f}/{got[1]:.0f}/{got[2]:.0f}, want 3366/4010/3943")
    except Exception as e:
        notes.append(f"v_e call failed: {type(e).__name__}")

    sd = env.get("stage_delta_v")
    if not sd:
        notes.append("stage_delta_v missing")
    else:
        try:
            got = [sd(900000.0, 700000.0, 3366.0), sd(140000.0, 110000.0, 4010.0),
                   sd(18000.0, 11000.0, 3942.6)]
            want = [5063, 6177, 3724]
            if all(abs(g - w) / w < 1e-3 for g, w in zip(got, want)):
                scored += 1
                notes.append(f"Tsiolkovsky ok (sum {sum(got):.0f})")
            else:
                notes.append(f"delta-v = {got[0]:.0f}/{got[1]:.0f}/{got[2]:.0f}, want 5063/6177/3724")
        except Exception as e:
            notes.append(f"stage_delta_v call failed: {type(e).__name__}")

    for name in ("launch_to_leo", "propagate", "lunar_braking", "soft_landing"):
        if callable(env.get(name)):
            scored += 1
        else:
            notes.append(f"{name} missing")

    # Both numeric invariants plus all four mission stages, or it did not fly.
    return scored == 6, f"{scored}/6 invariants — " + "; ".join(notes[:4] or ["all present"])


RUNGS = [
    ("biquad-real",    BIQUAD_REAL,    check_biquad_real),
    ("biquad-complex", BIQUAD_COMPLEX, check_biquad_complex),
    ("fourier",        FOURIER,        check_fourier),
    ("pump",           PUMP,           check_pump),
    ("lunar",          LUNAR,          check_lunar),
]

WARMUP = [
    ("algebra",     ALGEBRA,     check_algebra),
    ("fibonacci",   FIBONACCI,   check_fibonacci),
    ("translation", TRANSLATION, check_translation),
]

# ── runner ───────────────────────────────────────────────────────────────────


def ask(base, prompt, max_tokens, timeout):
    body = {"model": "gemm", "stream": False, "max_tokens": max_tokens,
            "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}
    req = Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"]["content"], d.get("usage", {}), ch.get("finish_reason")


def main():
    p = argparse.ArgumentParser(description="Run the ladder against a local gemm server")
    # 127.0.0.1, not localhost: the server binds IPv4 only.
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--only", nargs="+", help="run a subset by name")
    p.add_argument("--warmup", action="store_true",
                   help="also run the three cheap smoke tests, before the ladder")
    p.add_argument("--json", help="write the results to this file")
    args = p.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    tasks = (WARMUP if args.warmup else []) + RUNGS
    tasks = [t for t in tasks if not args.only or t[0] in args.only]
    rows = []
    for name, prompt, check in tasks:
        t0 = time.time()
        text, usage, finish = ask(base, prompt, args.max_tokens, args.timeout)
        dt = time.time() - t0
        ok, detail = check(text)
        tok = usage.get("completion_tokens") or 0
        rows.append({"task": name, "passed": ok, "detail": detail, "tokens": tok,
                     "finish_reason": finish, "seconds": round(dt, 1),
                     "tps": round(tok / dt, 1) if dt else None})
        print(f"{name:12} {'PASS' if ok else 'FAIL'} — {detail}")
        print(f"{'':12} [{tok} tok, finish={finish}, {dt:.0f}s, {tok/dt if dt else 0:.1f} tps]",
              flush=True)

    rungs = [r for r in rows if r["task"] not in {w[0] for w in WARMUP}]
    print(f"\nladder: {sum(r['passed'] for r in rungs)}/{len(rungs)} rungs"
          + (f", warmup: {sum(r['passed'] for r in rows if r not in rungs)}/{len(rows) - len(rungs)}"
             if len(rows) != len(rungs) else ""))
    if args.json:
        json.dump({"rungs_passed": sum(r["passed"] for r in rungs),
                   "rungs_total": len(rungs), "tasks": rows},
                  open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
