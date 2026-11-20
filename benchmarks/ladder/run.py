#!/usr/bin/env python3
"""The ladder to the moon: five rungs against a running `basalt serve --rest`.

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

    basalt serve --model <id> --rest &
    python3 benchmarks/ladder/run.py --port 8080

`--flight` appends one more step after the ladder, outside its score: the same
lunar mission with the data formats fixed, so the grader can run the four stages
and check them against physics it computes itself (`flight k/7`). See
lunar_task.md for the table of checks.

`--warmup` prepends three cheap smoke tests (algebra, Fibonacci, translation)
that any instruct model should clear; they are not part of the ladder score.

For reasoning models whose template opens <think> on every turn, start the
server with `--reasoning-effort none`: otherwise they routinely spend the whole
token budget thinking and never emit an answer, which shows up here as
finish=length.
"""
import argparse, json, math, re, signal, time
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

# The flight step asks the same mission and also fixes the data formats it will be run with.
_REQUIREMENTS = "## Требования к решению"
assert _REQUIREMENTS in LUNAR
LUNAR_FLIGHT = LUNAR.replace(
    _REQUIREMENTS,
    Path(__file__).with_name("lunar_flight_formats.md").read_text(encoding="utf-8") + _REQUIREMENTS, 1)

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


EXEC_TIMEOUT_S = 60


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
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(EXEC_TIMEOUT_S)
        try:
            exec(b, env)
        except Exception:
            continue
        finally:
            signal.alarm(0)
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


# ── lunar flight: the mission stages are run, not just found ─────────────────
# A separate, opt-in step (--flight). The lunar rung above only finds the four stage
# functions; this one calls them. Its prompt is the lunar prompt plus the data formats
# in lunar_flight_formats.md, so the rung's own prompt and score stay as published.
# Every check runs the model's code on inputs the prompt never showed and compares
# against physics the grader computes itself.

MU_SUN, MU_EARTH, MU_MOON = 1.32712440018e20, 3.986004418e14, 4.9048695e12
R_EARTH, R_MOON, OMEGA_EARTH = 6_371_000.0, 1_737_400.0, 7.292115e-5
LANDER_THRUST, LANDER_VE = 66_000.0, math.sqrt(2 * 0.58 * 13.4e6)
LANDER_M0, LANDER_FUEL, LANDER_DRY = 18_000.0, 11_000.0, 7_000.0
EARTH0 = ([1.491694e11, 1.472995e10, 0.0], [-3411.2, 29530.5, 0.0])
MOON_REL0 = ([-3.5900e8, 6.0167e7, -2.1359e7], [-191.66, -1059.53, -70.86])
# Outbound toward the Moon from 60,000 km: J2 is negligible there, the Moon is not.
CRAFT_REL0 = ([-5.909e7, 9.9e6, -3.5e6], [-3258.0, -61.0, -227.0])
LANDING0 = (2000.0, -40.0, 30.0, 9000.0)


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def _timed(fn, *args, seconds=120):
    """Model code may never return; a rung must."""
    def expired(*_):
        raise TimeoutError(f"no result in {seconds}s")
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _four_body_state0():
    earth = EARTH0
    return {"earth": earth,
            "moon": (_add(earth[0], MOON_REL0[0]), _add(earth[1], MOON_REL0[1])),
            "craft": (_add(earth[0], CRAFT_REL0[0]), _add(earth[1], CRAFT_REL0[1]))}


def _accelerations(r):
    """Sun fixed at the origin; Earth and Moon pull each other; the craft is a test mass."""
    def pull(at, source, mu):
        d = [s - a for a, s in zip(at, source)]
        k = mu / _norm(d) ** 3
        return [k * x for x in d]
    zero = [0.0, 0.0, 0.0]
    e, m, c = r
    return [_add(pull(e, zero, MU_SUN), pull(e, m, MU_MOON)),
            _add(pull(m, zero, MU_SUN), pull(m, e, MU_EARTH)),
            _add(_add(pull(c, zero, MU_SUN), pull(c, e, MU_EARTH)), pull(c, m, MU_MOON))]


def _reference_propagation(state0, seconds, dt=30.0):
    """Fixed-step RK4, small enough that its own error is metres at these scales."""
    names = ("earth", "moon", "craft")
    r = [list(state0[n][0]) for n in names]
    v = [list(state0[n][1]) for n in names]

    def shifted(base, delta, h):
        return [[b + h * d for b, d in zip(bb, dd)] for bb, dd in zip(base, delta)]

    for _ in range(round(seconds / dt)):
        k1r, k1v = v, _accelerations(r)
        k2r, k2v = shifted(v, k1v, dt / 2), _accelerations(shifted(r, k1r, dt / 2))
        k3r, k3v = shifted(v, k2v, dt / 2), _accelerations(shifted(r, k2r, dt / 2))
        k4r, k4v = shifted(v, k3v, dt), _accelerations(shifted(r, k3r, dt))
        for i in range(3):
            for j in range(3):
                r[i][j] += dt / 6 * (k1r[i][j] + 2 * k2r[i][j] + 2 * k3r[i][j] + k4r[i][j])
                v[i][j] += dt / 6 * (k1v[i][j] + 2 * k2v[i][j] + 2 * k3v[i][j] + k4v[i][j])
    return dict(zip(names, r))


def _system_energy_and_momentum(state):
    """Earth + Moon in the fixed Sun's field: both are conserved exactly by the physics."""
    (re, ve), (rm, vm) = state["earth"], state["moon"]
    d = _norm([a - b for a, b in zip(re, rm)])
    energy = (MU_EARTH * (0.5 * _norm(ve) ** 2 - MU_SUN / _norm(re))
              + MU_MOON * (0.5 * _norm(vm) ** 2 - MU_SUN / _norm(rm))
              - MU_EARTH * MU_MOON / d)
    lz = (MU_EARTH * (re[0] * ve[1] - re[1] * ve[0]) + MU_MOON * (rm[0] * vm[1] - rm[1] * vm[0]))
    return energy, lz


def flight_launch(env):
    lat, alt = -2.373, 200_000.0
    dv, gravity, drag, rotation = _timed(env["launch_to_leo"], lat, -44.396, alt, 90.0)
    want_rotation = OMEGA_EARTH * R_EARTH * math.cos(math.radians(lat))
    if abs(rotation - want_rotation) > 5:
        return f"rotation gain {rotation:.0f}, want {want_rotation:.0f} m/s"
    if not (gravity > 0 and drag > 0):
        return f"losses must be positive, got gravity {gravity:.0f}, drag {drag:.0f}"
    budget = math.sqrt(MU_EARTH / (R_EARTH + alt)) + gravity + drag - rotation
    if abs(dv - budget) > 1:
        return f"dv_required {dv:.0f} is not v_circ + losses - rotation = {budget:.0f}"


def _propagated(env, cache={}):
    # Keyed by the namespace itself, not id(): a freed namespace's id can be reused by the next answer.
    if cache.get("env") is not env:
        cache.clear()
        cache["trajectory"] = list(_timed(env["propagate"], _four_body_state0(), (0.0, 2 * 86400.0), 300.0))
        cache["env"] = env
    return cache["trajectory"]


def flight_transfer(env):
    trajectory = _propagated(env)
    t_end, final = trajectory[-1]
    if abs(t_end - 2 * 86400.0) > 1e-3:
        return f"trajectory ends at t = {t_end:.0f}s, want 172800"
    want = _reference_propagation(_four_body_state0(), 2 * 86400.0)
    for name, limit in (("earth", 50e3), ("moon", 50e3), ("craft", 100e3)):
        miss = _norm([a - b for a, b in zip(final[name][0], want[name])])
        if miss > limit:
            return f"{name} is {miss / 1e3:.0f} km from the reference after 2 days"


def flight_conservation(env):
    trajectory = _propagated(env)
    if len(trajectory) < 10:
        return f"only {len(trajectory)} trajectory samples"
    e0, l0 = _system_energy_and_momentum(trajectory[0][1])
    drift_e = max(abs(_system_energy_and_momentum(s)[0] - e0) for _, s in trajectory) / abs(e0)
    drift_l = max(abs(_system_energy_and_momentum(s)[1] - l0) for _, s in trajectory) / abs(l0)
    if drift_e > 1e-6 or drift_l > 1e-6:
        return f"energy drift {drift_e:.1e}, angular momentum drift {drift_l:.1e}, limit 1e-6"


def flight_braking(env):
    alt, speed = 100_000.0, 2500.0
    r_p = R_MOON + alt
    dv, duration, fuel = _timed(env["lunar_braking"], ([r_p, 0.0, 0.0], [0.0, speed, 0.0]), alt)
    want = speed - math.sqrt(MU_MOON / r_p)
    if not (0.999 * want <= dv <= 1.03 * want):
        return f"braking dv {dv:.0f}, the orbit energy change needs {want:.0f} m/s"
    want_fuel = LANDER_M0 * (1 - math.exp(-dv / LANDER_VE))
    if abs(fuel - want_fuel) > 0.01 * want_fuel or fuel > LANDER_FUEL:
        return f"fuel {fuel:.0f} kg, Tsiolkovsky for that dv gives {want_fuel:.0f}"
    if duration < 0.98 * fuel * LANDER_VE / LANDER_THRUST:
        return f"burn of {duration:.0f}s is shorter than full thrust allows"


def _landed(env, cache={}):
    if cache.get("env") is not env:
        cache.clear()
        touchdown, fuel, telemetry = _timed(env["soft_landing"], LANDING0, None)
        cache["landing"] = (tuple(touchdown), fuel, [tuple(row) for row in telemetry])
        cache["env"] = env
    return cache["landing"]


def flight_touchdown(env):
    if flight_replay(env):
        return "the telemetry does not replay, so the touchdown it reports is not credited"
    (h, vz, vh, _), _, telemetry = _landed(env)
    if abs(h) > 1:
        return f"stopped at altitude {h:.1f} m"
    if math.hypot(vz, vh) >= 2 or abs(vz) >= 1.5:
        return f"touchdown at {math.hypot(vz, vh):.2f} m/s (vertical {vz:.2f}), limits 2 and 1.5"
    if abs(telemetry[-1][6]) >= 5:
        return f"touchdown tilt {telemetry[-1][6]:.1f} deg, limit 5"


def flight_limits(env):
    if flight_replay(env):
        return "the telemetry does not replay, so its limits are not credited"
    (_, _, _, m_end), fuel, telemetry = _landed(env)
    if abs(fuel - (LANDING0[3] - m_end)) > 1:
        return f"fuel_used {fuel:.0f} kg, mass fell by {LANDING0[3] - m_end:.0f}"
    for before, after in zip(telemetry, telemetry[1:]):
        dt = after[0] - before[0]
        if not (0 < dt <= 0.05 + 1e-9):
            return f"inner loop step {dt:.3f}s at t = {before[0]:.1f}, limit 0.05"
        if after[4] > before[4] + 1e-9 or after[4] < LANDER_DRY:
            return f"mass {after[4]:.0f} kg at t = {after[0]:.1f}"
        if abs(after[6] - before[6]) > 5 * dt * 1.001 + 1e-9:
            return f"tilt turns {abs(after[6] - before[6]) / dt:.1f} deg/s at t = {before[0]:.1f}, limit 5"
    for row in telemetry:
        if not (row[5] == 0 or 0.1 - 1e-9 <= row[5] <= 1 + 1e-9):
            return f"throttle {row[5]:.3f} at t = {row[0]:.1f}, allowed 0 or 0.1..1"


def flight_replay(env):
    """Fly the reported controls through the grader's own dynamics."""
    (h_end, vz_end, vh_end, m_end), _, telemetry = _landed(env)
    _, h, vz, vh, m = telemetry[0][:5]
    for before, after in zip(telemetry, telemetry[1:]):
        dt, thrust, tilt = after[0] - before[0], before[5] * LANDER_THRUST, math.radians(before[6])
        for _ in range(4):
            step = dt / 4
            g = MU_MOON / (R_MOON + h) ** 2
            h += vz * step + 0.5 * (thrust * math.cos(tilt) / m - g) * step * step
            vz += (thrust * math.cos(tilt) / m - g) * step
            vh -= thrust * math.sin(tilt) / m * step
            m -= thrust / LANDER_VE * step
    if abs(h - h_end) > 20 or abs(vz - vz_end) > 1 or abs(vh - vh_end) > 1:
        return (f"replayed controls end at h {h:.0f} m, v {vz:.1f}/{vh:.1f} m/s; "
                f"reported {h_end:.0f} m, {vz_end:.1f}/{vh_end:.1f}")
    if abs(m - m_end) > 0.005 * m_end:
        return f"replayed mass {m:.0f} kg, reported {m_end:.0f}"


FLIGHT_CHECKS = [
    ("launch", flight_launch), ("transfer", flight_transfer),
    ("conservation", flight_conservation), ("braking", flight_braking),
    ("touchdown", flight_touchdown), ("limits", flight_limits), ("replay", flight_replay),
]


def lunar_flight(env):
    score, notes = 0, []
    for name, check in FLIGHT_CHECKS:
        try:
            problem = check(env)
        except Exception as e:
            problem = f"{type(e).__name__}: {e}"[:90]
        if problem:
            notes.append(f"{name}: {problem}")
        else:
            score += 1
    return score, notes


def check_lunar_flight(text):
    rung_ok, rung_detail = check_lunar(text)
    env = next(environments(text, "def exhaust_velocity"), None)
    if env is None:
        return False, rung_detail
    flight, notes = lunar_flight(env)
    detail = f"flight {flight}/{len(FLIGHT_CHECKS)}; {rung_detail.split(' — ')[0]}"
    if notes:
        detail += " — " + "; ".join(notes[:3])
    return rung_ok and flight == len(FLIGHT_CHECKS), detail


RUNGS = [
    ("biquad-real",    BIQUAD_REAL,    check_biquad_real),
    ("biquad-complex", BIQUAD_COMPLEX, check_biquad_complex),
    ("fourier",        FOURIER,        check_fourier),
    ("pump",           PUMP,           check_pump),
    ("lunar",          LUNAR,          check_lunar),
]

# Opt-in with --flight, after the rungs, and outside the ladder score like the warm-up.
FLIGHT = [
    ("lunar-flight", LUNAR_FLIGHT, check_lunar_flight),
]

WARMUP = [
    ("algebra",     ALGEBRA,     check_algebra),
    ("fibonacci",   FIBONACCI,   check_fibonacci),
    ("translation", TRANSLATION, check_translation),
]

# ── runner ───────────────────────────────────────────────────────────────────


def ask(base, prompt, max_tokens, timeout):
    body = {"model": "basalt", "stream": False, "max_tokens": max_tokens,
            "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}
    req = Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"]["content"], d.get("usage", {}), ch.get("finish_reason")


def main():
    p = argparse.ArgumentParser(description="Run the ladder against a local basalt server")
    # 127.0.0.1, not localhost: the server binds IPv4 only.
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--only", nargs="+", help="run a subset by name")
    p.add_argument("--warmup", action="store_true",
                   help="also run the three cheap smoke tests, before the ladder")
    p.add_argument("--flight", action="store_true",
                   help="also run the lunar flight step after the ladder; it is not part of the score")
    p.add_argument("--json", help="write the results to this file")
    args = p.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    tasks = (WARMUP if args.warmup else []) + RUNGS + (FLIGHT if args.flight or "lunar-flight" in (args.only or []) else [])
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

    def named(group):
        return [r for r in rows if r["task"] in {t[0] for t in group}]

    rungs, warmup, flight = named(RUNGS), named(WARMUP), named(FLIGHT)
    print(f"\nladder: {sum(r['passed'] for r in rungs)}/{len(rungs)} rungs"
          + (f", warmup: {sum(r['passed'] for r in warmup)}/{len(warmup)}" if warmup else "")
          + (f", flight: {'PASS' if flight[0]['passed'] else 'FAIL'}" if flight else ""))
    if args.json:
        json.dump({"rungs_passed": sum(r["passed"] for r in rungs),
                   "rungs_total": len(rungs), "tasks": rows},
                  open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
