#!/usr/bin/env python3
"""Check the ladder's validators without a model or a GPU: python3 selfcheck.py

The validators are the part that can silently lie — a grader that picks the
wrong code block fails a correct answer, and one that is too loose passes a
wrong one. Both directions are asserted here on fixed answers.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ladder", Path(__file__).with_name("run.py"))
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)

# A real answer shape: the working implementation first, then an abbreviated
# snippet. Grading only the last block would fail this — it did, once.
FIB_GOOD = """Here you go:

```python
def fibonacci(n, memo=None):
    if memo is None:
        memo = {}
    if n < 2:
        return n
    if n not in memo:
        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

print(fibonacci(10))
```

If you want 1-indexing instead:

```python
def fibonacci_1indexed(n):
    if n <= 0:
        return 0
    # ... rest of the logic
```
"""

FIB_BAD = """```python
def fibonacci(n):
    return n * 2
```
"""

PUMP_WRONG_UNITS = """```python
import math

def pump_head(length_m, diameter_mm, roughness_mm, lift_m, flow_lps):
    d = diameter_mm          # bug: already treated as metres
    v = (flow_lps / 1000.0) / (math.pi * d * d / 4.0)
    return v, 0.0, 0.02, 0.0, lift_m
```
"""

BIQUAD_GOOD = """```python
import cmath, math

def solve_biquadratic_real(a, b, c):
    roots = []
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    for y in {(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)}:
        if y > 0:
            roots += [math.sqrt(y), -math.sqrt(y)]
        elif y == 0:
            roots.append(0.0)
    return sorted(set(roots))

def solve_biquadratic_complex(a, b, c):
    disc = cmath.sqrt(b * b - 4 * a * c)
    out = []
    for y in ((-b + disc) / (2 * a), (-b - disc) / (2 * a)):
        r = cmath.sqrt(y)
        out += [r, -r]
    return out
```
"""

# The complex rung wants a second function too; this fixture has both.
BIQUAD_FULL = BIQUAD_GOOD + """
```python
import cmath

def solve_biquadratic_complex(a, b, c):
    disc = cmath.sqrt(b * b - 4 * a * c)
    out = []
    for y in ((-b + disc) / (2 * a), (-b - disc) / (2 * a)):
        r = cmath.sqrt(y)
        out += [r, -r]
    return out

def verify_solution(a, b, c, x, tol=1e-10):
    return abs(a * x ** 4 + b * x ** 2 + c) < tol
```
"""

REFERENCE = Path(__file__).with_name("lunar_reference.py").read_text()


def lunar_answer(source):
    return "Mission code:\n\n```python\n" + source + "\n```\n"


def broken(old, new):
    assert old in REFERENCE, old
    return lunar_answer(REFERENCE.replace(old, new))


def flight(answer, wanted):
    """(does the detail carry this flight verdict, detail): the rung's own pass rule is separate."""
    _, detail = ladder.check_lunar_flight(answer)
    return wanted in detail, detail


# Every stage is present and the two engine numbers are right, so the rung's six
# invariants pass. Nothing flies. This is the answer the flight score exists to expose.
LUNAR_STUB = lunar_answer("""import math

def exhaust_velocity(Q, eta):
    return math.sqrt(2 * eta * Q)

def stage_delta_v(m0, m_fuel, v_e):
    return v_e * math.log(m0 / (m0 - m_fuel))

def launch_to_leo(lat_deg, lon_deg, target_alt_m, azimuth_deg):
    return 9400.0, 1500.0, 150.0, 465.0

def propagate(state0, t_span, dt_max):
    return [(t_span[0], state0), (t_span[1], state0)]

def lunar_braking(approach_state, target_orbit_alt):
    return 800.0, 300.0, 3000.0

def soft_landing(initial_state, controller=None):
    h, vz, vh, m = initial_state
    return (0.0, 0.0, 0.0, m - 500.0), 500.0, [(0.0, h, vz, vh, m, 1.0, 0.0), (0.05, 0.0, 0.0, 0.0, m - 500.0, 1.0, 0.0)]
""")

LUNAR_NO_MOON_PULL = broken("pull(c, e, MU_EARTH), pull(c, m, MU_MOON))", "pull(c, e, MU_EARTH), [0.0, 0.0, 0.0])")
LUNAR_FREE_FUEL = broken("        m -= thrust / v_e * dt\n", "        m -= 0.5 * thrust / v_e * dt\n")
LUNAR_HARD_LANDING = broken("v_ref = -max(0.8, profile)", "v_ref = -max(6.0, profile)")
LUNAR_SNAP_TURN = broken("tilt += max(-5 * dt, min(5 * dt, tilt_cmd - tilt))", "tilt = tilt_cmd")

SPINS_AT_IMPORT = lunar_answer("""import math

def exhaust_velocity(Q, eta):
    return math.sqrt(2 * eta * Q)

while True:
    pass
""")


def spin_verdict():
    ladder.EXEC_TIMEOUT_S = 1
    try:
        return ladder.check_lunar(SPINS_AT_IMPORT)
    finally:
        ladder.EXEC_TIMEOUT_S = 60


checks = [
    ("algebra accepts the answer",        ladder.check_algebra("Therefore x = 5."),            True),
    ("algebra rejects a wrong answer",    ladder.check_algebra("Therefore x = 7."),            False),
    ("fibonacci finds the real block",    ladder.check_fibonacci(FIB_GOOD),                    True),
    ("fibonacci rejects a fake",          ladder.check_fibonacci(FIB_BAD),                     False),
    ("translation wants all three",       ladder.check_translation("French: … German: … Spanish: …"), True),
    ("translation catches a missing one", ladder.check_translation("French: … German: …"),     False),
    ("pump catches the mm/m bug",         ladder.check_pump(PUMP_WRONG_UNITS),                 False),
    ("biquad-real accepts a correct one", ladder.check_biquad_real(BIQUAD_GOOD),               True),
    ("biquad-complex wants verify_solution", ladder.check_biquad_complex(BIQUAD_GOOD),          False),
    ("biquad-complex accepts the full one", ladder.check_biquad_complex(BIQUAD_FULL),           True),
    ("lunar needs exhaust_velocity",      ladder.check_lunar("I would use a Hohmann transfer."), False),
    ("lunar reference passes the rung",   ladder.check_lunar(lunar_answer(REFERENCE)),         True),
    ("lunar-flight passes the reference", ladder.check_lunar_flight(lunar_answer(REFERENCE)), True),
    ("lunar-flight fails the stub",       ladder.check_lunar_flight(LUNAR_STUB),               False),
    ("lunar rung detail is unchanged",    ("flight" not in ladder.check_lunar(LUNAR_STUB)[1], ladder.check_lunar(LUNAR_STUB)[1]), True),
    ("lunar reference flies 7/7",         flight(lunar_answer(REFERENCE), "flight 7/7"),       True),
    ("lunar stub still passes the rung",  ladder.check_lunar(LUNAR_STUB),                      True),
    ("lunar stub does not fly",           flight(LUNAR_STUB, "flight 0/7"),                    True),
    ("lunar flight catches a missing Moon", flight(LUNAR_NO_MOON_PULL, "transfer: craft is"),  True),
    ("lunar flight catches free fuel",    flight(LUNAR_FREE_FUEL, "replay: replayed"),    True),
    ("lunar flight catches a hard landing", flight(LUNAR_HARD_LANDING, "touchdown: touchdown at"), True),
    ("lunar flight catches a snap turn",  flight(LUNAR_SNAP_TURN, "limits: tilt turns"), True),
    ("a script that never returns is cut off", spin_verdict(),                               False),
    ("fourier needs a code block",        ladder.check_fourier("I would use numpy."),          False),
]

failed = 0
for (label, (got, detail), expected) in checks:
    ok = got is expected
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label} — {detail}")

assert not failed, f"{failed} validator check(s) failed"
print(f"\n{len(checks)}/{len(checks)} validator checks passed")
