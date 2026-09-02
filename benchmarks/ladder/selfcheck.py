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

checks = [
    ("algebra accepts the answer",        ladder.check_algebra("Therefore x = 5."),            True),
    ("algebra rejects a wrong answer",    ladder.check_algebra("Therefore x = 7."),            False),
    ("fibonacci finds the real block",    ladder.check_fibonacci(FIB_GOOD),                    True),
    ("fibonacci rejects a fake",          ladder.check_fibonacci(FIB_BAD),                     False),
    ("translation wants all three",       ladder.check_translation("French: … German: … Spanish: …"), True),
    ("translation catches a missing one", ladder.check_translation("French: … German: …"),     False),
    ("pump catches the mm/m bug",         ladder.check_pump(PUMP_WRONG_UNITS),                 False),
    ("biquadratic accepts a correct one", ladder.check_biquadratic(BIQUAD_GOOD),               True),
    ("fourier needs a code block",        ladder.check_fourier("I would use numpy."),          False),
]

failed = 0
for (label, (got, detail), expected) in checks:
    ok = got is expected
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label} — {detail}")

assert not failed, f"{failed} validator check(s) failed"
print(f"\n{len(checks)}/{len(checks)} validator checks passed")
