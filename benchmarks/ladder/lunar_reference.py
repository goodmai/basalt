"""A reference answer to the lunar rung, used only by selfcheck.py.

It is never shown to a model. It exists so the flight checks in run.py are proven
passable, and so a change to a check that breaks a correct answer fails selfcheck.
"""
import math

MU_SUN, MU_EARTH, MU_MOON = 1.32712440018e20, 3.986004418e14, 4.9048695e12
R_EARTH, R_MOON, OMEGA_EARTH = 6_371_000.0, 1_737_400.0, 7.292115e-5
RHO0, SCALE_HEIGHT, DRAG_COEFFICIENT, AREA = 1.225, 8500.0, 0.3, 30.0
LANDER_THRUST = 66_000.0


def exhaust_velocity(Q, eta):
    return math.sqrt(2 * eta * Q)


def stage_delta_v(m0, m_fuel, v_e):
    return v_e * math.log(m0 / (m0 - m_fuel))


def launch_to_leo(lat_deg, lon_deg, target_alt_m, azimuth_deg):
    """Vertical rise then a linear pitch-over, stage 1 only, integrated for the losses."""
    v_e, mass, thrust, fuel = exhaust_velocity(10.3e6, 0.55), 900_000.0, 12.5e6, 700_000.0
    h = v = 0.0
    gravity_loss = drag_loss = 0.0
    dt, t = 0.1, 0.0
    while fuel > 0 and h < target_alt_m:
        pitch = math.radians(90.0 if t < 15 else max(0.0, 90.0 - (t - 15) * 0.6))
        g = MU_EARTH / (R_EARTH + h) ** 2
        drag = 0.5 * RHO0 * math.exp(-h / SCALE_HEIGHT) * v * v * DRAG_COEFFICIENT * AREA / mass
        gravity_loss += g * math.sin(pitch) * dt
        drag_loss += drag * dt
        v += (thrust / mass - drag - g * math.sin(pitch)) * dt
        h += v * math.sin(pitch) * dt
        burned = thrust / v_e * dt
        mass -= burned
        fuel -= burned
        t += dt
    rotation_gain = (OMEGA_EARTH * R_EARTH * math.cos(math.radians(lat_deg))
                     * math.sin(math.radians(azimuth_deg)))
    v_circ = math.sqrt(MU_EARTH / (R_EARTH + target_alt_m))
    return v_circ + gravity_loss + drag_loss - rotation_gain, gravity_loss, drag_loss, rotation_gain


def _derivative(y):
    def pull(at, source, mu):
        d = [s - a for a, s in zip(at, source)]
        k = mu / math.sqrt(sum(x * x for x in d)) ** 3
        return [k * x for x in d]
    e, m, c = y[0:3], y[6:9], y[12:15]
    sun = [0.0, 0.0, 0.0]
    a_e = [p + q for p, q in zip(pull(e, sun, MU_SUN), pull(e, m, MU_MOON))]
    a_m = [p + q for p, q in zip(pull(m, sun, MU_SUN), pull(m, e, MU_EARTH))]
    a_c = [p + q + r for p, q, r in zip(pull(c, sun, MU_SUN), pull(c, e, MU_EARTH), pull(c, m, MU_MOON))]
    return y[3:6] + a_e + y[9:12] + a_m + y[15:18] + a_c


def _rk4(y, h):
    k1 = _derivative(y)
    k2 = _derivative([a + h / 2 * b for a, b in zip(y, k1)])
    k3 = _derivative([a + h / 2 * b for a, b in zip(y, k2)])
    k4 = _derivative([a + h * b for a, b in zip(y, k3)])
    return [a + h / 6 * (p + 2 * q + 2 * r + s) for a, p, q, r, s in zip(y, k1, k2, k3, k4)]


def propagate(state0, t_span, dt_max):
    """RK4 with step doubling: the step halves until two half steps agree with one full."""
    names = ("earth", "moon", "craft")

    def unpack(y):
        return {n: (y[6 * i:6 * i + 3], y[6 * i + 3:6 * i + 6]) for i, n in enumerate(names)}

    y = [x for n in names for part in state0[n] for x in part]
    t, t_end = t_span
    h = dt_max
    out = [(t, unpack(y))]
    while t < t_end:
        h = min(h, dt_max, t_end - t)
        full = _rk4(y, h)
        half = _rk4(_rk4(y, h / 2), h / 2)
        error = max(abs(a - b) for a, b in zip(full[12:15], half[12:15]))
        if error > 1.0 and h > 1.0:
            h /= 2
            continue
        y, t = half, t + h
        out.append((t_end if t_end - t < 1e-9 else t, unpack(y)))
        if error < 1e-2:
            h *= 2
    return out


def lunar_braking(approach_state, target_orbit_alt):
    r, v = approach_state
    speed = math.sqrt(sum(x * x for x in v))
    dv = speed - math.sqrt(MU_MOON / (R_MOON + target_orbit_alt))
    v_e = exhaust_velocity(13.4e6, 0.58)
    fuel = 18_000.0 * (1 - math.exp(-dv / v_e))
    return dv, fuel * v_e / LANDER_THRUST, fuel


def soft_landing(initial_state, controller=None):
    """Outer loop at 1 Hz picks the thrust vector, inner loop at 0.05 s turns to it."""
    h, vz, vh, m = initial_state
    m_start = m
    v_e = exhaust_velocity(13.4e6, 0.58)
    dt, t = 0.05, 0.0
    tilt = tilt_cmd = 0.0
    throttle = 0.0
    telemetry = []
    step = 0
    while True:
        g = MU_MOON / (R_MOON + h) ** 2
        if step % 20 == 0:
            if controller is not None:
                throttle, tilt_cmd = controller((h, vz, vh, m))
            else:
                profile = math.sqrt(2 * 0.8 * max(h, 0.0))
                v_ref = -max(0.8, profile)
                feedforward = 0.8 if profile > 0.8 else 0.0
                a_vertical = max(0.3, g + feedforward + 0.8 * (v_ref - vz))
                a_horizontal = min(0.5 * vh, 0.5 * a_vertical)
                wanted = m * math.hypot(a_vertical, a_horizontal) / LANDER_THRUST
                throttle = min(1.0, max(0.1, wanted))
                tilt_cmd = math.degrees(math.atan2(a_horizontal, max(a_vertical, 1e-6)))
        tilt += max(-5 * dt, min(5 * dt, tilt_cmd - tilt))
        telemetry.append((t, h, vz, vh, m, throttle, tilt))
        thrust = throttle * LANDER_THRUST
        a_z = thrust * math.cos(math.radians(tilt)) / m - g
        h_next = h + vz * dt + 0.5 * a_z * dt * dt
        if h_next <= 0:
            root = (-vz - math.sqrt(max(vz * vz - 2 * a_z * h, 0.0))) / a_z if abs(a_z) > 1e-9 else -h / vz
            dt = min(dt, max(root, 1e-6))
            h_next = 0.0
        vz += a_z * dt
        vh -= thrust * math.sin(math.radians(tilt)) / m * dt
        m -= thrust / v_e * dt
        h, t, step = h_next, t + dt, step + 1
        if h <= 0:
            telemetry.append((t, h, vz, vh, m, throttle, tilt))
            return (h, vz, vh, m), m_start - m, telemetry
