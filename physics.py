"""
Artemis II Physics Engine
N-body integrator: Spacecraft under Earth + Moon + Sun gravity.
All units: km, s, km^3/s^2
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── Gravitational parameters ──────────────────────────────────────────────────
MU_EARTH = 398600.4418      # km^3/s^2
MU_MOON  = 4902.8000        # km^3/s^2
MU_SUN   = 1.32712440018e11 # km^3/s^2

R_EARTH  = 6378.137         # km
R_MOON   = 1737.4           # km

# Moon orbital parameters
MOON_SEMI_MAJOR = 384400.0      # km (mean)
MOON_PERIOD     = 27.3217 * 86400.0  # s (sidereal)
MOON_INC        = np.radians(5.145)  # inclination to ecliptic

AU               = 1.496e8       # km
EARTH_SUN_PERIOD = 365.25 * 86400.0

# Moon initial phase (radians at t=0).
# Set so that Moon is ~90° ahead of spacecraft launch direction at TLI,
# which places the Moon at the right position after the ~6.9-day transfer.
# Spacecraft launches from [R,0,0] in +Y direction; Moon needs to be near
# angle = 0° - 90.5° = -90.5° ≈ -π/2 at t=0 (i.e., at [0, -384400, 0])
# so that after ~6.9 days (90.5° of Moon orbit) it arrives at [0, 384400, 0].
# However, a faster (closer to Moon) trajectory is more realistic for Artemis II.
# We use MOON_PHASE_0 as an optimization variable set by the mission planner.
MOON_PHASE_0 = 0.0  # default; overridden by mission.py


def moon_position(t, phase0=None):
    """Moon position vector in Earth-centered inertial frame (km)."""
    p0 = MOON_PHASE_0 if phase0 is None else phase0
    theta = p0 + 2 * np.pi * t / MOON_PERIOD
    x = MOON_SEMI_MAJOR * np.cos(theta)
    y = MOON_SEMI_MAJOR * np.sin(theta) * np.cos(MOON_INC)
    z = MOON_SEMI_MAJOR * np.sin(theta) * np.sin(MOON_INC)
    return np.array([x, y, z])


def sun_position(t):
    """Sun position vector in Earth-centered inertial frame (km)."""
    theta = 2 * np.pi * t / EARTH_SUN_PERIOD
    return np.array([-AU * np.cos(theta), -AU * np.sin(theta), 0.0])


def make_eom(moon_phase0, include_sun=True):
    """Return an equations_of_motion function with the given Moon phase."""
    def equations_of_motion(t, state):
        pos = state[0:3]
        vel = state[3:6]
        r_sc = np.linalg.norm(pos)

        # Earth gravity
        a = -MU_EARTH * pos / r_sc**3

        # Moon gravity (using Battin's indirect term for frame consistency)
        r_moon = moon_position(t, phase0=moon_phase0)
        d_moon = pos - r_moon
        r_dm   = np.linalg.norm(d_moon)
        r_mn   = np.linalg.norm(r_moon)
        a += -MU_MOON * (d_moon / r_dm**3 + r_moon / r_mn**3)

        # Sun gravity
        if include_sun:
            r_sun = sun_position(t)
            d_sun = pos - r_sun
            r_ds  = np.linalg.norm(d_sun)
            r_sn  = np.linalg.norm(r_sun)
            a += -MU_SUN * (d_sun / r_ds**3 + r_sun / r_sn**3)

        return np.concatenate([vel, a])
    return equations_of_motion


def propagate(state0, t_span, moon_phase0=0.0, n_points=50000,
              include_sun=True, rtol=1e-10, atol=1e-10):
    """
    Integrate the equations of motion.
    Returns (t_array, states_6xN).
    Note: for visualization, truncate data at reentry_index to avoid post-impact artifacts.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    eom    = make_eom(moon_phase0, include_sun)
    sol = solve_ivp(
        eom,
        t_span,
        state0,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y


def closest_approach(t_arr, states, moon_phase0=0.0):
    """Return (index, time, distance) of closest approach to Moon."""
    moon_pos = np.array([moon_position(t, phase0=moon_phase0) for t in t_arr])
    sc_pos   = states[0:3, :].T
    diffs    = sc_pos - moon_pos
    dists    = np.linalg.norm(diffs, axis=1)
    idx      = np.argmin(dists)
    return idx, t_arr[idx], dists[idx]


def reentry_index(t_arr, states, altitude_km=120.0, min_days=2.0):
    """Return index where spacecraft first goes below altitude_km after min_days."""
    r_sc      = np.linalg.norm(states[0:3, :], axis=0)
    threshold = R_EARTH + altitude_km
    min_t     = min_days * 86400.0
    for i, ti in enumerate(t_arr):
        if ti >= min_t and r_sc[i] < threshold:
            return i
    return None
