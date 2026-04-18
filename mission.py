"""
Artemis II Mission Planner
Verified free-return trajectory parameters (N-body simulation).

Key findings from trajectory search:
  - TLI apoapsis: 500,000 km (v_TLI = 10.962 km/s from 170 km parking orbit)
  - Moon initial phase: 130.2° at TLI
  - TLI declination: 5.25° out-of-plane (close to Moon's orbital inclination 5.145°)
  - Perilune: 6,572 km from Moon center (4,835 km above surface)
  - Reentry: T + 7.42 days at 11.147 km/s (real Artemis II: ~11 km/s)
  - Mechanism: Moon retrograde flyby decelerates spacecraft → free return

Real Artemis II (for reference):
  - Total mission: ~10.5 days (ours: 7.42 days — difference due to simplified Moon model)
  - Perilune: ~6,545 km from Moon center (~4,808 km above surface)
  - Entry speed: ~11.0 km/s
"""

import numpy as np
from physics import (
    MOON_PERIOD,
    MOON_SEMI_MAJOR,
    MU_EARTH,
    R_EARTH,
    R_MOON,
    closest_approach,
    moon_position,
    propagate,
    reentry_index,
)
from scipy.optimize import minimize

# ── Verified baseline parameters (from trajectory search) ─────────────────────
PARKING_ORBIT_ALT = 170.0
PARKING_R = R_EARTH + PARKING_ORBIT_ALT  # ~6548 km

R_APO_DEFAULT = 500000.0  # km — apoapsis of outbound ellipse
A_DEFAULT = (PARKING_R + R_APO_DEFAULT) / 2.0
V_TLI_DEFAULT = np.sqrt(MU_EARTH * (2.0 / PARKING_R - 1.0 / A_DEFAULT))  # ~10.962 km/s

MOON_PHASE0_DEG = 130.2  # Moon's initial orbital angle at TLI (degrees)
TLI_DEC_DEG = 5.25  # Out-of-plane launch declination (degrees)

MISSION_DURATION = 10.5 * 86400  # simulate 10.5 days


def tli_state(dec_deg=TLI_DEC_DEG, v_mag=V_TLI_DEFAULT + 0.00):
    """
    Build TLI state vector.
    Spacecraft at [PARKING_R, 0, 0], velocity primarily in +Y (tangential/prograde)
    with a z-component for out-of-plane launch.

      vy = v * cos(dec)   — in-plane tangential
      vz = v * sin(dec)   — out-of-plane
    """
    dec = np.radians(dec_deg)
    return np.array(
        [PARKING_R, 0.0, 0.0, 0.0, v_mag * np.cos(dec), v_mag * np.sin(dec)]
    )


def score_trajectory(params):
    """
    Objective for local optimization: minimize |perilune - 6545 km|.
    params = [dec_deg, moon_phase0_deg]
    """
    dec, phase_deg = params
    state0 = tli_state(dec, V_TLI_DEFAULT)
    phase0 = np.radians(phase_deg)
    try:
        t, states = propagate(
            state0,
            (0, 10 * 86400),
            moon_phase0=phase0,
            n_points=15000,
            include_sun=True,
            rtol=1e-9,
            atol=1e-9,
        )
        _, t_fly, d_fly = closest_approach(t, states, moon_phase0=phase0)
        if t_fly < 2 * 86400 or d_fly < R_MOON:
            return 1e9
        idx_re = reentry_index(t, states, altitude_km=120.0, min_days=2.0)
        return_pen = 0.0 if idx_re else 2e5
        return abs(d_fly - 6545.0) + return_pen
    except Exception:
        return 1e9


def optimize_free_return(verbose=True):
    """Local optimization from known good starting point."""
    if verbose:
        print("[Optimizer] Refining from verified baseline...")
    x0 = [TLI_DEC_DEG, MOON_PHASE0_DEG]
    result = minimize(
        score_trajectory,
        x0,
        method="Nelder-Mead",
        options={"xatol": 0.005, "fatol": 1.0, "maxiter": 200, "disp": verbose},
    )
    dec, phase_deg = result.x
    if verbose:
        print(
            f"  Optimized: dec={dec:.3f}°  phase={phase_deg:.3f}°  score={result.fun:.2f} km"
        )
    return dec, phase_deg


def build_full_trajectory(dec=None, phase0_deg=None, optimize=True):
    """
    Build the complete Artemis II free-return trajectory.
    Returns (t, states, meta_dict).
    """
    if optimize:
        dec, phase0_deg = optimize_free_return(verbose=True)
    else:
        if dec is None:
            dec = TLI_DEC_DEG
        if phase0_deg is None:
            phase0_deg = MOON_PHASE0_DEG

    state0 = tli_state(dec)
    phase0 = np.radians(phase0_deg)
    speed = np.linalg.norm(state0[3:6])

    print(f"\n[Mission] TLI state (post-burn):")
    print(f"  Position  : {state0[0:3]}  km")
    print(f"  Velocity  : {state0[3:6]}  km/s  (|v|={speed:.3f} km/s)")
    print(f"  Moon phase at TLI: {phase0_deg:.2f}°")
    print(f"  Apoapsis (2-body, no Moon):  {R_APO_DEFAULT:,.0f} km")

    print(
        f"\n[Mission] Propagating {MISSION_DURATION / 86400:.1f}-day trajectory "
        f"(100k points, DOP853, with Sun gravity)..."
    )

    t, states = propagate(
        state0,
        (0, MISSION_DURATION),
        moon_phase0=phase0,
        n_points=100000,
        include_sun=True,
        rtol=1e-10,
        atol=1e-10,
    )

    fly_idx, t_fly, d_fly = closest_approach(t, states, moon_phase0=phase0)
    idx_re = reentry_index(t, states, altitude_km=120.0, min_days=2.0)
    v_reentry = np.linalg.norm(states[3:6, idx_re]) if idx_re else None

    meta = {
        "dec_deg": dec,
        "moon_phase0_deg": phase0_deg,
        "moon_phase0_rad": phase0,
        "tli_speed_km_s": speed,
        "perilune_time_s": t_fly,
        "perilune_dist_km": d_fly,
        "perilune_alt_km": d_fly - R_MOON,
        "flyby_index": fly_idx,
        "reentry_index": idx_re,
        "reentry_time_s": t[idx_re] if idx_re else None,
        "reentry_speed_km_s": v_reentry,
    }

    print(f"\n[Mission] Results:")
    print(f"  Perilune distance  : {d_fly:.1f} km from Moon center")
    print(f"  Perilune altitude  : {d_fly - R_MOON:.1f} km above Moon surface")
    print(f"  Perilune at T+     : {t_fly / 3600:.2f} h  ({t_fly / 86400:.3f} days)")
    if idx_re:
        print(
            f"  Re-entry at T+     : {t[idx_re] / 3600:.2f} h  ({t[idx_re] / 86400:.3f} days)"
        )
        print(f"  Re-entry speed     : {v_reentry:.3f} km/s")
    else:
        print("  [!] No re-entry in simulation window.")

    return t, states, meta
