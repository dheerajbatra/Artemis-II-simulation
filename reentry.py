"""
Artemis II Re-entry Heating Simulation
Models the Orion capsule's atmospheric entry from the re-entry interface
(~120 km) down to drogue chute deploy (~8 km), tracking aerodynamic heating
on the AVCOAT heat shield.

Physics (SI units internally — m, s, kg, K):
  • Exponential atmosphere:  rho(h) = rho0 * exp(-h/H)
  • Planar entry with aerodynamic lift (Orion L/D ≈ 0.30, bank fixed):
      dh/dt     = v * sin(gamma)                       [gamma < 0 → descent]
      dv/dt     = -D/m - g * sin(gamma)
      dgamma/dt = (v/(R_E+h) - g/v) * cos(gamma) + L_eff/(m*v)
      where  D     = 0.5 * rho * v^2 * Cd * A
             L_eff = 0.5 * rho * v^2 * Cl * A * cos(bank)     [Cl = Cd * L/D]
  • Stagnation-point convective heat flux (Sutton-Graves, Earth):
      q_dot = 1.7415e-4 * sqrt(rho/Rn) * v^3      [W/m^2]
  • Radiative-equilibrium wall temperature (back-face conduction ignored):
      T_wall = (q_dot / (eps * sigma))^(1/4)      [K]

Orion capsule parameters (public NASA figures):
  mass at entry     ~ 8,900 kg
  heat-shield dia   ~ 5.03 m  ->  A = pi * (d/2)^2  ~ 19.9 m^2
  effective Rn      ~ 6.0 m   (blunt spherical-section)
  drag coefficient  Cd ~ 1.20
  emissivity        eps ~ 0.85 (AVCOAT char layer)
  AVCOAT limit      ~ 3,000 K surface (~2,725 C)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Physical constants ────────────────────────────────────────────────────────
R_EARTH_M = 6_378_137.0           # m
MU_EARTH_M = 3.986004418e14       # m^3/s^2
SIGMA_SB = 5.670374419e-8         # W/m^2/K^4
K_SUTTON_GRAVES = 1.7415e-4       # Earth, SI units

# ── Atmosphere (simple exponential — good to ~100 km for entry heating) ──────
RHO0 = 1.225                      # kg/m^3 at sea level
H_SCALE = 7200.0                  # m (isothermal scale height)

# ── Orion capsule (Block 1) ───────────────────────────────────────────────────
CAPSULE_MASS = 8_900.0            # kg (entry mass)
CAPSULE_DIAMETER = 5.03           # m
CAPSULE_AREA = np.pi * (CAPSULE_DIAMETER / 2.0) ** 2   # m^2
NOSE_RADIUS = 6.0                 # m (effective blunt-body radius)
CD = 1.20                         # drag coefficient
L_OVER_D = 0.30                   # Orion hypersonic L/D (trim)
BANK_DEG = 75.0                   # lift-vector bank angle (positive lift-down component)
EMISSIVITY = 0.85                 # AVCOAT char layer

# AVCOAT heat-shield design limit (surface, short duration)
T_SHIELD_LIMIT_K = 3000.0


def atmosphere_density(h_m):
    """Exponential atmosphere density (kg/m^3) at altitude h (m)."""
    return RHO0 * np.exp(-h_m / H_SCALE)


def gravity(h_m):
    """Gravitational acceleration at altitude h (m)."""
    return MU_EARTH_M / (R_EARTH_M + h_m) ** 2


def heat_flux(rho, v):
    """Sutton-Graves stagnation-point heat flux (W/m^2)."""
    return K_SUTTON_GRAVES * np.sqrt(max(rho, 0.0) / NOSE_RADIUS) * v ** 3


def wall_temperature(q_dot):
    """Radiative-equilibrium wall temperature (K) from heat flux (W/m^2)."""
    if q_dot <= 0:
        return 0.0
    return (q_dot / (EMISSIVITY * SIGMA_SB)) ** 0.25


def reentry_eom(t, y):
    """State y = [h, v, gamma]. Lifting planar entry over a spherical Earth."""
    h, v, gamma = y
    rho = atmosphere_density(h)
    g = gravity(h)
    q_dyn_over_m = 0.5 * rho * v * v * CAPSULE_AREA / CAPSULE_MASS

    drag_acc = q_dyn_over_m * CD
    lift_acc = q_dyn_over_m * CD * L_OVER_D
    # Lift component in the vertical plane (positive upward):
    lift_vertical = lift_acc * np.cos(np.radians(BANK_DEG))

    dh = v * np.sin(gamma)
    dv = -drag_acc - g * np.sin(gamma)
    dgamma = (v / (R_EARTH_M + h) - g / v) * np.cos(gamma) + lift_vertical / v
    return [dh, dv, dgamma]


def simulate_reentry(v_entry_km_s, h_entry_km=120.0, gamma_entry_deg=-6.5,
                     t_max_s=900.0, n_points=4000):
    """
    Integrate the capsule descent.

    Parameters
    ----------
    v_entry_km_s    : inertial speed at interface (km/s) — from lunar return
    h_entry_km      : interface altitude (km)
    gamma_entry_deg : flight-path angle at interface (deg, negative = descending)

    Returns
    -------
    dict with arrays for t, h, v, gamma, rho, q_dot, T_wall, decel_g, heat_load
    """
    y0 = [h_entry_km * 1000.0, v_entry_km_s * 1000.0, np.radians(gamma_entry_deg)]

    def hit_ground(t, y):
        return y[0] - 8_000.0     # stop at 8 km (drogue chute deploy altitude)
    hit_ground.terminal = True
    hit_ground.direction = -1

    def went_up(t, y):
        return y[2]               # gamma crosses 0 → skip-out
    went_up.terminal = True
    went_up.direction = +1

    t_eval = np.linspace(0.0, t_max_s, n_points)
    sol = solve_ivp(
        reentry_eom,
        (0.0, t_max_s),
        y0,
        method="DOP853",
        t_eval=t_eval,
        events=[hit_ground, went_up],
        rtol=1e-9,
        atol=1e-9,
    )

    t = sol.t
    h = sol.y[0]
    v = sol.y[1]
    gamma = sol.y[2]

    rho = atmosphere_density(h)
    q_dot = K_SUTTON_GRAVES * np.sqrt(np.clip(rho, 0, None) / NOSE_RADIUS) * v ** 3
    T_wall = np.where(q_dot > 0, (q_dot / (EMISSIVITY * SIGMA_SB)) ** 0.25, 0.0)

    drag_acc = 0.5 * rho * v * v * CD * CAPSULE_AREA / CAPSULE_MASS
    decel_g = drag_acc / 9.80665

    # Cumulative integrated heat load (J/m^2) — trapezoid rule
    heat_load = np.concatenate(([0.0], np.cumsum(0.5 * (q_dot[1:] + q_dot[:-1]) * np.diff(t))))

    # Downrange (ground-track arc length along Earth's surface)
    dx_ds = v * np.cos(gamma)                                     # horizontal ground speed (m/s)
    downrange_m = np.concatenate(
        ([0.0], np.cumsum(0.5 * (dx_ds[1:] + dx_ds[:-1]) * np.diff(t)))
    )

    return {
        "t": t,
        "h_km": h / 1000.0,
        "v_km_s": v / 1000.0,
        "gamma_deg": np.degrees(gamma),
        "rho": rho,
        "q_dot_W_m2": q_dot,
        "T_wall_K": T_wall,
        "decel_g": decel_g,
        "heat_load_J_m2": heat_load,
        "downrange_km": downrange_m / 1000.0,
        "terminated_by": (
            "ground" if sol.t_events[0].size
            else "skip-out" if sol.t_events[1].size
            else "timeout"
        ),
    }


def summarize(result):
    """Return a dict of peak quantities for logging."""
    q_peak = float(np.max(result["q_dot_W_m2"]))
    i_qpeak = int(np.argmax(result["q_dot_W_m2"]))
    T_peak = float(np.max(result["T_wall_K"]))
    i_tpeak = int(np.argmax(result["T_wall_K"]))
    g_peak = float(np.max(result["decel_g"]))
    i_gpeak = int(np.argmax(result["decel_g"]))
    return {
        "peak_heat_flux_W_m2": q_peak,
        "peak_heat_flux_MW_m2": q_peak / 1e6,
        "peak_heat_flux_altitude_km": float(result["h_km"][i_qpeak]),
        "peak_heat_flux_time_s": float(result["t"][i_qpeak]),
        "peak_wall_temp_K": T_peak,
        "peak_wall_temp_C": T_peak - 273.15,
        "peak_wall_temp_altitude_km": float(result["h_km"][i_tpeak]),
        "peak_decel_g": g_peak,
        "peak_decel_altitude_km": float(result["h_km"][i_gpeak]),
        "total_heat_load_MJ_m2": float(result["heat_load_J_m2"][-1]) / 1e6,
        "final_altitude_km": float(result["h_km"][-1]),
        "final_speed_km_s": float(result["v_km_s"][-1]),
        "duration_s": float(result["t"][-1]),
        "terminated_by": result["terminated_by"],
    }


def print_reentry_log(res, summary):
    print("\n" + "─" * 65)
    print(f"  {'RE-ENTRY HEATING ANALYSIS (Orion + AVCOAT)':<63}")
    print("─" * 65)
    print(f"  Entry speed              : {res['v_km_s'][0]:.3f} km/s")
    print(f"  Entry altitude           : {res['h_km'][0]:.1f} km")
    print(f"  Entry flight-path angle  : {res['gamma_deg'][0]:.2f}°")
    print(f"  Descent duration         : {summary['duration_s']:.1f} s "
          f"({summary['duration_s']/60:.1f} min)  [{summary['terminated_by']}]")
    print()
    print(f"  Peak stag. heat flux     : {summary['peak_heat_flux_MW_m2']:.2f} MW/m²"
          f"  (at h={summary['peak_heat_flux_altitude_km']:.1f} km,"
          f" T+{summary['peak_heat_flux_time_s']:.0f} s)")
    print(f"  Peak shield wall temp    : {summary['peak_wall_temp_K']:.0f} K"
          f"  ({summary['peak_wall_temp_C']:.0f} °C)"
          f"  at h={summary['peak_wall_temp_altitude_km']:.1f} km")
    print(f"  Peak deceleration        : {summary['peak_decel_g']:.2f} g")
    print(f"  Total heat load          : {summary['total_heat_load_MJ_m2']:.1f} MJ/m²")
    print(f"  Shield design limit      : {T_SHIELD_LIMIT_K:.0f} K "
          f"({T_SHIELD_LIMIT_K-273.15:.0f} °C)")
    margin = T_SHIELD_LIMIT_K - summary['peak_wall_temp_K']
    status = "WITHIN LIMIT" if margin > 0 else "EXCEEDED"
    print(f"  Margin vs. limit         : {margin:+.0f} K  [{status}]")
    print("─" * 65)


def plot_reentry(res, summary, filename="artemis_reentry_heating.png"):
    """4-panel heating/trajectory diagnostic plot."""
    t = res["t"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Artemis II — Re-entry Heating on Orion Heat Shield",
                 fontsize=14, fontweight="bold")

    # 1. Altitude vs time
    ax = axes[0, 0]
    ax.plot(t, res["h_km"], color="#1f77b4", lw=2)
    ax.axhline(120, color="gray", ls=":", lw=0.8, label="Interface (120 km)")
    ax.set_xlabel("Time since interface (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude profile")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # 2. Speed vs altitude
    ax = axes[0, 1]
    ax.plot(res["v_km_s"], res["h_km"], color="#2ca02c", lw=2)
    ax.set_xlabel("Speed (km/s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Velocity profile (altitude vs speed)")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # 3. Stagnation heat flux
    ax = axes[1, 0]
    ax.plot(t, res["q_dot_W_m2"] / 1e6, color="#d62728", lw=2)
    ax.axhline(summary["peak_heat_flux_MW_m2"], color="#d62728", ls=":",
               lw=0.8, alpha=0.5)
    ax.set_xlabel("Time since interface (s)")
    ax.set_ylabel("Heat flux (MW/m²)")
    ax.set_title(f"Stagnation-point heat flux  "
                 f"(peak {summary['peak_heat_flux_MW_m2']:.1f} MW/m²)")
    ax.grid(alpha=0.3)

    # 4. Wall temperature
    ax = axes[1, 1]
    T_C = res["T_wall_K"] - 273.15
    ax.plot(t, T_C, color="#ff7f0e", lw=2, label="Shield surface")
    ax.axhline(T_SHIELD_LIMIT_K - 273.15, color="red", ls="--", lw=1.2,
               label=f"AVCOAT limit ({T_SHIELD_LIMIT_K-273.15:.0f} °C)")
    ax.axhline(summary["peak_wall_temp_C"], color="#ff7f0e", ls=":", lw=0.8,
               alpha=0.5)
    ax.set_xlabel("Time since interface (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Heat-shield surface temperature  "
                 f"(peak {summary['peak_wall_temp_C']:.0f} °C)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _temperature_to_rgb(T_K):
    """
    Blackbody-ish colour ramp for the plasma halo.
    Cool → dim red, hot → yellow-white.
    """
    # Normalise 800 K (dark red) ... 3000 K (white-hot)
    u = np.clip((T_K - 800.0) / (3000.0 - 800.0), 0.0, 1.0)
    # Red ramps up fast then saturates, green follows, blue last
    r = np.clip(0.25 + 1.2 * u, 0.0, 1.0)
    g = np.clip(-0.10 + 1.4 * u, 0.0, 1.0)
    b = np.clip(-0.50 + 1.6 * u, 0.0, 1.0)
    return (float(r), float(g), float(b))


def create_reentry_animation(res, summary, filename="artemis_reentry_animation.gif",
                             n_frames=180, fps=30):
    """
    Side-view animation of the capsule descending, with a plasma halo whose
    size scales with heat flux and colour with shield surface temperature.
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    t = res["t"]
    h = res["h_km"]
    dr = res["downrange_km"]
    q = res["q_dot_W_m2"]
    T = res["T_wall_K"]
    v = res["v_km_s"]

    # Resample to n_frames
    idxs = np.linspace(0, len(t) - 1, n_frames).astype(int)

    q_peak = max(np.max(q), 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#050510")
    ax.set_facecolor("#050510")

    # Atmosphere band (fade from space black → thin blue → denser cyan)
    dr_max = dr[-1] * 1.05
    ax.set_xlim(0, dr_max)
    ax.set_ylim(0, 130)
    ax.set_xlabel("Downrange (km)", color="#cccccc")
    ax.set_ylabel("Altitude (km)", color="#cccccc")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333344")
    ax.set_title("Artemis II — Orion Re-entry (plasma halo ∝ heat flux, colour ∝ T_shield)",
                 color="#eeeeee", fontsize=12, pad=12)

    # Shaded atmosphere layers
    for alt_band, col, al in [
        (120, "#0a0a20", 0.0),
        (90,  "#0d2040", 0.25),
        (60,  "#184880", 0.35),
        (30,  "#3070b0", 0.45),
        (0,   "#4c9cd8", 0.55),
    ]:
        ax.axhspan(0, alt_band, color=col, alpha=al, zorder=0)

    # Earth surface
    ax.axhline(0, color="#2a5c2a", lw=2, zorder=1)
    ax.axhspan(-5, 0, color="#1a3a1a", alpha=1, zorder=1)

    # Full flight path (faint)
    ax.plot(dr, h, color="#666677", lw=0.8, alpha=0.5, zorder=2)

    # Animated trail (brighter recent path)
    trail_line, = ax.plot([], [], color="#ffbb55", lw=1.6, alpha=0.85, zorder=3)

    # Halo layers (back-to-front, biggest/most transparent first)
    halos = []
    halo_scales = [4.0, 2.6, 1.6, 1.0]
    halo_alphas = [0.18, 0.28, 0.45, 0.75]
    for sc, al in zip(halo_scales, halo_alphas):
        c = Circle((0, 0), 0.5, color="orange", alpha=al, zorder=4 + len(halos))
        ax.add_patch(c)
        halos.append((c, sc))

    # The capsule itself
    capsule = Circle((0, 0), 0.4, color="#ffffff", zorder=10)
    ax.add_patch(capsule)

    # HUD text
    hud = ax.text(
        0.01, 0.97, "", transform=ax.transAxes, va="top", ha="left",
        color="#ffeecc", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#101025", ec="#333355", alpha=0.85),
    )

    # Peak-heating marker (will appear when we pass it)
    peak_idx = int(np.argmax(q))

    def frame(k):
        i = idxs[k]
        x, y = dr[i], h[i]

        # Trail: everything up to current point
        trail_line.set_data(dr[: i + 1], h[: i + 1])

        # Halo: radius scales with heat flux (plot-axis km)
        q_norm = q[i] / q_peak
        base_radius = 1.0 + 9.0 * q_norm    # 1 → 10 km visual radius at peak
        colour = _temperature_to_rgb(T[i])
        for patch, sc in halos:
            patch.center = (x, y)
            patch.radius = base_radius * sc * (0.6 + 0.4 * q_norm)
            patch.set_color(colour)

        # Capsule marker (small, white-hot)
        capsule.center = (x, y)
        capsule.radius = 0.8 + 1.2 * q_norm
        capsule.set_color(_temperature_to_rgb(min(T[i] + 200, 3200)))

        phase = "COAST"
        if q[i] > 0.1 * q_peak:
            phase = "PEAK HEATING" if abs(i - peak_idx) < 0.05 * len(t) else "PLASMA"
        if y < 30:
            phase = "SUBSONIC DESCENT" if v[i] < 0.34 else "LOWER ATMOSPHERE"

        hud.set_text(
            f"T+ {t[i]:6.1f} s    phase: {phase}\n"
            f"altitude : {y:6.1f} km\n"
            f"speed    : {v[i]:6.2f} km/s\n"
            f"heat flux: {q[i]/1e6:6.2f} MW/m²\n"
            f"T_shield : {T[i]-273.15:6.0f} °C\n"
            f"decel    : {res['decel_g'][i]:6.2f} g"
        )
        return (trail_line, capsule, hud, *[c for c, _ in halos])

    anim = animation.FuncAnimation(
        fig, frame, frames=n_frames, interval=1000 / fps, blit=True
    )
    path = os.path.join(OUTPUT_DIR, filename)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return path


def run_reentry_from_mission(meta, make_animation=True):
    """
    Chain the atmospheric-entry sim onto the lunar-return trajectory.
    Uses the re-entry speed from meta; flight-path angle assumed -6.5° (Orion nominal).
    """
    if meta.get("reentry_speed_km_s") is None:
        print("[Re-entry] No re-entry state in trajectory — skipping heat-shield sim.")
        return None

    v_entry = meta["reentry_speed_km_s"]
    print(f"\n[Re-entry] Propagating atmospheric descent from {v_entry:.3f} km/s, "
          f"gamma = -6.5°...")
    res = simulate_reentry(v_entry_km_s=v_entry, gamma_entry_deg=-6.5)
    summary = summarize(res)
    print_reentry_log(res, summary)
    path = plot_reentry(res, summary)
    print(f"[Re-entry] Saved: {os.path.basename(path)}")
    if make_animation:
        print("[Re-entry] Rendering plasma-halo animation...")
        anim_path = create_reentry_animation(res, summary)
        print(f"[Re-entry] Saved: {os.path.basename(anim_path)}")
    return res, summary
