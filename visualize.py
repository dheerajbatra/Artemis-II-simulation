"""
Artemis II Visualization
Produces:
  1. 3D trajectory plot (Earth + Moon + Orion path)
  2. Distance-from-Earth timeline
  3. Speed timeline
  4. Animated GIF of figure-8 trajectory
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import os

from physics import moon_position, R_EARTH, R_MOON, MOON_SEMI_MAJOR

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _starfield(ax, lim, n_bright=60, n_dim=350, seed=42):
    """Sprinkle a reproducible starfield onto a 2D axes."""
    rng = np.random.default_rng(seed)
    # Dim background stars
    xs = rng.uniform(-lim, lim, n_dim)
    ys = rng.uniform(-lim, lim, n_dim)
    ax.scatter(xs, ys, s=rng.uniform(0.1, 0.6, n_dim),
               color="#8899bb", alpha=0.35, zorder=0)
    # Brighter foreground stars
    xs = rng.uniform(-lim, lim, n_bright)
    ys = rng.uniform(-lim, lim, n_bright)
    sizes = rng.uniform(0.6, 2.0, n_bright)
    ax.scatter(xs, ys, s=sizes, color="#eef2ff", alpha=0.85, zorder=0)


def _plasma_colour(fraction):
    """
    Return (r, g, b) for a re-entry plasma glow.
    fraction = 0  → cool red  (just entering atmosphere)
    fraction = 1  → yellow-white hot (peak plasma)
    """
    u = float(np.clip(fraction, 0.0, 1.0))
    r = np.clip(0.25 + 1.2 * u, 0.0, 1.0)
    g = np.clip(-0.10 + 1.4 * u, 0.0, 1.0)
    b = np.clip(-0.50 + 1.6 * u, 0.0, 1.0)
    return (float(r), float(g), float(b))


def sphere(center, radius, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def plot_3d_trajectory(t, states, meta, filename="artemis_3d_trajectory.png"):
    fig = plt.figure(figsize=(14, 11), facecolor='#050510')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050510')
    fig.patch.set_facecolor('#050510')

    sc_x, sc_y, sc_z = states[0], states[1], states[2]
    phase0 = meta['moon_phase0_rad']

    n = len(t)
    flyby_idx = meta['flyby_index']
    re_idx    = meta['reentry_index'] or n - 1

    # Scale: km → 1000 km for readability
    scale = 1000.0

    # ── 3D starfield ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(7)
    lim_star = MOON_SEMI_MAJOR * 1.1 / scale
    n_stars = 300
    sx = rng.uniform(-lim_star, lim_star, n_stars)
    sy = rng.uniform(-lim_star, lim_star, n_stars)
    sz = rng.uniform(-lim_star * 0.5, lim_star * 0.5, n_stars)
    ax.scatter(sx, sy, sz, s=rng.uniform(0.2, 1.5, n_stars),
               color='#ccd4ee', alpha=0.45, depthshade=False)

    # Outbound leg
    ax.plot(sc_x[:flyby_idx]/scale, sc_y[:flyby_idx]/scale, sc_z[:flyby_idx]/scale,
            color='#00aaff', lw=1.8, label='Trans-Lunar Coast', alpha=0.95)
    # Flyby highlight
    fb_s = max(0, flyby_idx-500)
    fb_e = min(n-1, flyby_idx+500)
    ax.plot(sc_x[fb_s:fb_e]/scale, sc_y[fb_s:fb_e]/scale, sc_z[fb_s:fb_e]/scale,
            color='#ff9900', lw=3.0, label='Lunar Flyby', zorder=5)
    # Return leg
    ax.plot(sc_x[flyby_idx:re_idx]/scale, sc_y[flyby_idx:re_idx]/scale,
            sc_z[flyby_idx:re_idx]/scale,
            color='#ff4466', lw=1.8, label='Return Leg', alpha=0.95)

    # Earth sphere (scaled up 15x for visibility)
    ex, ey, ez = sphere([0, 0, 0], R_EARTH * 5 / scale, resolution=30)
    ax.plot_surface(ex, ey, ez, color='#1a6faf', alpha=0.9, zorder=5)

    # Moon at perilune time (correct phase)
    r_moon_fly = moon_position(meta['perilune_time_s'], phase0=phase0) / scale
    mx, my, mz = sphere(r_moon_fly, R_MOON * 10 / scale, resolution=20)
    ax.plot_surface(mx, my, mz, color='#aaaaaa', alpha=0.8, zorder=5)
    ax.text(r_moon_fly[0], r_moon_fly[1], r_moon_fly[2] + 15,
            'Moon', color='#aaaaaa', fontsize=8, ha='center')

    # Moon orbit trace
    theta = np.linspace(0, 2 * np.pi, 300)
    mo_x = MOON_SEMI_MAJOR * np.cos(theta) / scale
    mo_y = MOON_SEMI_MAJOR * np.sin(theta) * np.cos(np.radians(5.145)) / scale
    mo_z = MOON_SEMI_MAJOR * np.sin(theta) * np.sin(np.radians(5.145)) / scale
    ax.plot(mo_x, mo_y, mo_z, '--', color='#444466', lw=0.6, alpha=0.6)

    # TLI and re-entry markers
    ax.scatter(*states[0:3, 0]/scale, color='lime', s=60, zorder=10, label='TLI')
    if meta['reentry_index']:
        ax.scatter(*states[0:3, re_idx]/scale, color='#ff4466', s=80, marker='v',
                   zorder=10, label='Re-entry (~120 km alt)')

    # Axis bounds in 1000-km units
    lim = MOON_SEMI_MAJOR * 1.15 / scale
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-80, 80)
    ax.set_box_aspect((1, 1, 0.25))
    ax.view_init(elev=55, azim=-62)

    # Labels
    ax.set_xlabel('X (×1000 km)', color='white', labelpad=10)
    ax.set_ylabel('Y (×1000 km)', color='white', labelpad=10)
    ax.set_zlabel('Z (×1000 km)', color='white', labelpad=10)
    ax.tick_params(colors='#888888', labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#222244')
    ax.grid(True, color='#222244', lw=0.4)

    title = (f"Artemis II — Free-Return Trajectory\n"
             f"Perilune: {meta['perilune_dist_km']:.0f} km from Moon center  |  "
             f"Re-entry: {meta['reentry_speed_km_s']:.2f} km/s" if meta['reentry_speed_km_s']
             else "Artemis II — Free-Return Trajectory")
    ax.set_title(title, color='white', fontsize=12, pad=12)

    legend = ax.legend(loc='upper left', facecolor='#111133', edgecolor='#334466',
                       labelcolor='white', fontsize=8)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"[Viz] Saved {path}")
    return path


def plot_telemetry(t, states, meta, filename="artemis_telemetry.png"):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), facecolor='#0a0a1a')
    fig.suptitle("Artemis II — Mission Telemetry", color='white', fontsize=14)

    # Truncate at reentry to avoid post-impact numerical garbage
    re_idx = meta['reentry_index'] or len(t) - 1
    t      = t[:re_idx + 1]
    states = states[:, :re_idx + 1]

    t_days = t / 86400
    r_sc = np.linalg.norm(states[0:3, :], axis=0)
    v_sc = np.linalg.norm(states[3:6, :], axis=0)

    phase0 = meta.get('moon_phase0_rad', 0.0)
    moon_pos  = np.array([moon_position(ti, phase0=phase0) for ti in t])
    d_moon_sc = np.linalg.norm(states[0:3, :].T - moon_pos, axis=1)

    phase_colors = {
        'TLI':             ('#00ff88', 0),
        'Lunar Flyby':     ('#ff9900', meta['perilune_time_s'] / 86400),
        'Re-entry':        ('#ff4466', meta['reentry_time_s'] / 86400 if meta['reentry_time_s'] else None),
    }

    fly_day = meta['perilune_time_s'] / 86400
    re_day = meta['reentry_time_s'] / 86400 if meta['reentry_time_s'] else None

    for ax in axes:
        ax.set_facecolor('#0c0c22')
        ax.tick_params(colors='#888888')
        ax.spines[:].set_edgecolor('#334466')
        ax.grid(True, color='#1a1a3a', lw=0.4, alpha=0.6)
        # Shaded phase bands
        ax.axvspan(0, fly_day - 0.15, color='#00aaff', alpha=0.06, zorder=0)
        ax.axvspan(fly_day - 0.15, fly_day + 0.15, color='#ff9900', alpha=0.14, zorder=0)
        if re_day is not None:
            ax.axvspan(fly_day + 0.15, re_day - 0.05, color='#ff4466', alpha=0.05, zorder=0)
            ax.axvspan(re_day - 0.05, re_day, color='#ffee88', alpha=0.20, zorder=0)
        for label, (color, xpos) in phase_colors.items():
            if xpos is not None:
                ax.axvline(xpos, color=color, lw=1.0, alpha=0.8, linestyle='--')

    # Phase labels drawn INSIDE the first panel (top, slightly transparent)
    labels_top = [
        ((0 + (fly_day - 0.15)) / 2, 'Trans-Lunar Coast', '#66ccff'),
        (fly_day, 'Flyby', '#ffbb55'),
    ]
    if re_day is not None:
        labels_top.append(((fly_day + re_day) / 2, 'Return Coast', '#ff88aa'))
        labels_top.append((re_day, 'Re-entry', '#ffee88'))
    for x, txt, col in labels_top:
        axes[0].text(x, 0.92, txt, transform=axes[0].get_xaxis_transform(),
                     ha='center', va='top', color=col, fontsize=8,
                     fontweight='bold', alpha=0.9)

    # Panel 1: Distance from Earth
    axes[0].plot(t_days, r_sc - R_EARTH, color='#00aaff', lw=1.2)
    axes[0].set_ylabel('Alt above Earth (km)', color='white')
    axes[0].set_title('Altitude from Earth Surface', color='#aaaacc', fontsize=10)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Panel 2: Distance from Moon
    axes[1].plot(t_days, d_moon_sc, color='#aaaaaa', lw=1.2)
    axes[1].axhline(6545, color='#ff9900', lw=0.8, linestyle=':', alpha=0.8,
                    label='Target perilune 6,545 km')
    axes[1].set_ylabel('Distance from Moon (km)', color='white')
    axes[1].set_title('Distance from Moon Center', color='#aaaacc', fontsize=10)
    axes[1].legend(facecolor='#111133', edgecolor='#334466', labelcolor='white', fontsize=8)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Panel 3: Speed (clamped to physical range, reentry spike handled)
    v_clamp = np.clip(v_sc, 0, 15.0)  # physical max ≈ 11 km/s at reentry
    axes[2].plot(t_days, v_clamp, color='#ff4466', lw=1.2)
    axes[2].set_ylabel('Speed (km/s)', color='white')
    axes[2].set_xlabel('Mission Time (days)', color='white')
    axes[2].set_title('Spacecraft Speed', color='#aaaacc', fontsize=10)
    axes[2].set_ylim(0, 13)

    # Annotate perilune
    fly_day = meta['perilune_time_s'] / 86400
    fly_idx = np.argmin(np.abs(t_days - fly_day))
    axes[2].annotate(f"Perilune\n{v_sc[fly_idx]:.2f} km/s",
                     xy=(fly_day, min(v_sc[fly_idx], 12.5)),
                     xytext=(fly_day + 0.4, min(v_sc[fly_idx], 12.5) + 0.8),
                     color='#ff9900', fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='#ff9900', lw=0.8))
    # Annotate reentry speed
    if meta['reentry_time_s'] and meta['reentry_speed_km_s']:
        re_day = meta['reentry_time_s'] / 86400
        axes[2].annotate(f"Re-entry\n{meta['reentry_speed_km_s']:.2f} km/s",
                         xy=(re_day, meta['reentry_speed_km_s']),
                         xytext=(re_day - 1.2, meta['reentry_speed_km_s'] + 1.5),
                         color='#ff4466', fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='#ff4466', lw=0.8))

    for ax in axes:
        ax.set_xlim(t_days[0], t_days[-1])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('#888888')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"[Viz] Saved {path}")
    return path


def plot_rotating_frame(t, states, meta, filename="artemis_rotating_frame.png"):
    """
    Plot trajectory in the co-rotating Earth-Moon frame (figure-8 shape appears here).
    """
    n = len(t)
    x_rot = np.zeros(n)
    y_rot = np.zeros(n)

    phase0 = meta.get('moon_phase0_rad', 0.0)
    for i, ti in enumerate(t):
        # Rotate into frame where Moon is always on +X axis
        theta_moon = phase0 + 2 * np.pi * ti / (27.3217 * 86400)
        angle = -theta_moon  # negate to counter-rotate frame
        c, s = np.cos(angle), np.sin(angle)
        x = states[0, i]; y = states[1, i]
        x_rot[i] = c * x - s * y
        y_rot[i] = s * x + c * y

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#050510')
    ax.set_facecolor('#050510')

    flyby_idx = meta['flyby_index']
    re_idx = meta['reentry_index'] or n - 1

    # Truncate at re-entry
    x_plot = x_rot[: re_idx + 1] / 1000
    y_plot = y_rot[: re_idx + 1] / 1000
    t_plot = t[: re_idx + 1] / 86400

    # Starfield backdrop
    lim_km = 500  # frame half-width in 1000-km units
    _starfield(ax, lim_km, n_bright=40, n_dim=240, seed=9)

    # ── Time-coloured trajectory (multi-segment LineCollection) ─────────────
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'artemis', ['#00aaff', '#88cc66', '#ffaa44', '#ff4466'], N=256
    )
    pts = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(t_plot.min(), t_plot.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=1.3, alpha=0.95, zorder=4)
    lc.set_array(t_plot[:-1])
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Mission time (days)', color='#cccccc')
    cb.ax.tick_params(colors='#888888')
    cb.outline.set_edgecolor('#334466')

    # Earth with subtle atmosphere glow
    ax.add_patch(Circle((0, 0), R_EARTH / 1000 * 2.5, color='#3080d0',
                        alpha=0.15, zorder=4))
    earth_circle = Circle((0, 0), R_EARTH / 1000, color='#1a6faf', zorder=5)
    ax.add_patch(earth_circle)

    # Moon fixed at (384400, 0) in rotating frame
    moon_circle = Circle((384.4, 0), R_MOON / 1000 * 8, color='#aaaaaa', zorder=5)
    ax.add_patch(moon_circle)
    ax.text(384.4, 22, 'Moon', color='#cccccc', ha='center', fontsize=9)
    ax.text(0, 22, 'Earth', color='#88bbee', ha='center', fontsize=9)

    # L1 / L2 Lagrange points (Earth-Moon system), rough positions
    ax.scatter([326.45], [0], marker='x', s=40, color='#ffcc66',
               zorder=6, label='L1')
    ax.scatter([449.1], [0], marker='x', s=40, color='#ff8844',
               zorder=6, label='L2')

    ax.scatter(x_rot[0] / 1000, y_rot[0] / 1000, color='#00ff88', s=70,
               zorder=8, label='TLI', edgecolors='#003322', linewidths=0.5)
    if meta['reentry_index']:
        ax.scatter(x_rot[re_idx] / 1000, y_rot[re_idx] / 1000, color='#ff4466',
                   s=90, marker='v', zorder=8, label='Re-entry',
                   edgecolors='#441119', linewidths=0.5)

    ax.set_xlabel('X (×1000 km)', color='white')
    ax.set_ylabel('Y (×1000 km)', color='white')
    ax.set_title('Artemis II — Earth-Moon Rotating Frame\n(Figure-8 Free-Return)',
                 color='white', fontsize=12)
    ax.tick_params(colors='#888888')
    ax.spines[:].set_edgecolor('#334466')
    ax.legend(facecolor='#111133', edgecolor='#334466', labelcolor='white', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, color='#1a1a3a', lw=0.5)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"[Viz] Saved {path}")
    return path


def create_animation(t, states, meta, filename="artemis_animation.gif", n_frames=150):
    """
    Inertial-frame animation of the full Artemis II trajectory.
    Truncates at re-entry, adds a plasma halo to Orion as it plunges into
    Earth's atmosphere, and uses non-linear pacing so the flyby and re-entry
    moments aren't skimmed over in one frame.
    """
    print(f"[Viz] Creating animation ({n_frames} frames)...")
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#050510')
    ax.set_facecolor('#050510')
    ax.set_aspect('equal')

    sc_x, sc_y = states[0], states[1]
    phase0 = meta.get('moon_phase0_rad', 0.0)
    fly_idx = meta['flyby_index']
    re_idx = meta['reentry_index'] or len(t) - 1

    # ── truncate propagation at re-entry (beyond is unphysical) ─────────────
    end_idx = re_idx
    t = t[: end_idx + 1]
    sc_x = sc_x[: end_idx + 1]
    sc_y = sc_y[: end_idx + 1]
    n = len(t)

    # ── Moon positions (vectorized) ─────────────────────────────────────────
    moon_xy = np.array(
        [[moon_position(ti, phase0=phase0)[0],
          moon_position(ti, phase0=phase0)[1]] for ti in t]
    )

    lim = MOON_SEMI_MAJOR * 1.15 / 1000
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # ── Starfield backdrop ──────────────────────────────────────────────────
    _starfield(ax, lim)

    # ── Static elements ─────────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(MOON_SEMI_MAJOR * np.cos(theta) / 1000,
            MOON_SEMI_MAJOR * np.sin(theta) / 1000,
            '--', color='#333355', lw=0.6, zorder=1)

    # Earth glow (atmosphere ring)
    earth_glow = Circle((0, 0), R_EARTH / 1000 * 8, color='#3080d0',
                        alpha=0.15, zorder=3)
    ax.add_patch(earth_glow)
    earth_c = Circle((0, 0), R_EARTH / 1000 * 5, color='#1a6faf', zorder=5)
    ax.add_patch(earth_c)

    # ── Dynamic elements ────────────────────────────────────────────────────
    moon_dot, = ax.plot([], [], 'o', color='#cccccc', ms=12, zorder=6,
                       markeredgecolor='#777788', markeredgewidth=0.5)
    trail, = ax.plot([], [], color='#00aaff', lw=1.1, alpha=0.85, zorder=7)

    # Plasma-halo layers around Orion (hidden until re-entry approach)
    halo_scales = [5.0, 3.0, 1.8, 1.0]
    halo_alphas = [0.15, 0.28, 0.45, 0.75]
    halos = []
    for sc, al in zip(halo_scales, halo_alphas):
        c = Circle((0, 0), 0.1, color='#ffaa55', alpha=0.0,
                   zorder=8 + len(halos))
        ax.add_patch(c)
        halos.append((c, sc, al))

    orion, = ax.plot([], [], 'o', color='#ffdd00', ms=6, zorder=12)

    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        color='white', fontsize=10, va='top', family='monospace')
    phase_text = ax.text(0.02, 0.92, '', transform=ax.transAxes,
                         color='#aaaaff', fontsize=10, va='top', family='monospace')

    ax.set_xlabel('X (×1000 km)', color='#cccccc')
    ax.set_ylabel('Y (×1000 km)', color='#cccccc')
    ax.set_title('Artemis II — Inertial Frame',
                 color='white', fontsize=12, pad=10)
    ax.tick_params(colors='#888888')
    ax.spines[:].set_edgecolor('#334466')

    # ── Non-linear pacing: concentrate frames around flyby + re-entry ───────
    # Weight by inverse distance to the two "interesting" indices.
    idxs = np.arange(n)
    # Base density = uniform; bonus density around flyby and re-entry
    sigma_fly = 0.025 * n
    sigma_re = 0.02 * n
    density = (
        0.35
        + 1.6 * np.exp(-0.5 * ((idxs - fly_idx) / sigma_fly) ** 2)
        + 2.2 * np.exp(-0.5 * ((idxs - (n - 1)) / sigma_re) ** 2)
    )
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    frame_indices = np.searchsorted(cdf, np.linspace(0, 1, n_frames)).clip(0, n - 1)

    # Re-entry halo starts activating once spacecraft is closer than this
    r_sc_all = np.sqrt(sc_x ** 2 + sc_y ** 2)
    halo_activate_km = 80_000.0   # well outside Earth, inside inner vicinity
    halo_saturate_km = 200.0      # peak at atmospheric interface

    def phase_name(idx):
        if idx < fly_idx * 0.95:
            return "Trans-Lunar Coast"
        elif idx < fly_idx * 1.05:
            return ">> LUNAR FLYBY <<"
        elif idx >= end_idx - 3:
            return ">> PLASMA RE-ENTRY <<"
        elif r_sc_all[idx] < halo_activate_km:
            return "Atmospheric Entry"
        else:
            return "Return Coast"

    def update(frame_num):
        idx = int(frame_indices[frame_num])
        trail_start = max(0, idx - 1200)
        trail.set_data(sc_x[trail_start:idx] / 1000, sc_y[trail_start:idx] / 1000)
        orion.set_data([sc_x[idx] / 1000], [sc_y[idx] / 1000])
        moon_dot.set_data([moon_xy[idx, 0] / 1000], [moon_xy[idx, 1] / 1000])

        label = phase_name(idx)
        time_text.set_text(f"T+{t[idx]/86400:6.2f} days")
        phase_text.set_text(label)

        # ── Orion colour by phase ───────────────────────────────────────────
        if "LUNAR FLYBY" in label:
            orion.set_color('#ff9900')
        elif "PLASMA" in label or "Atmospheric" in label:
            orion.set_color('#ffe6b0')
        else:
            orion.set_color('#ffdd00')

        # ── Plasma halo (only activates as Orion falls into atmosphere) ─────
        r_sc = r_sc_all[idx]
        if r_sc < halo_activate_km:
            # 0 at halo_activate, 1 at halo_saturate (log-ish ramp for punchiness)
            t_frac = np.clip(
                (np.log10(halo_activate_km) - np.log10(max(r_sc, halo_saturate_km)))
                / (np.log10(halo_activate_km) - np.log10(halo_saturate_km)),
                0.0, 1.0,
            )
            colour = _plasma_colour(t_frac)
            base_radius = 2.0 + 28.0 * t_frac      # in plot-units (×1000 km)
            for patch, sc, al in halos:
                patch.center = (sc_x[idx] / 1000, sc_y[idx] / 1000)
                patch.radius = base_radius * sc
                patch.set_color(colour)
                patch.set_alpha(al * (0.35 + 0.65 * t_frac))
        else:
            for patch, _, _ in halos:
                patch.set_alpha(0.0)

        return (trail, orion, moon_dot, time_text, phase_text,
                *(p for p, _, _ in halos))

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=70, blit=True
    )
    path = os.path.join(OUTPUT_DIR, filename)
    ani.save(path, writer='pillow', fps=15, dpi=100)
    plt.close()
    print(f"[Viz] Saved {path}")
    return path
