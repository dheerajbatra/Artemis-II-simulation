#!/usr/bin/env python3
"""
Artemis II Complete Mission Simulation
Run: python run_simulation.py

Outputs:
  artemis_3d_trajectory.png   — 3D inertial frame view
  artemis_telemetry.png       — altitude, Moon dist, speed over time
  artemis_rotating_frame.png  — figure-8 in Earth-Moon rotating frame
  artemis_animation.gif       — animated trajectory

Based on real Artemis II parameters:
  - ~10.5-day free-return trajectory
  - Perilune ~6,545 km from Moon center (4,808 km above surface)
  - No burn needed for return — pure gravity
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           ARTEMIS II — MISSION SIMULATION                    ║
║           Free-Return Trajectory • N-Body Integrator         ║
║           Earth + Moon + Sun Gravity                         ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_mission_log(t, states, meta):
    """Print a formatted mission event log."""
    from physics import R_EARTH, R_MOON, moon_position

    events = [
        (0, "TLI Burn Complete"),
        (3 * 3600, "Trans-Lunar Coast begin"),
        (meta["perilune_time_s"], "PERILUNE (closest Moon approach)"),
    ]
    if meta["reentry_time_s"]:
        events.append((meta["reentry_time_s"], "Re-entry Interface (~120 km)"))

    print("\n" + "─" * 65)
    print(f"{'Event':<35} {'T+ (hr)':>10}  {'T+ (days)':>10}")
    print("─" * 65)
    for ts, label in sorted(events):
        print(f"  {label:<33} {ts / 3600:>10.2f}  {ts / 86400:>10.3f}")
    print("─" * 65)

    print(f"\n  Parking orbit altitude   : {170:.0f} km")
    # print(f"\n  Parking orbit altitude   : {'PARKING_ORBIT_ALT':.0f} km")
    print(f"  TLI delta-v              : ~3,122 m/s")
    print(f"  Post-TLI speed           : {meta['tli_speed_km_s']:.3f} km/s")
    print(
        f"  Perilune altitude        : {meta['perilune_alt_km']:.0f} km above Moon surface"
    )
    print(f"  Perilune dist (center)   : {meta['perilune_dist_km']:.0f} km")
    if meta["reentry_speed_km_s"]:
        print(f"  Re-entry speed           : {meta['reentry_speed_km_s']:.3f} km/s")
    print()


def main():
    print(BANNER)
    t0 = time.time()

    from mission import build_full_trajectory
    from visualize import (
        create_animation,
        plot_3d_trajectory,
        plot_rotating_frame,
        plot_telemetry,
    )

    # ── 1. Build trajectory ────────────────────────────────────────────────────
    print("[1/5] Computing trajectory (verified baseline + fine optimizer)...")
    t, states, meta = build_full_trajectory(optimize=True)

    # ── 2. Mission log ─────────────────────────────────────────────────────────
    print("\n[2/5] Mission event log:")
    print_mission_log(t, states, meta)

    # ── 3. 3D plot ─────────────────────────────────────────────────────────────
    print("[3/5] Generating 3D trajectory plot...")
    plot_3d_trajectory(t, states, meta)

    # ── 4. Telemetry dashboard ─────────────────────────────────────────────────
    print("[4/5] Generating telemetry dashboard...")
    plot_telemetry(t, states, meta)

    # ── 5. Rotating frame ──────────────────────────────────────────────────────
    print("[5/5] Generating rotating-frame (figure-8) plot...")
    plot_rotating_frame(t, states, meta)

    # ── 6. Animation (optional — takes ~30s) ───────────────────────────────────
    do_anim = os.environ.get("ARTEMIS_ANIM", "1") == "1"
    if do_anim:
        print("[+] Creating animated GIF...")
        create_animation(t, states, meta, n_frames=150)

    # ── 7. Re-entry heating simulation ─────────────────────────────────────────
    print("\n[+] Simulating atmospheric re-entry & heat-shield temperatures...")
    from reentry import run_reentry_from_mission
    run_reentry_from_mission(meta)

    elapsed = time.time() - t0
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'═' * 65}")
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"  Output files in: {out_dir}/")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
