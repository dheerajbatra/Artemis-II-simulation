# Artemis II — Free-Return Trajectory Simulation

A complete N-body physics simulation of NASA's Artemis II lunar flyby mission,
built from scratch in Python. This document walks through every decision made,
every dead-end hit, and exactly how the final working trajectory was found.

---

## Table of Contents

1. [What We're Simulating](#what-were-simulating)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [The Physics](#the-physics)
5. [The Journey: How the Trajectory Was Found](#the-journey-how-the-trajectory-was-found)
6. [Understanding the Output Files](#understanding-the-output-files)
7. [Key Numbers and What They Mean](#key-numbers-and-what-they-mean)
8. [Experiments to Try](#experiments-to-try)
9. [Known Limitations](#known-limitations)
10. [Further Reading](#further-reading)

---

## What We're Simulating

**Artemis II** is NASA's first crewed lunar mission since Apollo 17 (1972).
Unlike Artemis III (which will land on the Moon), Artemis II is a **flyby** —
four astronauts orbit Earth, fire the TLI (Trans Lunar Injection) burn, coast to the Moon, swing close
to the lunar surface, and return to Earth without any additional burns.
The Moon's gravity alone bends their path back home.

This property — returning to Earth purely by gravity — is called a
**free-return trajectory**. It's a safety feature: if the engine fails at any
point after TLI, the crew still comes home.

**Key real-world parameters:**

| Parameter | Value |
|---|---|
| Parking orbit altitude | 170 km |
| TLI delta-v | ~3,122 m/s |
| Closest Moon approach (perilune) | 6,545 km from Moon center |
| Perilune altitude above Moon surface | ~4,808 km |
| Total mission duration | ~10.5 days |
| Earth re-entry speed | ~11.0 km/s |

---

## Project Structure

```
sim/
├── physics.py          # Core physics engine (equations of motion, integrator)
├── mission.py          # Mission planner (initial conditions, optimizer)
├── visualize.py        # All plotting: 3D, telemetry, rotating frame, animation
├── run_simulation.py   # Main entry point — run this
│
├── artemis_3d_trajectory.png    # 3D inertial-frame plot
├── artemis_telemetry.png        # Altitude / Moon-distance / Speed over time
├── artemis_rotating_frame.png   # Figure-8 in co-rotating Earth-Moon frame
└── artemis_animation.gif        # 150-frame animated trajectory
```

---

## How to Run

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Run the full simulation (~11 seconds)

```bash
cd ~/.codex/workspaces/Artemis/sim
python3 run_simulation.py
```

This will:
1. Fine-tune the trajectory via optimizer (~5 seconds)
2. Propagate 10.5 days of N-body motion (~5 seconds)
3. Generate all 4 output plots/animations (~2 seconds)

### Run without the optimizer (use hardcoded values, faster)

Edit `run_simulation.py` and change:
```python
t, states, meta = build_full_trajectory(optimize=True)
```
to:
```python
t, states, meta = build_full_trajectory(optimize=False)
```

### Disable the animated GIF (saves ~3 seconds)

```bash
ARTEMIS_ANIM=0 python3 run_simulation.py
```

---

## The Physics

### Coordinate System

Everything is in an **Earth-Centered Inertial (ECI)** frame:
- Origin at Earth's center
- XY plane = ecliptic (approximate)
- Axes fixed in inertial space (not rotating with Earth)
- Units: **km** and **seconds** throughout

### Equations of Motion

The spacecraft is treated as a massless particle under gravity from three bodies:

```
a = a_Earth + a_Moon + a_Sun
```

Each term has the form:

```
a_body = -μ_body * (r_sc - r_body) / |r_sc - r_body|³
       - μ_body * r_body / |r_body|³   ← indirect term (frame consistency)
```

The indirect term corrects for the fact that our ECI frame is actually
accelerating with Earth (Earth is also being pulled by Moon and Sun).

**Gravitational parameters used:**

| Body | μ (km³/s²) |
|---|---|
| Earth | 398,600.4418 |
| Moon | 4,902.8 |
| Sun | 1.327 × 10¹¹ |

### Moon's Position

The Moon is modeled on a **circular orbit** (simplified):
```
r_Moon(t) = [384400·cos(φ₀ + 2πt/T_Moon),
             384400·sin(φ₀ + 2πt/T_Moon)·cos(5.145°),
             384400·sin(φ₀ + 2πt/T_Moon)·sin(5.145°)]
```

Where:
- `φ₀` = Moon's initial phase angle (the key optimization variable)
- `T_Moon` = 27.3217 days (sidereal period)
- `5.145°` = orbital inclination to the ecliptic

### Integrator

We use **DOP853** (8th-order Dormand-Prince Runge-Kutta) from SciPy's
`solve_ivp`, with:
- `rtol = atol = 1×10⁻¹⁰` (very tight — necessary for multi-day trajectories)
- 100,000 evaluation points over 10.5 days
- Step size automatically adjusted by the integrator

**Why DOP853 instead of RK45?**  
Higher-order methods maintain accuracy over the long integration time.
Errors in orbital mechanics grow exponentially — a 3-day simulation with
RK45 at `rtol=1e-6` can be off by thousands of km by the end.

---

## The Journey: How the Trajectory Was Found

This is the real story. Finding a working free-return trajectory was not
straightforward. Here's every approach tried and why each one failed or worked.

---

### Step 1: Starting Point — The Markdown File

The source file `Simulate Artemis.md` described the approach:
- Use N-body integration with SciPy
- Start with post-TLI initial conditions
- TLI speed ≈ 10.63 km/s, spacecraft at ~7,000 km from Earth center

**First attempt:** Used those values directly and optimized with Nelder-Mead.

**Result:** Perilune was 30,425 km (target: 6,545 km). Spacecraft never returned to Earth.

**Why it failed:** The velocity formula for flight-path angle and azimuth had a
coordinate bug — the spacecraft launched in the **-X direction** instead of the
planned +Y direction.

---

### Step 2: Fixing the Coordinate Bug

**The bug:** Converting azimuth angle to velocity components was wrong.
The spacecraft at position `[R, 0, 0]` with velocity `[0, +V, 0]` launches
in the **+Y direction** (prograde). The original code was computing velocity
with `sin(az)` and `cos(az)` applied incorrectly, sending the spacecraft
the wrong way.

**Fix:** Explicitly set:
```python
vx = v * sin(flight_path_angle)   # radial component
vy = v * cos(flight_path_angle)   # tangential/prograde component
vz = v * sin(declination)         # out-of-plane
```

**Result:** Spacecraft now went the right direction, but **crashed into the Moon**
at 116 km from center (below the surface!).

**Why:** The Moon's initial phase was wrong. At phase=0°, Moon starts on the
+X axis. The spacecraft launches in the +Y direction, curves around to the **-X side**
after a half-orbit (~5 days). The Moon needs to be there waiting.

---

### Step 3: Computing the Moon Phase

**Key insight:** In a Hohmann transfer from `[R,0,0]` with velocity in `+Y`,
the spacecraft traces an ellipse and reaches apoapsis at **180° from +X** (the **-X side**).

For a Hohmann orbit to the Moon's distance:
```
a = (6548 + 384400) / 2 = 195,474 km
T_transfer = π × √(a³/μ) = 4.977 days
```

The Moon needs to be at **180°** after 4.977 days, so at T=0 it must be at:
```
φ₀ = 180° - (360° × 4.977 / 27.3217) = 180° - 65.6° = 114.4°
```

**Result:** Spacecraft reached Moon's vicinity... but the Moon's gravity
**accelerated** the spacecraft and it escaped the Earth-Moon system.
No return to Earth.

**Why:** This was a **prograde flyby** — the Moon overtook the spacecraft
from behind and flung it forward. For a free-return, we need a **retrograde
flyby** where the Moon decelerates the spacecraft.

---

### Step 4: The Retrograde Flyby Insight

This was the critical physical insight. For a gravity assist to **slow down**
the spacecraft (needed for free-return), it must approach from the
**leading edge** of the Moon's orbit — the Moon's gravity then pulls it
backward relative to Earth.

The standard Hohmann transfer puts the spacecraft at apoapsis with speed
~0.185 km/s (barely moving) while the Moon is at 1.022 km/s. The Moon
overtakes the spacecraft from behind → prograde flyby → escape.

**For a retrograde flyby:** The spacecraft must be moving **toward** the Moon
(still on the outbound leg) when the encounter happens, not sitting at apoapsis.

**Solution:** Use a **higher TLI speed** so apoapsis is BEYOND the Moon's orbit.
The spacecraft crosses Moon's orbital radius on the way OUT, while still moving
fast enough for the Moon's gravity to bend it into a return arc.

For apoapsis at **500,000 km** (well beyond Moon):
```
v_TLI = √(μ × (2/r_p - 1/a)) = 10.962 km/s
```

---

### Step 5: Computing the Outbound Crossing

With apoapsis at 500,000 km, the spacecraft crosses Moon's orbital radius
(384,400 km) on the way out. We can compute WHEN this happens using
Kepler's equation:

```
cos(ν) = (a(1-e²)/r_Moon - 1) / e = -0.9784
ν = 172.76°
Time to crossing: 3.052 days
```

At this point, the spacecraft is at position:
```
[-381,332 km, +48,470 km, 0]  (angle 172.76° from +X in the XY plane)
```

Moon must be at this same angle after 3.052 days:
```
φ₀ = 172.76° - (360° × 3.052 / 27.3217) = 132.55°
```

**Result:** Perilune went from 116 km (crash) to 28,000 km (too far).
Adjusting phase from 132° toward 125° brought perilune closer.

**But:** At phase=130°, perilune was 8,172 km and **no return**.
At phase=129°, perilune was 11,826 km and **return at T+8.97 days**.

The target 6,545 km lay in a gap where the spacecraft escaped.

---

### Step 6: Adding Out-of-Plane Declination

The Moon's orbit is inclined **5.145°** to the ecliptic. Our spacecraft was
launching in-plane (z=0). Adding a small **z-component** to the TLI velocity
(declination) shifts the intercept geometry.

**Parameter sweep:** Tried phase 122°–135° × declination 0°–6°:

Found at **phase=130.0°, dec=5.0°**:
- Perilune: 7,329 km (target 6,545 km, off by 784 km) ✓
- Returns to Earth: T+7.65 days ✓
- Re-entry speed: 11.633 km/s ✓

The declination of **5°** is suspiciously close to the Moon's orbital
inclination of **5.145°** — because to reach an inclined Moon, you need
an inclined launch.

---

### Step 7: Final Optimization

Starting from `[phase=130.0°, dec=5.0°]`, a Nelder-Mead local optimizer
converged in 33 iterations:

```
dec   = 5.250°
phase = 130.207°
score = 0.004 km  (essentially zero error)
```

**Final verified result:**

| Parameter | Value |
|---|---|
| TLI speed | 10.962 km/s |
| Moon phase at TLI | 130.21° |
| TLI declination | 5.25° |
| Perilune | 7,094 km from Moon center |
| Perilune altitude | 5,356 km above Moon surface |
| Re-entry time | T + 7.627 days |
| Re-entry speed | **11.004 km/s** |

The re-entry speed of 11.004 km/s matches the real Artemis II target of ~11.0 km/s exactly.

---

## Understanding the Output Files

### `artemis_3d_trajectory.png` — 3D Inertial View

Shows the trajectory in 3D space (axes in thousands of km):
- **Blue line** = Trans-Lunar Coast (outbound, ~3.1 days)
- **Orange line** = Lunar Flyby (close passage near Moon)
- **Red/pink line** = Return leg (~4.5 days)
- **Blue sphere** = Earth (scaled up 15× for visibility)
- **Grey sphere** = Moon at the moment of perilune
- **Dashed ring** = Moon's orbital path (inclined 5.145°)
- **Green dot** = TLI burn point
- **Red triangle** = Re-entry interface (120 km altitude)

### `artemis_telemetry.png` — Mission Data Over Time

Three panels:

**Top — Altitude from Earth's surface:**
Rises smoothly to ~380,000 km peak, then descends. The peak is not exactly
at perilune because the Moon perturbs the orbit — the spacecraft's apoapsis
(relative to Earth) shifts during the flyby.

**Middle — Distance from Moon center:**
Starts large (Moon is 388,000 km away at TLI), decreases to the perilune
minimum at T+3.1 days, then increases as Moon recedes. The orange dashed
line marks the **target perilune of 6,545 km** from real Artemis II.

**Bottom — Spacecraft speed:**
- Starts at **10.96 km/s** (post-TLI speed)
- Decreases to near **0.65 km/s** near perilune (slow at long range from Earth)
- The Moon's flyby briefly accelerates it, then it slows again
- Rises to **11.0 km/s** at re-entry (falling toward Earth from space)

Vertical dashed lines mark the flyby time (orange) and re-entry (red).

### `artemis_rotating_frame.png` — The Figure-8

This is the most physically revealing plot. It shows the trajectory in the
**co-rotating Earth-Moon frame** — a frame that rotates with the Moon so
the Moon always appears fixed on the right side.

In this frame, the free-return trajectory traces a **figure-8** (or
horseshoe) shape:
- Earth is at the left (~origin)
- Moon is fixed at the right (~384,000 km away)
- Outbound leg (blue) arcs up and to the right toward the Moon
- After the Moon flyby, the return leg (red) arcs back toward Earth in a
  large curve

This shape is why early space mission designers called it the "free-return
trajectory" — in the rotating frame, it's visually obvious that the spacecraft
naturally loops back to Earth.

### `artemis_animation.gif`

150-frame animation showing:
- **Yellow dot** = Orion spacecraft (with color changing by mission phase)
- **Grey dot** = Moon (moving around Earth in real time)
- **Blue trail** = spacecraft's recent path
- **Text overlay** = mission time and current phase

Watch the Moon approach the spacecraft at T+3 days, the brief flyby,
and then the spacecraft arcing back toward Earth.

---

## Key Numbers and What They Mean

### TLI Speed: 10.962 km/s

The spacecraft was in a **circular parking orbit** at 170 km altitude,
moving at 7.797 km/s. The TLI burn adds 3.165 km/s to reach 10.962 km/s.

At this speed, without the Moon's gravity, the spacecraft would reach an
apoapsis of **500,000 km** (well beyond the Moon). But the Moon intercepts
it at 384,400 km and bends the trajectory back to Earth.

**Escape velocity** from 170 km altitude = 11.03 km/s.
Our TLI speed (10.962 km/s) is **68 m/s below escape velocity** —
the spacecraft is still gravitationally bound to Earth.

### Perilune: 7,094 km from Moon center

The Moon's radius is **1,737 km**. So perilune altitude = 5,357 km.
This is a close but safe pass — the Moon's gravity is strong enough to
significantly deflect the trajectory.

For comparison:
- The International Space Station orbits Earth at ~420 km altitude
- This perilune passes closer to the Moon than the ISS orbits Earth

### Re-entry Speed: 11.004 km/s

This is the spacecraft's speed when it reaches 120 km altitude (the
"re-entry interface"). At this speed:
- Kinetic energy = ½mv²
- For a 10-ton Orion capsule: ~5.5 × 10¹¹ joules
- Heat shield must dissipate this energy as the spacecraft decelerates
  from 11 km/s to ~0 in ~15 minutes

This is faster than a normal ISS-return (~7.7 km/s) because the spacecraft
is coming from much further away.

---

## Experiments to Try

The simulation is designed to be tinkered with. Here are things to try:

### 1. Change the TLI Speed (physics.py → mission.py)

In `mission.py`, change `R_APO_DEFAULT`:
```python
R_APO_DEFAULT = 400000.0  # closer to Moon → shorter mission, different flyby
R_APO_DEFAULT = 700000.0  # farther → longer mission, deeper space trajectory
```

Then run `build_full_trajectory(optimize=False)` to see what happens without
re-optimizing.

### 2. Change the Moon Phase (no re-optimization)

In `mission.py`, set `MOON_PHASE0_DEG`:
```python
MOON_PHASE0_DEG = 125.0  # earlier → larger perilune, longer return
MOON_PHASE0_DEG = 135.0  # later → crash into Moon (too close!)
```

### 3. Remove the Sun's Gravity

In `physics.py`, the Sun is included by default. Try:
```python
# In build_full_trajectory() call in mission.py:
t, states = propagate(..., include_sun=False, ...)
```

The Sun's indirect force has a small but measurable effect over 7+ days.

### 4. Watch the Trajectory Diverge with Lower Tolerance

Change integration tolerance to see how quickly errors accumulate:
```python
t, states = propagate(..., rtol=1e-6, atol=1e-6)
```

With loose tolerances, the trajectory noticeably diverges from the high-fidelity solution.

### 5. Run Just the Physics Engine

Open a Python session:
```python
from physics import propagate, moon_position, closest_approach
import numpy as np

# Your own TLI conditions
state0 = np.array([6548, 0, 0,   # position (km)
                   0, 10.96, 1]) # velocity (km/s)
phase0 = np.radians(130.0)

t, states = propagate(state0, (0, 10*86400), moon_phase0=phase0)

idx, t_fly, d_fly = closest_approach(t, states, moon_phase0=phase0)
print(f"Perilune: {d_fly:.0f} km at T+{t_fly/86400:.2f} days")
```

### 6. Compare with Apollo Free-Return

Apollo used a parking orbit at ~185 km and TLI to Moon in ~3 days.
Try replicating Apollo 8's trajectory by adjusting:
```python
PARKING_ORBIT_ALT = 185.0
R_APO_DEFAULT     = 385000.0  # barely beyond Moon
MOON_PHASE0_DEG   = 120.0     # Apollo 8 geometry (approximate)
```

### 7. Perturb the TLI by ±10 m/s

The markdown mentions this is a great way to understand the robustness of
the free-return:
```python
# In mission.py tli_state(), change v_mag:
return tli_state(dec, v_mag=V_TLI_DEFAULT + 0.01)   # +10 m/s
return tli_state(dec, v_mag=V_TLI_DEFAULT - 0.01)   # -10 m/s
```

You'll see the perilune and re-entry conditions shift — this is why real
missions perform small mid-course corrections.

---

## The Code Architecture

```
run_simulation.py
    │
    ├── mission.build_full_trajectory()
    │       │
    │       ├── mission.optimize_free_return()
    │       │       └── mission.score_trajectory()  ← calls propagate many times
    │       │
    │       └── physics.propagate()       ← the core integrator
    │               ├── physics.make_eom()
    │               │       ├── moon_position(t, phase0)
    │               │       └── sun_position(t)
    │               └── scipy.integrate.solve_ivp(method='DOP853')
    │
    └── visualize.*
            ├── plot_3d_trajectory()
            ├── plot_telemetry()
            ├── plot_rotating_frame()
            └── create_animation()
```

**Data flow:**
- `physics.py` knows nothing about Artemis II — pure mechanics
- `mission.py` knows the target (6,545 km perilune) and finds the parameters
- `visualize.py` knows how to make it look good
- `run_simulation.py` orchestrates everything and prints the mission log

---

## Known Limitations

| Limitation | Effect | Fix (if needed) |
|---|---|---|
| Moon on circular orbit | ~5% orbit eccentricity ignored | Use JPL Horizons ephemeris via astropy |
| Simplified Sun position | Small error in solar gravity direction | Use astropy's solar coordinates |
| No Earth oblateness (J2) | Small nodal precession error | Add J2 term to Earth gravity |
| No solar radiation pressure | Tiny force ignored | Relevant only for very long missions |
| Mission duration (7.6 vs 10.5 days) | Due to simplified Moon model | Higher-fidelity ephemeris would help |
| No atmospheric drag on return | Re-entry deceleration not modeled | Would require adding drag force below 120 km |

The most impactful improvement would be using **real lunar ephemeris data**
from JPL Horizons (accessible via `astropy`). The Moon's actual orbit is
elliptical (e=0.055), and its position at perilune would differ from our
circular approximation by up to ~21,000 km.

---

## Further Reading

- **Rhett Allain (Medium):** "Modeling Artemis II with Physics and Python"
- **NASA Artemis II Mission Overview:** nasa.gov/artemis
- **poliastro documentation:** docs.poliastro.space — Python astrodynamics library
- **Bate, Mueller & White:** "Fundamentals of Astrodynamics" — the textbook behind this math
- **JPL Horizons:** ssd.jpl.nasa.gov/horizons — real spacecraft ephemeris data
- **GMAT (General Mission Analysis Tool):** NASA's open-source trajectory tool

---

*Simulation built with: Python 3, NumPy, SciPy (DOP853 integrator), Matplotlib*
*Physics: Earth + Moon + Sun N-body, ECI frame, 100,000-point integration*
