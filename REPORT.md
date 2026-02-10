# Kepler-432b Interstellar Slingshot — Run Report

**Date:** 2026-02-09  
**Solver:** ThreeBodySolver v2.3  
**Config:** `config_interstellar_k432.yaml`  
**Results directory:** `results_Kepler-432_20260209_154118/`

---

## 1. System Parameters

| Parameter | Value |
|-----------|-------|
| Star mass | 1.19 M☉ (2.366 × 10³⁰ kg) |
| Star radius | 4.06 R☉ (2,825,760 km) |
| Planet mass | 5.2 M♃ |
| Planet radius | 1.155 R♃ (82,573 km) |
| Semi-major axis | 0.0896 AU (1.34 × 10⁷ km) |
| Orbital velocity | ~99.55 km/s |
| Star barycentric velocity | 0.4518 km/s |
| Hill sphere | 1,496,020 km (0.010 AU) |

Kepler-432b is a hot Jupiter on a tight orbit with an exceptionally high orbital velocity (~100 km/s). This makes it a strong candidate for planet-mediated gravitational slingshots, where an interstellar object extracts kinetic energy from the binary's orbital motion.

---

## 2. Run Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N particles | 24,000 | Compensates for tight spatial targeting |
| Velocity range | 15–150 km/s | Covers real ISOs ('Oumuamua ~26, Borisov ~32 km/s) |
| Impact parameter | 0.03–0.25 AU | Brackets planet orbit (0.0896 AU) ± few r_Hill |
| Angular range | ±90° | Full hemisphere for diverse approach geometries |
| Star filter | 2.0 R★ | Rejects unphysical star-penetrating trajectories |
| Selection metric | `bary_delta_v` | Raw scalar ΔV — primary research objective |
| Unbound requirement | both | Must be unbound on entry AND exit (interstellar) |
| ODE integrator | DOP853 (8th order) | rtol = atol = 10⁻¹⁰ |
| Softening | 1,000 km | Prevents numerical singularity at close approach |
| Flyby r_min threshold | 5 × r_Hill | Must approach within 5 Hill radii |
| Max integration time | 5 × 10⁷ s (~1.6 yr) | Allows full traversal of the binary system |

---

## 3. Monte Carlo Results

### 3.1 Acceptance Statistics

| Stage | Count | Fraction |
|-------|-------|----------|
| Total particles | 24,000 | 100% |
| **Successful** | **1,010** | **4.2%** |
| Rejected: unbound requirement | 11,322 | 47.2% |
| Rejected: flyby too distant | 7,872 | 32.8% |
| Rejected: analysis failed | 2,281 | 9.5% |
| Rejected: flyby incomplete | 824 | 3.4% |
| Rejected: **star too close** | **691** | **2.9%** |

The 4.2% acceptance rate reflects the tight impact parameter window targeting the planet's orbital radius. The star proximity filter (new in v2.3) successfully rejected 691 trajectories (2.9%) that would have passed within 2.0 R★ of the star — these are unphysical trajectories that would be destroyed in reality.

### 3.2 Population Statistics (1,010 successful particles)

| Metric | Value |
|--------|-------|
| Scalar ΔV range | −65.61 to +24.22 km/s |
| Scalar ΔV mean | −25.03 ± 11.45 km/s |
| Deflection range | −152.4° to +152.0° |
| Deflection mean | −44.2° ± 79.4° |

The population is strongly biased toward negative ΔV (deceleration), which is physically expected: most random encounters slow the particle as it exchanges energy with the gravitational field. The few particles with positive ΔV are the scientifically interesting slingshot candidates.

### 3.3 Top Candidate Selection

101 candidates selected (top 10% by scalar ΔV). All 101 passed the star proximity filter at high-resolution re-run with 0 star-proximity rejections in Phase 3.

---

## 4. Best Candidates

### 4.1 Best by Scalar ΔV — MC#14155

This is the primary research metric: maximum speed gain of the interstellar object.

| Metric | Value |
|--------|-------|
| **Scalar ΔV (|v_f| − |v_i|)** | **+24.22 km/s** |
| Vector |ΔV_vec| | 109.04 km/s |
| Scattering energy ½\|ΔV_vec\|² | 5,944.54 km²/s² |
| Deflection | 57.0° |
| r_min (planet) | 221,875 km (2.7 R_p) |
| r_min (star) | 9,256,129 km (3.3 R★) ✓ |
| Impact parameter | 0.1026 AU |
| ε_i (initial specific energy) | 3.077 × 10³ km²/s² |
| ε_f (final specific energy) | 7.069 × 10³ km²/s² |
| **Δε (energy gain)** | **+3,992 km²/s²** |

**Planet-frame diagnostics:**

| Metric | Value |
|--------|-------|
| v_rel planet (inbound) | 61.96 km/s |
| v_rel planet (outbound) | 157.17 km/s |
| Δv planet frame | +95.21 km/s |
| Planet deflection | 93.9° |
| ε from planet orbit | −0.048 km²/s² |
| Δε monopole (from stellar potential) | 3,992.07 km²/s² |

**Physical interpretation:** This object approached the planet at 62 km/s (relative to planet) and departed at 157 km/s — a 95 km/s speed gain in the planet's rest frame. The near-90° deflection in the planet frame combined with the planet's 100 km/s orbital velocity produces a classic powered slingshot effect. The object essentially "scooped" kinetic energy from the planet's orbital motion. The stellar potential contributes ~3,992 km²/s² to the energy change (the monopole term from the combined-mass point potential), while the planet-specific orbital energy extraction is negligibly small (−0.048 km²/s²), indicating that the energy gain is dominated by the gravitational assist geometry rather than tidal coupling to the binary orbit.

### 4.2 Best by Scattering Energy ½|ΔV_vec|² — MC#22695

This metric captures total velocity-vector change regardless of direction.

| Metric | Value |
|--------|-------|
| Scalar ΔV | +9.61 km/s |
| **Vector |ΔV_vec|** | **185.93 km/s** |
| **½\|ΔV_vec\|²** | **17,284.68 km²/s²** |
| Deflection | 115.9° |
| r_min (star) | 11,668,058 km (4.1 R★) ✓ |
| r_min (planet) | 160,639 km (1.9 R_p) |
| ε_i | 7.921 × 10² km²/s² |
| ε_f | 4.983 × 10³ km²/s² |
| **Δε** | **+4,191 km²/s²** |

**Planet-frame diagnostics:**

| Metric | Value |
|--------|-------|
| v_rel planet (inbound) | 124.34 km/s |
| v_rel planet (outbound) | 29.31 km/s |
| Δv planet frame | −95.03 km/s |
| ε from planet orbit | +4.24 km²/s² |

**Physical interpretation:** This object entered very fast (124 km/s relative to planet) and was strongly deflected (116°), exiting slowly (29 km/s) relative to the planet. Despite *losing* speed in the planet frame, it gained a massive change in direction that, when transformed back to the barycentric frame, produces a huge 186 km/s velocity-vector change. The scalar speed gain is modest (+9.6 km/s) because the direction change dominates over speed change. This candidate maximises raw momentum transfer rather than speed gain.

---

## 5. Two-Body Baselines (Narrowed)

The narrowed baseline sweep matches 2-body hyperbolic encounters to the velocity/geometry envelope of the 3-body top candidates, providing a fair "apples-to-apples" comparison.

### 5.1 Baseline Results

| Metric | Star 2-body | Planet 2-body | 3-body Best |
|--------|-------------|---------------|-------------|
| Max ½\|ΔV_vec\|² (km²/s²) | 22,447.42 | 57.33 | 17,284.68 |
| Max \|ΔV_vec\| (km/s) | 211.88 | 10.71 | 185.93 |
| Max ε = ½v∞² (km²/s²) | 48,879.37 | 48,880.94 | 17,284.68 |

### 5.2 CDF Analysis

| Population | Exceeds Star ceiling | Exceeds Planet ceiling |
|------------|---------------------|----------------------|
| Coarse MC (1,010 particles) | 0.0% | 94.3% |
| High-res re-run (101 candidates) | 0/101 (0.0%) | 101/101 (100.0%) |

**Planet slingshot amplification: 301.5× the planet-only 2-body ceiling.**

---

## 6. Key Findings

### 6.1 Star Filter Validation

The star proximity filter (new in v2.3) is working as intended:
- **691 trajectories rejected** at the MC stage for approaching within 2.0 R★
- Both best candidates pass the filter comfortably (3.3 R★ and 4.1 R★)
- Previous runs (pre-v2.3) found "best" candidates at 0.06 R★ — deep inside the star. These are now correctly excluded.

### 6.2 Three-Body vs Two-Body

**The 3-body system cannot surpass the star 2-body ceiling** (0/1,010 = 0%). This is expected and correct: a single deep stellar flyby at the same velocity produces maximum possible ΔV. The star is 1.19 M☉ and the test particle can dive to within 2 R★ in the 2-body case.

**The 3-body system massively exceeds the planet 2-body ceiling** (94.3% of all successful particles). This is the scientifically important result: the planet alone, as an isolated body, can only produce ½|ΔV|² ≈ 57 km²/s² — but embedded in the binary potential, the same planet mediates energy exchanges of up to 17,285 km²/s². This is **301.5× amplification** over the planet-only ceiling.

### 6.3 Energy Source

The energy gained by the interstellar object comes from the **binary's orbital kinetic energy**. The planet acts as a gravitational lever:

1. The object approaches on a hyperbolic orbit in the combined stellar+planetary potential
2. The planet deflects the object's trajectory (up to ~116° observed)
3. This deflection, timed with the planet's orbital motion, converts orbital KE into the object's kinetic energy
4. The star provides the deep potential well that determines the overall energy scale

The planet-frame speed change (Δv ≈ ±95 km/s in both best candidates) confirms this mechanism: the planet swings the velocity vector, and the planet's ~100 km/s orbital speed provides the energy reservoir.

### 6.4 Two Distinct Slingshot Modes

The two best candidates illustrate fundamentally different slingshot modes:

| Mode | MC#14155 (speed gain) | MC#22695 (scattering) |
|------|----------------------|----------------------|
| Scalar ΔV | +24.22 km/s | +9.61 km/s |
| \|ΔV_vec\| | 109 km/s | 186 km/s |
| Planet-frame Δv | +95 km/s (gains speed) | −95 km/s (loses speed) |
| Deflection | 57° | 116° |
| Primary effect | Speed boost | Direction change |
| Use case | Escape acceleration | Orbital redirection |

**MC#14155** is a classic Voyager-type slingshot: gain speed from the planet's orbital motion. **MC#22695** is a gravitational mirror: massive direction change with modest speed gain but enormous momentum transfer.

### 6.5 Acceptance Rate

The 4.2% acceptance rate is a natural consequence of the tight targeting (impact parameter 0.03–0.25 AU around a 0.09 AU orbit). The unbound requirement rejects 47% of particles — many arrive too slow to escape the combined potential. The flyby-too-distant rejection (33%) reflects particles whose impact parameters miss the planet's Hill sphere. This is appropriate for the research goal: we are deliberately targeting the narrow geometric window where planet-mediated slingshots operate.

---

## 7. Assessment & Limitations

### 7.1 Model Strengths
- **Full 3-body dynamics** in the barycentric (inertial) frame — no patched conics or restricted approximations
- **High-order integrator** (DOP853, tol = 10⁻¹⁰) ensures numerical energy conservation
- **Star proximity filter** eliminates unphysical trajectories
- **Narrowed baselines** provide fair 2-body comparison by matching the 3-body velocity envelope

### 7.2 Model Limitations
- **Planar (2D)**: All trajectories are coplanar with the binary orbit. Real interstellar objects approach from arbitrary inclinations. The 2D restriction overestimates encounter probability but correctly captures the energy exchange physics for in-plane encounters.
- **Circular orbit**: Kepler-432b has eccentricity e ≈ 0.51; the circular approximation overestimates the planet's speed at apoapsis and underestimates it at periapsis. The actual slingshot efficiency would vary with orbital phase.
- **Fixed planet phase**: The planet is placed at a single orbital phase at t = 0. A full study should sweep over planet phases to find the optimal encounter geometry.
- **No tidal/atmospheric effects**: At 1.9 R_p closest approach, tidal heating and atmospheric drag could be non-negligible for a real encounter.
- **Mass ratio**: The test particle approximation assumes the interstellar object's mass is negligible compared to the planet. Valid for any realistic ISO.

### 7.3 Recommendations for Next Steps
1. **Phase sweep**: Run the MC at multiple planet orbital phases to map the slingshot efficiency as a function of encounter timing
2. **Star filter sweep**: Vary `star_min_clearance_Rstar` from 1.0 to 5.0 to map how the optimal ΔV population changes with stellar proximity constraints
3. **Eccentric orbit**: Implement Kepler-432b's e = 0.51 eccentricity to assess the impact of orbital phase on slingshot efficiency
4. **3D extension**: Allow out-of-plane approaches to quantify the geometric cross-section reduction
5. **Velocity targeting**: Focus on the 'Oumuamua/Borisov velocity regime (20–40 km/s) for astrophysically realistic ISOs

---

## 8. Summary

This run validates the v2.3 interstellar slingshot pipeline on Kepler-432b with 24,000 Monte Carlo particles and the new star proximity filter. The key result is a **301.5× amplification** of the planet-only scattering energy ceiling through three-body dynamics: a planet that can only produce ~57 km²/s² of scattering energy in isolation mediates up to 17,285 km²/s² when embedded in the stellar potential. The best scalar speed gain is **+24.22 km/s**, achieved by a trajectory that passes 2.7 R_p from the planet and 3.3 R★ from the star — both physically valid. The star filter successfully excludes the 691 unphysical star-penetrating trajectories that contaminated earlier runs. No 3-body trajectory exceeds the star 2-body ceiling, confirming that the star sets the absolute energy scale while the planet acts as the lever that makes it accessible to interstellar objects on planet-crossing trajectories.
