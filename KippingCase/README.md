# Gravity assist + Oberth at perihelion (Python reproducibility)

This repo reproduces the same closed-form planar flyby model developed in the chat:

- Star moves at constant lab-frame velocity: `V* = (0, vstar0, 0)`
- Particle starts at `(xm0, ym0, 0)` with lab velocity `(um0, vm0, 0)`
- Work in the star frame (subtract star velocity), compute a hyperbolic flyby
- Optional: apply an impulsive tangential Δv at perihelion (“Oberth burn”), which creates a second hyperbola
- Output: deflection angle(s), perihelion distance/speed, final asymptotic velocity components in the lab frame, and ΔV in the lab frame

## 1) Dependencies

Core math uses only NumPy:

```bash
pip install numpy
