# Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2D Optical Coherence Tomography using a Plane Target

This repository provides MATLAB implementations, datasets, and evaluation utilities for cross-scale hand-eye calibration of 2D Optical Coherence Tomography (OCT) using a plane target.

## Environment

- **MATLAB**: R2024b

## Directory structure

```text
repository-root/
├─ data/
│  └─ oct/      Calibration datasets (plane and sphere target profile points, poses)
├─ evaluation/  Simulation and real-world experiments scripts
├─ fun/
│  ├─ calib/    Proposed calibration algorithms and baseline implementations
│  ├─ sim/      Simulation data generation functions
│  └─ misc/     Helper utilities
└─ result/      Output directory
```

## Getting started

1. Add the repository root to the MATLAB search path,
2. Run the scripts in `evaluation` to reproduce simulation and real-world experiments results.

For calibration with your own data, please read the comments on data format in `fun/calib`.