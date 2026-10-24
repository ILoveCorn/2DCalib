# Simulation agreement

The five drivers in `experiments/` follow `evaluation/sim_*.m` with the cuts in
the port plan: no `plane_Alt_GH_debug`, no GPU solver, and no inner or outer
iteration columns. NumPy's generator is not MATLAB's, so these runs are not a
bitwise reproduction of a MATLAB seed.

## What was compared

The published table digits are not in this repository. The paper
(DOI 10.1109/TRO.2026.3710397) is on IEEE Xplore and the open pages that were
checked do not include the simulation tables. The 15% median test against
those digits was therefore not scored. The numbers below are this port, seed
0, 20 trials per setting, run with `--n-jobs 4`. The noise-level plane-fit
loop used `--n-plane-fit 10` rather than 50; calibration errors do not use
those extra clouds.

Xu's position error and both of Tang's errors are expected to differ from the
paper even at a much larger sample. `line_rcim2022` uses the column fix for
`fun/calib/line_RCIM2022.m:58`. On the fixture that bug does not change the
orientation error, and it does change the position error once pose noise is
present (`bug-impact.md`). `sim_meas2023` clips the arc with `xMax` instead of
`zMax` (`fun/sim/simMEAS2023.m:90`), which moves Tang by millimeters on the
fixture. Kang uses `scipy.optimize.least_squares`, not `lsqnonlin`.

## Noise level

Isotropic position medians, in millimeters, at 0.01 / 0.1 / 0.5 / 1:

| Method | 0.01 | 0.1 | 0.5 | 1 |
|---|---:|---:|---:|---:|
| Xu | 0.046 | 0.349 | 1.820 | 2.279 |
| Proposed | 0.006 | 0.073 | 0.471 | 0.822 |
| Kang | 0.009 | 0.091 | 0.580 | 0.932 |
| Sharifzadeh | 0.034 | 0.247 | 1.304 | 2.194 |
| Carlson | 0.027 | 0.267 | 1.098 | 1.698 |
| Tang | 0.011 | 0.076 | 0.460 | 1.031 |

Isotropic orientation medians, in degrees:

| Method | 0.01 | 0.1 | 0.5 | 1 |
|---|---:|---:|---:|---:|
| Xu | 1.415 | 10.599 | 59.057 | 65.722 |
| Proposed | 0.007 | 0.057 | 0.401 | 0.750 |
| Kang | 0.030 | 0.305 | 1.650 | 2.488 |
| Sharifzadeh | 0.863 | 6.908 | 36.053 | 55.927 |
| Carlson | 0.686 | 7.589 | 30.882 | 45.309 |
| Tang | 0.027 | 0.350 | 1.831 | 2.696 |

On this draw the proposed orientation median is the smallest at every noise
setting, and every method's isotropic position and orientation median rises
as the noise setting goes from 0.01 to 1. Position is less tidy: at 0.5 mm
Tang's median (0.460) is slightly under the proposed median (0.471). The
anisotropic block in `sim_accuracy_vs_noise_level.txt` from the same run has
the same orientation ranking. That is the qualitative shape of the experiment.
It is not a claim that these medians sit inside the paper's confidence
intervals.

## Other scripts, same 20 trials

- Initial deviation, threshold 1.5 mm and 1.5°. All 20 trials at each of the
  five deviations (10°/1 mm through 50°/20 mm) finished inside the threshold.
  Position medians stayed between 0.31 mm and 0.40 mm. This sample does not
  show the convergence rate falling as the initial guess gets worse. A sample
  of 20 can miss failures that 100 trials would catch.
- Image noise, position in micrometers. Ideal 427 (IQR 332), speckle 427
  (0.10% above ideal), distortion 423 (0.79% below ideal). The shifts are
  much smaller than the IQR, which is what those sub-micrometer profile
  perturbations should do on top of 0.5 mm / 0.5° pose noise.
- Variance factors, truth 1 mm and 0.1°. The VFE medians were 0.985 mm and
  0.100°. Relative median errors were -1.5% and +0.17%. Equal weights, with
  `iOmax = 1` and `sigma0 = [1, 1]`, were worse: 1.32 mm and 1.53°, against
  0.76 mm and 0.26° with VFE. Ground-truth weights and VFE stayed close.
- Measurement count. Mean solver time rose from 36 measurements through 784
  for every method. Tang stayed the fastest. The step from 784 to 1024 was
  not monotone for every method in this timing sample; runtime is only a
  qualitative check.

## Re-run

```bash
python -m experiments.sim_accuracy_vs_noise_level --n-test 20 --n-plane-fit 10 --n-jobs 4 --no-show --seed 0
```

The other four scripts use the same `--n-test 20 --n-jobs 4 --no-show --seed 0`.
Default `--n-test` is 100, matching the MATLAB scripts.
