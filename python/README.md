# handeye2d

Python package for [2DCalib](https://github.com/ILoveCorn/2DCalib), the code
for *Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical
Coherence Tomography Using a Plane Target* (IEEE T-RO, 2026,
[10.1109/TRO.2026.3710397](https://doi.org/10.1109/TRO.2026.3710397)).

**This tree has the geometry utilities, the OCT YAML readers, the five
baseline solvers and their simulators, the three Gauss-Helmert plane
solvers, the five simulation experiment scripts, and the real-data drivers
`real_calibration` and `real_plane`.** `real_sphere` and `measure` are not
ported, and nothing here is published to PyPI.
The import name and the distribution name are both `handeye2d`.

The MATLAB code at the repository root is unchanged.

## Install

From this directory:

```bash
python -m pip install -e ".[dev,experiments]"
```

Python 3.10 or newer. Runtime dependencies are NumPy, SciPy, and PyYAML.
The `experiments` extra adds matplotlib and tqdm for the simulation scripts.

```bash
ruff check src tests tools experiments
pytest
```

## Quick start

This is the Python form of the minimal example in the repository README.
Run it from this directory. Paths are files on disk; the loaders do not use
MATLAB's search path. The initial pose is the first scan in
`data/measure_260219_1144.mat`, which is what `real_calibration` uses.
`plane_jnt_gh` returns a `PlaneGhResult` (`tO2H`, `RO2H`, `n`, `d`, `tH2W`,
`RH2W`, `sigma`, `time`).

```python
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from handeye2d.calib import plane_jnt_gh
from handeye2d.fitting import plane_fit
from handeye2d.io import load_plane_yaml

repo = Path("..")
measured = loadmat(repo / "data" / "measure_260219_1144.mat")
t0 = np.asarray(measured["tO2H"], dtype=float).reshape(3, -1)[:, 0]
R0 = np.asarray(measured["RO2H"], dtype=float).reshape(3, 3, -1)[:, :, 0]
session = repo / "data" / "oct" / "plane" / "oct_plane_250710_1057.yaml"
hand_t, hand_R, position, direction, scan = load_plane_yaml(session)
normal, distance, _residual = plane_fit(t0, R0, hand_t, hand_R, scan)
result = plane_jnt_gh(
    t0,
    R0,
    normal,
    distance,
    np.array([0.1, 0.01]),
    hand_t,
    hand_R,
    position,
    direction,
    30,
    50,
    1e-5,
)
print(result.tO2H)
print(result.RO2H)
```

## Layout

```text
python/
├─ pyproject.toml
├─ src/handeye2d/
│  ├─ so3.py               hat, exp, log, jr, jr_inv, random_rotvec
│  ├─ rotations.py         quaternion, Euler, axis-angle
│  ├─ fitting.py           plane_fit, circle_fit
│  ├─ metrics.py           calibration_error
│  ├─ io.py                OCT YAML readers
│  ├─ sim/                 sim_plane, sim_iros2015, sim_rcim2020,
│  │                       sim_rcim2022, sim_meas2023
│  └─ calib/               Gauss-Helmert solvers and the five baselines
├─ experiments/            sim_accuracy_vs_*.py, real_calibration, real_plane
├─ docs/bug-impact.md      measured effect of the two MATLAB bugs
├─ docs/simulation-agreement.md
├─ docs/real-data.md       plane reconstruction means on the saved mat
├─ tests/fixtures/         committed Octave oracles (MAT v7)
└─ tools/octave/
   ├─ export_fixtures.m    only .m file under python/
   ├─ export_fixtures.py   regenerate fixtures and MANIFEST.json
   ├─ compare_fixtures.py  tolerance check for the weekly job
   └─ shims/*.m.in         Octave stand-ins, never stored as .m
```

Names are snake_case. Batch axes come first. MATLAB stores the same poses
with the batch axis last.

| Quantity | MATLAB | Python |
|---|---|---|
| `tH2W` | `3×1×m` | `(m, 3)` |
| `RH2W` | `3×3×m` | `(m, 3, 3)` |
| `scanO` | `3×n×m` | `(m, n, 3)` |
| `tO2H` for one pose | `3×1` | `(3,)` |
| `RO2H` for one pose | `3×3` | `(3, 3)` |

`experiments/real_calibration.py` keeps the saved-mat layout in its `.npz`:
`alltO2He` is `(3, n, 5)` and `allRO2He` is `(3, 3, n, 5)`. There is no GPU
build and no replacement for the missing `*_debug` functions. NumPy's random
stream is not Octave's or MATLAB's, so a seed of `sim_*` does not reproduce a
committed fixture or a MATLAB experiment table. `plane_tro2016` calls
`scipy.optimize.least_squares` (`method='trf'`, tolerances `1e-6`) in place
of `lsqnonlin`.

## Function map

| MATLAB | Python |
|---|---|
| `hat`, `pagehat` | `so3.hat` |
| `ExpSO3`, `pageExpSO3` | `so3.exp` |
| `pageLogSO3` | `so3.log` |
| `JrSO3`, `pageJrSO3`, `invJrSO3` | `so3.jr`, `so3.jr_inv` |
| `randSO3` | `so3.random_rotvec(rng)` |
| `quat2rotm` | `rotations.quat_to_rotm` (scalar first) |
| `eul2rotm` | `rotations.eul_to_rotm(angles, seq)` |
| `rotm2eul` | `rotations.rotm_to_eul` |
| `axang2rotm` | `rotations.axang_to_rotm(axis, theta)` |
| `rotm2axang` | `rotations.rotm_to_angle` (angle only, radians) |
| `calibrationError` | `metrics.calibration_error` |
| `planeFit`, `circleFit` | `fitting.plane_fit`, `fitting.circle_fit` |
| `loadPlaneYAML`, `loadTestYAML`, `loadMEAS2023YAML` | `io.load_plane_yaml`, `io.load_test_yaml`, `io.load_meas2023_yaml` |
| `simPlane` | `sim.sim_plane` (`sim.sim_plane_batch` is the vectorized residual helper) |
| `simIROS2015`, `simRCIM2020`, `simRCIM2022`, `simMEAS2023` | `sim.sim_iros2015`, `sim.sim_rcim2020`, `sim.sim_rcim2022`, `sim.sim_meas2023` |
| `plane_Jnt_GH`, `plane_Alt_GH`, `plane_Jnt_GH_decoupled` | `calib.plane_jnt_gh`, `calib.plane_alt_gh`, `calib.plane_jnt_gh_decoupled` |
| `plane_TRO2016`, `plane_RCIM2020`, `plane_IROS2015` | `calib.plane_tro2016`, `calib.plane_rcim2020`, `calib.plane_iros2015` |
| `line_RCIM2022`, `sphere_MEAS2023` | `calib.line_rcim2022`, `calib.sphere_meas2023` |
| `evaluation/sim_accuracy_vs_*.m` | `python -m experiments.sim_accuracy_vs_*` |
| `evaluation/real_calibration.m`, `evaluation/real_plane.m` | `python -m experiments.real_calibration`, `python -m experiments.real_plane` |

`plane_fit` may return `n` or `-n`; `d` flips with the normal. `pagemtimes`
and `pagemrdivide` are `@` and `np.linalg.solve`. Quartiles in the experiment
tables use `np.percentile(..., method="hazen")`, which matches MATLAB
`prctile` plotting positions.

These MATLAB pieces are absent here: `plane_Jnt_GH_gpu`, every `*_debug`
solver, `sphereFit`, the LiDAR branch of the YAML loaders, `measure.m`,
`real_sphere.m`, and `real_filter_comparison.m`. The two bug fixes are in
[Differences from the MATLAB code](#differences-from-the-matlab-code).

## Simulation experiments

From this directory, with the `experiments` extra installed:

```bash
python -m experiments.sim_accuracy_vs_noise_level --n-test 100 --n-jobs 4 --no-show
python -m experiments.sim_accuracy_vs_number_of_measurements --n-test 100 --n-jobs 4 --no-show
python -m experiments.sim_accuracy_vs_initial_deviation --n-test 100 --n-jobs 4 --no-show
python -m experiments.sim_accuracy_vs_image_noise --n-test 100 --n-jobs 4 --no-show
python -m experiments.sim_accuracy_vs_VFE --n-test 100 --n-jobs 4 --no-show
```

Each script takes `--n-test`, `--seed`, `--n-jobs`, `--out`, and `--no-show`.
The noise-level script also takes `--n-plane-fit` (default 50). Every trial
gets its own `SeedSequence`, so the sample does not depend on `--n-jobs`.
Outputs are a `.npz` of the raw errors, a median (IQR) table, and the figures
under `--out` (default `experiments/output/`).

The noise-level and measurement-count scripts run Xu, the proposed
Gauss-Helmert method, Kang (TRO 2016), Sharifzadeh (RCIM 2020), Carlson
(IROS 2015), and Tang (MEAS 2023). They do not run the alternating
Gauss-Helmert ablation, the decoupled ablation, or a GPU solver. Kang and
Sharifzadeh reuse the plane cloud drawn for the proposed method. The 50
plane-fit clouds are drawn by `sim_plane_batch` on a second seed. The initial
deviation script calls `plane_jnt_gh` and does not report inner or outer
iteration counts. Its time column is milliseconds, which is what the MATLAB
`fprintf` prints; that file's column header says seconds.

Xu and Tang will not reproduce the paper's simulation tables. `line_rcim2022`
fixes the column index at `fun/calib/line_RCIM2022.m:58`, and `sim_meas2023`
clips the arc with `xMax` instead of `zMax` at `fun/sim/simMEAS2023.m:90`.
The measured effect of those two edits is in
[`docs/bug-impact.md`](docs/bug-impact.md). What a reduced NumPy run does and
does not say about the paper is in
[`docs/simulation-agreement.md`](docs/simulation-agreement.md).

## Real data

From this directory:

```bash
python -m experiments.real_calibration --out experiments/output
python -m experiments.real_plane --out experiments/output
```

`real_calibration` reads `data/measure_260219_1144.mat` and the ten plane and
ten sphere sessions. Every plane method starts from the first measured pose.
It writes `experiments/output/real_calibration.npz` and does not write into
`result/`. Poses are compared with `result/real_calibration_260219_1146.mat`.
Wall-clock time is not. Session order is the sorted file name, which matches
that mat after `oct_checkerboard` is renamed to `oct_plane`.

`real_plane` loads that mat unless `--calibration` points at an `.npz` from
the command above. For the three plane methods, the session used to calibrate
is left out. MEAS 2023 and the measured poses are scored on every test
session. The printed means are in [`docs/real-data.md`](docs/real-data.md).

`real_sphere` and `measure` are not ported.

## Octave fixtures

`tools/octave/export_fixtures.m` copies `fun/` into a temporary directory,
patches the two bugs below, and runs the simulators and solvers with Octave's
`rng`. Shim sources are copied from `shims/*.m.in` to `*.m` only inside that
temporary directory, so a MATLAB session that runs `addpath(genpath(pwd))`
cannot see `pagemtimes.m`, `eul2rotm.m`, or the other stand-ins.

`plane_TRO2016` is not part of the Octave set: Octave has no `lsqnonlin`.

Regenerate (Octave 8.4 is what produced the committed files):

```bash
python python/tools/octave/export_fixtures.py
```

`tests/fixtures/MANIFEST.json` hashes array contents. MAT headers include a
timestamp, so file bytes are not the checksum. `tests/fixtures/SOURCE.txt`
records the Octave version and the seeds.

The noiseless plane, line, and sphere fixtures recover the hand-eye pose used
to generate them. `plane_Alt_GH` is stopped at `iOmax = 5`, which improves on
the initial guess and is not yet the noiseless fixed point. `simIROS2015`
uses a 6 by 6 Euler grid: a 4 by 4 grid leaves only four shots per plane after
`round(mAlpha/sqrt(nPlane))`, and `plane_IROS2015` does not recover.

## Differences from the MATLAB code

Two bugs stay in `fun/` and are patched only on the temporary copy that builds
these fixtures. Python implements the corrected behavior and does not keep
the original one. The measured effect is in
[`docs/bug-impact.md`](docs/bug-impact.md). The simulation scripts use those
corrections, so the Xu and Tang columns are not a reproduction of the paper.
See [`docs/simulation-agreement.md`](docs/simulation-agreement.md).

| Location | MATLAB | Corrected reference |
|---|---|---|
| `fun/calib/line_RCIM2022.m:58` | `ub(:,(1:(mt-1))+(mt-1))` overwrites one block of columns on every pose, so only the last pose enters the line-direction estimate | `(1:(mt-1))+(mt-1)*(i-1)` |
| `fun/sim/simMEAS2023.m:90` | right edge of the sphere arc is `min(..., zMax)` | `min(..., xMax)` |

`tests/fixtures/bug_impact.mat` is the measurement of those two changes, checked
by `tests/test_bug_impact.py`:

- Unpatched `line_RCIM2022` leaves `mt - 1` nonzero columns in `ub`. The patch
  fills all `m*(mt - 1)` columns. Orientation error is unchanged, because the
  bug is after `Rsa` is solved. With no pose noise both position errors are
  ~1e-7 mm. At 0.1 mm / 0.1° the patched position error is smaller.
- With the default `xMax = 5` and `zMax = 2`, every unpatched sample stops at
  `x = zMax - 1e-4`. The patch reaches `xMax - 1e-4`. On the noisy sphere
  trials the unpatched calibration error is several times larger.

Also, relative to MATLAB: Octave's random stream is not MATLAB's; there is no
GPU variant; `*_debug` solvers are not provided. `so3.exp`, `so3.log`,
`so3.jr`, and `so3.jr_inv` share one small-angle threshold, `1e-10`, with the
Taylor coefficients from the `page*` forms. The scalar MATLAB functions use a
stricter cutoff and, below it, return the identity.

## Tests

| Test module | What it checks |
|---|---|
| `test_fixture_integrity.py` | manifest hashes, no shim `.m` files, patch text still matches `fun/`, shapes, noiseless recovery |
| `test_shim_scipy.py` | Euler, axis-angle, quaternion, and `rotm2eul` against SciPy; page products against NumPy / `scipy.linalg.solve` |
| `test_bug_impact.py` | column counts, arc limits, and the error change in `bug_impact.mat` |
| `test_so3.py`, `test_rotations.py` | Rodrigues, Jacobians, and the shim rotation cases |
| `test_fitting.py`, `test_metrics.py` | circle and plane fits; `calibration_error` against every fixture that stored it |
| `test_io.py` | units, MATLAB `round(linspace)`, MEAS grouping, no LiDAR branch |
| `test_sphere_meas2023.py` | closed form against `sphere_meas2023.mat` and a known pose |
| `test_plane_gh.py` | joint, alternating, and decoupled solvers against the Octave plane fixtures |
| `test_sim_plane.py` | Euler grid, plane incidence, and a reproducible NumPy stream |
| `test_baselines.py` | RCIM2020, IROS2015, and RCIM2022 against the Octave fixtures; the patched `ub` column index |
| `test_sim_baselines.py` | deterministic simulator orientations, noiseless pose recovery, and the `xMax` arc |
| `test_real_data.py` | ten MEAS2023, `plane_jnt_gh`, `plane_rcim2020`, and `plane_tro2016` sessions against `result/real_calibration_260219_1146.mat`; measured slot against `data/measure_260219_1144.mat` |
| `test_experiments.py` | `--n-test 2` smoke for the five simulation scripts, including parallel seeds |
| `test_real_drivers.py` | `real_calibration` poses against the saved mat; `real_plane` leave-one-out means |

## Citation

Cite the paper. This package is not on PyPI and has no Zenodo record.
The repository citation file is
[`../CITATION.cff`](../CITATION.cff); its keywords include `Python`.

```bibtex
@article{ma2026uncertainty,
  author  = {Ma, Yan and Lyu, Haitian and Lai, Jiewen and Zhang, Ruiyang and Yuan, Wu and Ren, Hongliang},
  title   = {Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical Coherence Tomography Using a Plane Target},
  journal = {IEEE Transactions on Robotics},
  year    = {2026},
  doi     = {10.1109/TRO.2026.3710397}
}
```

## License

Code is [MIT](LICENSE), the same license as the repository. Datasets under
`data/` and precomputed results under `result/` stay [CC BY 4.0](../data/LICENSE).
