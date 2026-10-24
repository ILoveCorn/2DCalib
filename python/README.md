# handeye2d

Python package for [2DCalib](https://github.com/ILoveCorn/2DCalib), the code
for *Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical
Coherence Tomography Using a Plane Target* (IEEE T-RO, 2026,
[10.1109/TRO.2026.3710397](https://doi.org/10.1109/TRO.2026.3710397)).

**This tree has the geometry utilities, the OCT YAML readers, the MEAS2023
closed form, the three Gauss-Helmert plane solvers, and `sim_plane`.** The
other baselines and the experiment scripts are not ported yet, and nothing
here is published to PyPI. The import name and the distribution name are both
`handeye2d`.

The MATLAB code at the repository root is unchanged.

## Install

From this directory:

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer. Runtime dependencies are NumPy, SciPy, and PyYAML.
`experiments` (matplotlib, tqdm) is reserved for the later experiment scripts.

```bash
ruff check src tests tools
pytest
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
│  ├─ sim/sim_plane.py     synthetic plane scans
│  └─ calib/
│     ├─ plane_gh.py       joint, alternating, and decoupled Gauss-Helmert
│     └─ sphere_meas2023.py
├─ tests/fixtures/         committed Octave oracles (MAT v7)
└─ tools/octave/
   ├─ export_fixtures.m    only .m file under python/
   ├─ export_fixtures.py   regenerate fixtures and MANIFEST.json
   ├─ compare_fixtures.py  tolerance check for the weekly job
   └─ shims/*.m.in         Octave stand-ins, never stored as .m
```

Names are snake_case. Batch dimensions come first (`tH2W` as `(m, 3)`,
`RH2W` as `(m, 3, 3)`, `scanO` as `(m, n_points, 3)`). There is no GPU build
and no replacement for the missing `*_debug` functions. The remaining
baseline solvers and the `evaluation/sim_*.m` experiment scripts are later
milestones.

Real-data scripts that come later follow scope B: the YAML loaders are in
place; `real_calibration` and `real_plane` are not.

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
these fixtures. Python will implement the corrected behavior and will not keep
the original one.

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
| `test_real_data.py` | ten MEAS2023 sessions and ten `plane_jnt_gh` sessions against `result/real_calibration_260219_1146.mat`; measured slot against `data/measure_260219_1144.mat` |

## License

Code is [MIT](LICENSE), the same license as the repository. Datasets under
`data/` and precomputed results under `result/` stay [CC BY 4.0](../data/LICENSE).
Please cite the paper above if you use either.
