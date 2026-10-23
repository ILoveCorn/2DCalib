# handeye2d

Python package for [2DCalib](https://github.com/ILoveCorn/2DCalib), the code
for *Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical
Coherence Tomography Using a Plane Target* (IEEE T-RO, 2026,
[10.1109/TRO.2026.3710397](https://doi.org/10.1109/TRO.2026.3710397)).

**This tree is the package skeleton and the Octave reference fixtures.** The
calibration and simulation algorithms are not ported yet, and nothing here is
published to PyPI. The import name and the distribution name are both
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
├─ src/handeye2d/          import package (algorithms arrive later)
├─ tests/fixtures/         committed Octave oracles (MAT v7)
└─ tools/octave/
   ├─ export_fixtures.m    only .m file under python/
   ├─ export_fixtures.py   regenerate fixtures and MANIFEST.json
   ├─ compare_fixtures.py  tolerance check for the weekly job
   └─ shims/*.m.in         Octave stand-ins, never stored as .m
```

Planned modules, not in this version: `so3`, `rotations`, `fitting`, `metrics`,
`io`, `calib`, and `sim`. Names will be snake_case. Batch dimensions come
first (`tH2W` as `(m, 3)`, `RH2W` as `(m, 3, 3)`). There is no GPU build and
no replacement for the missing `*_debug` functions.

Real-data scripts that come later follow scope B: YAML loaders,
`real_calibration`, and `real_plane`.

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
GPU variant; `*_debug` solvers are not provided.

## Tests

| Test module | What it checks |
|---|---|
| `test_fixture_integrity.py` | manifest hashes, no shim `.m` files, patch text still matches `fun/`, shapes, noiseless recovery |
| `test_shim_scipy.py` | Euler, axis-angle, quaternion, and `rotm2eul` against SciPy; page products against NumPy / `scipy.linalg.solve` |
| `test_bug_impact.py` | column counts, arc limits, and the error change in `bug_impact.mat` |

## License

Code is [MIT](LICENSE), the same license as the repository. Datasets under
`data/` and precomputed results under `result/` stay [CC BY 4.0](../data/LICENSE).
Please cite the paper above if you use either.
