# Changelog

## 0.0.0

Geometry, the Gauss-Helmert plane solvers, the five baseline methods, and the
five simulation experiment scripts. This version is not published to PyPI.

### Added

- `handeye2d` package skeleton (`numpy`, `scipy`, `pyyaml`), pytest, and ruff.
- GitHub Actions for the Python tree, plus a manual/weekly Octave fixture check.
- Octave reference fixtures under `tests/fixtures/`, including `bug_impact.mat`.
- `so3`, `rotations`, `fitting`, `metrics`, and the OCT YAML readers.
- `calib.sphere_meas2023`, checked against the Octave sphere fixture and the
  ten saved real sphere sessions.
- `calib.plane_jnt_gh`, `calib.plane_alt_gh`, and `calib.plane_jnt_gh_decoupled`,
  one shared Gauss-Helmert loop. Checked against the Octave plane fixtures and,
  for the joint solver, the ten saved real plane sessions.
- `sim.sim_plane` and `sim.sim_plane_batch`. A seed of `sim_plane` reproduces
  that function. `sim_plane_batch` draws many sessions for plane-fit residuals
  and does not replay sequential `sim_plane` calls.
- Baseline solvers `plane_tro2016`, `plane_rcim2020`, `plane_iros2015`, and
  `line_rcim2022` (`sphere_meas2023` was already present), plus simulators
  `sim_iros2015`, `sim_rcim2020`, `sim_rcim2022`, and `sim_meas2023`.
- `docs/bug-impact.md`, from `tests/fixtures/bug_impact.mat`.
- `experiments/sim_accuracy_vs_{noise_level,number_of_measurements,initial_deviation,image_noise,VFE}.py`,
  with figures, median (IQR) tables, and `.npz` output. `--n-jobs` gives each
  trial its own seed. The noise-level plane-fit loop is vectorized.
- `docs/simulation-agreement.md`.

### Fixed

- `load_plane_yaml` fits `pOn` and `vOn` on each original profile. Resampling
  to the shortest scan applies only to `scanO`, as in `loadPlaneYAML`.

### Fixed in Python only

MATLAB sources are unchanged. See `docs/bug-impact.md`.

- `line_rcim2022` writes pose `i` into columns `(1:(mt-1))+(mt-1)*(i-1)`.
  `fun/calib/line_RCIM2022.m:58` writes every pose into
  `ub(:,(1:(mt-1))+(mt-1))`.
- `sim_meas2023` clips the sample arc with `xMax`.
  `fun/sim/simMEAS2023.m:90` uses `zMax`.
- `plane_tro2016` stops on the flat shoulder of the Euler-angle cost, where
  `lsqnonlin` stops. SciPy's relative function tolerance would otherwise
  continue into a lower, distant basin. SciPy builds that still accept
  Python 3.10 have no `least_squares` callback, so that stop also runs from
  the Jacobian evaluated at each accepted point.
