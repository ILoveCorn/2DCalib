# Changelog

## 0.0.0

Geometry, the MEAS2023 closed form, and the Gauss-Helmert plane solvers.
This version is not published to PyPI. The other baselines are not included.

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
- `sim.sim_plane`. Its random stream is NumPy's, so it reproduces a seed of
  this function and not the committed Octave fixture.

### Fixed

- `load_plane_yaml` fits `pOn` and `vOn` on each original profile. Resampling
  to the shortest scan applies only to `scanO`, as in `loadPlaneYAML`.

### Noted bugs (fixed only when generating Python reference data)

MATLAB sources are unchanged.

- `fun/calib/line_RCIM2022.m:58` writes every pose's line direction into the
  same columns, `ub(:,(1:(mt-1))+(mt-1))`. The reference patch uses
  `(1:(mt-1))+(mt-1)*(i-1)`.
- `fun/sim/simMEAS2023.m:90` clips the sample arc with `zMax` instead of
  `xMax`.
