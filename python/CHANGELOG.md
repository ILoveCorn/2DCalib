# Changelog

## 0.0.0

Infrastructure only. This version is not published to PyPI and does not include
the calibration algorithms.

### Added

- `handeye2d` package skeleton (`numpy`, `scipy`, `pyyaml`), pytest, and ruff.
- GitHub Actions for the Python tree, plus a manual/weekly Octave fixture check.
- Octave reference fixtures under `tests/fixtures/`, including `bug_impact.mat`.
- `so3`, `rotations`, `fitting`, `metrics`, and the OCT YAML readers.
- `calib.sphere_meas2023`, checked against the Octave sphere fixture and the
  ten saved real sphere sessions. The other calibration routines are not in
  this version.

### Noted bugs (fixed only when generating Python reference data)

MATLAB sources are unchanged.

- `fun/calib/line_RCIM2022.m:58` writes every pose's line direction into the
  same columns, `ub(:,(1:(mt-1))+(mt-1))`. The reference patch uses
  `(1:(mt-1))+(mt-1)*(i-1)`.
- `fun/sim/simMEAS2023.m:90` clips the sample arc with `zMax` instead of
  `xMax`.
