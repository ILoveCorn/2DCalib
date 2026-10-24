"""Hand-eye calibration for 2-D profile sensors.

Distribution and import name: ``handeye2d``. This version has the SO(3)
utilities, rotation conversions, plane and circle fits, the calibration
error, the OCT YAML readers, the Gauss-Helmert plane solvers, and the five
baseline methods with their simulators. The five simulation drivers live in
``python/experiments`` and are not part of this import package.

Array batches come first (``tH2W`` as ``(m, 3)``, ``RH2W`` as ``(m, 3, 3)``,
``scanO`` as ``(m, n_points, 3)``). Function names are snake_case. Numeric
oracles are produced by ``python/tools/octave`` and checked in under
``python/tests/fixtures``. The saved real-calibration result is the oracle
for MEAS2023, the joint Gauss-Helmert solver, RCIM2020, and TRO2016.
"""

from handeye2d import calib, fitting, io, metrics, rotations, sim, so3

__version__ = "0.0.0"

__all__ = ["calib", "fitting", "io", "metrics", "rotations", "sim", "so3"]
