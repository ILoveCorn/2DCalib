"""Hand-eye calibration for 2-D profile sensors.

Distribution and import name: ``handeye2d``. This version has the SO(3)
utilities, rotation conversions, plane and circle fits, the calibration
error, the OCT YAML readers, the MEAS2023 closed form, the plane simulator,
and the Gauss-Helmert plane solvers. The other baselines are not ported yet.

Array batches come first (``tH2W`` as ``(m, 3)``, ``RH2W`` as ``(m, 3, 3)``,
``scanO`` as ``(m, n_points, 3)``). Function names are snake_case. Numeric
oracles are produced by ``python/tools/octave`` and checked in under
``python/tests/fixtures``. The saved real-calibration result is the oracle
for MEAS2023 and for the joint Gauss-Helmert solver.
"""

from handeye2d import calib, fitting, io, metrics, rotations, sim, so3

__version__ = "0.0.0"

__all__ = ["calib", "fitting", "io", "metrics", "rotations", "sim", "so3"]
