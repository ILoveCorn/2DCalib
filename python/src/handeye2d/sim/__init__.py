"""Simulation data generators.

One generator per calibration target. NumPy's random stream is not Octave's
or MATLAB's. ``sim_plane_batch`` draws many plane sessions for experiment
plane-fit residuals.
"""

from handeye2d.sim.sim_iros2015 import IrosSimulation, sim_iros2015
from handeye2d.sim.sim_meas2023 import SphereSimulation, sim_meas2023
from handeye2d.sim.sim_plane import PlaneSimulation, sim_plane, sim_plane_batch
from handeye2d.sim.sim_rcim2020 import Rcim2020Simulation, sim_rcim2020
from handeye2d.sim.sim_rcim2022 import LineSimulation, sim_rcim2022

__all__ = [
    "IrosSimulation",
    "LineSimulation",
    "PlaneSimulation",
    "Rcim2020Simulation",
    "SphereSimulation",
    "sim_iros2015",
    "sim_meas2023",
    "sim_plane",
    "sim_plane_batch",
    "sim_rcim2020",
    "sim_rcim2022",
]
