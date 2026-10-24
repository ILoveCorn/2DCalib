"""Simulation data generators.

Only the plane generator is included. The other simulators arrive with the
baseline methods.
"""

from handeye2d.sim.sim_plane import PlaneSimulation, sim_plane

__all__ = ["PlaneSimulation", "sim_plane"]
