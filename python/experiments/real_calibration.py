"""Hand-eye calibration on the saved OCT sessions.

Port of ``evaluation/real_calibration.m``. The MATLAB script calls
``rng('default')`` and then draws no random numbers, so this driver is
deterministic. It does not write into ``result/``.

Columns are Proposed, TRO 2016, RCIM 2020, MEAS 2023, and the measured
fiducial pose. Every plane method starts from the first pose in
``data/measure_260219_1144.mat``. Session order is the sorted YAML name,
which matches ``result/real_calibration_260219_1146.mat`` after
``oct_checkerboard`` is renamed to ``oct_plane``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from experiments._real import (
    DIAMETER,
    IIMAX,
    IOMAX,
    METHODS,
    MT,
    SIGMA0,
    TOL,
    RealCalibration,
    measured_poses,
    repo_root,
    save_calibration,
    session_names,
)
from handeye2d.calib import plane_jnt_gh, plane_rcim2020, plane_tro2016, sphere_meas2023
from handeye2d.fitting import plane_fit
from handeye2d.io import load_meas2023_yaml, load_plane_yaml


def _store(translation, rotation, elapsed, index, method, poses) -> None:
    poses.translation[:, index, method] = np.asarray(translation, dtype=float).reshape(3)
    poses.rotation[:, :, index, method] = np.asarray(rotation, dtype=float).reshape(3, 3)
    poses.time[index, method] = float(elapsed)


def calibrate(root: Path | None = None, verbose: bool = False) -> RealCalibration:
    """Run the five columns. ``root`` is the repository root."""
    root = repo_root() if root is None else Path(root)
    plane_files = session_names(root / "data" / "oct" / "plane", "oct_plane*.yaml")
    sphere_files = session_names(root / "data" / "oct" / "sphere", "oct_sphere*.yaml")
    if not plane_files:
        raise FileNotFoundError(f"no plane sessions under {root / 'data' / 'oct' / 'plane'}")
    measured_t, measured_R = measured_poses(root)
    if measured_t.shape[1] < len(plane_files):
        raise ValueError(
            f"{len(plane_files)} plane sessions but only {measured_t.shape[1]} measured poses"
        )
    count = len(plane_files)
    poses = RealCalibration(
        translation=np.zeros((3, count, len(METHODS))),
        rotation=np.zeros((3, 3, count, len(METHODS))),
        time=np.zeros((count, len(METHODS))),
        plane_files=plane_files,
        sphere_files=sphere_files,
    )
    initial_t = measured_t[:, 0]
    initial_R = measured_R[:, :, 0]
    for index, name in enumerate(plane_files):
        if verbose:
            print(f"Calibration using file - {index + 1} / {count}")
        hand_t, hand_R, position, direction, scan = load_plane_yaml(
            root / "data" / "oct" / "plane" / name
        )
        normal, distance, _residual = plane_fit(initial_t, initial_R, hand_t, hand_R, scan)
        if verbose:
            print("\tUsing method-1")
        joint = plane_jnt_gh(
            initial_t,
            initial_R,
            normal,
            distance,
            SIGMA0,
            hand_t,
            hand_R,
            position,
            direction,
            IIMAX,
            IOMAX,
            TOL,
        )
        _store(joint.tO2H, joint.RO2H, joint.time, index, 0, poses)
        if verbose:
            print("\tUsing method-2")
        tro_t, tro_R, _normal, tro_time = plane_tro2016(
            initial_t, initial_R, hand_t, hand_R, direction, scan
        )
        _store(tro_t, tro_R, tro_time, index, 1, poses)
        if verbose:
            print("\tUsing method-3")
        rcim_t, rcim_R, _normal, rcim_time = plane_rcim2020(
            initial_t, initial_R, hand_t, hand_R, position, TOL
        )
        _store(rcim_t, rcim_R, rcim_time, index, 2, poses)
        if verbose:
            print("\tUsing method-4")
        if index < len(sphere_files):
            sphere_t, sphere_R, centers, _scans = load_meas2023_yaml(
                root / "data" / "oct" / "sphere" / sphere_files[index], MT, DIAMETER
            )
            meas_t, meas_R, meas_time = sphere_meas2023(sphere_t, sphere_R, centers)
            _store(meas_t, meas_R, meas_time, index, 3, poses)
        else:
            _store(
                poses.translation[:, 0, 3],
                poses.rotation[:, :, 0, 3],
                poses.time[0, 3],
                index,
                3,
                poses,
            )
        _store(measured_t[:, index], measured_R[:, :, index], 0.0, index, 4, poses)
    return poses


def main(argv: list[str] | None = None) -> RealCalibration:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/output"),
        help="Directory for real_calibration.npz. Does not write into result/.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Repository root")
    args = parser.parse_args(argv)
    result = calibrate(args.root, verbose=True)
    path = args.out / "real_calibration.npz"
    save_calibration(path, result)
    print(path)
    return result


if __name__ == "__main__":
    main()
