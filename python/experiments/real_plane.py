"""Plane reconstruction error on the saved calibrations.

Port of ``evaluation/real_plane.m``. For the three plane methods the session
used to calibrate is left as NaN. MEAS 2023 and the measured poses are scored
on every test session. The residual is the mean absolute point-to-plane
distance from ``plane_fit``, in millimeters.

The default calibration is ``result/real_calibration_260219_1146.mat``.
Pass ``--calibration`` to score an ``.npz`` from ``real_calibration`` instead.
This driver does not write into ``result/``. There is no saved MATLAB
``real_plane`` mat in the repository, and the paper table is not checked here.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from experiments._common import write_outputs
from experiments._real import (
    METHODS,
    N_PLANE_METHODS,
    RealCalibration,
    load_calibration,
    reference_calibration,
    repo_root,
)
from handeye2d.fitting import plane_fit
from handeye2d.io import load_plane_yaml


def reconstruction_table(errors: np.ndarray, methods: tuple[str, ...] = METHODS) -> str:
    """MATLAB ``fprintf`` rows: per-test mean, then the mean over every finite entry."""
    if errors.ndim != 3 or errors.shape[2] != len(methods):
        raise ValueError(f"errors expected (n_test, n_calib, {len(methods)}), got {errors.shape}")
    lines = ["", "", "Plane reconstruction error:"]
    n_test = errors.shape[0]
    for index, name in enumerate(methods):
        cells = "".join(
            f"& {float(np.nanmean(errors[i_test, :, index])):.3f} " for i_test in range(n_test)
        )
        overall = float(np.nanmean(errors[:, :, index]))
        lines.append(f"\t{name}: {cells}& {overall:.3f}")
    return "\n".join(lines) + "\n"


def plane_errors(
    calibration: RealCalibration,
    root: Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Return ``ePlane`` with shape ``(n_test, n_calib, n_method)``."""
    root = repo_root() if root is None else Path(root)
    n_calib = len(calibration.plane_files)
    n_method = calibration.translation.shape[2]
    if calibration.translation.shape != (3, n_calib, n_method):
        raise ValueError(
            f"translation expected (3, {n_calib}, {n_method}), got {calibration.translation.shape}"
        )
    if calibration.rotation.shape != (3, 3, n_calib, n_method):
        raise ValueError(
            f"rotation expected (3, 3, {n_calib}, {n_method}), got {calibration.rotation.shape}"
        )
    sessions = []
    for name in calibration.plane_files:
        hand_t, hand_R, _position, _direction, scan = load_plane_yaml(
            root / "data" / "oct" / "plane" / name
        )
        sessions.append((hand_t, hand_R, scan))
    errors = np.full((n_calib, n_calib, n_method), np.nan)
    for i_calib in range(n_calib):
        if verbose:
            print(f"Plane reconstruction using calibration - {i_calib + 1} / {n_calib}")
        for i_test, (hand_t, hand_R, scan) in enumerate(sessions):
            for i_method in range(n_method):
                if i_method < N_PLANE_METHODS and i_test == i_calib:
                    continue
                estimated_t = calibration.translation[:, i_calib, i_method]
                estimated_R = calibration.rotation[:, :, i_calib, i_method]
                _normal, _distance, residual = plane_fit(
                    estimated_t, estimated_R, hand_t, hand_R, scan
                )
                errors[i_test, i_calib, i_method] = residual
    return errors


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/output"),
        help="Directory for the npz and the printed table. Does not write into result/.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Repository root")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Saved .mat or real_calibration.npz. Default: result/real_calibration_260219_1146.mat",
    )
    args = parser.parse_args(argv)
    root = repo_root() if args.root is None else Path(args.root)
    if args.calibration is None:
        calibration = reference_calibration(root)
    else:
        calibration = load_calibration(args.calibration)
    errors = plane_errors(calibration, root, verbose=True)
    table = reconstruction_table(errors)
    print(table)
    write_outputs(
        args.out,
        "real_plane",
        table,
        {"ePlane": errors, "planeFiles": np.asarray(calibration.plane_files)},
    )
    return {"ePlane": errors, "table": table, "out": args.out}


if __name__ == "__main__":
    main()
