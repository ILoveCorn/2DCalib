"""Proposed-method error under speckle and scanning distortion.

Port of ``evaluation/sim_accuracy_vs_image_noise.m``. All three conditions
start from the same ideal ``sim_plane`` draw. Position in the table is
micrometers, as in the MATLAB ``fprintf`` (``1000 * median`` of millimeters).
There is no figure in the MATLAB script; this port writes the table only.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from experiments._common import (
    DT_MM,
    DTHETA,
    IIMAX,
    IOMAX,
    TOL,
    add_common_arguments,
    median_iqr,
    parallel_map,
    trial_sequences,
    use_headless,
    write_outputs,
)
from experiments._methods import prepare_plane
from handeye2d.metrics import calibration_error

CONDITIONS = ("Ideal", "Speckle Noise", "Scanning Distortion")
SIGMAT = 0.5
SIGMA_RD = 0.5
M_ALPHA = 10
SPECKLE = (0.5513e-3, 0.5263e-3, float(np.deg2rad(0.0001)), float(np.deg2rad(0.0046)))
DISTORTION = (-0.3942e-3, 0.6997e-3, float(np.deg2rad(-0.0030)), float(np.deg2rad(0.0083)))


def _perturb(point, direction, mu_z, sigma_z, mu_angle, sigma_angle, rng):
    perturbed_p = np.array(point, copy=True)
    perturbed_v = np.array(direction, copy=True)
    perturbed_p[:, 2] = point[:, 2] + mu_z + sigma_z * rng.normal(size=point.shape[0])
    theta = np.arctan2(direction[:, 2], direction[:, 0])
    theta = theta + mu_angle + sigma_angle * rng.normal(size=point.shape[0])
    perturbed_v[:, 0] = np.cos(theta)
    perturbed_v[:, 1] = 0.0
    perturbed_v[:, 2] = np.sin(theta)
    return perturbed_p, perturbed_v


def _job(payload: dict) -> dict:
    rng = np.random.default_rng(payload["sequence"])
    translation, rotation, initial_t, initial_R, normal0, distance0, sigma0, plane = prepare_plane(
        rng,
        sigmat=SIGMAT,
        sigma_rd=SIGMA_RD,
        m_alpha=M_ALPHA,
        m_beta=M_ALPHA,
        dt=DT_MM,
        dtheta=DTHETA,
        init="box",
        sigma_before_scan=True,
    )
    errors = np.empty((2, len(CONDITIONS)))
    clouds = (None, SPECKLE, DISTORTION)
    for index, cloud in enumerate(clouds):
        point, direction = plane.pOn, plane.vOn
        if cloud is not None:
            point, direction = _perturb(plane.pOn, plane.vOn, *cloud, rng)
        solved = plane_with_profiles(
            initial_t, initial_R, normal0, distance0, sigma0, plane, point, direction
        )
        e_pos, e_rot = calibration_error(translation, rotation, solved.tO2H, solved.RO2H)
        errors[0, index] = float(e_pos)
        errors[1, index] = float(e_rot)
    return {"i_test": payload["i_test"], "errors": errors}


def plane_with_profiles(initial_t, initial_R, normal0, distance0, sigma0, plane, point, direction):
    from handeye2d.calib.plane_gh import plane_jnt_gh

    return plane_jnt_gh(
        initial_t,
        initial_R,
        normal0,
        distance0,
        sigma0,
        plane.tH2Wn,
        plane.RH2Wn,
        point,
        direction,
        IIMAX,
        IOMAX,
        TOL,
    )


def _table(e_pos, e_rot) -> str:
    pos_med = np.empty(e_pos.shape[1])
    pos_iqr = np.empty_like(pos_med)
    rot_med = np.empty_like(pos_med)
    rot_iqr = np.empty_like(pos_med)
    for index in range(e_pos.shape[1]):
        pos_med[index], pos_iqr[index] = median_iqr(e_pos[:, index])
        rot_med[index], rot_iqr[index] = median_iqr(e_rot[:, index])
    lines = [
        "Calibration error: median (IQR) and relative increase",
        "Condition | e_t (um) | e_R (deg) | delta e_t (%) | delta e_R (%)",
    ]
    for index, name in enumerate(CONDITIONS):
        if index == 0 or pos_med[0] == 0.0:
            pos_increase = "-"
        else:
            pos_increase = f"{(pos_med[index] - pos_med[0]) / pos_med[0] * 100.0:.2f}"
        if index == 0 or rot_med[0] == 0.0:
            rot_increase = "-"
        else:
            rot_increase = f"{(rot_med[index] - rot_med[0]) / rot_med[0] * 100.0:.2f}"
        lines.append(
            f"{name:30s} | {1000.0 * pos_med[index]:8.4f} ({1000.0 * pos_iqr[index]:7.4f}) | "
            f"{rot_med[index]:8.4f} ({rot_iqr[index]:7.4f}) | {pos_increase} | {rot_increase}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    if args.n_test < 1:
        raise SystemExit("n-test must be at least 1")
    use_headless(args.no_show)
    sequences = trial_sequences(args.seed, args.n_test)
    payloads = [{"sequence": sequences[i], "i_test": i} for i in range(args.n_test)]
    rows = parallel_map(_job, payloads, args.n_jobs, "image noise")
    e_pos = np.zeros((args.n_test, len(CONDITIONS)))
    e_rot = np.zeros_like(e_pos)
    for row in rows:
        e_pos[row["i_test"]] = row["errors"][0]
        e_rot[row["i_test"]] = row["errors"][1]
    table = _table(e_pos, e_rot)
    print(table)
    stem = "sim_accuracy_vs_image_noise"
    write_outputs(args.out, stem, table, {"e_pos": e_pos, "e_rot": e_rot})
    return {"e_pos": e_pos, "e_rot": e_rot, "out": args.out}


if __name__ == "__main__":
    main()
