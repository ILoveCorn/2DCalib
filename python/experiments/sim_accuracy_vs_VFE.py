"""Proposed-method error with and without variance-factor estimation.

Port of ``evaluation/sim_accuracy_vs_VFE.m``. The three settings share one
``sim_plane`` draw:

1. VFE, ``iOmax=50``, ``sigma0`` random in ``[0.75, 1.25]`` times the truth.
2. No VFE, ``iOmax=1``, ``sigma0`` equal to the ground-truth noise.
3. No VFE, ``iOmax=1``, ``sigma0 = [1, 1]``.

Reported orientation variance is in degrees. The solver itself is given radians.
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
    TOL,
    add_common_arguments,
    median_iqr,
    parallel_map,
    save_figure,
    trial_sequences,
    use_headless,
    write_outputs,
)
from experiments._methods import apply_joint, prepare_plane
from experiments._plotting import vfe_figure
from experiments._scene import draw_sigma0
from handeye2d.metrics import calibration_error

SIGMAT = 1.0
SIGMA_RD = 0.1
M_ALPHA = 10
LABELS = ("VFE", "GT", "EQ")


def _job(payload: dict) -> dict:
    rng = np.random.default_rng(payload["sequence"])
    translation, rotation, initial_t, initial_R, normal0, distance0, _sigma, plane = prepare_plane(
        rng,
        sigmat=SIGMAT,
        sigma_rd=SIGMA_RD,
        m_alpha=M_ALPHA,
        m_beta=M_ALPHA,
        dt=DT_MM,
        dtheta=DTHETA,
        init="box",
        sigma_before_scan=False,
    )
    sigma_r = float(np.deg2rad(SIGMA_RD))
    settings = (
        (50, draw_sigma0(rng, SIGMAT, sigma_r)),
        (1, np.array([SIGMAT, sigma_r])),
        (1, np.ones(2)),
    )
    e_pos = np.empty(3)
    e_rot = np.empty(3)
    sigma = np.zeros(2)
    for index, (iomax, sigma0) in enumerate(settings):
        result = apply_joint(
            initial_t, initial_R, normal0, distance0, sigma0, plane, IIMAX, iomax, TOL
        )
        position, angle = calibration_error(translation, rotation, result.tO2H, result.RO2H)
        e_pos[index] = float(position)
        e_rot[index] = float(angle)
        if index == 0:
            sigma = np.asarray(result.sigma, dtype=float).reshape(2)
    return {"i_test": payload["i_test"], "e_pos": e_pos, "e_rot": e_rot, "sigma": sigma}


def _table(e_pos, e_rot, sigma_deg) -> str:
    truth = np.array([SIGMAT, SIGMA_RD])
    error = sigma_deg - truth
    relative = 100.0 * error / truth
    sigma_med = [median_iqr(sigma_deg[:, i]) for i in range(2)]
    abs_med = [median_iqr(error[:, i]) for i in range(2)]
    rel_med = [median_iqr(relative[:, i]) for i in range(2)]
    lines = [
        "Accuracy of variance factor estimation (VFE)",
        f"Ground truth: sigma_t = {SIGMAT:.4f} mm, sigma_R = {SIGMA_RD:.4f} deg",
        f"Estimated median (IQR): {sigma_med[0][0]:.4f} ({sigma_med[0][1]:.4f})    "
        f"{sigma_med[1][0]:.4f} ({sigma_med[1][1]:.4f})",
        f"Absolute error median (IQR): {abs_med[0][0]:.4f} ({abs_med[0][1]:.4f})    "
        f"{abs_med[1][0]:.4f} ({abs_med[1][1]:.4f})",
        f"Relative error median (IQR) %: {rel_med[0][0]:.2f} ({rel_med[0][1]:.2f})    "
        f"{rel_med[1][0]:.2f} ({rel_med[1][1]:.2f})",
        "",
        "Calibration error median (IQR)",
    ]
    for index, label in enumerate(LABELS):
        pos = median_iqr(e_pos[:, index])
        rot = median_iqr(e_rot[:, index])
        lines.append(
            f"{label}: e_t {pos[0]:.4f} ({pos[1]:.4f}) mm, e_R {rot[0]:.4f} ({rot[1]:.4f}) deg"
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
    rows = parallel_map(_job, payloads, args.n_jobs, "VFE")
    e_pos = np.zeros((args.n_test, 3))
    e_rot = np.zeros_like(e_pos)
    sigma = np.zeros((args.n_test, 2))
    for row in rows:
        e_pos[row["i_test"]] = row["e_pos"]
        e_rot[row["i_test"]] = row["e_rot"]
        sigma[row["i_test"]] = row["sigma"]
    sigma_deg = np.column_stack((sigma[:, 0], np.rad2deg(sigma[:, 1])))
    table = _table(e_pos, e_rot, sigma_deg)
    print(table)
    stem = "sim_accuracy_vs_VFE"
    write_outputs(
        args.out,
        stem,
        table,
        {"e_pos": e_pos, "e_rot": e_rot, "sigma": sigma, "sigma_deg": sigma_deg},
    )
    figure = vfe_figure(e_rot, e_pos)
    save_figure(figure, args.out, stem)
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return {"e_pos": e_pos, "e_rot": e_rot, "sigma": sigma, "out": args.out}


if __name__ == "__main__":
    main()
