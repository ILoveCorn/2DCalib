"""Convergence of the proposed method against the initial guess.

Port of ``evaluation/sim_accuracy_vs_initial_deviation.m``. The MATLAB script
calls ``plane_Jnt_GH_debug`` and cannot run as shipped. This port calls
``plane_jnt_gh`` and drops the inner and outer iteration columns. The time
column is milliseconds: the MATLAB header says seconds, but the ``fprintf``
prints ``1000 * compTime``.

A trial counts as converged when both errors are below three times the pose
noise (1.5 mm and 1.5 degrees).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from experiments._common import (
    IIMAX,
    IOMAX,
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
from experiments._plotting import deviation_figures
from handeye2d.metrics import calibration_error

DTHETA_DEG = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
DT = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
SIGMAT = 0.5
SIGMA_RD = 0.5
M_ALPHA = 10
POS_THRESHOLD = 3.0 * SIGMAT
ROT_THRESHOLD = 3.0 * SIGMA_RD


def _job(payload: dict) -> dict:
    rng = np.random.default_rng(payload["sequence"])
    translation, rotation, initial_t, initial_R, normal0, distance0, sigma0, plane = prepare_plane(
        rng,
        sigmat=SIGMAT,
        sigma_rd=SIGMA_RD,
        m_alpha=M_ALPHA,
        m_beta=M_ALPHA,
        dt=payload["dt"],
        dtheta=payload["dtheta"],
        init="fixed",
        sigma_before_scan=True,
    )
    result = apply_joint(initial_t, initial_R, normal0, distance0, sigma0, plane, IIMAX, IOMAX, TOL)
    e_pos, e_rot = calibration_error(translation, rotation, result.tO2H, result.RO2H)
    converged = float(e_pos) < POS_THRESHOLD and float(e_rot) < ROT_THRESHOLD
    return {
        "i_test": payload["i_test"],
        "i_dev": payload["i_dev"],
        "e_pos": float(e_pos),
        "e_rot": float(e_rot),
        "time": result.time,
        "converged": float(converged),
    }


def _table(e_pos, e_rot, elapsed, converged) -> str:
    lines = [
        "Convergence rate vs initial deviation (proposed method)",
        f"Noise level: sigma_t = {SIGMAT:.2f} mm, sigma_R = {SIGMA_RD:.2f} deg",
        f"Threshold: e_t < {POS_THRESHOLD:.2f} mm, e_R < {ROT_THRESHOLD:.2f} deg",
        "Initial deviation | converged/total | rate",
    ]
    n_test = e_pos.shape[0]
    rates = converged.mean(axis=0) * 100.0
    for index, (angle, shift) in enumerate(zip(DTHETA_DEG, DT, strict=True)):
        count = int(converged[:, index].sum())
        lines.append(
            f"[{angle:.0f} deg, {shift:.1f} mm] | {count} / {n_test} | {rates[index]:.1f} %"
        )
    lines.append("")
    lines.append("Initial deviation | position (mm) | orientation (deg) | comp. time (ms)")
    lines.append("Iteration columns are omitted; plane_jnt_gh does not return them.")
    for index, (angle, shift) in enumerate(zip(DTHETA_DEG, DT, strict=True)):
        pos = median_iqr(e_pos[:, index])
        rot = median_iqr(e_rot[:, index])
        timing = median_iqr(1000.0 * elapsed[:, index])
        lines.append(
            f"[{angle:.0f} deg, {shift:.1f} mm] | "
            f"{pos[0]:.3f} ({pos[1]:.3f}) | {rot[0]:.3f} ({rot[1]:.3f}) | "
            f"{timing[0]:.2f} ({timing[1]:.2f})"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    if args.n_test < 1:
        raise SystemExit("n-test must be at least 1")
    use_headless(args.no_show)
    n_dev = len(DT)
    sequences = trial_sequences(args.seed, args.n_test * n_dev)
    payloads = []
    cursor = 0
    for i_dev in range(n_dev):
        for i_test in range(args.n_test):
            payloads.append(
                {
                    "sequence": sequences[cursor],
                    "i_test": i_test,
                    "i_dev": i_dev,
                    "dt": float(DT[i_dev]),
                    "dtheta": float(np.deg2rad(DTHETA_DEG[i_dev])),
                }
            )
            cursor += 1
    rows = parallel_map(_job, payloads, args.n_jobs, "initial deviation")
    e_pos = np.zeros((args.n_test, n_dev))
    e_rot = np.zeros_like(e_pos)
    elapsed = np.zeros_like(e_pos)
    converged = np.zeros_like(e_pos)
    for row in rows:
        e_pos[row["i_test"], row["i_dev"]] = row["e_pos"]
        e_rot[row["i_test"], row["i_dev"]] = row["e_rot"]
        elapsed[row["i_test"], row["i_dev"]] = row["time"]
        converged[row["i_test"], row["i_dev"]] = row["converged"]
    table = _table(e_pos, e_rot, elapsed, converged)
    print(table)
    stem = "sim_accuracy_vs_initial_deviation"
    write_outputs(
        args.out,
        stem,
        table,
        {
            "e_pos": e_pos,
            "e_rot": e_rot,
            "time": elapsed,
            "converged": converged,
            "dtheta_deg": DTHETA_DEG,
            "dt": DT,
        },
    )
    labels = [
        f"[{angle:.0f} deg, {shift:.1f} mm]" for angle, shift in zip(DTHETA_DEG, DT, strict=True)
    ]
    bar, boxes, histograms = deviation_figures(
        converged.mean(axis=0) * 100.0, e_pos, e_rot, labels, SIGMAT, SIGMA_RD
    )
    save_figure(bar, args.out, stem + "_convergence")
    save_figure(boxes, args.out, stem + "_boxplot")
    save_figure(histograms, args.out, stem + "_histogram")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return {
        "e_pos": e_pos,
        "e_rot": e_rot,
        "time": elapsed,
        "converged": converged,
        "out": args.out,
    }


if __name__ == "__main__":
    main()
