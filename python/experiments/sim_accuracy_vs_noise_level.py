"""Calibration accuracy against pose-noise level.

Port of ``evaluation/sim_accuracy_vs_noise_level.m`` without method 3
(``plane_Alt_GH_debug``) and without the unused iteration arrays. The six
methods are Xu, Proposed, Kang, Sharifzadeh, Carlson, and Tang. Plane-fit
clouds are drawn with ``sim_plane_batch``. Trials run in a process pool;
each trial has its own ``SeedSequence``.

Xu (``line_RCIM2022.m:58``) and Tang (``simMEAS2023.m:90``) use the Python
corrections, so their medians are not the paper's.
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
    mediqr,
    parallel_map,
    save_figure,
    trial_sequences,
    use_headless,
    write_outputs,
)
from experiments._methods import NOISE_METHODS, calibrate_suite
from experiments._plotting import noise_figure

SIGMAT = np.array([0.01, 0.1, 0.5, 1.0])
SIGMA_RD = np.array([0.01, 0.1, 0.5, 1.0])
COV_NAMES = ("iso.", "aniso.")
COV_T = (np.eye(3), np.diag([0.5, 0.5, 2.0]))
COV_R = (np.eye(3), np.diag([0.5, 1.0, 1.5]))
M_ALPHA = 10


def _job(payload: dict) -> dict:
    sequence = payload["sequence"]
    calib_ss, fit_ss = sequence.spawn(2)
    e_pos, e_rot, elapsed, residual = calibrate_suite(
        np.random.default_rng(calib_ss),
        np.random.default_rng(fit_ss),
        methods=NOISE_METHODS,
        sigmat=payload["sigmat"],
        sigma_rd=payload["sigma_rd"],
        cov_t=payload["cov_t"],
        cov_r=payload["cov_r"],
        m_alpha=M_ALPHA,
        m_beta=M_ALPHA,
        n_plane_fit=payload["n_plane_fit"],
        dt=DT_MM,
        dtheta=DTHETA,
        iimax=IIMAX,
        iomax=IOMAX,
        tol=TOL,
    )
    payload = dict(payload)
    payload.update(e_pos=e_pos, e_rot=e_rot, time=elapsed, residual=residual)
    del payload["sequence"], payload["cov_t"], payload["cov_r"]
    return payload


def _table(e_pos, e_rot, elapsed, residual) -> str:
    labels = [item[2] for item in NOISE_METHODS]
    blocks = []
    specs = (
        ("Position calibration error - Median (IQR)", e_pos, 3, 1.0),
        ("Orientation calibration error - Median (IQR)", e_rot, 3, 1.0),
        ("Computation time - Median (IQR) (ms)", elapsed, 3, 1000.0),
        ("Plane fitting residual - Median (IQR) (mm)", residual, 4, 1.0),
    )
    for title, values, digits, scale in specs:
        if values.size == 0:
            blocks.append(title + "\n(no plane-fit samples)\n")
            continue
        lines = [title]
        for i_method, label in enumerate(labels):
            cells = []
            for i_vf in range(values.shape[1]):
                for i_cov in range(values.shape[2]):
                    sample = values[:, i_vf, i_cov, i_method]
                    if sample.ndim > 1:
                        sample = sample.ravel()
                    median, iqr = median_iqr(sample)
                    cells.append(mediqr(scale * median, scale * iqr, digits))
            lines.append(label + "\n" + " ".join(f"& {cell}" for cell in cells) + " \\\\")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--n-plane-fit",
        type=int,
        default=50,
        help="Extra sim_plane clouds per calibration (MATLAB default 50)",
    )
    args = parser.parse_args(argv)
    if args.n_test < 1 or args.n_plane_fit < 0:
        raise SystemExit("n-test must be >= 1 and n-plane-fit must be >= 0")
    use_headless(args.no_show)
    n_vf = len(SIGMAT)
    n_cov = len(COV_NAMES)
    n_method = len(NOISE_METHODS)
    sequences = trial_sequences(args.seed, args.n_test * n_vf * n_cov)
    payloads = []
    cursor = 0
    for i_cov in range(n_cov):
        for i_vf in range(n_vf):
            for i_test in range(args.n_test):
                payloads.append(
                    {
                        "sequence": sequences[cursor],
                        "i_test": i_test,
                        "i_vf": i_vf,
                        "i_cov": i_cov,
                        "sigmat": float(SIGMAT[i_vf]),
                        "sigma_rd": float(SIGMA_RD[i_vf]),
                        "cov_t": COV_T[i_cov],
                        "cov_r": COV_R[i_cov],
                        "n_plane_fit": args.n_plane_fit,
                    }
                )
                cursor += 1
    rows = parallel_map(_job, payloads, args.n_jobs, "noise level")
    e_pos = np.zeros((args.n_test, n_vf, n_cov, n_method))
    e_rot = np.zeros_like(e_pos)
    elapsed = np.zeros_like(e_pos)
    residual = np.zeros((args.n_test, n_vf, n_cov, n_method, args.n_plane_fit))
    for row in rows:
        e_pos[row["i_test"], row["i_vf"], row["i_cov"]] = row["e_pos"]
        e_rot[row["i_test"], row["i_vf"], row["i_cov"]] = row["e_rot"]
        elapsed[row["i_test"], row["i_vf"], row["i_cov"]] = row["time"]
        if args.n_plane_fit:
            residual[row["i_test"], row["i_vf"], row["i_cov"]] = row["residual"]
    stem = "sim_accuracy_vs_noise_level"
    table = _table(e_pos, e_rot, elapsed, residual)
    print(table)
    write_outputs(
        args.out,
        stem,
        table,
        {
            "e_pos": e_pos,
            "e_rot": e_rot,
            "time": elapsed,
            "plane_fit_res": residual,
            "sigmat": SIGMAT,
            "sigma_rd": SIGMA_RD,
            "method_id": np.array([item[0] for item in NOISE_METHODS]),
        },
    )
    figure = noise_figure(
        e_pos.mean(axis=0),
        e_rot.mean(axis=0),
        [item[0] for item in NOISE_METHODS],
        [item[3] for item in NOISE_METHODS],
        COV_NAMES,
    )
    save_figure(figure, args.out, stem)
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return {
        "e_pos": e_pos,
        "e_rot": e_rot,
        "time": elapsed,
        "plane_fit_res": residual,
        "out": args.out,
    }


if __name__ == "__main__":
    main()
