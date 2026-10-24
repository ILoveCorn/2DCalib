"""Runtime against the number of plane measurements.

Port of ``evaluation/sim_accuracy_vs_number_of_measurements.m`` without method
3 (GPU) and without the unused iteration arrays. The plotted quantity is the
mean solver time. The log-scaled figure follows the MATLAB script.
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
    parallel_map,
    save_figure,
    trial_sequences,
    use_headless,
    write_outputs,
)
from experiments._methods import COUNT_METHODS, calibrate_suite
from experiments._plotting import count_figure

M_ALPHA = np.array([6, 8, 10, 14, 20, 28, 32])
SIGMAT = 0.5
SIGMA_RD = 0.5
MARKERS = ("o", "*", "s", "d", "^", "v")


def _job(payload: dict) -> dict:
    sequence = payload["sequence"]
    calib_ss, fit_ss = sequence.spawn(2)
    e_pos, e_rot, elapsed, _residual = calibrate_suite(
        np.random.default_rng(calib_ss),
        np.random.default_rng(fit_ss),
        methods=COUNT_METHODS,
        sigmat=SIGMAT,
        sigma_rd=SIGMA_RD,
        cov_t=np.eye(3),
        cov_r=np.eye(3),
        m_alpha=payload["m_alpha"],
        m_beta=payload["m_alpha"],
        n_plane_fit=0,
        dt=DT_MM,
        dtheta=DTHETA,
        iimax=IIMAX,
        iomax=IOMAX,
        tol=TOL,
    )
    return {
        "i_test": payload["i_test"],
        "i_m": payload["i_m"],
        "e_pos": e_pos,
        "e_rot": e_rot,
        "time": elapsed,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    if args.n_test < 1:
        raise SystemExit("n-test must be at least 1")
    use_headless(args.no_show)
    n_m = len(M_ALPHA)
    n_method = len(COUNT_METHODS)
    sequences = trial_sequences(args.seed, args.n_test * n_m)
    payloads = []
    cursor = 0
    for i_m, m_alpha in enumerate(M_ALPHA):
        for i_test in range(args.n_test):
            payloads.append(
                {
                    "sequence": sequences[cursor],
                    "i_test": i_test,
                    "i_m": i_m,
                    "m_alpha": int(m_alpha),
                }
            )
            cursor += 1
    rows = parallel_map(_job, payloads, args.n_jobs, "measurement count")
    e_pos = np.zeros((args.n_test, n_m, n_method))
    e_rot = np.zeros_like(e_pos)
    elapsed = np.zeros_like(e_pos)
    for row in rows:
        e_pos[row["i_test"], row["i_m"]] = row["e_pos"]
        e_rot[row["i_test"], row["i_m"]] = row["e_rot"]
        elapsed[row["i_test"], row["i_m"]] = row["time"]
    measurements = M_ALPHA.astype(float) ** 2
    legend = " ".join(item[3] for item in COUNT_METHODS)
    lines = [
        "Mean runtime (s) vs number of measurements",
        "m " + legend,
    ]
    for i_m, count in enumerate(measurements):
        means = " ".join(f"{elapsed[:, i_m, i].mean():.6f}" for i in range(n_method))
        lines.append(f"{int(count)} {means}")
    table = "\n".join(lines) + "\n"
    print(table)
    stem = "sim_accuracy_vs_number_of_measurements"
    write_outputs(
        args.out,
        stem,
        table,
        {
            "e_pos": e_pos,
            "e_rot": e_rot,
            "time": elapsed,
            "m_alpha": M_ALPHA,
            "m": measurements,
            "method_id": np.array([item[0] for item in COUNT_METHODS]),
        },
    )
    figure = count_figure(
        measurements,
        elapsed.mean(axis=0),
        MARKERS,
        [item[3] for item in COUNT_METHODS],
    )
    save_figure(figure, args.out, stem)
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return {"e_pos": e_pos, "e_rot": e_rot, "time": elapsed, "m": measurements, "out": args.out}


if __name__ == "__main__":
    main()
