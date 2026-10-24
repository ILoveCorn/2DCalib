"""CLI, per-trial seeds, and median tables shared by the simulation scripts."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar

import numpy as np
from tqdm import tqdm

X_MAX = 5.0
Z_MAX = 2.0
ALPHA_MAX = float(np.deg2rad(45.0))
BETA_MAX = float(np.deg2rad(45.0))
LENGTH = 50.0
SPHERE_RADIUS = 7.5
MT = 3
N_PLANES = 3
IIMAX = 30
IOMAX = 50
TOL = 1e-5
DT_MM = 5.0
DTHETA = float(np.deg2rad(5.0))

T = TypeVar("T")


def matlab_round(value: float) -> int:
    """MATLAB ``round``: half away from zero."""
    sign = 1.0 if value >= 0.0 else -1.0
    return int(np.floor(abs(value) + 0.5) * sign)


def line_grid(m_alpha: int, m_beta: int, mt: int = MT) -> int:
    """``round(sqrt(m / mt))``, the reduced Euler count used by Xu and Tang."""
    return matlab_round(float(np.sqrt((m_alpha * m_beta) / mt)))


def median_iqr(values: np.ndarray) -> tuple[float, float]:
    """Median and interquartile range.

    The quartile positions follow NumPy's Hazen rule, which is the plotting
    position MATLAB's ``prctile`` documents: ``(k - 0.5) / n``.
    """
    data = np.asarray(values, dtype=float).ravel()
    median = float(np.median(data))
    low, high = np.percentile(data, [25.0, 75.0], method="hazen")
    return median, float(high - low)


def mediqr(median: float, iqr: float, digits: int) -> str:
    """One ``\\mediqr{median}{iqr}`` cell."""
    return f"\\mediqr{{{median:.{digits}f}}}{{{iqr:.{digits}f}}}"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-test", type=int, default=100, help="Monte Carlo repeats per setting")
    parser.add_argument("--seed", type=int, default=0, help="SeedSequence root")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes. 1 runs in this process.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/output"),
        help="Directory for npz, tables, and figures",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening a window",
    )


def trial_sequences(seed: int, count: int) -> list[np.random.SeedSequence]:
    """One independent ``SeedSequence`` per trial, independent of ``--n-jobs``."""
    if count < 1:
        raise ValueError("n-test must be at least 1")
    return list(np.random.SeedSequence(seed).spawn(count))


def parallel_map(
    function: Callable[[T], dict],
    payloads: Sequence[T],
    n_jobs: int,
    description: str,
) -> list[dict]:
    """Run ``function`` over payloads. Results stay in payload order."""
    if not payloads:
        return []
    workers = max(1, n_jobs)
    if workers == 1 or len(payloads) == 1:
        return [function(item) for item in tqdm(payloads, desc=description)]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        mapped = pool.map(function, payloads)
        return list(tqdm(mapped, total=len(payloads), desc=description))


def write_outputs(out: Path, stem: str, table: str, arrays: dict[str, np.ndarray]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{stem}.npz", **arrays)
    path = out / f"{stem}.txt"
    path.write_text(table)
    return path


def save_figure(figure, out: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / f"{stem}.png", dpi=200)
    figure.savefig(out / f"{stem}.pdf")
    plt.close(figure)


def use_headless(no_show: bool) -> None:
    import matplotlib

    if no_show:
        matplotlib.use("Agg", force=True)
