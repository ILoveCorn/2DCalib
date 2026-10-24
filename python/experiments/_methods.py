"""One Monte Carlo trial of the six published methods.

Cases that share a plane cloud do so explicitly: Proposed draws ``sim_plane``,
and Kang and Sharifzadeh reuse that draw. Plane-fit residuals use
``sim_plane_batch`` on a separate seed, so the 50 extra clouds are vectorized
and do not interleave with the calibration stream.
"""

from __future__ import annotations

import numpy as np

from experiments._common import (
    ALPHA_MAX,
    BETA_MAX,
    LENGTH,
    MT,
    N_PLANES,
    SPHERE_RADIUS,
    X_MAX,
    Z_MAX,
    line_grid,
)
from experiments._scene import (
    draw_box_init,
    draw_fixed_init,
    draw_hand_eye,
    draw_multiplane_normals,
    draw_plane_init,
    draw_sigma0,
    draw_unit_uniform,
    multiplane_centers,
)
from handeye2d.calib.line_rcim2022 import line_rcim2022
from handeye2d.calib.plane_gh import plane_jnt_gh, plane_jnt_gh_decoupled
from handeye2d.calib.plane_iros2015 import plane_iros2015
from handeye2d.calib.plane_rcim2020 import plane_rcim2020
from handeye2d.calib.plane_tro2016 import plane_tro2016
from handeye2d.calib.sphere_meas2023 import sphere_meas2023
from handeye2d.fitting import plane_fit
from handeye2d.metrics import calibration_error
from handeye2d.sim.sim_iros2015 import sim_iros2015
from handeye2d.sim.sim_meas2023 import sim_meas2023
from handeye2d.sim.sim_plane import sim_plane, sim_plane_batch
from handeye2d.sim.sim_rcim2022 import sim_rcim2022

# Noise-level script, in testedMethods order. The integer is the MATLAB index.
NOISE_METHODS = (
    (1, "xu", "1 - RCIM-2022", "Xu"),
    (2, "proposed", "2 - Proposed", "Proposed"),
    (5, "kang", "5 - TRO-2016", "Kang"),
    (6, "sharifzadeh", "6 - RCIM-2020", "Sharifzadeh"),
    (7, "carlson", "7 - IROS-2015", "Carlson"),
    (8, "tang", "8 - MEAS-2023", "Tang"),
)

# Measurement-count script. Method 3 (GPU) is omitted.
COUNT_METHODS = (
    (1, "xu", "1 - RCIM-2022", "Xu"),
    (2, "proposed", "2 - Proposed", "Proposed (CPU)"),
    (4, "kang", "4 - TRO-2016", "Kang"),
    (5, "sharifzadeh", "5 - RCIM-2020", "Sharifzadeh"),
    (6, "carlson", "6 - IROS-2015", "Carlson"),
    (7, "tang", "7 - MEAS-2023", "Tang"),
)


def calibrate_suite(
    rng: np.random.Generator,
    fit_rng: np.random.Generator,
    *,
    methods: tuple[tuple[int, str, str, str], ...],
    sigmat: float,
    sigma_rd: float,
    cov_t: np.ndarray,
    cov_r: np.ndarray,
    m_alpha: int,
    m_beta: int,
    n_plane_fit: int,
    dt: float,
    dtheta: float,
    iimax: int,
    iomax: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return position error, orientation error, solver time, and plane-fit residuals."""
    sigma_r = float(np.deg2rad(sigma_rd))
    position_cov = sigmat**2 * np.asarray(cov_t, dtype=float)
    orientation_cov = sigma_r**2 * np.asarray(cov_r, dtype=float)
    translation, rotation = draw_hand_eye(rng)
    direction = draw_unit_uniform(rng)
    normals = draw_multiplane_normals(rng)
    initial_t, initial_R = draw_box_init(rng, translation, rotation, dt, dtheta)
    normal0, distance0 = draw_plane_init(rng, direction)
    sigma0 = draw_sigma0(rng, sigmat, sigma_r)
    m_line = line_grid(m_alpha, m_beta)
    centers, lengths = multiplane_centers()
    # Drawn on the first plane method so Xu's simulator consumes the generator
    # before sim_plane, matching the MATLAB switch order.
    plane = None
    n_method = len(methods)
    e_pos = np.empty(n_method)
    e_rot = np.empty(n_method)
    elapsed = np.empty(n_method)
    residuals = np.empty((n_method, n_plane_fit))
    names = [item[1] for item in methods]
    if "proposed" not in names and ("kang" in names or "sharifzadeh" in names):
        raise ValueError("Kang and Sharifzadeh reuse the Proposed plane draw")

    for index, name in enumerate(names):
        if plane is None and name in {"proposed", "kang", "sharifzadeh", "decoupled"}:
            plane = sim_plane(
                np.zeros(3),
                direction,
                LENGTH,
                X_MAX,
                Z_MAX,
                translation,
                rotation,
                position_cov,
                orientation_cov,
                0.0,
                ALPHA_MAX,
                BETA_MAX,
                m_alpha,
                m_beta,
                rng,
            )
        estimated_t, estimated_R, seconds = _one_method(
            name,
            rng,
            translation,
            rotation,
            direction,
            normals,
            centers,
            lengths,
            initial_t,
            initial_R,
            normal0,
            distance0,
            sigma0,
            position_cov,
            orientation_cov,
            plane,
            m_alpha,
            m_beta,
            m_line,
            iimax,
            iomax,
            tol,
        )
        position_error, angle_error = calibration_error(
            translation, rotation, estimated_t, estimated_R
        )
        e_pos[index] = float(position_error)
        e_rot[index] = float(angle_error)
        elapsed[index] = seconds
        if n_plane_fit:
            residuals[index] = _plane_residuals(
                fit_rng,
                estimated_t,
                estimated_R,
                direction,
                translation,
                rotation,
                position_cov,
                orientation_cov,
                m_alpha,
                m_beta,
                n_plane_fit,
            )
    return e_pos, e_rot, elapsed, residuals


def prepare_plane(
    rng: np.random.Generator,
    *,
    sigmat: float,
    sigma_rd: float,
    m_alpha: int,
    m_beta: int,
    dt: float,
    dtheta: float,
    init: str,
    sigma_before_scan: bool,
):
    """Draw the isotropic plane scene used by the single-method scripts.

    ``init`` is ``box`` (uniform offsets) or ``fixed`` (given magnitudes).
    The random variance factor is drawn before the scan except in the VFE
    script, which draws it after ``sim_plane``.
    """
    sigma_r = float(np.deg2rad(sigma_rd))
    position_cov = sigmat**2 * np.eye(3)
    orientation_cov = sigma_r**2 * np.eye(3)
    translation, rotation = draw_hand_eye(rng)
    normal = draw_unit_uniform(rng)
    if init == "box":
        initial_t, initial_R = draw_box_init(rng, translation, rotation, dt, dtheta)
    elif init == "fixed":
        initial_t, initial_R = draw_fixed_init(rng, translation, rotation, dt, dtheta)
    else:
        raise ValueError(f"unknown init {init}")
    normal0, distance0 = draw_plane_init(rng, normal)
    sigma0 = draw_sigma0(rng, sigmat, sigma_r) if sigma_before_scan else None
    plane = sim_plane(
        np.zeros(3),
        normal,
        LENGTH,
        X_MAX,
        Z_MAX,
        translation,
        rotation,
        position_cov,
        orientation_cov,
        0.0,
        ALPHA_MAX,
        BETA_MAX,
        m_alpha,
        m_beta,
        rng,
    )
    return translation, rotation, initial_t, initial_R, normal0, distance0, sigma0, plane


def apply_joint(initial_t, initial_R, normal0, distance0, sigma0, plane, iimax, iomax, tol):
    return plane_jnt_gh(
        initial_t,
        initial_R,
        normal0,
        distance0,
        sigma0,
        plane.tH2Wn,
        plane.RH2Wn,
        plane.pOn,
        plane.vOn,
        iimax,
        iomax,
        tol,
    )


def _one_method(
    name,
    rng,
    translation,
    rotation,
    direction,
    normals,
    centers,
    lengths,
    initial_t,
    initial_R,
    normal0,
    distance0,
    sigma0,
    position_cov,
    orientation_cov,
    plane,
    m_alpha,
    m_beta,
    m_line,
    iimax,
    iomax,
    tol,
):
    if name == "xu":
        simulated = sim_rcim2022(
            np.zeros(3),
            direction,
            LENGTH,
            X_MAX,
            Z_MAX,
            translation,
            rotation,
            position_cov,
            orientation_cov,
            0.0,
            ALPHA_MAX,
            BETA_MAX,
            m_line,
            m_line,
            MT,
            rng,
        )
        estimated_t, estimated_R, seconds = line_rcim2022(
            simulated.tH2Wn, simulated.RH2Wn, simulated.pOn
        )
        return estimated_t, estimated_R, seconds
    if name == "proposed":
        result = plane_jnt_gh(
            initial_t,
            initial_R,
            normal0,
            distance0,
            sigma0,
            plane.tH2Wn,
            plane.RH2Wn,
            plane.pOn,
            plane.vOn,
            iimax,
            iomax,
            tol,
        )
        return result.tO2H, result.RO2H, result.time
    if name == "decoupled":
        result = plane_jnt_gh_decoupled(
            initial_t,
            initial_R,
            normal0,
            distance0,
            sigma0,
            plane.tH2Wn,
            plane.RH2Wn,
            plane.pOn,
            plane.vOn,
            iimax,
            iomax,
            tol,
        )
        return result.tO2H, result.RO2H, result.time
    if name == "kang":
        estimated_t, estimated_R, _, seconds = plane_tro2016(
            initial_t, initial_R, plane.tH2Wn, plane.RH2Wn, plane.vOn, plane.scanO
        )
        return estimated_t, estimated_R, seconds
    if name == "sharifzadeh":
        estimated_t, estimated_R, _, seconds = plane_rcim2020(
            initial_t, initial_R, plane.tH2Wn, plane.RH2Wn, plane.pOn, tol
        )
        return estimated_t, estimated_R, seconds
    if name == "carlson":
        simulated = sim_iros2015(
            N_PLANES,
            centers,
            normals,
            lengths,
            X_MAX,
            Z_MAX,
            translation,
            rotation,
            position_cov,
            orientation_cov,
            0.0,
            ALPHA_MAX,
            BETA_MAX,
            m_alpha,
            m_beta,
            rng,
        )
        estimated_t, estimated_R, _, seconds = plane_iros2015(
            initial_t, initial_R, simulated.tH2Wn, simulated.RH2Wn, simulated.pOn, tol
        )
        return estimated_t, estimated_R, seconds
    if name == "tang":
        simulated = sim_meas2023(
            np.zeros(3),
            SPHERE_RADIUS,
            X_MAX,
            Z_MAX,
            translation,
            rotation,
            position_cov,
            orientation_cov,
            0.0,
            ALPHA_MAX,
            BETA_MAX,
            m_line,
            m_line,
            MT,
            rng,
        )
        estimated_t, estimated_R, seconds = sphere_meas2023(
            simulated.tH2Wn, simulated.RH2Wn, simulated.pcOe
        )
        return estimated_t, estimated_R, seconds
    raise ValueError(f"unknown method {name}")


def _plane_residuals(
    rng,
    estimated_t,
    estimated_R,
    direction,
    translation,
    rotation,
    position_cov,
    orientation_cov,
    m_alpha,
    m_beta,
    n_plane_fit,
):
    batch = sim_plane_batch(
        n_plane_fit,
        np.zeros(3),
        direction,
        LENGTH,
        X_MAX,
        Z_MAX,
        translation,
        rotation,
        position_cov,
        orientation_cov,
        0.0,
        ALPHA_MAX,
        BETA_MAX,
        m_alpha,
        m_beta,
        rng,
    )
    residual = np.empty(n_plane_fit)
    for index in range(n_plane_fit):
        _, _, residual[index] = plane_fit(
            estimated_t, estimated_R, batch.tH2Wn[index], batch.RH2Wn[index], batch.scanO[index]
        )
    return residual
