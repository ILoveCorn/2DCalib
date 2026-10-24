# Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical Coherence Tomography Using a Plane Target

[![Paper](https://img.shields.io/badge/IEEE%20T--RO-10.1109%2FTRO.2026.3710397-00629B)](https://doi.org/10.1109/TRO.2026.3710397)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](data/LICENSE)
![MATLAB R2024b](https://img.shields.io/badge/MATLAB-R2024b-orange)
[![Python package](https://img.shields.io/badge/python-handeye2d-3776AB)](python/README.md)

Official MATLAB implementation, datasets, and evaluation scripts for the IEEE Transactions on Robotics article:

> Yan Ma, Haitian Lyu, Jiewen Lai, Ruiyang Zhang, Wu Yuan, and Hongliang Ren,
> "Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical Coherence Tomography Using a Plane Target,"
> *IEEE Transactions on Robotics*, 2026. DOI: [10.1109/TRO.2026.3710397](https://doi.org/10.1109/TRO.2026.3710397)

## Overview

Optical coherence tomography (OCT) provides micrometer-resolution cross-sectional images within a millimeter-scale field of view, so robot pose uncertainty is no longer negligible at the OCT scale. This repository calibrates the hand-eye transformation of a **2-D OCT** probe from cross-sectional images of a **single plane target**:

- Robot pose uncertainty is modeled explicitly with the **Gauss-Helmert model**.
- The hand-eye transformation and plane parameters are **jointly estimated** on a composite manifold with analytical Jacobians.
- When pose covariances are unknown, **variance factors are estimated** from the measurements.

The same formulation applies to other 2-D profile sensors such as laser profilers and 2-D LiDAR.

## Requirements

Tested with **MATLAB R2024b**.

| Dependency | Needed by |
|---|---|
| [MATLAB YAML](https://github.com/MartinKoch123/yaml) **v1.6.0 or newer** (third-party) | All real-data experiments (`yaml.loadFile`) |
| Robotics System Toolbox | Rotation conversions (`quat2rotm`, `eul2rotm`, `axang2rotm`, `rotm2axang`, ...) |
| Optimization Toolbox | TRO-2016 baseline (`lsqnonlin`) |
| Computer Vision Toolbox | Sphere fitting in `real_sphere.m` (`pointCloud`, `pcfitsphere`, `pcshow`) |
| Statistics and Machine Learning Toolbox | Box plots in some simulation scripts (`boxplot`) |
| Parallel Computing Toolbox | Optional, GPU variant `plane_Jnt_GH_gpu` only |

The YAML package must be v1.6.0 or newer: from that version on, 1-D sequences are loaded as column vectors, which the data loaders in `fun/misc` rely on.

## Getting started

```bash
git clone https://github.com/ILoveCorn/2DCalib.git
```

1. Install the MATLAB YAML package (via Add-On Explorer, or download it from GitHub and add it to the path).
2. In MATLAB, set the **repository root as the current folder**. All scripts use paths relative to it.
3. Add the repository to the search path:

   ```matlab
   addpath(genpath(pwd))
   ```

   The data folders must be on the path as well, because the loaders open YAML files by file name.
4. Run any script in `evaluation/`.

Shims used to regenerate the Python fixtures are stored as
`python/tools/octave/shims/*.m.in`, not as `.m` files, so this `addpath` does
not shadow MATLAB built-ins.

## Python implementation

A Python package is in [`python/`](python/README.md). The import name is
`handeye2d`. This repository's MATLAB code is unchanged. The Python tree has
the geometry utilities, the OCT YAML readers, the MEAS2023 closed form, the
Gauss-Helmert plane solvers, the plane simulator, tests, and Octave reference
fixtures. The other baselines are not ported yet.

## Repository structure

```text
2DCalib/
├─ data/
│  ├─ oct/plane/       Plane target datasets (10 sessions)
│  ├─ oct/sphere/      Sphere target datasets (10 sessions)
│  └─ measure_*.mat    Reference hand-eye transformations from fiducial registration
├─ evaluation/         Simulation and real-world experiment scripts
├─ python/             Python package (geometry, Gauss-Helmert, MEAS2023) and Octave fixtures
├─ fun/
│  ├─ calib/           Proposed method, its variants, and baseline methods
│  ├─ sim/             Synthetic data generators
│  └─ misc/            Data loaders, fitting, SO(3) utilities, error metrics
└─ result/             Experiment outputs (includes a precomputed calibration result)
```

## Methods

| Function | Description |
|---|---|
| `plane_Jnt_GH` | **Proposed**: joint estimation with the Gauss-Helmert model and variance factor estimation |
| `plane_Jnt_GH_gpu` | Proposed method, GPU implementation |
| `plane_Alt_GH` | Ablation: alternating estimation (plane parameters recovered by SVD) |
| `plane_Jnt_GH_decoupled` | Ablation: orientation estimated without positional constraints |
| `plane_TRO2016` | Baseline, plane target, [10.1109/TRO.2016.2596769](https://doi.org/10.1109/TRO.2016.2596769) |
| `plane_RCIM2020` | Baseline, plane target, [10.1016/j.rcim.2019.101823](https://doi.org/10.1016/j.rcim.2019.101823) |
| `plane_IROS2015` | Baseline, multi-plane target, [10.1109/IROS.2015.7353884](https://doi.org/10.1109/IROS.2015.7353884) |
| `line_RCIM2022` | Baseline, line target, [10.1016/j.rcim.2021.102221](https://doi.org/10.1016/j.rcim.2021.102221) |
| `sphere_MEAS2023` | Baseline, sphere target, [10.1016/j.measurement.2023.112936](https://doi.org/10.1016/j.measurement.2023.112936) |

Baselines are our re-implementations of the cited methods. The matching synthetic data generators are in `fun/sim`.

## Reproducing the experiments

### Simulation

| Script | Experiment |
|---|---|
| `sim_accuracy_vs_noise_level.m` | Accuracy under different robot pose noise levels |
| `sim_accuracy_vs_number_of_measurements.m` | Accuracy versus number of measurements |
| `sim_accuracy_vs_initial_deviation.m` | Convergence versus initial hand-eye deviation |
| `sim_accuracy_vs_image_noise.m` | Accuracy under OCT image noise and distortion |
| `sim_accuracy_vs_VFE.m` | Effect of variance factor estimation |

Simulation scripts print and plot results but do not save them by default.

### Real-world experiments

Run in this order. Each step writes a time-stamped `.mat` file to `result/`.

1. `measure.m`: computes reference hand-eye transformations from fiducial registration. Its output is already provided as `data/measure_260219_1144.mat`.
2. `real_calibration.m`: calibrates with each plane and sphere session using the proposed and baseline methods.
3. `real_plane.m` and `real_sphere.m`: evaluate the calibrations by plane and sphere reconstruction error.

The step 3 scripts load the precomputed `result/real_calibration_260219_1146.mat`, so they can be run directly. To evaluate your own run of step 2, change the file name in the `load` call.

### Known limitations

- `sim_accuracy_vs_initial_deviation.m` calls `plane_Jnt_GH_debug`, which reports iteration counts and is not included in this repository.
- Method 3 in `sim_accuracy_vs_noise_level.m` calls `plane_Alt_GH_debug`, which is not included. It is disabled by default.
- `real_filter_comparison.m` needs the unfiltered raw recording `oct_checkerboard_250714_2030_distorted.yaml`, which is not included.
- `fun/calib/line_RCIM2022.m` (line 58) and `fun/sim/simMEAS2023.m` (line 90) each contain a known bug. They are left unchanged here. The Python fixtures measure the effect; see [`python/README.md`](python/README.md).

## Using your own data

### Notation

Variables follow `<quantity><From>2<To><suffix>`: `tO2H` and `RO2H` are the position and orientation of the OCT frame `O` in the hand (robot flange) frame `H`, and `H2W` is the hand pose in the world (robot base) frame `W`. The suffix `n` denotes noisy measurements and `e` denotes estimates. Each B-scan lies in the x-z plane of the OCT frame. Positions are in millimeters.

### Data format

Each YAML file holds one calibration session. Pass the bare file name (not a path) to the loaders, and keep the `oct` prefix: `loadPlaneYAML` uses it to select the sensor type.

```yaml
captures:
  - pose:                        # hand pose in world frame
      position: {x: ..., y: ..., z: ...}           # meters, converted to mm on load
      orientation: {w: ..., x: ..., y: ..., z: ...} # unit quaternion
      center_side: {sign_y: ...}                    # sphere datasets only
    points:                      # profile points in OCT frame, mm
      x: [...]
      z: [...]
```

### Minimal example

```matlab
% initial guess of the hand-eye transformation
tO2H0 = ...;   % 3 x 1
RO2H0 = ...;   % 3 x 3

% load measurements and initialize plane parameters
[tH2Wn, RH2Wn, pOn, vOn, scanO] = loadPlaneYAML('oct_plane_250710_1057.yaml');
[n0, d0] = planeFit(tO2H0, RO2H0, tH2Wn, RH2Wn, scanO);

% calibrate
sigma0 = [0.1 0.01];   % initial variance factors [translation (mm), rotation (rad)]
[tO2He, RO2He, ne, de, tH2We, RH2We, sigmae] = plane_Jnt_GH( ...
    tO2H0, RO2H0, n0, d0, sigma0, tH2Wn, RH2Wn, pOn, vOn, 30, 50, 1e-5);
```

Input and output dimensions are documented in the header of each function in `fun/calib`.

## Citation

If you use this code or data, please cite:

```bibtex
@article{ma2026uncertainty,
  author  = {Ma, Yan and Lyu, Haitian and Lai, Jiewen and Zhang, Ruiyang and Yuan, Wu and Ren, Hongliang},
  title   = {Uncertainty-Aware Cross-Scale Hand-Eye Calibration of 2-D Optical Coherence Tomography Using a Plane Target},
  journal = {IEEE Transactions on Robotics},
  year    = {2026},
  doi     = {10.1109/TRO.2026.3710397}
}
```

GitHub also offers this citation through **Cite this repository** ([`CITATION.cff`](CITATION.cff)).

## License

- Code is released under the [MIT License](LICENSE).
- Datasets in `data/` and precomputed results in `result/` are released under [CC BY 4.0](data/LICENSE).
