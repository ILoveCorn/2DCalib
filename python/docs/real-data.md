# Real calibration and plane reconstruction

`experiments/real_calibration.py` follows `evaluation/real_calibration.m`.
`experiments/real_plane.py` follows `evaluation/real_plane.m`. Both are
deterministic. They do not write into `result/`. `real_sphere` and `measure`
are not ported.

## Calibration

`real_calibration` is checked against `result/real_calibration_260219_1146.mat`.

| Column | Bound against the saved poses |
|---|---|
| Proposed (`plane_jnt_gh`) | translation and rotation, atol 1e-8 |
| TRO 2016 | position ≤ 0.05 mm, angle ≤ 0.01 deg |
| RCIM 2020 | translation atol 1e-8, angle ≤ 1e-4 deg |
| MEAS 2023 | translation and rotation, atol 1e-8 |
| Measured | exact copy of `data/measure_260219_1144.mat` |

The solver's reported time is stored and is not compared. The saved
`planeFiles` still use the old `oct_checkerboard_*` names. The driver lists
`data/oct/plane/oct_plane*.yaml` in sorted order, which is the same order
after that rename. Sphere files are sorted the same way. Both counts are 10,
so the MATLAB branch that copies the first sphere result is not used on this
dataset.

## Plane reconstruction

There is no `result/real_plane_*.mat` in the repository. The table below is
`plane_fit`'s mean absolute residual, in millimeters, on the saved calibration
mat. Plane methods omit the session that produced that column. Each row is one
test session, and the last number is the mean of every finite entry. This is
the same `fprintf` as `real_plane.m`.

```text
1 - Proposed:   0.171  0.180  0.193  0.150  0.139  0.275  0.123  0.122  0.220  0.130  0.170
2 - TRO-2016:   1.259  1.208  1.262  1.090  1.280  1.194  1.226  1.326  1.063  1.043  1.195
3 - RCIM-2020:  0.247  0.244  0.252  0.222  0.221  0.341  0.208  0.223  0.291  0.230  0.248
4 - MEAS-2023:  0.950  0.934  0.880  0.915  0.782  0.917  0.894  0.922  0.727  0.753  0.867
5 - Measured:   0.366  0.391  0.452  0.280  0.294  0.573  0.285  0.322  0.145  0.325  0.343
```

On this mat the proposed column has the smallest overall mean (0.170 mm).
The published table is not in the repository, and the open abstract does not
give these digits, so this run was not scored against the paper.
