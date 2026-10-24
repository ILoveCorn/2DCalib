# Effect of the two MATLAB bugs

Python implements the corrected behavior. The MATLAB sources in `fun/` are
unchanged. `tests/fixtures/bug_impact.mat` compares the original functions with
the same two edits the Octave exporter applies before building the other
fixtures. This note reports that file. Each noise setting uses 6 repetitions.
Position error is millimeters and orientation error is degrees, both from
`calibrationError` against the pose that generated the cloud.

## `line_RCIM2022` column index

`fun/calib/line_RCIM2022.m:58` is

```matlab
ub(:,(1:(mt-1))+(mt-1)) = ...
```

Every pose overwrites one block of `mt - 1` columns. The probe in the fixture
counts nonzero columns: the original leaves `mt - 1` of them, and the patch
`(1:(mt-1))+(mt-1)*(i-1)` fills all `m*(mt-1)` columns. Here `m = 9` and
`mt = 3`, so that is 2 columns against 18.

The bug is after `Rsa` is solved, so the orientation error is the same for
both versions at every noise level (the two median columns below match to the
printed digits). With no pose noise the position error is about 1e-7 mm either
way. With noise, dropping the other poses changes the position error. At
0.1 mm / 0.1° the patched median is smaller. At 1 mm / 1° this 6-draw sample
goes the other way. Medians:

| Pose noise | Original position | Patched position | Orientation (both) |
|---|---:|---:|---:|
| 0 | 1.7e-7 | 1.4e-7 | 6.6e-6 |
| 0.1 | 0.88 | 0.61 | 27 |
| 0.5 | 2.75 | 2.71 | 88 |
| 1 | 2.77 | 3.44 | 105 |

`line_rcim2022` uses the patched columns. The committed `line_rcim2022.mat`
fixture is that patched solver, and its noiseless draw does not show the bug.

## `simMEAS2023` field of view

`fun/sim/simMEAS2023.m:90` is

```matlab
xr = min(pcOx(it)+rO(it), zMax) - 1e-4;
```

The right edge of the sphere arc should use `xMax`. The defaults are
`xMax = 5` and `zMax = 2`. In the fixture every original right edge is
`1.9999`, and every patched right edge reaches `4.9999`. The left edge stays
`1e-4`.

On 6 noisy draws (`sigmat = 0.2`, `sigmaO = 0.05`) the clipped arc moves the
calibration much farther from the generating pose:

| | Position median (min, max) | Orientation median (min, max) |
|---|---|---|
| Original (`zMax`) | 8.53 (7.81, 8.92) | 151 (119, 155) |
| Patched (`xMax`) | 0.48 (0.25, 1.16) | 1.91 (0.86, 7.08) |

`sim_meas2023` samples with `xMax`. Later Monte Carlo comparisons with the
paper should expect the Xu and Tang numbers to move because of these two
corrections. Those experiment scripts are not in this tree.
