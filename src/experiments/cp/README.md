# CP tensor fitting

This folder contains the final CP experiment: **30 × 30 × 30, rank 5, ten paired
initializations**, with 10% fixed replacement corruption, 2% fixed fitting
perturbation, and 5% fresh query noise. It includes the filtered methods and
both unfiltered controls used in the paper figure.

[`cp_tensor_experiments_normalized.ipynb`](cp_tensor_experiments_normalized.ipynb)
defines the data, initializations, and alternating CP updates. Its inner
quantile updates call the repository's shared
[`filtered_kaczmarz`](../../utils/methods/noisy_methods_optimized.py).
`run_cp.py` loads the notebook definitions and saves the notebook and runner
snapshots for each group.

## View the included figure

Open the [paper figure (PDF)](figures/cp30_2pct_full10.pdf) or
[PNG](figures/cp30_2pct_full10.png) directly. Python setup is needed for the
commands below.

## Environment

The shared-solver workflow was tested with **Python 3.9.6, NumPy 1.23.5, and
Matplotlib 3.7.0**. Create a local virtual environment from the repository
root. If `python3 --version` reports Python 3.9, you can use `python3` in
place of `python3.9` below.

```bash
python3.9 -m venv .venv-cp
source .venv-cp/bin/activate
python -m pip install -r src/experiments/cp/requirements.txt
```

Git ignores the virtual environment. Use the activated environment for the
commands below. The command-line workflow works without Jupyter.

If you use the notebook, select a Jupyter kernel with the CP dependencies.
Start Jupyter in `src/experiments/cp/` with the repository root on the Python
import path. From the repository root, the launch command for JupyterLab is:

```bash
(cd src/experiments/cp && PYTHONPATH=../../.. python -m jupyterlab)
```

This requires JupyterLab in the activated environment. Run All replots the
archive with the default `RUN_EXPERIMENTS = False` and writes the files under
`src/experiments/cp/generated_figures/`. To start a new full run from the
notebook, set `RUN_EXPERIMENTS = True` and choose a fresh `RUN_DIRECTORY`.
For a resume, keep the notebook and runner files identical to those used
to start the run.

## Replot the final figure without rerunning

With the environment activated, run this from the repository root:

```bash
MPLBACKEND=Agg python src/experiments/cp/plot_cp.py --output cp_figures/cp30_2pct_full10
```

The plotter reads `src/experiments/cp/saved_runs/paper/` and writes a PDF, PNG,
and figure-source JSON under `cp_figures/` without running a solver. To
validate the archive separately, run `python src/experiments/cp/check_results.py`
from the repository root. Plotting details may vary across platforms.

## Verify the code and archived trajectories

Run these commands from the repository root with the environment activated:

```bash
MPLBACKEND=Agg PYTHONPATH=. python src/experiments/cp/check_solver.py
PYTHONPATH=. python src/experiments/cp/run_cp.py --smoke --workers 4 --output cp_runs/smoke
python src/experiments/cp/check_results.py --smoke-run cp_runs/smoke
```

The smoke run uses the full tensor, one initialization and only two outer
sweeps. The last command compares all 25 method/case trajectories with the
corresponding prefixes of the archived paper runs. The solver checks cover
normalization, zero rows, quantile ties, q=1, repeated-row fresh queries,
adaptive stopping, and agreement with the archived notebook on seeded inner
solves. The archived notebook for the regenerated runs also imports the
shared solver from this checkout.

## Rerun the complete paper experiment

Run these commands from the repository root with the environment activated:

```bash
PYTHONPATH=. python src/experiments/cp/run_cp.py --workers 4 --trials 10 --output cp_runs/paper
MPLBACKEND=Agg python src/experiments/cp/plot_cp.py --run cp_runs/paper --output cp_figures/cp30_2pct_full10
```

This is a substantial computation. The first command runs the filtered group
and then the unfiltered group, saving each completed method/trial fit atomically.
It uses independent processes and one BLAS thread per process. Times from these
parallel runs are not intended as method speed comparisons.

Repeat the command from the same checkout and environment to resume missing
fits. The runner checks the notebook, runner, settings, and saved inputs.
Use the same shared solver and dependencies when resuming. To run one group,
pass `--group filtered` or `--group unfiltered`. Each group uses its matching
subdirectory under `--output`. To run a three-trial pilot, use `--trials 3` and
a fresh output directory. The plotter requires matching settings and inputs
for both groups.

Use a fresh output directory for a new run. Resuming requires the checkpoints
from the original run directory.

## Rebuild the compact archive

After completing the ten-trial rerun above, package it into a new directory
and validate it there. Run both commands from the repository root with the
environment activated:

```bash
python src/experiments/cp/package_cp.py --run cp_runs/paper --output cp_runs/compact-paper
python src/experiments/cp/check_results.py --archive cp_runs/compact-paper
```

Pass `--run cp_runs/compact-paper` to the plotter to use the rebuilt archive.

The packager requires both groups to be complete and to share identical inputs.
It preserves the notebook and runner snapshots from the generated run in
`original_sources.zip`, writes checksums to `manifest.json`, and excludes the
individual trial checkpoints from the compact archive. The destination must be
new or empty.

## Exact experiment settings

- The exact tensor has independent standard Gaussian factors (seed 0).
- The approximate target adds one fixed Gaussian perturbation (seed 7), scaled
  to 2% of the exact tensor's Frobenius norm. The target is fixed across methods
  and initializations.
- Corruption replaces 2,700 entries chosen without replacement (seed 42).
  Replacement values are uniform on `[-2*tau, 2*tau]`, where `tau` is the
  90th percentile of the target's absolute entries. The exact range is about
  **[-7.31334, 7.31334]** for the exact target and **[-7.30201, 7.30201]** for the
  approximate target. The corruption locations and underlying uniform draws
  are shared across cases. These are replacement values, not additive errors.
  No rounding to [-7, 7] or change to [-10, 10] was made.
- The query-noise case uses the exact target and the same fixed replacement
  corruption. Every query adds fresh independent Gaussian noise with standard
  deviation `0.05 * norm(exact_target) / sqrt(30**3)` = 0.1159435713, before
  row normalization. This 5% query noise is distinct from the 2% fixed perturbation.
- Initialization seeds are 1 through 10, sampling seeds 124, 126, ..., 142, and
  query seeds 70124, 70126, ..., 70142. All methods share the same initial factors
  within a trial, also shared across the three cases. The clean targets and
  fixed corruptions do not change between trials.
- Inner coefficient rows and RHS entries are normalized together once per mode
  update; zero rows are omitted. Query-noise standard deviations are divided
  by the corresponding row norms. Factor rows are warm-started. Column
  normalization preserves the represented tensor by absorbing weights into
  the first factor.
- The filtered methods use q=0.85, alpha=1, full screening, s=1, 5, 10, 50 or all
  eligible rows. Quantiles use NumPy's linear interpolation convention and
  retain ties. Sampled update rows are drawn with replacement. One partition
  supplies the quantile and stopping statistic; deterministic scalar products
  are computed once per update and reused.
- In the query-noise case, every sampled update occurrence gets fresh noise,
  including duplicate row occurrences. All-row updates reuse the screening
  observations. With fixed observations, AQRK(s=1) coincides with QRK; with fresh
  update queries it is not the conventional reused-observation QRK.
- The unfiltered controls use the same solver with q=1, s=10 or all rows.
  No other numerical setting changes. The s=10 query-noise control uses fresh
  update observations; the all-row control reuses screening observations.
- ALS solves the original, unnormalized least-squares systems with ridge
  `1e-8 * I`. In the query-noise case it fits one fixed noisy tensor (seed 70042),
  so this is not an equal-query-budget comparison. Clean-target ALS is retained
  in the saved approximate-case results but is not plotted.
- Outer sweeps: exact 120, query noise 120, approximate 60. The figure displays
  the first 60, 30 and 30, respectively. The saved histories include the full runs.

## Inner stopping rule

The statistic is the RMS of the smallest `ceil(q*m)` absolute row-normalized
residuals, where m is the number of nonzero coefficient rows. Thus the q=1
controls use all residuals. Starting at update 40, check every 20 updates:
compare the mean statistic over the latest 20 with the preceding 20. Stop
after **two consecutive** checks with relative improvement below 0.1%.
Sufficient improvement resets the counter; worsening counts as insufficient
improvement. The earliest plateau stop is therefore at update 60.

At the same checks, also stop if the current statistic is at most `1e-12`
times the RMS of the first observed, normalized RHS. The hard cap is 500 updates.
Query-noise stopping uses noisy observations, not the clean target. This is a
computational stopping heuristic, not a convergence certificate. Changing its
parameters or the statistic requires new results.

## Figure and archive contents

The figure shows median relative errors against the uncorrupted target and
25th–75th percentile bands over ten paired initializations. These bands are
not confidence intervals. The dotted horizontal line in panel (c), described
in the caption rather than the legend, is
`norm(approximate_target-exact_target)/norm(approximate_target)` = 0.01999615118.
It is the error of the generating rank-5 tensor, not a lower bound on all fits.
[`paper_caption.tex`](paper_caption.tex) contains a caption for the included PDF.

`saved_runs/paper/filtered/` contains the regenerated filtered and ALS histories.
`unfiltered/` contains the regenerated q=1 s=10 and all-row histories. Both
groups use identical inputs and include settings, histories, summaries, and
provenance. `manifest.json` gives checksums, and `original_sources.zip` holds the
notebook and runner snapshots. The shared solver is part of the repository
source. A full run also produces per-fit checkpoints with raw diagnostics.
