# Averaged quantile Kaczmarz methods

Code for the synthetic, tomography, and CP tensor fitting experiments in
*Beyond the error horizon: averaged quantile Kaczmarz methods for corrupted and noisy linear systems*.

## CP tensor fitting

The [CP experiment guide](src/experiments/cp/README.md) provides the final
30 x 30 x 30, rank-5 experiment with ten paired initializations, both unfiltered
controls, and a compact archive for replotting. Open the
[CP paper figure (PDF)](src/experiments/cp/figures/cp30_2pct_full10.pdf) directly,
or follow the guide to replot or rerun the experiment.

## Synthetic and tomography experiments

### View the included figures

Open the [synthetic and tomography paper figures](src/experiments/synthetic_and_tomo/paper/figures_style_consistency_median/)
to view the PDFs directly. The [individual experiment plots](src/experiments/synthetic_and_tomo/figures/)
are also available. To regenerate the paper figures from saved results, use the
replot command below.

### Set up the environment

The synthetic and tomography workflow was tested with Python 3.9.6, NumPy
2.0.2, Matplotlib 3.9.4, and SciPy 1.13.1. From the repository root, create and
activate a local virtual environment. If `python3 --version` reports Python
3.9, you can use `python3` in place of `python3.9` below.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/experiments/synthetic_and_tomo/requirements.txt
```

### Replot the paper figures from saved results

With the environment activated, run this from the repository root:

```bash
python src/experiments/synthetic_and_tomo/generate_all_figures_style_consistency.py
```

The script reads the saved archives in `src/experiments/synthetic_and_tomo/results/`
and writes the PDFs to the [paper figures folder](src/experiments/synthetic_and_tomo/paper/figures_style_consistency_median/).
Replotting skips the solvers and replaces any PDFs with the same names.

### Rerun the experiments

With the environment activated, start from the repository root and run the
scripts for the experiments you want to repeat. The tomography script reads
the included `tomo_test.mat` from its working directory. Each rerun replaces
that experiment's saved outputs in `results/`, `data/`, and `figures/`.

```bash
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
cd src/experiments/synthetic_and_tomo
python base_experiments.py
python fixed_noise_experiments.py
python corruption_magnitude_sweep.py
python s_sweep.py
python scheduled_aqrk_horizon.py
python tomography_gaussian_noise.py
```

To replot after a rerun, run `python generate_all_figures_style_consistency.py`
from this directory. Each plot uses the current archive for its experiment.
The fixed-noise comparison table uses both the base and fixed-noise archives.

## Methods

Each linear-system run starts at `x0 = 0` and records `||x_k - x*||` before
update `k`. A run of `N` updates records iterates 0 through `N - 1`. The
experiments divide by `||x*||` for the plotted relative errors.

The shared filtered solver uses the observed right-hand side minus the
prediction. RK and ARK use prediction minus the right-hand side with the
corresponding subtraction in the update.

| Method | Implementation | Result label | Update |
|---|---|---|---|
| RK | `qrk.RK` | `RK` | One uniformly random row with a fresh noisy residual |
| ARK (s) | `qrk.ARK` | `ARK` | Average over `s` rows sampled uniformly with replacement, with fresh query noise for each occurrence |
| QRK | `qrk.quantileRK` | `QRK` | Screen all rows and update with one uniformly selected eligible row using its screening residual |
| AQRK (fresh) | `filtered_kaczmarz`, `requery=True` | `AQRK` | Average over `s` eligible rows sampled with replacement, with fresh query noise for each occurrence |
| AQRK (reused) | `filtered_kaczmarz`, `requery=False` | `NR-AQRK` | Sample as in fresh AQRK and reuse the screening residuals |
| AQRK-all (QABK) | `filtered_kaczmarz`, `batch_size="all"` | `QABK` | Average over all eligible rows using their screening residuals |
| Scheduled AQRK | `filtered_kaczmarz` with per-iteration stepsizes and batch sizes | Schedule name | Fresh AQRK with stepsize `alpha_k` and batch size `s_k` |

Filtered methods screen all `m` rows and keep those at or below the
`q`-quantile of absolute residuals. The experiments use
`q = 1 - 0.1 - beta` and stepsize 1, except for the decreasing-stepsize
schedule. Query noise is specified by `noise_scale` in the RK, ARK and QRK
wrappers and by `query_width` in the shared solver. The value is the half-width
for uniform noise and the standard deviation for Gaussian noise. The
fixed-noise cases perturb the right-hand side once and use zero query noise.

## Experiments

### Synthetic systems

All synthetic experiments share the following setting, created by `rand_system.normalized_gaussian`:

- `A` is 2000 × 100 with rows drawn from a standard Gaussian and normalized to unit length.
- `x* ~ N(0, I)` and `b* = A x*`.
- 600 rows (β = 0.3) are corrupted.
- In the query-noise experiments, the noise is `Uniform[-0.02, 0.02]` and the separation scale is `sqrt(n) r_inf = 0.2`.
- The filtered methods use `q = 0.6` and screen all 2000 rows.

Corruption models in `rand_system.py`:

| Model | Function | Description |
|---|---|---|
| Oblivious | `add_uniform_corr(err, l, r, b, rng)` | Random support; add `Uniform[l, r]` with positive bounds in these experiments |
| Sign flip | `add_signflip_corr(err, b, rng)` | Random support; `b_i -> -b_i` |
| Singular-vector | `add_adversarial_corr(err, f, b, u)` | Support at the `err` largest entries of `abs(u)`, where `u` is the left singular vector for the smallest singular value; add `f * sign(u_i)` |

| Script | Methods | Corruption | s | Iterations | Output directory |
|---|---|---|---|---|---|
| `base_experiments.py` | RK, ARK, QRK, AQRK, NR-AQRK, QABK | Oblivious [0.1, 0.4]; oblivious [2, 3]; sign flip; singular-vector with f = 2 | 100 | 7000 | `results/results_base_10r/` |
| `fixed_noise_experiments.py` | RK, ARK, QRK, AQRK, QABK | As in the base experiment | 100 | 7000 | `results/results_fnoise_10r/` |
| `corruption_magnitude_sweep.py` | AQRK | Oblivious [ℓ, ℓ+1]·0.2, ℓ = 1..5; singular-vector with f = ℓ·0.2, ℓ = 1..6 | 100 | 7000 | `results/results_separation_sweep/` |
| `s_sweep.py` | AQRK | Oblivious [1, 1.2] | 1, 10, 100, 1000 | 6000 | `results/results_s_sweep_10_runs/` |
| `scheduled_aqrk_horizon.py` | AQRK, scheduled AQRK | Oblivious [1, 1.2] | see below | budget-limited | `results/results_scheduled_aqrk_horizon/` |

In `fixed_noise_experiments.py`, a vector of independent `Uniform[-0.02, 0.02]` entries is added once to the corrupted right-hand side, and every method runs with noise-free queries. With noise-free queries, the fresh and reused AQRK variants coincide, so only `AQRK` is run.

`scheduled_aqrk_horizon.py` compares five schedules under a shared cap of
21 million residual queries. Each update costs `m + s_k` queries, and the final
diagnostic scan is excluded. A schedule stops when its next update would
exceed the cap. The individual experiment plots show all five schedules.
The paper figures show the constant, decreasing-stepsize, and exponential
schedules.

| Name | Stepsize | Batch size | Iterations |
|---|---|---|---|
| `constant` | 1 | 100 | 10000 |
| `decreasing_stepsize` | 1 for k < 2500, then 2500/k | 100 | 10000 |
| `exponential_batches` | 1 | `ceil(100 * 0.999^(-k))` | 4751 |
| `geometric_batches_T500` | 1 | `100 * 2^floor(k/500)` | 4009 |
| `geometric_batches_T1000` | 1 | `100 * 2^floor(k/1000)` | 6321 |

### Tomography

`tomography_gaussian_noise.py` uses the tomography test problem from Regularization Tools (Hansen), stored in `tomo_test.mat`:

- The file contains `A` (9000 × 100), `b_ex` and the reference image `x_ex`.
- Rows are normalized and the zero rows removed, which leaves a 1148 × 100 system with condition number about 62.
- `b* = A x*` is recomputed from the normalized matrix.
- In each trial, `floor(0.12 * m)` entries are selected without replacement from the normalized system. Each selected entry receives a corruption with random sign and magnitude `Uniform[5, 6]`. For 1148 rows, this gives 137 entries (β ≈ 0.119).
- `q = 1 - 0.1 - 137/1148 ≈ 0.781`, with full screening and `alpha = 1`.
- ARK and AQRK use `s = n = 100`.
- 20000 iterations.

| Case key | Noise |
|---|---|
| `uniform_only` | Gaussian query noise, σ = 0.06 |
| `uniform_plus_fixed_noise` | Independent Gaussian noise, σ = 0.06 per entry, added once to `b`, with noise-free queries |

Both cases share the same corruption within a trial. The matrix `A` and
reference solution `x*` are fixed across trials. Archives are written under
`results/results_tomography_gaussian_noise_10r/`.

## Reproducibility

- Trial `i` uses `trial_seed = 42 + i`.
- Row sampling, screening noise, and fresh update noise use separate
  `numpy.random.default_rng` streams seeded with offsets from `trial_seed`.
- Within a trial, methods and sweep levels share the same clean system.
  Corruption-model and sweep comparisons reuse the method seeds, and the
  scheduled comparison uses distinct stream seeds for each schedule.
- The seeds of each trial are stored in its `trial_data` entry in the results archive.

Results can differ slightly across NumPy versions if the random-number streams change.

## Outputs

Each linear-system experiment uses ten trials and writes a compressed archive
under `results/`, a summary pickle
under `data/`, and quick-look PDFs under `figures/`. The paper-figure script
uses the saved archives. For the sample-size sweep, the paper figure reports
the relative error at the last recorded iterate, `k = 5999` for a 6000-update
run. The quick-look plot uses `k = 5000`.
