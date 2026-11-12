
#%%
# =============================================================================
# SAMPLE-SIZE SWEEP
# Compare fresh AQRK convergence for several sample sizes on paired trials.
# Each trial uses deterministic seeds and saves raw and normalized trajectories.
#
# n = 100
# s in {1,10,100,1000}
# uniform corruption at a fixed magnitude
# 10 runs per setting
#
# AQRK runs on filtered_kaczmarz with three independently seeded streams:
#     algorithm        row sampling (the batch);
#     selection_noise  noise on the residuals used to select rows;
#     update_noise     noise on freshly queried residuals used in the update.
# Every s uses the same seeds within a trial. The selection noise draws m
# values per iteration whatever s is, so every s sees exactly the same
# selection noise; the batch and update-noise draws depend on s.
# All rows are scanned; there is no candidate subsampling.
# =============================================================================

from src.utils.datagen import rand_system
from src.utils.methods import noisy_methods_optimized

from dataclasses import dataclass
from pathlib import Path
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt

#%%
# =============================================================================
# CONFIG
# =============================================================================


@dataclass
class ExperimentConfig:
    m: int = 2000
    n: int = 100
    err: int = 600
    noise: float = 0.02

    alpha_quantile: float = 0.1

    nruns: int = 10
    n_iter: int = 6000

    ark_alpha: float = 1.0
    aqrk_alpha: float = 1.0

    seed: int = 42


CFG = ExperimentConfig()

q = 1 - CFG.alpha_quantile - CFG.err / CFG.m

#%%
# =============================================================================
# SWEEP PARAMETERS
# =============================================================================

s_values = [1, 10, 100, 1000]

# MAGNITUDE
CORRUPTION_LEVEL = 5

#%%
# =============================================================================
# SEEDS
# Each stream's seed is trial_seed + method offset + stream offset; the
# corruption uses trial_seed + its offset and the data uses trial_seed.
# Every seed is therefore fixed by the trial, and each trial is reproducible
# on its own.
# =============================================================================

CORRUPTION_SEED_OFFSET = 1000

METHOD_SEED_OFFSET = 50000

STREAM_SEED_OFFSETS = {
    "algorithm": 0,
    "selection_noise": 5000,
    "update_noise": 7000,
}

# Offsets must be at least nruns apart, or two streams in different trials
# would receive the same seed.
_offsets = sorted(
    [0, CORRUPTION_SEED_OFFSET]
    + [METHOD_SEED_OFFSET + stream for stream in STREAM_SEED_OFFSETS.values()]
)
assert len(set(_offsets)) == len(_offsets) and np.min(np.diff(_offsets)) >= CFG.nruns, \
    "seed offsets are too close together for this many runs"


def stream_seeds(trial_seed):
    """Seeds of the three AQRK streams in the trial with ``trial_seed``."""
    base = trial_seed + METHOD_SEED_OFFSET
    return {stream: base + offset for stream, offset in STREAM_SEED_OFFSETS.items()}


def make_rngs(seeds):
    """Fresh generators for a dict of stream seeds."""
    return {stream: np.random.default_rng(seed) for stream, seed in seeds.items()}

#%%
# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RESULTS_DIR = Path("results/results_s_sweep_10_runs")
FIGURES_DIR = Path("figures/figures_s_sweep_10_runs")
DATA_DIR = Path("data/data_s_sweep_10_runs")

for d in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

#%%
# =============================================================================
# PLOTTING STYLE
# =============================================================================

plt.rcParams.update({
    "figure.figsize": (7, 5),
    "font.size": 15.5,
    "axes.labelsize": 16.5,
    "axes.titlesize": 17.5,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CB_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
]

LINESTYLES = [
    "-",                 # solid
    "--",                # dashed
    "-.",                # dash-dot
    ":",                 # dotted
    (0, (5, 1)),         # densely dashed
    (0, (3, 1, 1, 1)),   # dash-dot-dot
]

#%%
# =============================================================================
# HELPERS
# =============================================================================

def normalize_curve(curve, x_true):

    return (
        np.asarray(curve)
        / np.linalg.norm(x_true)
    )


def summarize(curves):

    curves = np.asarray(curves)
    if curves.ndim < 2 or curves.shape[0] < 5:
        raise ValueError("At least five runs are required for trimmed bounds")

    sorted_curves = np.sort(curves, axis=0)

    return {
        "mean": curves.mean(axis=0),
        "std": curves.std(axis=0),
        "median": np.median(curves, axis=0),
        "lower_bound": sorted_curves[2],
        "upper_bound": sorted_curves[-3],
    }


def run_aqrk(
    A,
    b,
    x_true,
    s,
    noise,
    rngs,
):
    """AQRK with fresh residuals: filtered_kaczmarz from zero for CFG.n_iter
    iterations with uniform noise and adaptive stopping off. Returns the error
    before each update, length CFG.n_iter."""
    settings = dict(
        q=q,
        alpha=CFG.aqrk_alpha,
        inner_iterations=CFG.n_iter,
        adaptive_stopping=False,
        window=1,                # required key, unused without adaptive stopping
        residual_tolerance=0.0,  # required key, unused without adaptive stopping
    )
    _, _, history = noisy_methods_optimized.filtered_kaczmarz(
        A,
        np.asarray(b, dtype=float).reshape(-1),
        np.zeros(A.shape[1]),
        s,
        settings,
        rngs["algorithm"],
        query_width=noise,
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
        requery=True,
        soln=x_true,
    )
    return history["errors"]

#%%
# =============================================================================
# STORAGE
# =============================================================================

results = {
    s: []
    for s in s_values
}

raw_results = {
    s: []
    for s in s_values
}

trial_data = {}

#%%
# =============================================================================
# MAIN LOOP
# =============================================================================

for s in s_values:

    print(f"\nRunning s = {s}")

    for trial in range(CFG.nruns):

        trial_seed = CFG.seed + trial

        A, b, x_true = rand_system.normalized_gaussian(
            CFG.m,
            CFG.n,
            seed=trial_seed,
        )

        f = (
            math.sqrt(CFG.n)
            * CFG.noise
        )
        corruption_seed = trial_seed + CORRUPTION_SEED_OFFSET
        rng_corr = np.random.default_rng(
            corruption_seed
        )

        b_corr = rand_system.add_uniform_corr(
            CFG.err,
            CORRUPTION_LEVEL * f,
            (CORRUPTION_LEVEL + 1) * f,
            b,
            rng_corr,
        )

        seeds = stream_seeds(trial_seed)

        trial_data.setdefault(trial, {
            "trial": trial,
            "trial_seed": trial_seed,
            "A": A,
            "b": b,
            "x_true": x_true,
            "b_corr": b_corr,
            "corruption_seed": corruption_seed,
            "method_seeds": {},
        })
        trial_data[trial]["method_seeds"][s] = seeds

        raw_curve = np.asarray(
            run_aqrk(
                A,
                b_corr,
                x_true,
                s,
                CFG.noise,
                make_rngs(seeds),
            )
        )

        raw_results[s].append(raw_curve)
        results[s].append(normalize_curve(raw_curve, x_true))

#%%
# =============================================================================
# SUMMARIZE
# =============================================================================

summary_results = {}

for s in s_values:

    summary_results[s] = summarize(
        results[s]
    )

#%%
# =============================================================================
# SAVE DATA
# =============================================================================

np.savez_compressed(
    RESULTS_DIR / "s_sweep.npz",
    config=vars(CFG),
    q=q,
    s_values=np.asarray(s_values),
    corruption_level=CORRUPTION_LEVEL,
    corruption_interval=np.asarray(
        [
            CORRUPTION_LEVEL * math.sqrt(CFG.n) * CFG.noise,
            (CORRUPTION_LEVEL + 1) * math.sqrt(CFG.n) * CFG.noise,
        ]
    ),
    f_scale=math.sqrt(CFG.n) * CFG.noise,
    seed_offsets={
        "corruption": CORRUPTION_SEED_OFFSET,
        "method": METHOD_SEED_OFFSET,
        "stream": STREAM_SEED_OFFSETS,
    },
    trial_data=trial_data,
    unnormalized_results=raw_results,
    raw_results=results,
    summary_results=summary_results,
)

with open(
    DATA_DIR / "s_sweep.pkl",
    "wb",
) as f:

    pickle.dump(
        summary_results,
        f,
    )

#%%
# =============================================================================
# PLOTTING
# =============================================================================
main_data = np.load(
    RESULTS_DIR / "s_sweep.npz",
    allow_pickle=True,
)
summary_results = main_data["summary_results"].item()

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for idx, s in enumerate(s_values):

    level_summary = summary_results[s]
    median_curve = level_summary["median"]
    lower_bound = level_summary["lower_bound"]
    upper_bound = level_summary["upper_bound"]

    ax.plot(
        median_curve,
        color=CB_COLORS[idx],
        linewidth=2.2,
        linestyle = LINESTYLES[idx],
        label=rf"$s={s}$",
    )

    ax.fill_between(
        np.arange(len(median_curve)),
        lower_bound,
        upper_bound,
        color=CB_COLORS[idx],
        alpha=0.15,
    )

ax.set_yscale("log")
#ax.set_ylim(5e-4, 1.8)

ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Error")

# ax.set_title(
#     rf"$n={CFG.n}$, "
#     rf"corruption={CORRUPTION_LEVEL}\sqrt{{n}}r_\infty$"
# )

ax.legend(
    frameon=True,
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "s_sweep_convergence.pdf"
)

plt.close()

print("\nFinished.")
# %%
#%%
# =============================================================================
# FINAL ERROR AT ITERATION 5000 VS s
# =============================================================================

HORIZON = 5000

s_plot = []
err_plot = []

for s in s_values:

    median_curve = summary_results[s]["median"]
    
    idx = min(HORIZON, len(median_curve) - 1)

    s_plot.append(s)
    err_plot.append(median_curve[idx])

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    s_plot,
    err_plot,
    marker="o",
    linewidth=2,
)


ax.set_yscale("log")
ax.set_xscale("log")

ax.tick_params(axis='both', labelsize=10)

ax.set_xlabel(r"Sample size $s$", fontsize = 10)
ax.set_ylabel(r"Relative Error at $k=5000$", fontsize = 10)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "error_vs_s_horizon5000.pdf"
)

plt.close()

print("\nFinished.")
#%%
