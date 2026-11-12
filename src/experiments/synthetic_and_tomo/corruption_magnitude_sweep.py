#%%
# =============================================================================
# CORRUPTION MAGNITUDE SWEEP
# AQRK ONLY, with uniform and adversarial corruption magnitude sweeps.
#
# Uniform:
#   [f,2f], [2f,3f], ..., [5f,6f]
#
# Adversarial:
#   1f, 2f, ..., 6f
#
# Each magnitude level uses the same seeded matrix for a paired comparison.
#
# AQRK runs on filtered_kaczmarz with fresh residuals and three independently
# seeded streams:
#     algorithm        row sampling (the batch);
#     selection_noise  noise on the residuals used to select rows;
#     update_noise     noise on freshly queried residuals used in the update.
# Every level uses the same seeds within a trial, and each stream draws the
# same amount per iteration at every level, so all levels of a trial see
# identical random draws: differences between levels come from the
# corruption alone. All rows are scanned; there is no candidate subsampling.
# =============================================================================

from src.utils.datagen import rand_system
from src.utils.methods import noisy_methods_optimized

from dataclasses import dataclass
from pathlib import Path
import math
import pickle
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
    n_iter: int = 7000

    ark_alpha: float = 1.0
    aqrk_alpha: float = 1.0

    s: int = 100

    seed: int = 42


CFG = ExperimentConfig()

q = 1 - CFG.alpha_quantile - CFG.err / CFG.m

# =============================================================================
# SEEDS
# Each stream's seed is trial_seed + method offset + stream offset; the
# uniform corruption uses trial_seed + its offset and the data uses
# trial_seed. The adversarial corruption is deterministic.
# =============================================================================

UNIFORM_CORRUPTION_SEED_OFFSET = 1000

METHOD_SEED_OFFSET = 50000

STREAM_SEED_OFFSETS = {
    "algorithm": 0,
    "selection_noise": 5000,
    "update_noise": 7000,
}

# Offsets must be at least nruns apart, or two streams in different trials
# would receive the same seed.
_offsets = sorted(
    [0, UNIFORM_CORRUPTION_SEED_OFFSET]
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

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RESULTS_DIR = Path("results/results_separation_sweep")
FIGURES_DIR = Path("figures/figures_separation_sweep")
DATA_DIR = Path("data/data_separation_sweep")

for d in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

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

# =============================================================================
# HELPERS
# =============================================================================

def normalize_curve(curve, x_true):
    return np.asarray(curve) / np.linalg.norm(x_true)


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


# =============================================================================
# METHOD WRAPPER
# =============================================================================

def run_aqrk(A, b, x_true, rngs):
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
        CFG.s,
        settings,
        rngs["algorithm"],
        query_width=CFG.noise,
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
        requery=True,
        soln=x_true,
    )
    return history["errors"]

#%%
# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

uniform_levels = list(range(1, 6))
adversarial_levels = list(range(1, 7))

results = {
    "uniform": {},
    "adversarial": {},
}

raw_results = {
    "uniform": {},
    "adversarial": {},
}

trial_data = {}

for trial in range(CFG.nruns):
    trial_seed = CFG.seed + trial
    A, b, x_true = rand_system.normalized_gaussian(
        CFG.m,
        CFG.n,
        seed=trial_seed,
    )
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    trial_data[trial] = {
        "trial": trial,
        "trial_seed": trial_seed,
        "A": A,
        "b": b,
        "x_true": x_true,
        "uniform_corruptions": {},
        "adversarial_corruptions": {},
        "seeds": {
            "uniform_corruption": trial_seed + UNIFORM_CORRUPTION_SEED_OFFSET,
            "adversarial_corruption": None,
            "method": stream_seeds(trial_seed),
        },
        "u_min": U[:, -1],
    }

f = math.sqrt(CFG.n) * CFG.noise
#%%

# =============================================================================
# UNIFORM SWEEP
# =============================================================================

for level in uniform_levels:

    print(
        f"\nUniform sweep "
        f"[{level}f,{level+1}f]"
    )

    results["uniform"][level] = []
    raw_results["uniform"][level] = []

    for trial in range(CFG.nruns):

        trial_data_entry = trial_data[trial]
        A = trial_data_entry["A"]
        b = trial_data_entry["b"]
        x_true = trial_data_entry["x_true"]
        seeds = trial_data_entry["seeds"]

        rng_corr = np.random.default_rng(
            seeds["uniform_corruption"]
        )

        b_corr = rand_system.add_uniform_corr(
            CFG.err,
            level * f,
            (level + 1) * f,
            b,
            rng_corr,
        )

        trial_data_entry["uniform_corruptions"][level] = b_corr

        raw_curve = np.asarray(
            run_aqrk(
                A,
                b_corr,
                x_true,
                make_rngs(seeds["method"]),
            )
        )

        raw_results["uniform"][level].append(raw_curve)
        results["uniform"][level].append(
            normalize_curve(raw_curve, x_true)
        )
#%%
# =============================================================================
# ADVERSARIAL SWEEP
# =============================================================================

for level in adversarial_levels:

    print(
        f"\nAdversarial sweep "
        f"{level}f"
    )

    results["adversarial"][level] = []
    raw_results["adversarial"][level] = []

    for trial in range(CFG.nruns):

        trial_data_entry = trial_data[trial]
        A = trial_data_entry["A"]
        b = trial_data_entry["b"]
        x_true = trial_data_entry["x_true"]
        seeds = trial_data_entry["seeds"]
        u_min = trial_data_entry["u_min"]

        b_corr = rand_system.add_adversarial_corr(
            CFG.err,
            level * f,
            b,
            u_min,
        )

        trial_data_entry["adversarial_corruptions"][level] = b_corr

        raw_curve = np.asarray(
            run_aqrk(
                A,
                b_corr,
                x_true,
                make_rngs(seeds["method"]),
            )
        )

        raw_results["adversarial"][level].append(raw_curve)
        results["adversarial"][level].append(
            normalize_curve(raw_curve, x_true)
        )
#%%
# =============================================================================
# SUMMARIZE
# =============================================================================

summary_results = {
    "uniform": {},
    "adversarial": {},
}

for level in uniform_levels:

    summary_results["uniform"][level] = summarize(
        results["uniform"][level]
    )
#%%
for level in adversarial_levels:

    summary_results["adversarial"][level] = summarize(
        results["adversarial"][level]
    )

# =============================================================================
# SAVE DATA
# =============================================================================

np.savez_compressed(
    RESULTS_DIR / "corruption_magnitude_sweep.npz",
    config=vars(CFG),
    q=q,
    uniform_levels=np.asarray(uniform_levels),
    adversarial_levels=np.asarray(adversarial_levels),
    f_scale=math.sqrt(CFG.n) * CFG.noise,
    seed_offsets={
        "uniform_corruption": UNIFORM_CORRUPTION_SEED_OFFSET,
        "method": METHOD_SEED_OFFSET,
        "stream": STREAM_SEED_OFFSETS,
    },
    trial_data=trial_data,
    unnormalized_results=raw_results,
    raw_results=results,
    summary_results=summary_results,
)

with open(
    DATA_DIR / "corruption_magnitude_sweep.pkl",
    "wb",
) as f:
    pickle.dump(summary_results, f)

#%%
#%%
# =============================================================================
# PLOTTING: UNIFORM SWEEP (ALL LEVELS)
# =============================================================================

main_data = np.load(
    RESULTS_DIR / "corruption_magnitude_sweep.npz",
    allow_pickle=True,
)
summary_results = main_data["summary_results"].item()

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

fig, ax = plt.subplots(figsize=(8, 6))

for idx, level in enumerate(uniform_levels):

    level_summary = summary_results["uniform"][level]
    median_curve = level_summary["median"]
    lower_bound = level_summary["lower_bound"]
    upper_bound = level_summary["upper_bound"]

    color = CB_COLORS[idx % len(CB_COLORS)]
    linestyle = LINESTYLES[idx]

    ax.plot(
        median_curve,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=rf"$[{level},{(level+1)}]\sqrt{{n}}r_\infty$",
    )

    ax.fill_between(
        np.arange(len(median_curve)),
        lower_bound,
        upper_bound,
        color=color,
        alpha=0.15,
        )

ax.set_yscale("log")
ax.set_ylim(5e-4, 1.8)

ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Error")
#ax.set_title("Uniform Corruption Magnitude Sweep")

ax.legend(
    ncol=2,
    fontsize=14,
    frameon=True,
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "uniform_magnitude_sweep.pdf"
)

plt.close()
#%%
# =============================================================================
# PLOTTING: ADVERSARIAL SWEEP (ALL LEVELS)
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

for idx, level in enumerate(adversarial_levels):

    level_summary = summary_results["adversarial"][level]
    median_curve = level_summary["median"]
    lower_bound = level_summary["lower_bound"]
    upper_bound = level_summary["upper_bound"]

    color = CB_COLORS[idx % len(CB_COLORS)]
    linestyle = LINESTYLES[idx]

    ax.plot(
        median_curve,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=rf"${level}\sqrt{{n}}r_\infty$",
    )

    ax.fill_between(
        np.arange(len(median_curve)),
        lower_bound,
        upper_bound,
        color=color,
        alpha=0.15,
    )

ax.set_yscale("log")
ax.set_ylim(5e-4, 1.8)

ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Error")
#ax.set_title("Singular Vector Corruption Magnitude Sweep")

ax.legend(
    ncol=2,
    fontsize=14,
    frameon=True,
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "adversarial_magnitude_sweep.pdf"
)

plt.close()
# %%
