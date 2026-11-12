#%%
# =============================================================================
# BASE EXPERIMENTS
# Compare RK, ARK, QRK, AQRK, no-resampling AQRK and QABK across four
# corruption types. Each trial uses deterministic seeds and saves raw and
# normalized trajectories.
#
# Every method gets three independently seeded random streams:
#     algorithm        row sampling (the chosen row or batch);
#     selection_noise  noise on the residuals used to select rows;
#     update_noise     noise on freshly queried residuals used in the update.
# A method only draws from the streams it needs; unused ones are left alone.
# All methods scan every row; there is no candidate subsampling.
# =============================================================================

from src.utils.datagen import rand_system
from src.utils.methods import qrk
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
# corruptions use trial_seed + corruption offset and the data uses trial_seed.
# Every seed is therefore fixed by the trial, and each trial is reproducible
# on its own.
# =============================================================================

CORRUPTION_SEED_OFFSETS = {
    "small_uniform": 1000,
    "large_uniform": 2000,
    "sign_flip": 3000,
}

METHOD_SEED_OFFSETS = {
    "RK": 10000,
    "ARK": 20000,
    "QRK": 30000,
    "AQRK": 40000,
    "NR-AQRK": 50000,
    "QABK": 60000,
}

STREAM_SEED_OFFSETS = {
    "algorithm": 0,
    "selection_noise": 5000,
    "update_noise": 7000,
}

# Offsets must be at least nruns apart, or two streams in different trials
# would receive the same seed.
_offsets = sorted(
    [0, *CORRUPTION_SEED_OFFSETS.values()]
    + [method + stream
       for method in METHOD_SEED_OFFSETS.values()
       for stream in STREAM_SEED_OFFSETS.values()]
)
assert len(set(_offsets)) == len(_offsets) and np.min(np.diff(_offsets)) >= CFG.nruns, \
    "seed offsets are too close together for this many runs"


def stream_seeds(trial_seed, method):
    """Seeds of the three streams of ``method`` in the trial with ``trial_seed``."""
    base = trial_seed + METHOD_SEED_OFFSETS[method]
    return {stream: base + offset for stream, offset in STREAM_SEED_OFFSETS.items()}


def make_rngs(seeds):
    """Fresh generators for a dict of stream seeds."""
    return {stream: np.random.default_rng(seed) for stream, seed in seeds.items()}

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RESULTS_DIR = Path("results/results_base_10r")
FIGURES_DIR = Path("figures/figures_base_10r")
DATA_DIR = Path("data/data_base_10r")

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

# Color-blind-friendly palette (Okabe-Ito style).
COLORS = {
    "RK": "#0072B2",       # blue
    "ARK": "#E69F00",      # orange
    "QRK": "#009E73",      # green
    "NR-AQRK": "#D55E00",  # vermillion
    "AQRK": "#CC79A7",     # reddish purple
    "QABK": "#56B4E9",     # sky blue
    "L2": "#000000",      # black
}

LINESTYLES = {
    "RK": "-",                 # solid
    "ARK": "--",                # dashed
    "QRK": "-.",                # dash-dot
    "AQRK": ":",                 # dotted
    "NR-AQRK": (0, (5, 1)),         # densely dashed
    "QABK": "-",
    "L2": (0, (3, 1, 1, 1)),   # dash-dot-dot
}


# =============================================================================
# HELPERS
# =============================================================================

def relative_l2_error(A, b_corrupted, x_true):
    x_l2, *_ = np.linalg.lstsq(A, b_corrupted, rcond=None)
    return np.linalg.norm(x_l2 - x_true) / np.linalg.norm(x_true)


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
# METHOD WRAPPERS
# Each takes the generators from make_rngs as ``rngs`` and returns the error
# before each update, length CFG.n_iter.
# =============================================================================

def run_rk(A, b, x_true, noise, rngs):
    return qrk.RK(
        A, b,
        CFG.n_iter,
        x_true,
        noise,
        rngs["algorithm"],
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
    )


def run_ark(A, b, x_true, noise, rngs, s=CFG.s, n_iter=CFG.n_iter):
    return qrk.ARK(
        A, b,
        noise,
        n_iter,
        x_true,
        CFG.ark_alpha,
        s,
        rngs["algorithm"],
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
    )


def run_qrk(A, b, x_true, noise, rngs):
    return qrk.quantileRK(
        A, b,
        q,
        CFG.n_iter,
        x_true,
        noise,
        rngs["algorithm"],
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
    )


def _run_filtered(A, b, x_true, noise, rngs, batch_size, requery):
    """Run filtered_kaczmarz from zero for CFG.n_iter iterations with uniform
    noise and adaptive stopping off."""
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
        batch_size,
        settings,
        rngs["algorithm"],
        query_width=noise,
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
        requery=requery,
        soln=x_true,
    )
    return history["errors"]


def run_aqrk(A, b, x_true, noise, rngs):
    return _run_filtered(A, b, x_true, noise, rngs, CFG.s, requery=True)


def run_aqrk_no_resample(A, b, x_true, noise, rngs):
    return _run_filtered(A, b, x_true, noise, rngs, CFG.s, requery=False)


def run_qabk(A, b, x_true, noise, rngs):
    return _run_filtered(A, b, x_true, noise, rngs, "all", requery=False)


RUNNERS = {
    "RK": run_rk,
    "ARK": run_ark,
    "QRK": run_qrk,
    "AQRK": run_aqrk,
    "NR-AQRK": run_aqrk_no_resample,
    "QABK": run_qabk,
}

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

experiments = {
    "small_uniform": "Small Uniform Corruption",
    "large_uniform": "Large Uniform Corruption",
    "sign_flip": "Sign Flip Corruption",
    "adversarial": "Singular Vector Adversary",
}

methods = ["RK", "ARK", "QRK", "AQRK", "NR-AQRK", "QABK"]

results = {
    e: {m: [] for m in methods}
    for e in experiments
}

raw_results = {
    e: {m: [] for m in methods}
    for e in experiments
}

l2_errors = {e: [] for e in experiments}
trial_data = []
#%%

# =============================================================================
# MAIN LOOP
# =============================================================================

for trial in range(CFG.nruns):

    trial_seed = CFG.seed + trial

    print(f"Trial {trial+1}/{CFG.nruns}")

    A, b, x_true = rand_system.normalized_gaussian(
        CFG.m,
        CFG.n,
        seed=trial_seed,
    )

    f = math.ceil(
        10
        * math.sqrt(CFG.n)
        * CFG.noise
    )

    corruption_seeds = {
        name: trial_seed + offset
        for name, offset in CORRUPTION_SEED_OFFSETS.items()
    }
    method_seeds = {
        method: stream_seeds(trial_seed, method)
        for method in methods
    }

    rng_small = np.random.default_rng(corruption_seeds["small_uniform"])
    rng_large = np.random.default_rng(corruption_seeds["large_uniform"])
    rng_sign = np.random.default_rng(corruption_seeds["sign_flip"])

    b_small = rand_system.add_uniform_corr(
        CFG.err, 0.1, 0.4, b, rng_small
    )

    b_large = rand_system.add_uniform_corr(
        CFG.err, f, f + 1, b, rng_large
    )

    b_sign = rand_system.add_signflip_corr(
        CFG.err, b, rng_sign
    )

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    u_min = U[:, -1]

    b_adv = rand_system.add_adversarial_corr(
        CFG.err, f, b, u_min
    )

    corruptions = {
        "small_uniform": b_small,
        "large_uniform": b_large,
        "sign_flip": b_sign,
        "adversarial": b_adv,
    }

    trial_data.append({
        "trial": trial,
        "trial_seed": trial_seed,
        "A": A,
        "b": b,
        "x_true": x_true,
        "corruptions": corruptions,
        "corruption_seeds": corruption_seeds,
        "method_seeds": method_seeds,
    })

    for exp_name, b_corr in corruptions.items():

        l2_errors[exp_name].append(
            relative_l2_error(A, b_corr, x_true)
        )

        # Fresh generators for every corruption type, so each method sees the
        # same random streams across the four experiments of a trial.
        method_curves = {
            method: RUNNERS[method](
                A, b_corr, x_true, CFG.noise, make_rngs(method_seeds[method])
            )
            for method in methods
        }

        for method, curve in method_curves.items():
            raw_results[exp_name][method].append(np.asarray(curve))
            results[exp_name][method].append(
                normalize_curve(curve, x_true)
            )
#%%

# =============================================================================
# SUMMARIZE
# =============================================================================

summary_results = {}
for exp_name in experiments:
    summary_results[exp_name] = {
        method: summarize(results[exp_name][method])
        for method in methods
    }

    l2_errors[exp_name] = {
        "per_trial": np.asarray(l2_errors[exp_name]),
        "mean": float(np.mean(l2_errors[exp_name])),
    }


# =============================================================================
# SAVE DATA
# =============================================================================

np.savez_compressed(
    RESULTS_DIR / "experiment_results.npz",
    config=vars(CFG),
    q=q,
    seed_offsets={
        "corruption": CORRUPTION_SEED_OFFSETS,
        "method": METHOD_SEED_OFFSETS,
        "stream": STREAM_SEED_OFFSETS,
    },
    trial_data=trial_data,
    unnormalized_results=raw_results,
    raw_results=results,
    summary_results=summary_results,
    l2_errors=l2_errors,
)

with open(DATA_DIR / "summary_results.pkl", "wb") as f:
    pickle.dump(summary_results, f)

#%%
# =============================================================================
# PLOTTING
# =============================================================================
# =============================================================================
# LOAD SAVED RESULTS
# =============================================================================

data = np.load(
    RESULTS_DIR / "experiment_results.npz",
    allow_pickle=True,
)

raw_stored_res = data["raw_results"].item()

summary_results = data["summary_results"].item()
l2_errors = data["l2_errors"].item()


# =============================================================================
# PLOTTING
# =============================================================================

experiments = {
    "small_uniform": "Small Uniform Corruption",
    "large_uniform": "Large Uniform Corruption",
    "sign_flip": "Sign Flip Corruption",
    "adversarial": "Singular Vector Adversary",
}

for exp_name, title in experiments.items():

    fig, ax = plt.subplots()

    for method in methods:

        method_summary = summary_results[exp_name][method]
        median_curve = method_summary["median"]
        lower_bound = method_summary["lower_bound"]
        upper_bound = method_summary["upper_bound"]

        x = np.arange(len(median_curve))

        ax.plot(
            x,
            median_curve,
            label="ARK (s = 100)" if method == "ARK" else method,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.2,
        )

        ax.fill_between(
            x,
            lower_bound,
            upper_bound,
            color=COLORS[method],
            alpha=0.15,
        )

    ax.axhline(
        l2_errors[exp_name]["mean"],
        color=COLORS["L2"],
        linestyle="--",
        linewidth=2,
        label="Least Squares",
    )

    ax.set_yscale("log")
    ax.set_ylim(5e-4, 1.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Error")
    #ax.set_title(title)

    ax.legend(
        frameon=True,
        loc="lower left",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{exp_name}_10_run.pdf"
    )

    plt.close()

print("Finished.")
# %%
