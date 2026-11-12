#%%
# =============================================================================
# TOMOGRAPHY GAUSSIAN-NOISE EXPERIMENT
# Compare RK, ARK, QRK, AQRK, and QABK under two tomography conditions:
#   1) 12% uniform corruption (with Gaussian query noise)
#   2) 12% uniform corruption plus fixed additive Gaussian noise
#      (exact queries)
#
# This script follows the same reproducible experiment template used elsewhere:
# deterministic per-trial seeds, saved raw trajectories, and trimmed summary bands
# computed from the median and the 3rd smallest / 3rd largest values.
#
# Every method gets three independently seeded random streams:
#     algorithm        row sampling (the chosen row or batch);
#     selection_noise  noise on the residuals used to select rows;
#     update_noise     noise on freshly queried residuals used in the update.
# A method only draws from the streams it needs. All methods scan every row;
# there is no candidate subsampling.
# =============================================================================

from src.utils.methods import qrk
from src.utils.methods import noisy_methods_optimized
from scipy.io import loadmat
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

TOMOGRAPHY_NOISE_DISTRIBUTION = "gaussian"


def row_normalize(A, b):
    rows, cols = A.shape
    zerorows = []
    A = A.copy()
    b = b.copy()

    for i in range(rows):
        norm = np.linalg.norm(A[i, :])
        if norm != 0:
            A[i, :] /= norm
            b[i] /= norm
        else:
            zerorows.append(i)

    return A, b, zerorows


def add_uniform_corr(b, n_corr, low, high, rng):
    b_corr = b.copy()

    idx = rng.choice(len(b), size=n_corr, replace=False)

    corruptions = rng.uniform(low, high, size=(n_corr, 1))
    signs = rng.choice([-1, 1], size=(n_corr, 1))

    b_corr[idx] += signs * corruptions
    return b_corr


def add_fixed_noise(b, sigma, rng):
    return b + sigma * rng.standard_normal(size=b.shape)


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


@dataclass
class ExperimentConfig:
    corruption_fraction: float = 0.12
    noise: float = 0.06
    fixed_noise_sigma: float = 0.06
    alpha_quantile: float = 0.1
    nruns: int = 10
    n_iter: int = 20000
    ark_alpha: float = 1.0
    aqrk_alpha: float = 1.0
    seed: int = 42


CFG = ExperimentConfig()

# =============================================================================
# SEEDS
# The corruption uses trial_seed itself and the fixed noise trial_seed + 7000,
# as before. Each method stream's seed is trial_seed + method offset + stream
# offset, laid out as in the base experiment.
# =============================================================================

CORRUPTION_SEED_OFFSET = 0
FIXED_NOISE_SEED_OFFSET = 7000

METHOD_SEED_OFFSETS = {
    "RK": 10000,
    "ARK": 20000,
    "QRK": 30000,
    "AQRK": 40000,
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
    [CORRUPTION_SEED_OFFSET, FIXED_NOISE_SEED_OFFSET]
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


tomo = loadmat("tomo_test.mat")

A_raw = tomo["A"]
x_true = tomo["x_ex"].reshape(-1, 1)

A_norm, _, zero_rows = row_normalize(A_raw, tomo["b_ex"])

all_rows = np.arange(A_norm.shape[0])
good_rows = np.setdiff1d(all_rows, zero_rows)
A = A_norm[good_rows, :]

b_clean = A @ x_true

m, n = A.shape
err = int(CFG.corruption_fraction * m)

q = 1 - CFG.alpha_quantile - err / m

RESULTS_DIR = Path("results/results_tomography_gaussian_noise_10r")
FIGURES_DIR = Path("figures/figures_tomography_gaussian_noise_10r")
DATA_DIR = Path("data/data_tomography_gaussian_noise_10r")

for d in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)


# =============================================================================
# METHOD WRAPPERS
# Each takes the generators from make_rngs as ``rngs`` and returns the error
# before each update, length CFG.n_iter. ARK and AQRK use batch size n.
# =============================================================================

def run_rk(A, b, x_true, noise, rngs):
    return qrk.RK(
        A, b, CFG.n_iter, x_true, noise, rngs["algorithm"],
        noise_distribution=TOMOGRAPHY_NOISE_DISTRIBUTION,
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
    )


def run_ark(A, b, x_true, noise, rngs):
    return qrk.ARK(
        A, b, noise, CFG.n_iter, x_true,
        CFG.ark_alpha, n, rngs["algorithm"],
        noise_distribution=TOMOGRAPHY_NOISE_DISTRIBUTION,
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
    )


def run_qrk(A, b, x_true, noise, rngs):
    return qrk.quantileRK(
        A, b, q, CFG.n_iter, x_true,
        noise, rngs["algorithm"],
        noise_distribution=TOMOGRAPHY_NOISE_DISTRIBUTION,
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
    )


def _run_filtered(A, b, x_true, noise, rngs, batch_size, requery):
    """Run filtered_kaczmarz from zero for CFG.n_iter iterations with
    adaptive stopping off."""
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
        noise_distribution=TOMOGRAPHY_NOISE_DISTRIBUTION,
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
        requery=requery,
        soln=x_true,
    )
    return history["errors"]


def run_aqrk(A, b, x_true, noise, rngs):
    return _run_filtered(A, b, x_true, noise, rngs, n, requery=True)


def run_qabk(A, b, x_true, noise, rngs):
    return _run_filtered(A, b, x_true, noise, rngs, "all", requery=False)


RUNNERS = {
    "RK": run_rk,
    "ARK": run_ark,
    "QRK": run_qrk,
    "AQRK": run_aqrk,
    "QABK": run_qabk,
}

methods = ["RK", "ARK", "QRK", "AQRK", "QABK"]

experiment_cases = {
    "uniform_only": {
        "extra_noise": 0.0,
        "alg_noise": CFG.noise,
        "noise_distribution": TOMOGRAPHY_NOISE_DISTRIBUTION,
    },
    "uniform_plus_fixed_noise": {
        "extra_noise": CFG.fixed_noise_sigma,
        "alg_noise": 0.0,
        "noise_distribution": TOMOGRAPHY_NOISE_DISTRIBUTION,
    },
}

results = {
    case: {method: [] for method in methods}
    for case in experiment_cases
}
raw_results = {
    case: {method: [] for method in methods}
    for case in experiment_cases
}
trial_data = []

for trial in range(CFG.nruns):
    trial_seed = CFG.seed + trial
    print(f"Trial {trial + 1}/{CFG.nruns}")

    corruption_seed = trial_seed + CORRUPTION_SEED_OFFSET
    rng_corr = np.random.default_rng(corruption_seed)
    b_corr = add_uniform_corr(
        b_clean,
        err,
        5.0,
        6.0,
        rng_corr,
    )

    method_seeds = {
        method: stream_seeds(trial_seed, method)
        for method in methods
    }

    trial_case_data = {
        "trial": trial,
        "trial_seed": trial_seed,
        "A": A,
        "b_clean": b_clean,
        "b_corr": b_corr,
        "x_true": x_true,
        "corruption_fraction": CFG.corruption_fraction,
        "corruption_seeds": {
            "uniform": corruption_seed,
            "fixed_noise": None,
        },
        # The same seeds are used in both cases, with fresh generators.
        "method_seeds": method_seeds,
        "cases": {},
    }

    for case_name, case_cfg in experiment_cases.items():
        b_exp = b_corr.copy()

        if case_cfg["extra_noise"] > 0:
            fixed_noise_seed = trial_seed + FIXED_NOISE_SEED_OFFSET
            rng_noise = np.random.default_rng(fixed_noise_seed)
            b_exp = add_fixed_noise(
                b_exp,
                case_cfg["extra_noise"],
                rng_noise,
            )
            trial_case_data["corruption_seeds"]["fixed_noise"] = fixed_noise_seed

        trial_case_data["cases"][case_name] = {
            "b_exp": b_exp,
            "alg_noise": case_cfg["alg_noise"],
            "extra_noise": case_cfg["extra_noise"],
            "noise_distribution": case_cfg["noise_distribution"],
        }

        for method in methods:
            curve = np.asarray(
                RUNNERS[method](
                    A,
                    b_exp,
                    x_true,
                    case_cfg["alg_noise"],
                    make_rngs(method_seeds[method]),
                )
            )

            raw_results[case_name][method].append(curve)
            results[case_name][method].append(
                normalize_curve(curve, x_true)
            )

    trial_data.append(trial_case_data)

summary_results = {}

for case in experiment_cases:
    summary_results[case] = {}
    for method in methods:
        summary_results[case][method] = summarize(
            results[case][method]
        )

np.savez_compressed(
    RESULTS_DIR / "tomography_gaussian_noise_results.npz",
    config=vars(CFG),
    q=q,
    seed_offsets={
        "corruption": CORRUPTION_SEED_OFFSET,
        "fixed_noise": FIXED_NOISE_SEED_OFFSET,
        "method": METHOD_SEED_OFFSETS,
        "stream": STREAM_SEED_OFFSETS,
    },
    trial_data=trial_data,
    raw_results=results,
    unnormalized_results=raw_results,
    summary_results=summary_results,
)

with open(
    DATA_DIR / "tomography_gaussian_noise_results.pkl",
    "wb",
) as f:
    pickle.dump(summary_results, f)

COLORS = {
    "RK": "#0072B2",
    "ARK": "#E69F00",
    "QRK": "#009E73",
    "AQRK": "#CC79A7",
    "QABK": "#56B4E9",
    "L2": "#000000",
}

LINESTYLES = {
    "RK": "-",
    "ARK": "--",
    "QRK": "-.",
    "AQRK": ":",
    "QABK": "-",
    "L2": "--",
}

CASE_LINESTYLES = {
    "uniform_only": "-",
    "uniform_plus_fixed_noise": "--",
}

FIGURE_NAMES = {
    "uniform_only": (
        "tomography_uniform_corruption_12pct_5to6_"
        "gaussian_query_noise_sigma0p06.pdf"
    ),
    "uniform_plus_fixed_noise": (
        "tomography_uniform_corruption_12pct_5to6_"
        "gaussian_fixed_noise_sigma0p06.pdf"
    ),
}

for case_name in experiment_cases:
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in methods:
        method_summary = summary_results[case_name][method]
        median_curve = method_summary["median"]
        lower_bound = method_summary["lower_bound"]
        upper_bound = method_summary["upper_bound"]

        ax.plot(
            median_curve,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.2,
            label=method,
        )

        ax.fill_between(
            np.arange(len(median_curve)),
            lower_bound,
            upper_bound,
            color=COLORS[method],
            alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Error")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / FIGURE_NAMES[case_name])
    plt.close()

print("Finished.")

# %%