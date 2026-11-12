"""Compare AQRK schedules under a shared residual-query budget.

The experiment compares constant and scheduled step sizes or batch sizes
using the same total ``m+s`` budget: each iteration queries all ``m`` rows
once for the selection scan and ``s`` fresh residuals for the update. The
corruption model is uniform: 30% of the measurement rows receive independent
additive corruptions uniformly distributed in ``[1, 1.2]``. Results are shown
both against iteration count and cumulative budget used, with medians and the
same trimmed bands as the base experiment.

AQRK runs on filtered_kaczmarz with fresh residuals and three independently
seeded streams: algorithm (the batch), selection noise and update noise.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import matplotlib.ticker as ticker
import numpy as np

from src.utils.datagen import rand_system
from src.utils.methods import noisy_methods_optimized


@dataclass(frozen=True)
class ExperimentConfig:
    m: int = 2000
    n: int = 100
    beta: float = 0.3
    corruption_low: float = 1.0
    corruption_high: float = 1.2
    q: float = 0.6
    noise: float = 0.02
    nruns: int = 10
    budget: int = 21_000_000
    s0: int = 100
    batch_lambda: float = 0.999
    k_star: int = 2500
    seed: int = 42


CFG = ExperimentConfig()
RESULTS_DIR = Path("results/results_scheduled_aqrk_horizon")
DATA_DIR = Path("data/data_scheduled_aqrk_horizon")
FIGURES_DIR = Path("figures/figures_scheduled_aqrk_horizon")

# =============================================================================
# SEEDS
# Each stream's seed is trial_seed + schedule offset + stream offset; the
# corruption uses trial_seed + its offset and the data uses trial_seed.
# Schedule offsets are 10000, 20000, ... in the order of make_schedules().
# =============================================================================

CORRUPTION_SEED_OFFSET = 1000
SCHEDULE_SEED_STEP = 10000

STREAM_SEED_OFFSETS = {
    "algorithm": 0,
    "selection_noise": 5000,
    "update_noise": 7000,
}


def schedule_seed_offsets(names):
    return {
        name: SCHEDULE_SEED_STEP * (index + 1)
        for index, name in enumerate(names)
    }


def check_seed_offsets(schedule_offsets):
    """Offsets must be distinct and at least nruns apart, or two streams in
    different trials would receive the same seed."""
    offsets = sorted(
        [0, CORRUPTION_SEED_OFFSET]
        + [schedule + stream
           for schedule in schedule_offsets.values()
           for stream in STREAM_SEED_OFFSETS.values()]
    )
    if len(set(offsets)) != len(offsets) or np.min(np.diff(offsets)) < CFG.nruns:
        raise ValueError("seed offsets are too close together for this many runs")


def stream_seeds(trial_seed, schedule_offset):
    base = trial_seed + schedule_offset
    return {stream: base + offset for stream, offset in STREAM_SEED_OFFSETS.items()}


def make_rngs(seeds):
    return {stream: np.random.default_rng(seed) for stream, seed in seeds.items()}


def format_scheduled_axes(ax):
    x_formatter = ticker.ScalarFormatter(useMathText=True)
    x_formatter.set_scientific(True)
    x_formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))


def summarize(curves):
    """Return medians and the base experiment's trimmed bands."""
    curves = np.asarray(curves)
    if curves.shape[0] != CFG.nruns:
        raise ValueError(f"expected {CFG.nruns} runs, got {curves.shape[0]}")

    sorted_curves = np.sort(curves, axis=0)
    return {
        "median": np.median(curves, axis=0),
        "lower_bound": sorted_curves[2],
        "upper_bound": sorted_curves[-3],
    }


def normalize_curve(curve, x_true):
    return np.asarray(curve) / np.linalg.norm(x_true)


def make_schedules():
    constant_iterations = CFG.budget // (CFG.m + CFG.s0)
    constant_cost = constant_iterations * (CFG.m + CFG.s0)
    constant_alpha = np.ones(constant_iterations)
    constant_s = np.full(constant_iterations, CFG.s0, dtype=int)
    decreasing_alpha = np.ones(constant_iterations)
    iterations = np.arange(constant_iterations)
    decreasing_alpha[CFG.k_star:] = CFG.k_star / iterations[CFG.k_star:]

    def geometric_schedule(period):
        batches = []
        cost = 0
        iteration = 0
        while cost + CFG.m + CFG.s0 * 2 ** (iteration // period) <= CFG.budget:
            batch = CFG.s0 * 2 ** (iteration // period)
            batches.append(batch)
            cost += CFG.m + batch
            iteration += 1
        return np.ones(iteration), np.asarray(batches, dtype=int), cost

    if not 0 < CFG.batch_lambda < 1:
        raise ValueError("batch_lambda must be between zero and one")

    batches = []
    cost = 0
    iteration = 0
    while True:
        batch = int(np.ceil(
            CFG.s0 * CFG.batch_lambda ** (-iteration)
        ))
        if cost + CFG.m + batch > CFG.budget:
            break
        batches.append(batch)
        cost += CFG.m + batch
        iteration += 1
    exponential_schedule = (
        np.ones(iteration),
        np.asarray(batches, dtype=int),
        cost,
    )

    return {
        "constant": (constant_alpha, constant_s, constant_cost),
        "decreasing_stepsize": (decreasing_alpha, constant_s, constant_cost),
        "geometric_batches_T500": geometric_schedule(500),
        "geometric_batches_T1000": geometric_schedule(1000),
        "exponential_batches": exponential_schedule,
    }


def run_schedule(A, b, x_true, alpha, batches, cost, rngs):
    """AQRK with fresh residuals under step-size and batch-size schedules.

    Returns the error before each update, one entry per schedule entry, and
    checks that the residual queries made equal the schedule's budget.
    """
    settings = dict(
        q=CFG.q,
        alpha=alpha,
        inner_iterations=len(alpha),
        adaptive_stopping=False,
        window=1,                # required key, unused without adaptive stopping
        residual_tolerance=0.0,  # required key, unused without adaptive stopping
    )
    _, stats, history = noisy_methods_optimized.filtered_kaczmarz(
        A,
        np.asarray(b, dtype=float).reshape(-1),
        np.zeros(A.shape[1]),
        batches,
        settings,
        rngs["algorithm"],
        query_width=CFG.noise,
        noise_distribution="uniform",
        selection_rng=rngs["selection_noise"],
        update_rng=rngs["update_noise"],
        requery=True,
        soln=x_true,
    )
    if CFG.noise > 0:
        # stats[6] and stats[7] count the selection and update queries; the
        # final diagnostic scan is not part of the budget.
        used = stats[6] + stats[7]
        if used != cost:
            raise RuntimeError(f"queried {used:,} residuals, budget says {cost:,}")
    return history["errors"]


def run_experiment():
    if not 0 < CFG.beta < 1:
        raise ValueError("beta must be between zero and one")

    for directory in (RESULTS_DIR, DATA_DIR, FIGURES_DIR):
        directory.mkdir(exist_ok=True)

    schedules = make_schedules()
    schedule_offsets = schedule_seed_offsets(schedules)
    check_seed_offsets(schedule_offsets)
    errors = {name: [] for name in schedules}
    trial_data = []
    n_corrupted = round(CFG.beta * CFG.m)

    for trial in range(CFG.nruns):
        trial_seed = CFG.seed + trial
        print(f"Trial {trial + 1}/{CFG.nruns}")

        A, b, x_true = rand_system.normalized_gaussian(
            CFG.m,
            CFG.n,
            seed=trial_seed,
        )
        corruption_seed = trial_seed + CORRUPTION_SEED_OFFSET
        corruption_rng = np.random.default_rng(corruption_seed)
        b_corrupted = rand_system.add_uniform_corr(
            n_corrupted,
            CFG.corruption_low,
            CFG.corruption_high,
            b,
            corruption_rng,
        )
        method_seeds = {
            name: stream_seeds(trial_seed, offset)
            for name, offset in schedule_offsets.items()
        }

        trial_data.append({
            "trial": trial,
            "trial_seed": trial_seed,
            "A": A,
            "b": b,
            "x_true": x_true,
            "b_corrupted": b_corrupted,
            "corruption_seed": corruption_seed,
            "method_seeds": method_seeds,
        })

        for name, (alpha, s, cost) in schedules.items():
            if trial == 0:
                print(
                    f"  {name}: {len(alpha)} iterations, "
                    f"budget used {cost:,}/{CFG.budget:,}"
                )
            curve = run_schedule(
                A, b_corrupted, x_true, alpha, s, cost,
                make_rngs(method_seeds[name]),
            )
            errors[name].append(normalize_curve(curve, x_true))

    summary_results = {
        name: summarize(curves)
        for name, curves in errors.items()
    }
    raw_results = {
        name: np.asarray(curves)
        for name, curves in errors.items()
    }
    schedule_metadata = {
        name: {
            "iterations": len(alpha),
            "budget_used": cost,
        }
        for name, (alpha, s, cost) in schedules.items()
    }

    np.savez_compressed(
        RESULTS_DIR / "experiment_results.npz",
        config=vars(CFG),
        schedule_metadata=schedule_metadata,
        seed_offsets={
            "corruption": CORRUPTION_SEED_OFFSET,
            "schedule": schedule_offsets,
            "stream": STREAM_SEED_OFFSETS,
        },
        raw_results=raw_results,
        summary_results=summary_results,
        trial_data=trial_data,
    )
    with open(DATA_DIR / "summary_results.pkl", "wb") as file:
        pickle.dump(summary_results, file)

    plot_results(summary_results)
    plot_budget_results(errors, schedules)
    return summary_results


def plot_results(summary_results):
    import matplotlib.pyplot as plt

    colors = {
        "constant": "#0072B2",
        "decreasing_stepsize": "#D55E00",
        "geometric_batches_T500": "#009E73",
        "geometric_batches_T1000": "#CC79A7",
        "exponential_batches": "#E69F00",
    }
    labels = {
        "constant": r"(i) $\alpha=1,\ s=100$",
        "decreasing_stepsize": r"(ii) decreasing $\alpha_k,\ s=100$",
        "geometric_batches_T500": r"$s_k=100\cdot 2^{\lfloor k/500\rfloor}$",
        "geometric_batches_T1000": r"$s_k=100\cdot 2^{\lfloor k/1000\rfloor}$",
        "exponential_batches": (
            rf"$s_k=\left\lceil100\cdot"
            rf"\left(1/{CFG.batch_lambda:g}\right)^k\right\rceil$"
        ),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, summary in summary_results.items():
        x = np.arange(len(summary["median"]))
        ax.plot(x, summary["median"], label=labels[name], color=colors[name])
        ax.fill_between(
            x,
            summary["lower_bound"],
            summary["upper_bound"],
            color=colors[name],
            alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Error")
    format_scheduled_axes(ax)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scheduled_aqrk_horizon.pdf", dpi=300)
    plt.close(fig)


def plot_budget_results(errors, schedules):
    """Plot median error at uniformly spaced cumulative ``m+s`` budgets."""
    import matplotlib.pyplot as plt

    colors = {
        "constant": "#0072B2",
        "decreasing_stepsize": "#D55E00",
        "geometric_batches_T500": "#009E73",
        "geometric_batches_T1000": "#CC79A7",
        "exponential_batches": "#E69F00",
    }
    labels = {
        "constant": r"(i) $\alpha=1,\ s=100$",
        "decreasing_stepsize": r"(ii) decreasing $\alpha_k,\ s=100$",
        "geometric_batches_T500": r"$s_k=100\cdot 2^{\lfloor k/500\rfloor}$",
        "geometric_batches_T1000": r"$s_k=100\cdot 2^{\lfloor k/1000\rfloor}$",
        "exponential_batches": (
            rf"$s_k=\left\lceil100\cdot"
            rf"\left(1/{CFG.batch_lambda:g}\right)^k\right\rceil$"
        ),
    }
    budget_step = 10_000

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, curves in errors.items():
        alpha, batches, _ = schedules[name]
        per_iteration_cost = CFG.m + batches
        cumulative_cost = np.concatenate(
            ([0], np.cumsum(per_iteration_cost[:-1]))
        )
        targets = budget_step * np.arange(
            1,
            int(cumulative_cost[-1] // budget_step) + 1,
        )
        if len(targets) == 0:
            continue

        curves = np.asarray(curves)
        sampled_curves = np.asarray([
            np.interp(targets, cumulative_cost, curve)
            for curve in curves
        ])
        sorted_curves = np.sort(sampled_curves, axis=0)
        ax.plot(
            targets,
            np.median(sampled_curves, axis=0),
            label=labels[name],
            color=colors[name],
        )
        ax.fill_between(
            targets,
            sorted_curves[2],
            sorted_curves[-3],
            color=colors[name],
            alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Cumulative $m+s$ budget used")
    ax.set_ylabel("Relative Error")
    format_scheduled_axes(ax)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "scheduled_aqrk_error_vs_budget.pdf",
        dpi=300,
    )
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()