"""Regenerate all saved experiment figures with a consistent publication style.

This script serves as the canonical way to rebuild the project figures with a
consistent publication style and interquartile uncertainty bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    LogFormatterMathtext,
    ScalarFormatter,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "paper" / "figures_style_consistency_median"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# Color-blind-friendly palette with distinct hues and good contrast.
METHOD_COLORS = {
    "RK": "#0072B2",
    "ARK": "#E69F00",
    "QRK": "#009E73",
    "AQRK": "#CC79A7",
    "NR-AQRK": "#D55E00",
    "QABK": "#56B4E9",
    "L2": "#000000",
}

METHOD_LINESTYLES = {
    "RK": "-",
    "ARK": "-",
    "QRK": "-",
    "AQRK": "-",
    "NR-AQRK": "-",
    "QABK": "-",
    "L2": "--",
}

METHOD_MARKERS = {
    "RK": "o",
    "ARK": "s",
    "QRK": "^",
    "AQRK": "D",
    "NR-AQRK": "v",
    "QABK": "P",
    "L2": None,
}

METHOD_MARKER_OFFSETS = {
    "RK": 0,
    "ARK": 1,
    "QRK": 2,
    "AQRK": 3,
    "NR-AQRK": 4,
    "QABK": 5,
}

METHOD_LABELS = {
    "RK": "RK",
    "ARK": "ARK (s = 100)",
    "QRK": "QRK",
    "AQRK": "AQRK (s = 100, fresh)",
    "NR-AQRK": "AQRK (s = 100, reused)",
    "QABK": "AQRK-all",
    "L2": "Least Squares",
}

PLOT_STYLE = {
    "figure.figsize": (7, 5),
    "font.family": "DejaVu Sans",
    "font.size": 19.5,
    "axes.labelsize": 20.5,
    "axes.titlesize": 21.5,
    "legend.fontsize": 21.5,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.color": "#D6D6D6",
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
LEGEND_FONTSIZE = 17
BASE_LEGEND_FONTSIZE = 19
COMPOSITE_LEGEND_FONTSIZE = PLOT_STYLE["axes.titlesize"]


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def normalize_method_name(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return method


def format_iteration_ticks(ax: plt.Axes) -> None:
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_scientific(True)
    x_formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))


def format_scheduled_axes(ax: plt.Axes) -> None:
    format_iteration_ticks(ax)


def format_log_x_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))


def percentile_bounds(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pointwise 25th and 75th percentiles of trial curves."""
    curves = np.asarray(curves)
    if curves.ndim < 2:
        raise ValueError("At least two dimensions are required for trial curves")
    return tuple(np.percentile(curves, [25, 75], axis=0))


def least_squares_median(l2_value: float | dict | None) -> float | None:
    if l2_value is None:
        return None
    if not isinstance(l2_value, dict):
        return float(l2_value)
    if "per_trial" in l2_value:
        return float(np.median(np.asarray(l2_value["per_trial"])))
    if "median" in l2_value:
        return float(l2_value["median"])
    if "mean" in l2_value:
        return float(l2_value["mean"])
    raise KeyError(
        "Least-squares data must contain 'per_trial', 'median', or 'mean'"
    )


def plot_summary_family(
    ax: plt.Axes,
    summary: dict,
    methods: list[str],
    l2_value: float | dict | None = None,
    include_l2: bool = True,
    method_labels: dict[str, str] | None = None,
    legend_loc: str = "lower left",
    show_legend: bool = True,
    legend_ncol: int = 1,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_fontsize: float = LEGEND_FONTSIZE,
) -> None:
    for method in methods:
        if method not in summary:
            continue
        method_summary = summary[method]
        central_curve = np.asarray(method_summary["median"])
        lower_bound = np.asarray(method_summary["lower_bound"])
        upper_bound = np.asarray(method_summary["upper_bound"])
        label = (
            method_labels[method]
            if method_labels is not None and method in method_labels
            else normalize_method_name(method)
        )
        color = METHOD_COLORS.get(method, METHOD_COLORS["QABK"])
        marker = METHOD_MARKERS.get(method, METHOD_MARKERS["QABK"])
        marker_step = max(1, len(central_curve) // 8)
        marker_offset = (
            METHOD_MARKER_OFFSETS.get(method, 0) * marker_step // 6
        )

        ax.plot(
            central_curve,
            label=label,
            color=color,
            linestyle="-",
            linewidth=1.9,
            marker=marker,
            markevery=(marker_offset, marker_step) if marker else None,
            markersize=8,
        )
        ax.fill_between(
            np.arange(len(central_curve)),
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.15,
        )

    if include_l2:
        l2_value = least_squares_median(l2_value)
        if l2_value is not None:
            ax.axhline(
                float(l2_value),
                color=METHOD_COLORS["L2"],
                linestyle="--",
                linewidth=1.8,
                label="Least Squares",
            )

    ax.set_yscale("log")
    ax.set_ylim(5e-4, 1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Error")
    format_iteration_ticks(ax)
    if show_legend:
        ax.legend(
            frameon=False,
            loc=legend_loc,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            bbox_to_anchor=legend_bbox_to_anchor,
        )


def save_figure(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_scheduled_aqrk_figures(
    output_dir: Path,
) -> None:
    """Generate iteration- and budget-normalized scheduled-AQRK figures."""
    archive = (
        RESULTS_DIR
        / "results_scheduled_aqrk_horizon"
        / "experiment_results.npz"
    )
    if not archive.exists():
        return

    data = np.load(archive, allow_pickle=True)
    raw_results = data["raw_results"].item()
    config = data["config"].item()
    screening_rows = int(config["t"] if "t" in config else config["m"])
    s0 = int(config["s0"])
    batch_lambda = float(config.get("batch_lambda", 0.999))
    budget_step = 10_000
    colors = {
        "constant": "#0072B2",
        "decreasing_stepsize": "#D55E00",
        "exponential_batches": "#E69F00",
    }
    labels = {
        "constant": r"$\alpha, s$",
        "decreasing_stepsize": r"$\alpha_k, s$",
        "exponential_batches": (
            rf"$\alpha, s_k$"
        ),
    }
    scheduled_methods = [
        "constant",
        "decreasing_stepsize",
        "exponential_batches",
    ]
    methods = [name for name in scheduled_methods if name in raw_results]
    if not methods:
        return
    markers = ["o", "s", "^", "D", "v"]

    def batch_schedule(name: str, length: int) -> np.ndarray:
        if name in {"constant", "decreasing_stepsize"}:
            return np.full(length, s0, dtype=int)
        if name == "exponential_batches":
            return np.ceil(
                s0 * batch_lambda ** (-np.arange(length))
            ).astype(int)
        period = 500 if name.endswith("T500") else 1000
        return s0 * 2 ** (np.arange(length) // period)

    def curves_for_stat(curves: np.ndarray) -> tuple[np.ndarray, ...]:
        central = np.median(curves, axis=0)
        lower, upper = percentile_bounds(curves)
        return central, lower, upper

    iteration_fig, iteration_ax = plt.subplots(figsize=(7, 5))
    budget_fig, budget_ax = plt.subplots(figsize=(7, 5))

    for index, name in enumerate(methods):
        curves = np.asarray(raw_results[name])
        central, lower, upper = curves_for_stat(curves)
        marker_step = max(1, len(central) // 8)
        marker = markers[index]

        iteration_ax.plot(
            central,
            label=labels[name],
            color=colors[name],
            linestyle="-",
            linewidth=1.9,
            marker=marker,
            markevery=(index * marker_step // 4, marker_step),
            markersize=8,
        )
        iteration_ax.fill_between(
            np.arange(len(central)),
            lower,
            upper,
            color=colors[name],
            alpha=0.15,
        )

        batches = batch_schedule(name, len(central))
        cumulative_budget = np.concatenate(
            ([0], np.cumsum((screening_rows + batches)[:-1]))
        )
        targets = budget_step * np.arange(
            1,
            int(cumulative_budget[-1] // budget_step) + 1,
        )
        budget_curves = np.asarray([
            np.interp(targets, cumulative_budget, curve)
            for curve in curves
        ])
        budget_central, budget_lower, budget_upper = curves_for_stat(
            budget_curves
        )
        budget_marker_step = max(1, len(targets) // 8)
        budget_ax.plot(
            targets,
            budget_central,
            label=labels[name],
            color=colors[name],
            linestyle="-",
            linewidth=1.9,
            marker=marker,
            markevery=(index * budget_marker_step // 4, budget_marker_step),
            markersize=8,
        )
        budget_ax.fill_between(
            targets,
            budget_lower,
            budget_upper,
            color=colors[name],
            alpha=0.15,
        )

    for ax in (iteration_ax, budget_ax):
        ax.set_yscale("log")
        ax.set_ylabel("Relative Error")
        format_scheduled_axes(ax)
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    iteration_ax.set_xlabel("Iteration")
    budget_ax.set_xlabel("Budget")
    save_figure(
        iteration_fig,
        "scheduled_aqrk_horizon.pdf",
        output_dir,
    )
    save_figure(
        budget_fig,
        "scheduled_aqrk_error_vs_budget.pdf",
        output_dir,
    )


def load_summary_archive(
    archive: Path,
    required_methods: list[str],
) -> tuple[dict, dict]:
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    data = np.load(archive, allow_pickle=True)
    summary = data["summary_results"].item()
    raw_results = data["raw_results"].item()
    for case_name, case_summary in summary.items():
        for method_name, method_summary in case_summary.items():
            lower_bound, upper_bound = percentile_bounds(
                raw_results[case_name][method_name]
            )
            method_summary["lower_bound"] = lower_bound
            method_summary["upper_bound"] = upper_bound
    l2_errors = data["l2_errors"].item() if "l2_errors" in data else {}
    missing = [
        f"{case}:{method}"
        for case, case_summary in summary.items()
        for method in required_methods
        if method not in case_summary
    ]
    if missing:
        raise KeyError(
            f"Missing required method summaries in {archive}: "
            + ", ".join(missing)
        )
    return summary, l2_errors


# =============================================================================
# BASE EXPERIMENT
# Source: results_base_10r/experiment_results.npz
# =============================================================================

def generate_base_figures(output_dir: Path) -> None:
    archive = RESULTS_DIR / "results_base_10r" / "experiment_results.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    summary_results, l2_errors = load_summary_archive(
        archive,
        ["RK", "ARK", "QRK", "AQRK", "NR-AQRK", "QABK"],
    )

    composite_cases = [
        ("large_uniform", "(a) Oblivious"),
        ("sign_flip", "(b) Sign-flip"),
        ("adversarial", "(c) Singular-vector-based"),
    ]
    fig, axes = plt.subplots(
        1,
        len(composite_cases),
        figsize=(18, 6.5),
        sharey=True,
    )
    for ax, (exp_name, panel_title) in zip(axes, composite_cases):
        plot_summary_family(
            ax,
            dict(summary_results[exp_name]),
            ["RK", "ARK", "QRK", "AQRK", "NR-AQRK", "QABK"],
            l2_value=l2_errors[exp_name],
            legend_loc="upper right",
            show_legend=False,
        )
        ax.set_title(panel_title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=COMPOSITE_LEGEND_FONTSIZE,
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(
        output_dir / "base_large_sign_adversarial.pdf"
    )
    plt.close(fig)


# =============================================================================
# FIXED-NOISE EXPERIMENT
# Source: results_fnoise_10r/experiment_results.npz
# =============================================================================

def generate_fixed_noise_figures(output_dir: Path) -> None:
    archive = RESULTS_DIR / "results_fnoise_10r" / "experiment_results.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    summary_results, l2_errors = load_summary_archive(
        archive,
        ["RK", "ARK", "QRK", "AQRK", "QABK"],
    )
    base_archive = RESULTS_DIR / "results_base_10r" / "experiment_results.npz"
    base_summary_results, _ = load_summary_archive(
        base_archive,
        ["RK", "ARK", "QRK", "AQRK", "NR-AQRK", "QABK"],
    )

    for exp_name, title in {
        "small_uniform": "Small Uniform Corruption",
        "large_uniform": "Large Uniform Corruption",
        "sign_flip": "Sign Flip Corruption",
        "adversarial": "Singular Vector Adversary",
    }.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_summary = dict(summary_results[exp_name])
        plot_summary_family(
            ax,
            plot_summary,
            ["RK", "ARK", "QRK", "AQRK", "QABK"],
            l2_value=l2_errors[exp_name],
            method_labels={"AQRK": "AQRK (s = 100, sampled)"},
        )
        save_figure(fig, f"fixed_noise_{exp_name}.pdf", output_dir)

    table_methods = ["RK", "ARK", "QRK", "AQRK", "QABK"]
    table_labels = {
        "RK": "RK",
        "ARK": "ARK (s = 100)",
        "QRK": "QRK",
        "AQRK": "AQRK-sampled (fresh)",
        "QABK": "AQRK-all",
    }
    table_cases = [
        ("large_uniform", "Oblivious"),
        ("sign_flip", "Sign-flip"),
        ("adversarial", "S-V-based"),
    ]
    start_iteration = 6000
    end_iteration = 7000
    def table_value(
        summaries: dict,
        case_name: str,
        method: str,
    ) -> str:
        method_summary = summaries[case_name][method]
        if end_iteration > len(method_summary["median"]):
            raise ValueError(
                f"{case_name} {method} summary is shorter than iterations "
                f"{start_iteration}--{end_iteration - 1}"
            )
        value = float(
            np.mean(
                method_summary["median"][start_iteration:end_iteration]
            )
        )
        return f"{value:.3e}"

    table_data = []
    for method in table_methods:
        row = [table_labels[method]]
        for case_name, _ in table_cases:
            row.append(table_value(summary_results, case_name, method))
        for case_name, _ in table_cases:
            row.append(table_value(base_summary_results, case_name, method))
        table_data.append(row)

    table_fig, table_ax = plt.subplots(figsize=(26, 4.5))
    table_ax.axis("off")
    column_widths = [0.22] + [0.13] * 6
    table = table_ax.table(
        cellText=table_data,
        colLabels=[
            "",
            *[label for _, label in table_cases],
            *[label for _, label in table_cases],
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths,
        bbox=[0, 0, 1, 0.76],
    )
    for cell in table.get_celld().values():
        cell.get_text().set_wrap(True)
    fixed_center = column_widths[0] + sum(column_widths[1:4]) / 2
    query_center = (
        column_widths[0]
        + sum(column_widths[1:4])
        + sum(column_widths[4:7]) / 2
    )
    table_ax.text(
        fixed_center,
        0.88,
        "Fixed noise",
        ha="center",
        va="center",
        fontsize=30,
        transform=table_ax.transAxes,
    )
    table_ax.text(
        query_center,
        0.88,
        "Query noise",
        ha="center",
        va="center",
        fontsize=30,
        transform=table_ax.transAxes,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(30)
    table.scale(1, 1.35)
    save_figure(table_fig, "fixed_noise_error_table.pdf", output_dir)


# =============================================================================
# TOMOGRAPHY EXPERIMENT
# Source: results_tomography_gaussian_noise_10r/
#         tomography_gaussian_noise_results.npz
# =============================================================================

def generate_tomography_figures(output_dir: Path) -> None:
    archive = (
        RESULTS_DIR
        / "results_tomography_gaussian_noise_10r"
        / "tomography_gaussian_noise_results.npz"
    )
    summary_results, _ = load_summary_archive(
        archive,
        ["RK", "ARK", "QRK", "AQRK", "QABK"],
    )

    tomography_cases = [
        ("uniform_only", "(a) Uniform corruption"),
        (
            "uniform_plus_fixed_noise",
            "(b) Uniform corruption with fixed Gaussian noise",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        len(tomography_cases),
        figsize=(14, 5.5),
    )

    for ax, (case_name, panel_title) in zip(axes, tomography_cases):
        plot_summary = dict(summary_results[case_name])
        tomography_methods = ["QRK", "AQRK", "QABK"]
        plotted_bounds = []

        for method in tomography_methods:
            if method not in plot_summary:
                continue
            method_summary = plot_summary[method]
            central_curve = np.asarray(method_summary["median"])
            lower_bound = np.asarray(method_summary["lower_bound"])
            upper_bound = np.asarray(method_summary["upper_bound"])
            plotted_bounds.extend((lower_bound, upper_bound))
            label = (
                "AQRK (s = 100, sampled)"
                if method == "AQRK"
                else normalize_method_name(method)
            )
            color = METHOD_COLORS.get(method, METHOD_COLORS["QABK"])
            marker = METHOD_MARKERS.get(method, METHOD_MARKERS["QABK"])
            marker_step = max(1, len(central_curve) // 8)
            marker_offset = (
                METHOD_MARKER_OFFSETS.get(method, 0) * marker_step // 6
            )

            ax.plot(
                central_curve,
                label=label,
                color=color,
                linestyle="-",
                linewidth=1.9,
                marker=marker,
                markevery=(marker_offset, marker_step) if marker else None,
                markersize=8,
            )
            ax.fill_between(
                np.arange(len(central_curve)),
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.15,
            )

        ax.set_yscale("log")
        positive_bounds = np.concatenate(plotted_bounds)
        positive_bounds = positive_bounds[positive_bounds > 0]
        if positive_bounds.size == 0:
            raise ValueError(
                f"Tomography case {case_name} has no positive plotted values"
            )
        ax.set_ylim(
            positive_bounds.min() / 1.25,
            positive_bounds.max() * 1.25,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Error")
        format_iteration_ticks(ax)
        ax.set_title(panel_title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=COMPOSITE_LEGEND_FONTSIZE,
        ncol=len(handles),
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(
        output_dir / "tomography_gaussian_noise_composite.pdf"
    )
    plt.close(fig)


# =============================================================================
# SAMPLE-SIZE SWEEP
# Source: results_s_sweep_10_runs/s_sweep.npz
# =============================================================================

def generate_sweep_figures(output_dir: Path) -> None:
    s_archive = RESULTS_DIR / "results_s_sweep_10_runs" / "s_sweep.npz"
    if s_archive.exists():
        s_data = np.load(s_archive, allow_pickle=True)
        s_summary = s_data["summary_results"].item()
        s_raw_results = s_data["raw_results"].item()
        s_values = list(s_data["s_values"].astype(int))

        # A 6000-update run records the errors of iterates 0 through 5999.
        horizon_iteration = min(len(s_summary[s]["median"]) for s in s_values) - 1
        horizon_iteration = min(5999, horizon_iteration)
        errors = [float(s_summary[s]["median"][horizon_iteration]) for s in s_values]
        s_array = np.asarray(s_values, dtype=float)
        error_array = np.asarray(errors, dtype=float)
        if np.any(s_array <= 0) or np.any(error_array <= 0):
            raise ValueError(
                "Sample-size horizon plotting requires positive sample sizes "
                "and errors"
            )

        r_infty = 0.02

        def plot_convergence(ax: plt.Axes) -> None:
            for idx, s in enumerate(s_values):
                summary = s_summary[s]
                median = np.asarray(summary["median"])
                lower, upper = percentile_bounds(s_raw_results[s])
                color = METHOD_COLORS["QABK"] if idx == 0 else [
                    "#0072B2",
                    "#E69F00",
                    "#009E73",
                    "#D55E00",
                ][idx % 4]
                marker = ["o", "s", "^", "D"][idx % 4]
                marker_step = max(1, len(median) // 8)
                ax.plot(
                    median,
                    color=color,
                    linewidth=2.2,
                    linestyle="-",
                    marker=marker,
                    markevery=(idx * marker_step // 4, marker_step),
                    markersize=7,
                    label=rf"$s={s}$",
                )
                ax.fill_between(
                    np.arange(len(median)),
                    lower,
                    upper,
                    color=color,
                    alpha=0.15,
                )
            ax.set_yscale("log")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Relative Error")
            format_iteration_ticks(ax)
            ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)

        def plot_horizon(ax: plt.Axes) -> None:
            ax.scatter(
                s_array,
                error_array,
                marker="o",
                s=8**2,
                color=METHOD_COLORS["QABK"],
                label="Horizon",
            )
            reference_s_values = np.geomspace(
                s_array.min(),
                s_array.max(),
                200,
            )
            ax.plot(
                reference_s_values,
                r_infty * reference_s_values ** (-0.5),
                color=METHOD_COLORS["L2"],
                linewidth=1.8,
                linestyle="--",
                label=r"$f(s)=r_\infty s^{-1/2}$",
            )
            ax.set_yscale("log")
            ax.set_xscale("log")
            format_log_x_axis(ax)
            ax.set_xlabel(r"Sample size $s$")
            ax.set_ylabel(
                rf"Relative Error at $k={horizon_iteration}$"
            )
            ax.legend(
                frameon=False,
                fontsize=LEGEND_FONTSIZE,
                loc="upper right",
            )

        convergence_fig, convergence_ax = plt.subplots(figsize=(7, 5))
        plot_convergence(convergence_ax)
        save_figure(
            convergence_fig,
            "sample_size_sweep.pdf",
            output_dir,
        )

        horizon_fig, horizon_ax = plt.subplots(figsize=(7, 5))
        plot_horizon(horizon_ax)
        save_figure(
            horizon_fig,
            f"error_vs_s_horizon{horizon_iteration}.pdf",
            output_dir,
        )

        composite_fig, composite_axes = plt.subplots(
            1,
            2,
            figsize=(14, 5),
        )
        plot_convergence(composite_axes[0])
        plot_horizon(composite_axes[1])
        composite_axes[0].set_title("(a) Convergence curves")
        composite_axes[1].set_title(
            rf"(b) Error at iteration $k={horizon_iteration}$"
        )
        save_figure(
            composite_fig,
            "sample_size_sweep_and_horizon.pdf",
            output_dir,
        )


# =============================================================================
# CORRUPTION-MAGNITUDE SWEEP
# Source: results_separation_sweep/corruption_magnitude_sweep.npz
# =============================================================================

    separation = RESULTS_DIR / "results_separation_sweep" / "corruption_magnitude_sweep.npz"
    if separation.exists():
        sep_data = np.load(separation, allow_pickle=True)
        summary = sep_data["summary_results"].item()
        raw_results = sep_data["raw_results"].item()
        uniform_levels = sep_data["uniform_levels"].tolist()
        adversarial_levels = sep_data["adversarial_levels"].tolist()
        f_scale = float(sep_data["f_scale"])

        palette = [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#D55E00",
            "#CC79A7",
            "#56B4E9",
        ]
        markers = ["o", "s", "^", "D", "v", "P"]

        uniform_fig, uniform_ax = plt.subplots(figsize=(7, 5))
        for idx, level in enumerate(uniform_levels):
            level_summary = summary["uniform"][level]
            median = np.asarray(level_summary["median"])
            lower, upper = percentile_bounds(raw_results["uniform"][level])
            uniform_ax.plot(
                median,
                color=palette[idx % len(palette)],
                linestyle="-",
                linewidth=1.8,
                marker=markers[idx % len(markers)],
                markevery=(
                    idx * max(1, len(median) // 8) // 6,
                    max(1, len(median) // 8),
                ),
                markersize=8,
                label=(
                    rf"$[a,b] = [{level * f_scale:g},"
                    rf"{(level + 1) * f_scale:g}]$"
                ),
            )
            uniform_ax.fill_between(
                np.arange(len(median)),
                lower,
                upper,
                color=palette[idx % len(palette)],
                alpha=0.15,
            )
        uniform_ax.set_yscale("log")
        uniform_ax.set_ylim(5e-4, 1.8)
        uniform_ax.set_xlabel("Iteration")
        uniform_ax.set_ylabel("Relative Error")
        format_iteration_ticks(uniform_ax)
        uniform_ax.legend(
            ncol=1,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
        )
        save_figure(uniform_fig, "uniform_magnitude_sweep.pdf", output_dir)

        adversarial_fig, adversarial_ax = plt.subplots(figsize=(7, 5))
        for idx, level in enumerate(adversarial_levels):
            level_summary = summary["adversarial"][level]
            median = np.asarray(level_summary["median"])
            lower, upper = percentile_bounds(
                raw_results["adversarial"][level]
            )
            adversarial_ax.plot(
                median,
                color=palette[idx % len(palette)],
                linestyle="-",
                linewidth=1.8,
                marker=markers[idx % len(markers)],
                markevery=(
                    idx * max(1, len(median) // 8) // 6,
                    max(1, len(median) // 8),
                ),
                markersize=8,
                label=rf"$f = {level * f_scale:g}$",
            )
            adversarial_ax.fill_between(
                np.arange(len(median)),
                lower,
                upper,
                color=palette[idx % len(palette)],
                alpha=0.15,
            )
        adversarial_ax.set_yscale("log")
        adversarial_ax.set_ylim(5e-4, 1.8)
        adversarial_ax.set_xlabel("Iteration")
        adversarial_ax.set_ylabel("Relative Error")
        format_iteration_ticks(adversarial_ax)
        adversarial_ax.legend(
            ncol=1,
            fontsize=17,
            frameon=False,
        )
        save_figure(adversarial_fig, "adversarial_magnitude_sweep.pdf", output_dir)

        composite_fig, composite_axes = plt.subplots(
            1,
            2,
            figsize=(14, 5.5),
            sharey=True,
        )
        for ax, levels, case_name in (
            (composite_axes[0], uniform_levels, "uniform"),
            (composite_axes[1], adversarial_levels, "adversarial"),
        ):
            for idx, level in enumerate(levels):
                level_summary = summary[case_name][level]
                median = np.asarray(level_summary["median"])
                lower, upper = percentile_bounds(
                    raw_results[case_name][level]
                )
                if case_name == "uniform":
                    median = median[:6000]
                    lower = lower[:6000]
                    upper = upper[:6000]
                if case_name == "uniform":
                    label = (
                        rf"$[a,b] = [{level * f_scale:g},"
                        rf"{(level + 1) * f_scale:g}]$"
                    )
                else:
                    label = rf"$f = {level * f_scale:g}$"
                ax.plot(
                    median,
                    color=palette[idx % len(palette)],
                    linewidth=1.8,
                    marker=markers[idx % len(markers)],
                    markevery=(
                        idx * max(1, len(median) // 8) // 6,
                        max(1, len(median) // 8),
                    ),
                    markersize=8,
                    label=label,
                )
                ax.fill_between(
                    np.arange(len(median)),
                    lower,
                    upper,
                    color=palette[idx % len(palette)],
                    alpha=0.15,
                )
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 1.8)
            ax.set_xlabel("Iteration")
            format_iteration_ticks(ax)
            ax.legend(
                ncol=1,
                fontsize=LEGEND_FONTSIZE,
                frameon=False,
            )
        composite_axes[0].set_ylabel("Relative Error")
        composite_axes[0].set_title("(a) Oblivious corruption")
        composite_axes[1].set_title("(b) Singular-vector corruption")
        save_figure(
            composite_fig,
            "magnitude_sweeps_composite.pdf",
            output_dir,
        )


apply_plot_style()
generate_base_figures(OUTPUT_DIR)
generate_fixed_noise_figures(OUTPUT_DIR)
generate_tomography_figures(OUTPUT_DIR)
generate_sweep_figures(OUTPUT_DIR)
generate_scheduled_aqrk_figures(OUTPUT_DIR)
print(f"Saved median plots to {OUTPUT_DIR}")
