"""Replot the archived 30-cubed CP experiment without running any solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
SPECS = [
    ("quantile_s1", r"AQRK ($s=1$)", "#E41A1C", (0, (5, 2))),
    ("quantile_s5", r"AQRK ($s=5$)", "#FF8C00", (0, (1, 1.5))),
    ("quantile_s10", r"AQRK ($s=10$)", "#0066FF", "-"),
    ("quantile_s50", r"AQRK ($s=50$)", "#8A2BE2", (0, (5, 2, 1, 2))),
    ("quantile_all", "AQRK-all", "#00A65A", (0, (10, 2.5))),
    ("als_observed", "ALS (corrupted data)", "#777777", "-"),
]
CASES = [("exact", "Exact fitting"),
         ("query_noise", "Exact fitting + query noise"),
         ("approximate", "Approximate fitting")]


def plot_cp(run_folder, output_stem=None, unfiltered_run=None, omit_s50=False, unfiltered_s10_run=None):
    run_folder = Path(run_folder).resolve()
    config = json.loads((run_folder / "settings.json").read_text())
    if config.get("filtering_residuals") != "row_normalized":
        raise ValueError("Expected row-normalized results.")
    if config["shape"] != [30, 30, 30] or config["rank"] != 5:
        raise ValueError("This figure is for the 30 x 30 x 30, rank-5 experiment.")
    with np.load(run_folder / "histories.npz") as saved:
        curves = {key: saved[key] for key in saved.files}
    specs = [item for item in SPECS if not (omit_s50 and item[0] == "quantile_s50")]
    def load_control(unfiltered_run):
        if unfiltered_run is None:
            return None, None
        unfiltered_run = Path(unfiltered_run).resolve()
        control_config = json.loads((unfiltered_run / "settings.json").read_text())
        differences = {k for k in config.keys() | control_config.keys()
                       if config.get(k) != control_config.get(k)}
        if differences != {"q"} or config["q"] != .85 or control_config["q"] != 1:
            raise ValueError("Expected matched settings differing only in q.")
        with np.load(run_folder / "inputs.npz") as original, np.load(unfiltered_run / "inputs.npz") as control:
            if set(original.files) != set(control.files) or not all(np.array_equal(original[k], control[k]) for k in original.files):
                raise ValueError("Control inputs do not match the baseline.")
        with np.load(unfiltered_run / "histories.npz") as saved:
            control_curves = {key: saved[key] for key in saved.files}
        control_manifest = {"run": str(unfiltered_run), "settings": control_config,
            "sha256": {name: hashlib.sha256((unfiltered_run / name).read_bytes()).hexdigest()
                       for name in ("settings.json", "histories.npz", "inputs.npz")}}
        return control_curves, control_manifest
    control_curves, control_manifest = load_control(unfiltered_run)
    s10_curves, s10_manifest = load_control(unfiltered_s10_run)
    with np.load(run_folder / "inputs.npz") as saved:
        approximate = saved["approximate_target"]
        generating = saved["exact_target"]
        if (approximate.shape != tuple(config["shape"])
                or generating.shape != approximate.shape
                or not np.all(np.isfinite(approximate))
                or not np.all(np.isfinite(generating))
                or np.linalg.norm(approximate) == 0):
            raise ValueError("Invalid saved targets for the generating-tensor reference")
        generating_error = float(np.linalg.norm(approximate - generating)
                                 / np.linalg.norm(approximate))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.6))
    fig.subplots_adjust(left=.085, right=.99, bottom=.34, top=.80, wspace=.29)
    final_numbers = {}
    for col, (case, title) in enumerate(CASES):
        ax = axes[col]
        limit = min(60 if case == "exact" else 30, config["outer_sweeps"][case])
        for method, label, color, dash in specs:
            suffix = "_fresh" if case == "query_noise" and method.startswith("quantile_s") else ""
            key = f"{case}__{method}{suffix}__target_error"
            values = curves[key]
            expected = (config["trials"], config["outer_sweeps"][case] + 1)
            if values.shape != expected or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Invalid history: {key}")
            low, center, high = np.percentile(values, [25, 50, 75], axis=0)
            x = np.arange(limit + 1)
            ax.fill_between(x, np.maximum(low[:limit+1], 1e-18),
                            np.maximum(high[:limit+1], 1e-18), color=color, alpha=.13, lw=0)
            ax.plot(x, np.maximum(center[:limit+1], 1e-18), color=color,
                    linestyle=dash, linewidth=1.8, label=label)
            final_numbers[f"{case}__{method}"] = {
                "displayed_sweep": limit, "median_at_displayed_sweep": float(center[limit]),
                "final_sweep": values.shape[1]-1, "final_median": float(center[-1])}
        if control_curves is not None:
            values = control_curves[f"{case}__quantile_all__target_error"]
            expected = (config["trials"], config["outer_sweeps"][case] + 1)
            if values.shape != expected or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Invalid unfiltered control history.")
            low, center, high = np.percentile(values, [25, 50, 75], axis=0)
            x = np.arange(limit + 1)
            ax.fill_between(x, np.maximum(low[:limit+1], 1e-18), np.maximum(high[:limit+1], 1e-18),
                            color="black", alpha=.08, lw=0)
            ax.plot(x, np.maximum(center[:limit+1], 1e-18), color="black", linestyle=(0,(4,2)),
                    linewidth=1.8, label="Unfiltered (all rows)")
            final_numbers[f"{case}__unfiltered_all"] = {
                "displayed_sweep": limit, "median_at_displayed_sweep": float(center[limit]),
                "final_sweep": values.shape[1]-1, "final_median": float(center[-1])}
        if s10_curves is not None:
            suffix = "_fresh" if case == "query_noise" else ""
            values = s10_curves[f"{case}__quantile_s10{suffix}__target_error"]
            expected = (config["trials"], config["outer_sweeps"][case] + 1)
            if values.shape != expected or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Invalid unfiltered s10 history.")
            low, center, high = np.percentile(values, [25, 50, 75], axis=0)
            x = np.arange(limit + 1)
            ax.fill_between(x, np.maximum(low[:limit+1],1e-18), np.maximum(high[:limit+1],1e-18),
                            color="#8C564B", alpha=.09, lw=0)
            ax.plot(x, np.maximum(center[:limit+1],1e-18), color="#8C564B", linestyle=(0,(5,2,1,2)),
                    linewidth=1.8, label=r"Unfiltered ($s=10$)")
            final_numbers[f"{case}__unfiltered_s10"] = {
                "displayed_sweep": limit, "median_at_displayed_sweep": float(center[limit]),
                "final_sweep": values.shape[1]-1, "final_median": float(center[-1])}
        if case == "approximate":
            ax.axhline(generating_error, color="black", linestyle=":",
                       linewidth=1.3, zorder=6)
            final_numbers["approximate__generating_tensor"] = {
                "definition": "norm(approximate_target - exact_target) / norm(approximate_target)",
                "relative_error": generating_error,
                "interpretation": "attainable rank-5 reference error, not a lower bound"}
        ax.set(yscale="log", xlim=(0, limit), xlabel="Outer CP sweep")
        ax.set_title(f"({chr(97+col)}) {title}", fontsize=12.5, pad=12)
        ax.set_xticks(np.arange(0, limit + 1, 15 if case == "exact" else 5))
        if case == "exact":
            ax.set_ylim(1e-17, 10)
            ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
            ax.set_yticklabels([r"$10^{-16}$", r"$10^{-12}$", r"$10^{-8}$",
                                r"$10^{-4}$", r"$10^0$"])
        else:
            ticks = ([.003, .01, .1, 1]
                     if case == "query_noise" else [.02, .05, .2, 1])
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{value:g}" for value in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color="#DDDDDD", linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Relative reconstruction error")
    method_handles = [Line2D([], [], color=color, linestyle=dash, linewidth=1.8, label=label)
                      for method, label, color, dash in specs if method != "als_observed"]
    reference_handles = [Line2D([], [], color="#777777", linewidth=1.8, label="ALS (corrupted data)")]
    if s10_curves is not None:
        reference_handles.append(Line2D([], [], color="#8C564B", linestyle=(0,(5,2,1,2)), linewidth=1.8,
                                        label=r"Unfiltered ($s=10$)"))
    if control_curves is not None:
        reference_handles.append(Line2D([], [], color="black", linestyle=(0,(4,2)), linewidth=1.8,
                                        label="Unfiltered (all rows)"))
    fig.legend(handles=method_handles, loc="lower center", bbox_to_anchor=(.5,.13),
               ncol=len(method_handles), frameon=False, fontsize=13.5, handlelength=3.4,
               columnspacing=1.05, handletextpad=.5)
    fig.legend(handles=reference_handles, loc="lower center", bbox_to_anchor=(.5,.025),
               ncol=len(reference_handles), frameon=False, fontsize=13.5, handlelength=3.4,
               columnspacing=1.2, handletextpad=.5)
    if output_stem is None:
        output_stem = HERE / "figures" / "cp30_three_cases"
    output_stem = Path(output_stem).resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix("." + extension), dpi=180,
                    bbox_inches="tight", pad_inches=.08)
    manifest = {"layout": "1 row x 3 cases; no insets", "shape": config["shape"],
                "rank": config["rank"], "settings": config,
                "run": os.path.relpath(run_folder, output_stem.parent),
                "cases": [case for case, _ in CASES], "sampled_query_variant": "fresh",
                "display_sweeps": {"exact": 60, "query_noise": 30, "approximate": 30},
                "y_ticks": {case: list(ax.get_yticks()) for (case, _), ax in zip(CASES, axes)},
                "unfiltered_control": control_manifest,
                "unfiltered_s10_control": s10_manifest,
                "omit_s50": omit_s50,
                "legend_font_points": 13.5,
                "legend_handle_length_font_units": 3.4,
                "method_linewidth_points": 1.8,
                "reference_linewidth_points": 1.3,
                "overall_title": None,
                "query_policy": "Sampled AQRK uses fresh observations in panel (b); AQRK-all reuses screening observations. With fixed observations, s=1 coincides with QRK.",
                "curve_statistic": "median with 25th-75th percentile shading",
                "reference": "horizontal generating-tensor relative error; not an optimal-error bound",
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numbers": final_numbers,
                "sha256": {name: hashlib.sha256((run_folder / name).read_bytes()).hexdigest()
                           for name in ("settings.json", "histories.npz", "inputs.npz")}}
    for key in ("unfiltered_control", "unfiltered_s10_control"):
        if manifest[key] is not None:
            manifest[key]["run"] = os.path.relpath(manifest[key]["run"], output_stem.parent)
    output_stem.with_suffix(".sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "saved_runs/paper",
                        help="Paper run directory containing filtered/ and unfiltered/.")
    parser.add_argument("--output", type=Path, default=HERE / "figures/cp30_2pct_full10")
    args = parser.parse_args()
    figure = plot_cp(args.run / "filtered", args.output,
                     unfiltered_run=args.run / "unfiltered",
                     unfiltered_s10_run=args.run / "unfiltered")
    plt.close(figure)
    print(f"Saved {args.output.with_suffix('.pdf')}")
