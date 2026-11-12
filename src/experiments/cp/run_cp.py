"""Run the final CP paper methods and q=1 controls, with per-trial checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

# Each worker runs independent small linear algebra; avoid nested BLAS threads.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "cp_tensor_experiments_normalized.ipynb"
NS = None
DESTINATION = None


def load_context(notebook, overrides=None):
    cells = json.loads(Path(notebook).read_text())["cells"]
    context = {}
    for index in (1, 3, 5, 7, 9, 11):
        exec(compile("".join(cells[index]["source"]), f"{notebook}:cell{index}", "exec"), context)
        if index == 3 and overrides:
            context["settings"].update(overrides)
    return context


def initialize_worker(notebook, destination, overrides):
    global NS, DESTINATION
    NS = load_context(notebook, overrides)
    DESTINATION = Path(destination)


def methods_for(case, settings):
    if settings["q"] == 1.0:
        suffix = "_fresh" if case == "query_noise" else ""
        return [("quantile_s10" + suffix, 10, case == "query_noise"),
                ("quantile_all", "all", False)]
    methods = [("als_observed", "als", False)]
    if case == "query_noise":
        methods += [(f"quantile_s{s}_fresh", s, True) for s in settings["batch_sizes"]]
    else:
        methods += [(f"quantile_s{s}", s, False) for s in settings["batch_sizes"]]
    methods += [("quantile_all", "all", False)]
    if case == "approximate":
        methods += [("als_clean", "als", False)]
    return methods


def trial_name(task):
    case, method, _, _, trial = task
    return f"{case}__{method}__trial{trial}"


def run_trial(task):
    case, method, batch, requery, trial = task
    name = trial_name(task)
    ns = NS
    values = ns["data"][case]
    observed = values["target"] if method == "als_clean" else values["observed"]
    if case == "query_noise" and batch == "als":
        observed = values["als_observed"]
    noisy = case == "query_noise" and batch != "als"
    query_seed = int(ns["query_seeds"][trial]) if noisy else None
    seed = int(ns["sampling_seeds"][trial])
    initial_seed = int(ns["initialization_seeds"][trial])
    result = ns["fit_cp"](
        observed, values["target"], ns["initializations"][trial], batch,
        ns["settings"]["outer_sweeps"][case], ns["settings"], seed,
        query_sigma=ns["query_sigma"] if noisy else 0., query_seed=query_seed,
        requery=requery)
    temp_path = DESTINATION / (name + ".tmp")
    with temp_path.open("wb") as stream:
        np.savez_compressed(stream, **result, seed=seed, initialization_seed=initial_seed,
                            query_seed=-1 if query_seed is None else query_seed,
                            query_sigma=ns["query_sigma"] if noisy else 0.,
                            requery_updates=requery, inner_columns=np.array(ns["inner_columns"]))
    temp_path.replace(DESTINATION / (name + ".npz"))
    return name, float(result["target_error"][-1]), float(result["elapsed_seconds"][-1])


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def collect_results(folder, settings):
    histories, summary = {}, {}
    for case in ("exact", "query_noise", "approximate"):
        for method, _, _ in methods_for(case, settings):
            records = []
            for trial in range(settings["trials"]):
                name = f"{case}__{method}__trial{trial}"
                path = folder / (name + ".npz")
                if not path.is_file():
                    continue
                with np.load(path) as saved:
                    r = {key: saved[key] for key in saved.files}
                records.append(r)
                d = r["inner_diagnostics"]
                record = dict(final_error=float(r["target_error"][-1]),
                              seconds=float(r["elapsed_seconds"][-1]),
                              initialization_seed=int(r["initialization_seed"]),
                              sampling_seed=int(r["seed"]), query_seed=int(r["query_seed"]),
                              requery_updates=bool(r["requery_updates"]))
                if len(d):
                    record.update(inner_solves=len(d), median_inner_updates=float(np.median(d[:, 3])),
                                  cap_hits=int(np.count_nonzero(d[:, 6] == 1)),
                                  residual_scans=int(d[:, 4].sum()), row_corrections=int(d[:, 5].sum()),
                                  selection_queries=int(d[:, 9].sum()), update_queries=int(d[:, 10].sum()),
                                  diagnostic_queries=int(d[:, 11].sum()))
                summary[name] = record
            if len(records) == settings["trials"]:
                for metric in ("target_error", "observed_error", "elapsed_seconds"):
                    histories[f"{case}__{method}__{metric}"] = np.stack([r[metric] for r in records])
    np.savez_compressed(folder / "histories.npz", **histories)
    atomic_json(folder / "summary.json", summary)
    return summary


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_inputs(folder, ns):
    inputs = {key: ns[key] for key in ("dense_noise", "initialization_seeds", "sampling_seeds", "query_seeds")}
    inputs["query_sigma"] = np.array(ns["query_sigma"])
    for mode in range(3):
        inputs[f"initial_factor_{mode}"] = np.stack([f[mode] for f in ns["initializations"]])
        inputs[f"true_factor_{mode}"] = ns["true_factors"][mode]
    for case, values in ns["data"].items():
        inputs.update({f"{case}_{key}": value for key, value in values.items()})
    with (folder / "inputs.tmp").open("wb") as stream:
        np.savez_compressed(stream, **inputs)
    (folder / "inputs.tmp").replace(folder / "inputs.npz")


def run_group(folder, workers, trials, group, smoke=False):
    folder = Path(folder).resolve()
    overrides = {"trials": trials, "query_update_variants": ["fresh"]}
    if group == "unfiltered":
        overrides["q"] = 1.0
    if smoke:
        overrides["outer_sweeps"] = {"exact": 2, "query_noise": 2, "approximate": 2}
    source_hash = sha256(NOTEBOOK)
    runner_hash = sha256(__file__)
    provenance_path = folder / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("archive_kind") == "compact":
            raise ValueError("This is a read-only compact paper archive. Choose a new --output directory.")
        if (provenance.get("source_sha256") != source_hash
                or provenance.get("runner_sha256") != runner_hash
                or provenance.get("settings_overrides") != overrides
                or provenance.get("group") != group):
            raise ValueError("Resume requires identical notebook, runner, settings and method group.")
        if sha256(folder / NOTEBOOK.name) != source_hash or sha256(folder / "run_cp.py") != runner_hash:
            raise ValueError("A saved source snapshot has changed.")
        if sha256(folder / "inputs.npz") != provenance["inputs_sha256"]:
            raise ValueError("Saved inputs have changed.")
        if sha256(folder / "settings.json") != provenance["settings_sha256"]:
            raise ValueError("Saved settings have changed.")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Output directory is not empty and has no compatible provenance.")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOTEBOOK, folder / NOTEBOOK.name)
        shutil.copy2(Path(__file__), folder / "run_cp.py")
        provenance = dict(status="prepared", source_notebook=NOTEBOOK.name,
                          source_sha256=source_hash, runner_sha256=runner_hash,
                          settings_overrides=overrides, group=group,
                          numpy_version=np.__version__, workers=workers,
                          started_utc=datetime.now(timezone.utc).isoformat(),
                          execution="independent trials; paired initializations and fixed random seeds")
    ns = load_context(folder / NOTEBOOK.name, overrides)
    settings = ns["settings"]
    if group == "filtered" and settings["q"] != .85:
        raise ValueError("The filtered paper group requires q=0.85.")
    if not (folder / "settings.json").exists():
        atomic_json(folder / "settings.json", settings)
        save_inputs(folder, ns)
        provenance["settings_sha256"] = sha256(folder / "settings.json")
        provenance["inputs_sha256"] = sha256(folder / "inputs.npz")
    tasks = [(case, method, batch, requery, trial)
             for trial in range(settings["trials"])
             for case in ("approximate", "exact", "query_noise")
             for method, batch, requery in methods_for(case, settings)]
    pending = [task for task in tasks if not (folder / (trial_name(task) + ".npz")).is_file()]
    completed = len(tasks) - len(pending)
    provenance.update(status="running", expected_trials=len(tasks), completed_trials=completed)
    atomic_json(provenance_path, provenance)
    print(f"RUN {folder}: {len(pending)} pending method/trial fits, {workers} workers", flush=True)
    start = time.perf_counter()
    try:
        if pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                     initargs=(folder / NOTEBOOK.name, folder, overrides)) as executor:
                futures = [executor.submit(run_trial, task) for task in pending]
                for future in as_completed(futures):
                    name, error, seconds = future.result()
                    completed += 1
                    provenance.update(completed_trials=completed,
                                      latest_completion_utc=datetime.now(timezone.utc).isoformat())
                    atomic_json(provenance_path, provenance)
                    print(f"{completed}/{len(tasks)} {name}: error={error:.7g}, seconds={seconds:.1f}", flush=True)
        summary = collect_results(folder, settings)
        if len(summary) != len(tasks):
            raise ValueError("Missing results after execution.")
        provenance.update(status="complete", completed_trials=len(summary),
                          finished_utc=datetime.now(timezone.utc).isoformat(),
                          wall_seconds_this_invocation=time.perf_counter()-start)
    except BaseException as error:
        provenance.update(status="interrupted", error=repr(error))
        raise
    finally:
        atomic_json(provenance_path, provenance)
    print("COMPLETE", folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--group", choices=("paper", "filtered", "unfiltered"), default="paper",
                        help="paper runs both groups in separate subdirectories")
    parser.add_argument("--smoke", action="store_true",
                        help="Run two sweeps per case; defaults to one trial. Not paper results.")
    args = parser.parse_args()
    trials = args.trials if args.trials is not None else (1 if args.smoke else 10)
    if args.workers < 1 or trials < 1:
        parser.error("workers and trials must be positive")
    groups = ("filtered", "unfiltered") if args.group == "paper" else (args.group,)
    for group in groups:
        run_group(args.output / group, args.workers, trials, group, args.smoke)


if __name__ == "__main__":
    main()
