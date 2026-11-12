"""Package a completed CP run into the compact reproducibility archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
GROUPS = ("filtered", "unfiltered")
COMPACT_FILES = ("histories.npz", "inputs.npz", "settings.json", "summary.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def validate_run(run_root):
    provenance = {}
    for group in GROUPS:
        folder = run_root / group
        data = load_json(folder / "provenance.json")
        if data.get("status") != "complete":
            raise ValueError(f"{folder} is not complete.")
        for name in COMPACT_FILES:
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        provenance[group] = data
    if provenance["filtered"]["inputs_sha256"] != provenance["unfiltered"]["inputs_sha256"]:
        raise ValueError("Filtered and unfiltered runs do not share identical inputs.")
    return provenance


def write_sources(archive, run_root, provenance):
    source_zip = archive / "original_sources.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as sources:
        for group in GROUPS:
            source_folder = run_root / group
            sources.write(
                source_folder / "cp_tensor_experiments_normalized.ipynb",
                f"{group}/cp_tensor_experiments_normalized.ipynb",
            )
            sources.write(source_folder / "run_cp.py", f"{group}/run_cp.py")
            provenance[group]["original_sources"] = {
                group: {
                    "source_sha256": sha256(
                        source_folder / "cp_tensor_experiments_normalized.ipynb"
                    ),
                    "runner_sha256": sha256(source_folder / "run_cp.py"),
                }
            }


def package(run_root, destination):
    run_root = Path(run_root).resolve()
    destination = Path(destination).resolve()
    provenance = validate_run(run_root)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"{destination.name}.", dir=destination.parent
    ) as temporary:
        archive = Path(temporary)
        for group in GROUPS:
            target = archive / group
            target.mkdir()
            source = run_root / group
            for name in COMPACT_FILES:
                shutil.copy2(source / name, target / name)

        write_sources(archive, run_root, provenance)
        for group in GROUPS:
            group_provenance = dict(provenance[group])
            group_provenance.update(
                archive_kind="compact",
                archive_source=str(run_root),
                archived_files=list(COMPACT_FILES),
            )
            (archive / group / "provenance.json").write_text(
                json.dumps(group_provenance, indent=2) + "\n"
            )
        files = {}
        for path in sorted(archive.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(archive))] = sha256(path)
        (archive / "manifest.json").write_text(
            json.dumps({"sha256": files}, indent=2) + "\n"
        )
        archive.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "generated_runs/paper")
    parser.add_argument("--output", type=Path, default=HERE / "saved_runs/paper")
    args = parser.parse_args()
    package(args.run, args.output)
    print(f"Packaged {args.run} -> {args.output}")


if __name__ == "__main__":
    main()
