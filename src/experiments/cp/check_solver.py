"""Focused checks for the CP quantile convention and plateau patience."""
import json
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load_solver(notebook_path):
    notebook = notebook_path if isinstance(notebook_path, dict) else json.loads(Path(notebook_path).read_text())
    namespace = {}
    # Import and definition cells only: never execute experiments or plotting.
    for index in (1, 5, 7, 9):
        exec(compile("".join(notebook["cells"][index]["source"]),
                     f"{notebook_path}:cell{index}", "exec"), namespace)
    return namespace


def main():
    current = load_solver(HERE / "cp_tensor_experiments_normalized.ipynb")
    with zipfile.ZipFile(HERE / "saved_runs/paper/original_sources.zip") as z:
        archived = load_solver(json.loads(z.read("filtered/cp_tensor_experiments_normalized.ipynb")))
    config = json.loads((HERE / "saved_runs/paper/filtered/settings.json").read_text())
    # A stationary inconsistent unit-row system makes the plateau unambiguous.
    config.update(q=1., inner_iterations=100)
    for patience, expected_updates in ((1, 40), (2, 60)):
        config["plateau_patience"] = patience
        _, info = current["filtered_kaczmarz"](
            np.ones((2, 1)), np.array([1., -1.]), np.zeros(1), "all", config,
            np.random.default_rng(0))
        assert info[0] == expected_updates and info[3] == 2, info
    config["adaptive_stopping"] = False
    _, info = current["filtered_kaczmarz"](
        np.ones((2, 1)), np.array([1., -1.]), np.zeros(1), "all", config,
        np.random.default_rng(0))
    assert info[0] == 100 and info[3] == 0
    config.update(adaptive_stopping=True, inner_iterations=120, alpha=0.)

    class ImprovingThenFlat:
        """Two flat windows, one genuine improvement, then a sustained plateau."""
        def __init__(self):
            self.calls = 0

        def normal(self, loc, scale, size):
            score = 1. if self.calls < 40 else .5
            self.calls += 1
            return loc + scale * np.array([-score, score])

    _, info = current["filtered_kaczmarz"](
        np.ones((2, 1)), np.zeros(2), np.zeros(1), "all", config,
        np.random.default_rng(0), query_std=np.ones(2),
        selection_rng=ImprovingThenFlat())
    assert info[0] == 100 and info[3] == 2, info  # Progress at 60 resets patience.
    config.update(alpha=1., inner_iterations=50)
    _, info = current["filtered_kaczmarz"](
        np.ones((2, 1)), np.array([1., -1.]), np.zeros(1), "all", config,
        np.random.default_rng(0))
    assert info[0] == 50 and info[3] == 1, info
    config["inner_iterations"] = 100
    config.update(adaptive_stopping=True, q=.85, plateau_patience=1)
    rng = np.random.default_rng(14)
    M = rng.normal(size=(900, 5))
    M /= np.linalg.norm(M, axis=1)[:, None]
    b = rng.normal(size=900)
    x0 = rng.normal(size=5)
    for batch in (1, 10, "all"):
        for noisy in (False, True):
            kwargs = dict(query_std=np.linspace(.01, .1, 900),
                          selection_rng=np.random.default_rng(8),
                          update_rng=np.random.default_rng(9), requery=batch != "all") if noisy else {}
            old_x, old_info = archived["filtered_kaczmarz"](
                M, b, x0, batch, config, np.random.default_rng(7), **kwargs)
            if noisy:
                kwargs.update(selection_rng=np.random.default_rng(8), update_rng=np.random.default_rng(9))
            new_x, new_info = current["filtered_kaczmarz"](
                M, b, x0, batch, config, np.random.default_rng(7), **kwargs)
            np.testing.assert_array_equal(new_x, old_x)
            np.testing.assert_array_equal(new_info, old_info)
    # Match the main-code threshold, including ties, on the paper's row count.
    for values in (rng.random(900), rng.integers(0, 12, 900).astype(float)):
        old_threshold = np.partition(values, 764)[764]
        new_threshold = np.quantile(values, .85)
        np.testing.assert_array_equal(values <= old_threshold, values <= new_threshold)
    # Row/RHS scaling invariance, including omitted zero rows.
    matrix = np.array([[3., 4.], [0., 0.], [0., -2.]])
    rhs = np.array([[10., 99., 6.], [5., 9., -8.]])
    unit, scaled, inverse = current["normalize_inner_systems"](matrix, rhs)
    np.testing.assert_allclose(unit, [[.6, .8], [0., -1.]])
    np.testing.assert_allclose(scaled, [[2., 3.], [1., -4.]])
    scales = np.array([2., 3., 4.])
    unit2, scaled2, inverse2 = current["normalize_inner_systems"](matrix * scales[:, None], rhs * scales)
    np.testing.assert_allclose(unit2, unit)
    np.testing.assert_allclose(scaled2, scaled)
    np.testing.assert_allclose(scales[[0, 2]] * inverse2, inverse)

    # Analytic one-step results distinguish interpolated quantiles, ties and q=1.
    config.update(adaptive_stopping=False, inner_iterations=1, alpha=1.)
    for q, expected in ((.85, 2.25), (.4, 5./3.), (1., 18./5.)):
        config["q"] = q
        x, _ = current["filtered_kaczmarz"](
            np.ones((5, 1)), np.array([1., 2., 2., 4., 9.]), np.zeros(1),
            "all", config, np.random.default_rng(0))
        np.testing.assert_allclose(x, [expected])

    # Five samples from two rows must repeat rows; each occurrence gets fresh noise.
    class FixedNoise:
        def __init__(self, offset):
            self.offset, self.counts = offset, []
        def normal(self, loc, scale, size):
            self.counts.append(size)
            return loc + scale * (np.arange(size, dtype=float) + self.offset)
    select, update = FixedNoise(1.), FixedNoise(3.)
    config["q"] = 1.
    x, _ = current["filtered_kaczmarz"](
        np.ones((2, 1)), np.zeros(2), np.zeros(1), 5, config,
        np.random.default_rng(0), query_std=np.ones(2), selection_rng=select,
        update_rng=update, requery=True)
    np.testing.assert_allclose(x, [5.])  # Mean of five independent occurrence values 3,...,7.
    assert update.counts == [5]
    print("PASS: stopping, normalization, quantiles/ties, q=1, repeated-row fresh queries, agreement with the archived notebook on seeded inner solves.")


if __name__ == "__main__":
    main()
