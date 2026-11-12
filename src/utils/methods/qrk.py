import numpy as np

from .noisy_methods_optimized import filtered_kaczmarz, sample_noise

# Random streams follow filtered_kaczmarz:
#     rng            row sampling (the chosen row or batch);
#     selection_rng  noise on the residuals used to select rows;
#     update_rng     noise on freshly queried residuals used in the update.
# A method draws only from the streams it needs. If selection_rng or
# update_rng is None, that noise is drawn from rng instead (single stream).
# Every method uses all m rows; there is no candidate subsampling.


def sample_noise(rng, scale, distribution="uniform", size=None):
    """Sample residual noise with ``scale`` as half-width or standard deviation."""
    if distribution == "uniform":
        return rng.uniform(-scale, scale, size=size)
    if distribution == "gaussian":
        return rng.normal(0.0, scale, size=size)
    raise ValueError(
        "distribution must be either 'uniform' or 'gaussian'"
    )



def _noise_streams(rng, selection_rng, update_rng):
    """Resolve the noise streams, falling back to ``rng`` when one is None."""
    return (
        rng if selection_rng is None else selection_rng,
        rng if update_rng is None else update_rng,
    )


def quantileRK(
    A, b, q, N, soln, noise_scale, rng, noise_distribution="uniform",
    *, selection_rng=None, update_rng=None,
):
    """Quantile RK with cached noisy selection residuals.

    Selection rule:
        - compute one noisy residual for every row once;
        - keep the subset whose residual magnitudes are at or below the ``q``-quantile;
        - choose one row uniformly from that selected subset.

    Update rule:
        - batch size = 1 (one selected row per update);
        - averaging normalization = none (single row projection);
        - step size = 1;
        - the update uses the cached residual from the selected row, so the residual is
          reused and not freshly queried again during the same iteration.

    Implemented by :func:`filtered_kaczmarz` with batch size 1, step size 1
    and no requerying. Returns the error before each update, length ``N``.

    Random streams: ``rng`` draws the selected row, ``selection_rng`` the
    selection-residual noise. ``update_rng`` is accepted
    for a uniform interface but unused, since residuals are never requeried.
    """
    selection_rng, _ = _noise_streams(rng, selection_rng, update_rng)
    settings = dict(
        q=q,
        alpha=1.0,
        inner_iterations=N,
        adaptive_stopping=False,
        window=1,                # required key, unused without adaptive stopping
        residual_tolerance=0.0,  # required key, unused without adaptive stopping
    )
    _, _, history = filtered_kaczmarz(
        A,
        np.asarray(b, dtype=float).reshape(-1),
        np.zeros(A.shape[1]),
        1,
        settings,
        rng,
        query_width=noise_scale,
        noise_distribution=noise_distribution,
        selection_rng=selection_rng,
        requery=False,
        soln=soln,
    )
    return history["errors"]


def RK(
    A, b, N, soln, noise_scale, rng, noise_distribution="uniform",
    *, selection_rng=None, update_rng=None,
):
    """Randomized Kaczmarz with a fresh residual query per update.

    Selection rule:
        - choose one row uniformly at random from all rows.

    Update rule:
        - batch size = 1;
        - averaging normalization = none;
        - step size = 1;
        - the residual is freshly queried for that update and not reused.

    Random streams: ``rng`` draws the row, ``update_rng`` the residual noise.
    ``selection_rng`` is accepted for a uniform interface but unused, since
    rows are chosen without looking at residuals.
    """
    _, update_rng = _noise_streams(rng, selection_rng, update_rng)
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    x_true = np.asarray(soln).reshape(-1)
    errors = np.empty(N, dtype=float)
    x = np.zeros(n, dtype=float)

    for k in range(N):
        errors[k] = np.linalg.norm(x_true - x)
        i = rng.choice(m)
        residual = A[i] @ x - b[i]
        residual += sample_noise(update_rng, noise_scale, noise_distribution)
        x -= residual * A[i]
             
    return errors


def ARK(
    A,
    b,
    noise_scale,
    N,
    soln,
    alpha,
    s,
    rng,
    noise_distribution="uniform",
    *,
    selection_rng=None,
    update_rng=None,
):
    """Averaged randomized Kaczmarz with fresh residuals for each occurrence.

    Selection rule:
        - draw ``s`` row indices uniformly with replacement from the full row set;
        - repeated rows are allowed and are treated as distinct update occurrences.

    Update rule:
        - batch size = ``s``;
        - averaging normalization = divide the sum of projected updates by ``s``;
        - step size = ``alpha``;
        - every sampled occurrence gets its own newly queried residual, so duplicates do not
          reuse any cached residual value.

    Random streams: ``rng`` draws the batch, ``update_rng`` the residual noise.
    ``selection_rng`` is accepted for a uniform interface but unused, since
    rows are chosen without looking at residuals.
    """
    _, update_rng = _noise_streams(rng, selection_rng, update_rng)
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    x_true = np.asarray(soln).reshape(-1)
    errors = np.empty(N, dtype=float)
    x = np.zeros(n, dtype=float)

    for k in range(N):
        errors[k] = np.linalg.norm(x_true - x)
        indices = rng.choice(m, size=s, replace=True)
        base_residuals = A[indices] @ x - b[indices]
        residuals = base_residuals + sample_noise(
            update_rng,
            noise_scale,
            noise_distribution,
            size=s,
        )
        unique_indices, inverse = np.unique(
            indices,
            return_inverse=True,
        )
        aggregated_residuals = np.bincount(
            inverse,
            weights=residuals,
        )
        x -= alpha * (
            A[unique_indices].T @ aggregated_residuals
        ) / s
             
    return errors