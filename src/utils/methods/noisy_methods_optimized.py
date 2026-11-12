import numpy as np


def sample_noise(rng, scale, distribution="uniform", size=None):
    """Sample residual noise with ``scale`` as half-width or standard deviation."""
    if distribution == "uniform":
        return rng.uniform(-scale, scale, size=size)
    if distribution == "gaussian":
        return rng.normal(0.0, scale, size=size)
    raise ValueError(
        "distribution must be either 'uniform' or 'gaussian'"
    )

def _per_iteration(value, length, name):
    """Return ``value`` as a float vector with one entry per iteration.

    A scalar is repeated for every iteration; a vector must already have
    exactly ``length`` entries.
    """
    schedule = np.asarray(value, dtype=float)
    if schedule.ndim == 0:
        return np.full(length, float(schedule))
    if schedule.ndim != 1 or schedule.size != length:
        raise ValueError(
            f"{name} must be a scalar or a one-dimensional vector of length "
            f"inner_iterations ({length})"
        )
    return schedule


def filtered_kaczmarz(A, b, x0, batch_size, settings, rng, *,
                      query_width=None, noise_distribution="gaussian",
                      selection_rng=None, update_rng=None, requery=False,
                      soln=None, return_history=False):
    """QRK/AQRK; A already has unit rows and b is scaled identically.

    ``batch_size`` is ``"all"``, a positive integer, or a vector of positive
    integers with one entry per inner iteration. ``settings["alpha"]`` is a
    scalar or a vector with one entry per inner iteration. Entry ``k`` of a
    vector is used at iteration ``k``; if adaptive stopping ends the run
    early, the remaining entries are unused.

    ``query_width`` is ``None`` (no noise), a scalar, or one width per row. It
    is the half-width for ``noise_distribution="uniform"`` and the standard
    deviation for ``"gaussian"``, as in :func:`sample_noise`.

    The random streams stay separate: ``rng`` draws the batches,
    ``selection_rng`` draws the selection and final diagnostic noise, and
    ``update_rng`` draws the requeried update noise.

    History: if ``soln`` is given or ``return_history`` is true, the function
    returns ``(x, stats, history)`` instead of ``(x, stats)``. ``history`` is a
    dict with one entry per scan of the loop, each measured on the iterate
    *before* that iteration's update (the AQRK convention):

        - ``history["errors"][k]`` = ``||soln - x_k||`` (``None`` without ``soln``);
        - ``history["scores"][k]`` = the trimmed residual RMS of ``x_k`` that
          adaptive stopping uses (computed from the noisy residuals).

    Without adaptive stopping both arrays have length ``inner_iterations``.
    If adaptive stopping ends the run, they have one more entry than there
    were updates, and the last entry belongs to the returned ``x``.
    """
    x = np.array(x0, dtype=float, copy=True)
    m = A.shape[0]
    iterations = settings["inner_iterations"]

    use_all = isinstance(batch_size, str)
    if use_all:
        if batch_size != "all":
            raise ValueError('batch_size must be "all", a positive integer, '
                             "or a vector of positive integers")
        batch_schedule = None
    else:
        batch_schedule = _per_iteration(batch_size, iterations, "batch_size")
        if (not np.all(np.isfinite(batch_schedule))
                or np.any(batch_schedule <= 0)
                or np.any(batch_schedule != np.floor(batch_schedule))):
            raise ValueError("batch_size must contain positive integers")
        batch_schedule = batch_schedule.astype(int)
    alpha_schedule = _per_iteration(settings["alpha"], iterations, 'settings["alpha"]')
    if not np.all(np.isfinite(alpha_schedule)):
        raise ValueError('settings["alpha"] must contain only finite values')
    if noise_distribution not in ("uniform", "gaussian"):
        raise ValueError(
            "noise_distribution must be either 'uniform' or 'gaussian'")

    adaptive = settings["adaptive_stopping"]
    updates = scans = corrections = 0
    plateau_checks = 0
    selection_queries = update_queries = diagnostic_queries = 0
    stop_code = 1 if adaptive else 0
    x_true = None if soln is None else np.asarray(soln, dtype=float).reshape(-1)
    want_history = return_history or x_true is not None
    errors = None if x_true is None else []
    score_trace = []
    if m == 0:
        stats = [0, 0, 0, 4, 0.0, 0.0, 0, 0, 0]
        if want_history:
            return x, stats, {
                "errors": None if errors is None else np.empty(0),
                "scores": np.empty(0)}
        return x, stats
    noisy = query_width is not None and np.any(np.asarray(query_width) > 0)
    if noisy:
        assert selection_rng is not None
        assert not requery or update_rng is not None
        # A scalar width applies to every row.
        query_width = np.broadcast_to(np.asarray(query_width, dtype=float), (m,))
    order = int(np.ceil(settings["q"] * m)) - 1
    # NumPy's default linear quantile lies between these two order statistics.
    quantile_position = (m - 1) * settings["q"]
    lower = int(np.floor(quantile_position))
    upper = int(np.ceil(quantile_position))
    fraction = quantile_position - lower
    partition_at = sorted({order, lower, upper})
    scores = []
    window = settings["window"]

    for step in range(iterations):
        if errors is not None:
            errors.append(np.linalg.norm(x_true - x))
        prediction = A @ x
        observed_rhs = b
        if noisy:
            observed_rhs = b + sample_noise(
                selection_rng, query_width, noise_distribution, size=m)
            selection_queries += m
        residuals = observed_rhs - prediction
        if not np.all(np.isfinite(residuals)):
            raise FloatingPointError("Nonfinite inner residuals.")
        # One partition supplies both the stopping score and the filter threshold.
        magnitudes = np.partition(np.abs(residuals), partition_at)
        score = float(np.linalg.norm(magnitudes[:order + 1]) / np.sqrt(order + 1))
        scans += 1
        score_trace.append(score)
        if step == 0:
            initial_score = score
            floor = settings["residual_tolerance"] * max(
                float(np.linalg.norm(observed_rhs) / np.sqrt(m)),
                np.finfo(float).tiny)
        if adaptive:
            if step >= settings["min_inner_iterations"] and step % window == 0:
                if score <= floor:
                    stop_code = 3
                    break
                before, after = np.mean(scores[:window]), np.mean(scores[window:])
                improvement = (before - after) / max(before, floor, np.finfo(float).tiny)
                if improvement < settings["plateau_tolerance"]:
                    plateau_checks += 1
                else:
                    plateau_checks = 0
                if plateau_checks >= settings.get("plateau_patience", 1):
                    stop_code = 2
                    break
            scores.append(score)
            if len(scores) > 2 * window:
                del scores[0]
        lo, hi = magnitudes[lower], magnitudes[upper]
        # Match NumPy's linear interpolation, including its upper-half formula.
        if fraction < 0.5:
            threshold = lo + (hi - lo) * fraction
        else:
            threshold = hi - (hi - lo) * (1.0 - fraction)
        eligible = np.flatnonzero(np.abs(residuals) <= threshold)
        if use_all:
            sampled = eligible
        else:
            sampled = rng.choice(eligible, size=batch_schedule[step], replace=True)
        update_residuals = residuals[sampled]
        if noisy and requery:
            # Each occurrence gets fresh noise, even if a row repeats.
            update_residuals = (b[sampled] - prediction[sampled]
                                + sample_noise(update_rng, query_width[sampled],
                                               noise_distribution,
                                               size=sampled.size))
            update_queries += sampled.size
        x += alpha_schedule[step] * np.mean(
            update_residuals[:, None] * A[sampled], axis=0)
        updates += 1
        corrections += sampled.size

    final_residuals = b - A @ x
    if noisy:
        final_residuals += sample_noise(
            selection_rng, query_width, noise_distribution, size=m)
        diagnostic_queries += m
    smallest = np.partition(np.abs(final_residuals), order)[:order + 1]
    final_score = float(np.linalg.norm(smallest) / np.sqrt(order + 1))
    if not np.all(np.isfinite(x)) or not np.isfinite(final_score):
        raise FloatingPointError("Nonfinite inner solution.")
    stats = [updates, scans + 1, corrections, stop_code, initial_score,
             final_score, selection_queries, update_queries, diagnostic_queries]
    if want_history:
        return x, stats, {
            "errors": None if errors is None else np.array(errors),
            "scores": np.array(score_trace)}
    return x, stats