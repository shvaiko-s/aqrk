import copy
import numpy as np

# =============================================================================
# SYSTEM GENERATION
# =============================================================================

def normalized_gaussian(rows, cols, seed=42):

    rng = np.random.default_rng(seed)

    def normal_gauss_vect(n):
        g = rng.normal(0, 1, n)
        return g / np.linalg.norm(g)

    A = np.zeros((rows, cols))

    for i in range(rows):
        A[i] = normal_gauss_vect(cols)

    x = rng.normal(0, 1, (cols, 1))
    b = A @ x

    return A, b, x


# =============================================================================
# NOISE MODELS
# =============================================================================

def add_normal_noise(mean, std, b, rng):

    return b + rng.normal(
        mean,
        std,
        size=b.shape,
    )


def add_uniform_noise(l, r, b, rng):

    return b + rng.uniform(
        l,
        r,
        size=b.shape,
    )


# =============================================================================
# CORRUPTION MODELS
# =============================================================================

def add_uniform_corr(errors, l, r, b, rng):

    ans = copy.deepcopy(b)

    bad_rows = rng.choice(
        len(b),
        errors,
        replace=False,
    )

    ans[bad_rows] += rng.uniform(
        l,
        r,
        size=(errors, 1),
    )

    return ans


def add_signflip_corr(errors, b, rng):

    ans = copy.deepcopy(b)

    bad_rows = rng.choice(
        len(b),
        errors,
        replace=False,
    )

    ans[bad_rows] *= -1

    return ans


def add_separated_corr(errors, b, rng):

    ans = copy.deepcopy(b)

    bad_rows = rng.choice(
        len(b),
        errors,
        replace=False,
    )

    k = len(bad_rows) // 2

    ans[bad_rows[:k]] += rng.uniform(
        1,
        5,
        size=(k, 1),
    )

    ans[bad_rows[k:]] += rng.uniform(
        100,
        500,
        size=(len(bad_rows) - k, 1),
    )

    return ans

def add_adversarial_corr(
    err,
    magnitude,
    b,
    direction,
):

    b_corr = b.copy().reshape(-1)
    direction = direction.reshape(-1)

    idx = np.argsort(
        np.abs(direction)
    )[-err:]

    b_corr[idx] += (
        magnitude
        * np.sign(direction[idx])
    )

    return b_corr.reshape(-1, 1)