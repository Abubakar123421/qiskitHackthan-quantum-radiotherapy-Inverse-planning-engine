"""Classical (normal-computer) baselines to compare QAOA against."""
import itertools
import time
import numpy as np


def greedy(qubo):
    """Add one beam at a time, always the one that lowers the cost most."""
    chosen = []
    for _ in range(qubo.k):
        best_b, best_f = None, np.inf
        for b in range(qubo.n):
            if b in chosen:
                continue
            x = np.zeros(qubo.n)
            x[chosen + [b]] = 1
            f = qubo.cost(x)          # cost of the partial plan (each beam = 1/k of the dose)
            if f < best_f:
                best_b, best_f = b, f
        chosen.append(best_b)
    x = np.zeros(qubo.n, int)
    x[chosen] = 1
    return x, qubo.cost(x)


def simulated_annealing(qubo, seed=0, steps=400, t0=1.0, t1=1e-3):
    """Swap one ON beam with one OFF beam; accept worse swaps with falling probability."""
    rng = np.random.default_rng(seed)
    x = np.zeros(qubo.n, int)
    x[rng.choice(qubo.n, qubo.k, replace=False)] = 1
    f = qubo.cost(x)
    best_x, best_f = x.copy(), f
    for s in range(steps):
        T = t0 * (t1 / t0) ** (s / steps)
        on, off = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
        y = x.copy()
        y[rng.choice(on)] = 0
        y[rng.choice(off)] = 1
        g = qubo.cost(y)
        if g < f or rng.random() < np.exp(-(g - f) / T):
            x, f = y, g
            if f < best_f:
                best_x, best_f = x.copy(), f
    return best_x, best_f


def evenly_spaced(qubo):
    """Naive plan: spread the k beams evenly over the list of candidate directions."""
    x = np.zeros(qubo.n, int)
    x[[int(round(i * qubo.n / qubo.k)) % qubo.n for i in range(qubo.k)]] = 1
    return x, qubo.cost(x)


def random_plans(qubo, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    costs = []
    for _ in range(n_samples):
        x = np.zeros(qubo.n, int)
        x[rng.choice(qubo.n, qubo.k, replace=False)] = 1
        costs.append(qubo.cost(x))
    return np.array(costs)


def clinical_plan(anat, weights=None):
    """The hospital's own beam angles, re-computed in OUR simple dose model with equal weights.

    (The real plan used different beam weights, wedges and energies, so only compare trends.)
    """
    from .dose import beam_dose
    from .qubo import dose_metrics, objective_from_dose
    D = np.mean([beam_dose(anat, a) for a in anat.clinical_angles], axis=0)
    return D, dose_metrics(anat, D), objective_from_dose(anat, D, weights)


def timed(fn, *a, **kw):
    t = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t
