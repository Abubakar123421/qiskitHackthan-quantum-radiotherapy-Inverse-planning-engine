"""Beam-angle selection as a QUBO (Quadratic Unconstrained Binary Optimisation).

x_b = 1  -> use beam b,   x_b = 0 -> don't.
With exactly k beams switched on, each delivers 1/k of the prescription, so the
dose in pixel v is   D_v(x) = sum_b x_b * a_bv,   a_bv = dose_b(v) / k.

Cost (lower is better) — the classic "least-squares" planning objective:
    f(x) = w_PTV * mean_{v in tumour} (D_v - 1)^2          (cover tumour evenly)
         + sum_organs w_o * mean_{v in organ} D_v^2          (spare healthy organs)
Because D_v is linear in x, f(x) is EXACTLY quadratic in x -> a QUBO, no approximation.
"Use exactly k beams" (limited machine time) is either a penalty  lam*(sum x - k)^2
or is enforced by the quantum circuit itself (XY mixer).
"""
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp

DEFAULT_WEIGHTS = {
    "PTV": 10.0,        # tumour must get the full dose
    "Heart": 5.0,       # heart damage -> long-term heart disease
    "Lung_L": 1.5,
    "Lung_R": 1.5,
    "SpinalCord": 3.0,
    "Breast_R": 2.0,    # healthy other breast
    "OtherTissue": 0.3,
}


class BeamQUBO:
    def __init__(self, dose_maps, anat, k, weights=None):
        self.n = dose_maps.shape[0]
        self.k = k
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.dose_maps = dose_maps
        self.anat = anat
        Q = np.zeros((self.n, self.n))
        c = np.zeros(self.n)
        const = 0.0
        for name, w in self.weights.items():
            if name not in anat.organs:      # e.g. real data has no spinal-cord outline
                continue
            mask = anat.organs[name]
            if w == 0 or mask.sum() == 0:
                continue
            A = dose_maps[:, mask] / k               # (n, |V|)
            nv = mask.sum()
            Q += w * (A @ A.T) / nv
            if name == "PTV":                        # (D-1)^2 = D^2 - 2D + 1
                c += w * (-2.0 * A.sum(axis=1) / nv)
                const += w
        # binary variables: x_i^2 = x_i -> move diagonal into the linear part
        c += np.diag(Q).copy()
        Q = Q - np.diag(np.diag(Q))
        self.Q, self.c, self.const = Q, c, const      # f = x^T Q x + c x + const

    # ---------- classical evaluation ----------
    def cost(self, x):
        x = np.asarray(x, float)
        return float(x @ self.Q @ x + self.c @ x + self.const)

    def penalty_lambda(self):
        """Penalty weight for breaking the "exactly k beams" rule.

        Rule of thumb: the spread of the valid plans' costs.  Big enough that the
        cheapest penalised bitstring is a valid plan (checked in tests/run), small
        enough not to drown the real objective.
        """
        f, X = self.diagonal(lam=None)
        feas = f[X.sum(axis=1) == self.k]
        return float(feas.max() - feas.min())

    def brute_force(self):
        """Check every k-subset (exact optimum).  C(12,4)=495 subsets."""
        best, best_x, all_costs = np.inf, None, []
        for combo in itertools.combinations(range(self.n), self.k):
            x = np.zeros(self.n)
            x[list(combo)] = 1
            f = self.cost(x)
            all_costs.append(f)
            if f < best:
                best, best_x = f, x
        return best_x.astype(int), best, np.array(all_costs)

    # ---------- quantum encoding ----------
    def diagonal(self, lam=None):
        """Cost of every one of the 2^n bitstrings (index i -> qubit j = bit j of i).

        If lam is given, adds the penalty lam*(sum x - k)^2.
        """
        idx = np.arange(2 ** self.n)
        X = ((idx[:, None] >> np.arange(self.n)) & 1).astype(float)
        f = np.einsum("bi,ij,bj->b", X, self.Q, X) + X @ self.c + self.const
        if lam is not None:
            f = f + lam * (X.sum(axis=1) - self.k) ** 2
        return f, X

    def to_ising(self, lam=None):
        """Turn the QUBO into a qubit Hamiltonian using x_i = (1 - Z_i)/2."""
        n = self.n
        Q, c, const = self.Q.copy(), self.c.copy(), self.const
        if lam is not None:                              # lam*(sum x - k)^2
            Q = Q + lam * (np.ones((n, n)) - np.eye(n))  # off-diagonal x_i x_j (both orders)
            c = c + lam * (1 - 2 * self.k)
            const = const + lam * self.k ** 2
        terms, offset = [], const
        h = np.zeros(n)
        for i in range(n):
            offset += c[i] / 2
            h[i] -= c[i] / 2
        for i in range(n):
            for j in range(n):
                if i == j or Q[i, j] == 0:
                    continue
                q = Q[i, j]
                offset += q / 4
                h[i] -= q / 4
                h[j] -= q / 4
                if i < j:
                    terms.append((self._zz(i, j), (Q[i, j] + Q[j, i]) / 4))
        for i in range(n):
            if h[i] != 0:
                terms.append((self._z(i), h[i]))
        op = SparsePauliOp.from_list(terms).simplify()
        return op, offset

    def _z(self, i):
        s = ["I"] * self.n
        s[self.n - 1 - i] = "Z"
        return "".join(s)

    def _zz(self, i, j):
        s = ["I"] * self.n
        s[self.n - 1 - i] = "Z"
        s[self.n - 1 - j] = "Z"
        return "".join(s)


def dose_metrics(anat, D):
    """Mean / max dose in every structure for a dose map D (1.0 = 100 % of the prescription)."""
    out = {}
    for name, m in anat.organs.items():
        if m.sum():
            out[name] = {"mean": float(D[m].mean()), "max": float(D[m].max())}
    ptv = D[anat.organs["PTV"]]
    out["PTV"]["V95"] = float((ptv >= 0.95).mean())      # share of target getting >= 95 % dose
    out["PTV"]["min"] = float(ptv.min())
    return out


def objective_from_dose(anat, D, weights=None):
    """Same cost f as the QUBO, evaluated on any dose map (used for the hospital's own angles)."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    f = 0.0
    for name, w in weights.items():
        if name not in anat.organs or anat.organs[name].sum() == 0:
            continue
        d = D[anat.organs[name]]
        f += w * float(np.mean((d - 1.0) ** 2 if name == "PTV" else d ** 2))
    return f


def plan_dose(qubo, x):
    """Total dose map of beam selection x (each selected beam gives 1/k of the prescription)."""
    return np.tensordot(np.asarray(x, float) / qubo.k, qubo.dose_maps, axes=1)


def organ_doses(qubo, x):
    D = plan_dose(qubo, x)
    return dose_metrics(qubo.anat, D), D
