"""QAOA for beam-angle selection, built directly on Qiskit 2.x (no archived packages).

Plain-words version
-------------------
The circuit alternates two kinds of layers, `reps` (= p) times:
  * a COST layer   – gives every plan a phase proportional to how bad it is;
  * a MIXER layer  – lets probability flow between plans.
After p layers, good plans have a high chance of being measured.

Two ways of handling the rule "use exactly k beams":
  * variant="penalty" – textbook QAOA.  Start in the superposition of ALL 2^n on/off
        patterns (Hadamard on every qubit), X mixer, and add lam*(sum x - k)^2 to the cost.
        Many measured answers break the rule and are wasted.
  * variant="xy"      – constraint-preserving QAOA (Hadfield et al. 2019).  Start in an equal
        superposition of ONLY the valid k-beam plans (a "Dicke state") and use an XY mixer,
        which can only move a beam from one angle to another.  Every measured answer is valid.

How the layer angles are chosen
-------------------------------
Linear-ramp schedule ("LR-QAOA", a digitised quantum-annealing schedule):
      gamma_i = d_gamma * (i+1)/p        (cost strength ramps UP)
      beta_i  = d_beta  * (1 - i/p)      (mixing strength ramps DOWN)
so there are only TWO numbers to tune (d_gamma, d_beta) whatever the depth p.
They are tuned once, on one patient, by a small grid search on the average measured cost
(no knowledge of the true optimum is used) and then re-used unchanged on every other patient.
Optionally all 2p angles can then be fine-tuned with COBYLA (`refine_all_angles`).
"""
import itertools
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
try:
    from qiskit.circuit.library import qaoa_ansatz              # Qiskit >= 2.1
except ImportError:                                             # older Qiskit 2.0
    from qiskit.circuit.library import QAOAAnsatz as qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorSampler

BASIS = ["rz", "rx", "ry", "h", "x", "cx", "rzz", "rxx", "ryy"]
TWO_QUBIT = ("cx", "rzz", "rxx", "ryy")

# default search grid for the two ramp numbers
GRID_DGAMMA = (0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
GRID_DBETA = (-0.8, -0.4, -0.2, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def dicke_vector(n, k):
    """Equal superposition of all bitstrings with exactly k ones."""
    vec = np.zeros(2 ** n)
    for combo in itertools.combinations(range(n), k):
        vec[sum(1 << q for q in combo)] = 1.0
    return vec / np.linalg.norm(vec)


_DICKE_PREP = {}


def dicke_prep_circuit(n, k):
    """Gate-level circuit that prepares the Dicke state (generic state preparation, cached).

    This generic method needs thousands of gates; a dedicated O(n*k) construction exists
    (Baertschi & Eidenbenz 2019) and would be used on real hardware.
    """
    if (n, k) not in _DICKE_PREP:
        qc = QuantumCircuit(n)
        qc.append(StatePreparation(dicke_vector(n, k)), range(n))
        _DICKE_PREP[(n, k)] = transpile(qc, basis_gates=BASIS, optimization_level=0)
    return _DICKE_PREP[(n, k)]


def xy_mixer(n):
    """All-pairs XY mixer: sum_{i<j} (X_i X_j + Y_i Y_j)/2.

    Each pair term swaps |01> <-> |10>, so the number of beams switched on never changes.
    (X_iX_j and Y_iY_j of the same pair are kept next to each other: they commute, so the
    product of their two rotations is exactly the pair's XY rotation.)
    """
    terms = []
    for i, j in itertools.combinations(range(n), 2):
        for P in ("X", "Y"):
            s = ["I"] * n
            s[n - 1 - i] = P
            s[n - 1 - j] = P
            terms.append(("".join(s), 0.5))
    return SparsePauliOp.from_list(terms)


def ramp_angles(p, d_gamma, d_beta):
    i = np.arange(p)
    return d_gamma * (i + 1) / p, d_beta * (1 - i / p)


class QAOABeamSolver:
    def __init__(self, qubo, variant="xy", reps=1, lam=None):
        self.qubo, self.variant, self.reps = qubo, variant, reps
        n, k = qubo.n, qubo.k
        self.lam = (qubo.penalty_lambda() if lam is None else lam) if variant == "penalty" else None

        cost_op, offset = qubo.to_ising(lam=self.lam)          # qubit Hamiltonian (Z, ZZ terms)
        self.diag, X = qubo.diagonal(lam=self.lam)             # its value on every bitstring
        self.true_cost, _ = qubo.diagonal(lam=None)            # real objective, no penalty
        self.feasible = X.sum(axis=1) == k
        feas_costs = self.true_cost[self.feasible]
        # divide by the spread of valid-plan costs so the same angles work for every patient
        self.scale = float(feas_costs.max() - feas_costs.min())
        self.cost_op = (cost_op / self.scale).simplify()
        self.diag_scaled = (self.diag - offset) / self.scale

        if variant == "xy":
            ansatz = qaoa_ansatz(self.cost_op, reps=reps, mixer_operator=xy_mixer(n),
                                 initial_state=QuantumCircuit(n))
            # The Dicke start state is loaded straight into the simulator.  On hardware it has
            # an efficient O(n*k)-gate circuit (Baertschi & Eidenbenz 2019).
            self.start_state = Statevector(dicke_vector(n, k))
        elif variant == "penalty":
            ansatz = qaoa_ansatz(self.cost_op, reps=reps)      # H on all qubits + X mixer
            self.start_state = Statevector.from_label("0" * n)
        else:
            raise ValueError(variant)
        self.ansatz = ansatz
        self.circuit = transpile(ansatz, basis_gates=BASIS, optimization_level=1)
        self.params = list(self.circuit.parameters)
        self.n_params = len(self.params)
        names = [str(t) for t in self.params]                  # "β[i]" (mixer), "γ[i]" (cost)
        self.gamma_idx = [names.index(f"γ[{i}]") for i in range(reps)]
        self.beta_idx = [names.index(f"β[{i}]") for i in range(reps)]
        self.theta = None
        self.n_evals = 0
        self.opt_time = 0.0

    # ------------------------------------------------------------------ simulation
    def pack(self, gammas, betas):
        theta = np.zeros(self.n_params)
        theta[self.gamma_idx] = gammas
        theta[self.beta_idx] = betas
        return theta

    def probabilities(self, theta):
        """Exact measurement probabilities of all 2^n bitstrings (noise-free simulation)."""
        bound = self.circuit.assign_parameters(dict(zip(self.params, theta)))
        return self.start_state.evolve(bound).probabilities()

    def expected_cost(self, theta):
        """Average (normalised) cost of the measured plans = what QAOA minimises."""
        self.n_evals += 1
        return float(self.probabilities(theta) @ self.diag_scaled)

    # ------------------------------------------------------------------ choosing the angles
    def set_ramp(self, d_gamma, d_beta):
        self.d_gamma, self.d_beta = float(d_gamma), float(d_beta)
        self.gammas, self.betas = ramp_angles(self.reps, d_gamma, d_beta)
        self.theta = self.pack(self.gammas, self.betas)
        return self

    def tune_ramp(self, grid_gamma=GRID_DGAMMA, grid_beta=GRID_DBETA, refine=True):
        """Grid search (+ short COBYLA polish) of the two ramp numbers."""
        t0 = time.time()
        best = (np.inf, None, None)
        for dg in grid_gamma:
            for db in grid_beta:
                e = self.expected_cost(self.pack(*ramp_angles(self.reps, dg, db)))
                if e < best[0]:
                    best = (e, dg, db)
        _, dg, db = best
        if refine:
            f = lambda z: self.expected_cost(self.pack(*ramp_angles(self.reps, z[0], z[1] / 10)))
            res = minimize(f, [dg, db * 10], method="COBYLA",
                           options={"maxiter": 40, "rhobeg": 0.2 * abs(dg)})
            if res.fun < best[0]:
                dg, db = res.x[0], res.x[1] / 10
        self.set_ramp(dg, db)
        self.opt_time += time.time() - t0
        return self.d_gamma, self.d_beta

    def refine_all_angles(self, maxiter=300):
        """Optional: let COBYLA adjust all 2p angles freely, starting from the current ones."""
        t0 = time.time()
        p = self.reps
        f = lambda z: self.expected_cost(self.pack(z[:p], z[p:] / 10))
        res = minimize(f, np.concatenate([self.gammas, self.betas * 10]), method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.1 * max(abs(self.gammas).max(), 1e-3)})
        if res.fun < self.expected_cost(self.theta):
            self.gammas, self.betas = res.x[:p].copy(), res.x[p:] / 10
            self.theta = self.pack(self.gammas, self.betas)
        self.opt_time += time.time() - t0
        return self

    # ------------------------------------------------------------------ measuring
    def measured_circuit(self):
        """Full circuit with state preparation + measurement (what would run on hardware)."""
        qc = QuantumCircuit(self.qubo.n)
        if self.variant == "xy":
            qc.compose(dicke_prep_circuit(self.qubo.n, self.qubo.k), inplace=True)
        qc.compose(self.circuit.assign_parameters(dict(zip(self.params, self.theta))), inplace=True)
        qc.measure_all()
        return qc

    def sample(self, shots=100, seed=0):
        """Measure the circuit `shots` times with Qiskit's StatevectorSampler (V2 primitive).

        Returns one bitstring per shot, e.g. "000100110001".
        """
        sampler = StatevectorSampler(seed=seed)
        result = sampler.run([self.measured_circuit()], shots=shots).result()
        return result[0].data.meas.get_bitstrings()

    def bits_to_x(self, bits):
        """Qiskit bitstrings are little-endian: the LAST character is qubit 0 (= beam 0)."""
        return np.array([int(c) for c in bits[::-1]])

    # ------------------------------------------------------------------ scoring
    def evaluate(self, best_cost, worst_cost, shots=100, repeats=5, seed=0):
        """All the numbers we report for one circuit.

        "Exact" numbers come from the simulated probabilities; "sampled" numbers come from
        `repeats` independent experiments of `shots` measurements each.
        """
        n = self.qubo.n
        p = self.probabilities(self.theta)
        feas_idx = np.nonzero(self.feasible)[0]
        order = feas_idx[np.argsort(self.true_cost[feas_idx])]        # valid plans, best first
        rank_of = {int(i): r + 1 for r, i in enumerate(order)}
        idx_opt = int(order[0])
        p_feas = float(p[self.feasible].sum())
        exp_cost = float(p[self.feasible] @ self.true_cost[self.feasible] / max(p_feas, 1e-12))
        p_opt = float(p[idx_opt])

        shots_all = [int(b, 2) for b in self.sample(shots * repeats, seed)]
        found, gaps, ranks, picks = [], [], [], []
        for r in range(repeats):
            chunk = [i for i in shots_all[r * shots:(r + 1) * shots] if self.feasible[i]]
            if not chunk:                               # every shot broke the k-beam rule
                found.append(False)
                picks.append(None)
                continue
            best_i = min(chunk, key=lambda i: self.true_cost[i])
            picks.append(best_i)
            found.append(best_i == idx_opt)
            gaps.append(100 * (self.true_cost[best_i] - best_cost) / best_cost)
            ranks.append(rank_of[best_i])
        valid = float(np.mean([self.feasible[i] for i in shots_all]))
        ops = self.circuit.count_ops()
        first = picks[0]
        return {
            "variant": self.variant,
            "reps": self.reps,
            "d_gamma": getattr(self, "d_gamma", None),
            "d_beta": getattr(self, "d_beta", None),
            # ---- exact (from simulated probabilities) ----
            "p_optimal": p_opt,                                  # chance ONE shot is the optimum
            "p_top5": float(p[order[:5]].sum()),
            "p_feasible": p_feas,                                # chance ONE shot obeys "k beams"
            "approx_ratio": float((worst_cost - exp_cost) / (worst_cost - best_cost)),
            "expected_cost_valid": exp_cost,
            "shots_for_95pct": (float(np.log(0.05) / np.log(1 - p_opt)) if 0 < p_opt < 1 else None),
            # ---- sampled (StatevectorSampler) ----
            "shots": shots,
            "repeats": repeats,
            "valid_shot_fraction": valid,
            "found_optimum_rate": float(np.mean(found)),         # share of experiments that hit it
            "best_sampled_gap_pct": float(np.mean(gaps)) if gaps else None,
            "best_sampled_rank": float(np.mean(ranks)) if ranks else None,
            "x_best": None if first is None else [(first >> j) & 1 for j in range(n)],
            # ---- circuit size ----
            "depth": int(self.circuit.depth()),
            "two_qubit_gates": int(sum(v for g, v in ops.items() if g in TWO_QUBIT)),
            "tune_time_s": self.opt_time,
            "circuit_evals": self.n_evals,
            "probs": p,                                          # stripped before saving JSON
            "rank_order": order,                                 # stripped before saving JSON
        }
