#!/usr/bin/env python
"""Live demo for judges: one patient, step by step, every Qiskit object shown.

    python demo.py            # patient P03 (left breast)
    python demo.py P04        # any patient ID, e.g. a right-sided one

Takes about 15 seconds.  Saves figures/demo_circuit.png and figures/demo_measurements.png.
"""
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qurie.dose import all_beams, candidate_angles
from qurie.qaoa_solver import QAOABeamSolver
from qurie.qubo import BeamQUBO, organ_doses
from qurie.real_data import load_patient

ROOT = Path(__file__).resolve().parent
pid = sys.argv[1] if len(sys.argv) > 1 else "P03"
P_DEPTH = 10


def step(n, text):
    print(f"\n{'-' * 78}\n STEP {n}: {text}\n{'-' * 78}")


# 1 -------------------------------------------------------------------------
step(1, f"Load real patient {pid}")
anat = load_patient(pid)
angles = candidate_angles(anat)
side = "LEFT" if anat.side == "L" else "RIGHT"
print(f" {side} breast  |  hospital used beams at {[round(a) for a in anat.clinical_angles]} degrees")
print(f" 12 candidate beam angles (1 qubit each): {[int(a) for a in angles]}")

# 2 -------------------------------------------------------------------------
step(2, "Dose model -> QUBO (the optimisation problem)")
qubo = BeamQUBO(all_beams(anat, angles), anat, k=4)
print(" variables : x_0 .. x_11  (1 = use that beam)")
print(" constraint: exactly 4 beams switched on")
print(" objective : cover the target evenly + keep heart and lung dose low")
print(f" Q matrix  : {qubo.Q.shape}, {np.count_nonzero(qubo.Q)} beam-pair interaction terms")

# 3 -------------------------------------------------------------------------
step(3, "QUBO -> qubit Hamiltonian (Qiskit SparsePauliOp), x = (1 - Z)/2")
H, offset = qubo.to_ising()
terms = sorted(zip(H.paulis.to_labels(), H.coeffs.real), key=lambda t: -abs(t[1]))
print(f" {len(H)} Pauli terms on {H.num_qubits} qubits. Largest five:")
for lab, c in terms[:5]:
    print(f"    {c:+.4f}  {lab}")

# 4 -------------------------------------------------------------------------
step(4, f"Build the QAOA circuit (XY mixer, depth p={P_DEPTH})")
ramps = json.loads((ROOT / "results" / "ramp_params.json").read_text())
solver = QAOABeamSolver(qubo, "xy", P_DEPTH).set_ramp(*ramps["xy"][str(P_DEPTH)])
ops = solver.circuit.count_ops()
print(f" qubits: {solver.circuit.num_qubits}   depth: {solver.circuit.depth()}   "
      f"two-qubit gates: {sum(v for g, v in ops.items() if g in ("cx", "rzz", "rxx", "ryy"))}")
print(f" ramp angles (tuned once on patient {ramps['tuned_on']}): "
      f"d_gamma={solver.d_gamma:.3f}, d_beta={solver.d_beta:.4f}")
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter
from qurie.qaoa_solver import xy_mixer
g, b = Parameter("γ"), Parameter("β")                   # block view of one QAOA layer
demo_qc = QuantumCircuit(12)
demo_qc.append(PauliEvolutionGate(xy_mixer(12) * 0, label="Dicke start: all valid 4-beam plans"), range(12))
demo_qc.barrier()
demo_qc.append(PauliEvolutionGate(solver.cost_op, time=g, label="COST layer  e^(-iγH)"), range(12))
demo_qc.append(PauliEvolutionGate(xy_mixer(12), time=b, label="XY MIXER  e^(-iβH_XY)"), range(12))
demo_qc.measure_all(add_bits=True)
fig = demo_qc.draw("mpl", scale=0.8, fold=-1)
fig.savefig(ROOT / "figures" / "demo_circuit.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(" circuit picture (one layer, block view; repeated p times) saved: figures/demo_circuit.png")

# 5 -------------------------------------------------------------------------
step(5, "Run it: measure the circuit 100 times with Qiskit's StatevectorSampler")
t = time.time()
shots = solver.sample(shots=100, seed=7)
print(f" done in {time.time() - t:.1f}s")
counts = Counter(shots)
valid = sum(c for b, c in counts.items() if b.count("1") == 4)
print(f" valid 4-beam plans among the 100 shots: {valid}/100")
print(" most frequent measured plans:")
for bits, c in counts.most_common(5):
    x = solver.bits_to_x(bits)
    print(f"    {bits}  -> beams {sorted(int(a) for a in angles[x == 1])}   seen {c} times   cost {qubo.cost(x):.3f}")
best_bits = min(counts, key=lambda b: qubo.cost(solver.bits_to_x(b)))
x_q = solver.bits_to_x(best_bits)

# 6 -------------------------------------------------------------------------
step(6, "Check against the exact answer (normal computer tries all 495 plans)")
t = time.time()
x_opt, f_opt, all_costs = qubo.brute_force()
print(f" brute force took {1000 * (time.time() - t):.0f} ms")
m_q, _ = organ_doses(qubo, x_q)
print(f" QUANTUM pick : beams {sorted(int(a) for a in angles[x_q == 1])}  cost {qubo.cost(x_q):.4f}")
print(f" EXACT best   : beams {sorted(int(a) for a in angles[x_opt == 1])}  cost {f_opt:.4f}")
print("  ==> " + ("MATCH: the quantum circuit found the true optimum" if np.array_equal(x_q, x_opt)
                  else f"off by {100 * (qubo.cost(x_q) - f_opt) / f_opt:.1f}%"))
print(f" plan quality: heart {100 * m_q['Heart']['mean']:.0f}%, lung on treated side "
      f"{100 * m_q['Lung_ipsi']['mean']:.0f}%, target covered (V95) {100 * m_q['PTV']['V95']:.0f}%  (of prescription)")
p = solver.probabilities(solver.theta)
print(f" chance ONE shot is the optimum: {100 * p[int(''.join(map(str, x_opt[::-1])), 2)]:.0f}%"
      f"  (random guess: {100 / len(all_costs):.1f}%)")

# bar chart of measured plans
top = counts.most_common(8)
labels = ["/".join(str(int(a)) for a in sorted(angles[solver.bits_to_x(b) == 1])) for b, _ in top]
cols = ["#2a78d6" if np.array_equal(solver.bits_to_x(b), x_opt) else "#b9b8b2" for b, _ in top]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(range(len(top)), [c for _, c in top], color=cols)
ax.set_xticks(range(len(top))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("times measured (of 100)")
ax.set_title(f"Patient {pid}: what the quantum circuit returned (blue = exact optimum)", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(ROOT / "figures" / "demo_measurements.png", dpi=150, bbox_inches="tight")
print("\n measurement chart saved: figures/demo_measurements.png")
