"""QURIE web app.   Run locally:  streamlit run app.py

Pick a real patient, run the quantum optimiser (QAOA in Qiskit), and compare it with the
exact answer and the angles the hospital used.
"""
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qurie import plots as P
from qurie.classical import clinical_plan, greedy
from qurie.dose import all_beams, candidate_angles
from qurie.qaoa_solver import QAOABeamSolver
from qurie.qubo import DEFAULT_WEIGHTS, BeamQUBO, organ_doses
from qurie.real_data import list_patients, load_patient, patient_table

ROOT = Path(__file__).resolve().parent
RAMPS = json.loads((ROOT / "results" / "ramp_params.json").read_text())
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text())
DEPTHS = [1, 2, 3, 5, 10, 20]

st.set_page_config(page_title="QURIE: quantum radiotherapy planning", page_icon="🎯", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  h1 {font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0;}
  .lede {font-size: 1.08rem; color: #3d4b5c; max-width: 62ch; margin-top: .3rem;}
  .verdict {border-left: 5px solid #1c5cab; background: #eef4fc; padding: .9rem 1.1rem;
            border-radius: 4px; font-size: 1.05rem; margin: .4rem 0 1rem 0;}
  .verdict.miss {border-left-color: #eb6834; background: #fdf1eb;}
  [data-testid="stMetricValue"] {font-size: 1.7rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached heavy work
@st.cache_data(show_spinner=False)
def patient_options():
    t = patient_table()
    return {pid: f"{pid}  ·  {'left' if t[pid]['side'] == 'L' else 'right'} breast" for pid in list_patients()}


@st.cache_resource(show_spinner=False)
def patient_and_beams(pid):
    anat = load_patient(pid)
    angles = candidate_angles(anat)
    return anat, angles, all_beams(anat, angles)


@st.cache_resource(show_spinner=False)
def solve(pid, k, p, heart_w, lung_w, shots, seed):
    anat, angles, dose_maps = patient_and_beams(pid)
    weights = dict(DEFAULT_WEIGHTS, Heart=heart_w,
                   **({"Lung_L": lung_w} if anat.side == "L" else {"Lung_R": lung_w}))
    qubo = BeamQUBO(dose_maps, anat, k, weights)
    t0 = time.time()
    x_opt, f_opt, all_costs = qubo.brute_force()
    t_brute = time.time() - t0

    t0 = time.time()
    solver = QAOABeamSolver(qubo, "xy", p).set_ramp(*RAMPS["xy"][str(p)])
    probs = solver.probabilities(solver.theta)
    samples = solver.sample(shots=shots, seed=seed)
    t_quantum = time.time() - t0
    counts = Counter(samples)
    best_bits = min(counts, key=lambda b: qubo.cost(solver.bits_to_x(b)))
    x_q = solver.bits_to_x(best_bits)
    i_opt = int("".join(map(str, x_opt[::-1])), 2)
    x_g, f_g = greedy(qubo)
    ops = solver.circuit.count_ops()
    return dict(
        anat=anat, angles=angles, qubo=qubo, x_opt=x_opt, f_opt=f_opt, n_valid=len(all_costs),
        x_q=x_q, f_q=qubo.cost(x_q), counts=counts, p_opt=float(probs[i_opt]),
        valid=sum(c for b, c in counts.items() if b.count("1") == k) / shots,
        x_g=x_g, f_g=f_g, t_brute=t_brute, t_quantum=t_quantum,
        depth=solver.circuit.depth(),
        twoq=sum(v for g, v in ops.items() if g in ("cx", "rzz", "rxx", "ryy")),
    )


def ang(x, angles):
    return sorted(int(a) for a in angles[np.asarray(x) == 1])


# ------------------------------------------------------------------ header
st.title("QURIE")
st.markdown('<p class="lede">Picks the radiation beam angles that cover a breast-cancer target and '
            'spare the heart and lungs. The choice is solved by a quantum algorithm (QAOA in Qiskit) '
            'and checked against the exact answer, on real patient CT scans.</p>', unsafe_allow_html=True)

tab_run, tab_study, tab_how = st.tabs(["Plan a patient", "Results on 48 patients", "How it works"])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    opts = patient_options()
    pid = st.selectbox("Patient", list(opts), index=list(opts).index("P03"), format_func=lambda k_: opts[k_])
    k = st.select_slider("Beams allowed", options=[2, 3, 4, 5], value=4,
                         help="More beams = better coverage but more machine time.")
    p = st.select_slider("Quantum circuit depth (p)", options=DEPTHS, value=10,
                         help="Number of QAOA layers. Deeper circuits find the best plan more often.")
    priority = st.slider("Protect the lung  ↔  protect the heart", 0.0, 1.0, 0.5, 0.1,
                         help="Shifts weight between heart dose and dose to the lung on the treated side.")
    shots = st.select_slider("Measurements (shots)", options=[20, 50, 100, 200, 500], value=100)
    seed = st.number_input("Random seed", 0, 9999, 7)
    st.caption("Default settings reproduce the study in the report.")

heart_w = 10 * priority if priority != 0.5 else DEFAULT_WEIGHTS["Heart"]
lung_w = 10 * (1 - priority) if priority != 0.5 else DEFAULT_WEIGHTS["Lung_L"]

# ------------------------------------------------------------------ tab 1: run one patient
with tab_run:
    with st.spinner(f"Running a {12}-qubit QAOA circuit for patient {pid}..."):
        R = solve(pid, k, p, heart_w, lung_w, shots, int(seed))
    anat, angles, qubo = R["anat"], R["angles"], R["qubo"]
    match = np.array_equal(R["x_q"], R["x_opt"])
    gap = 100 * (R["f_q"] - R["f_opt"]) / R["f_opt"]

    if match:
        st.markdown(f'<div class="verdict">The quantum circuit found the <b>exact best plan</b>: beams at '
                    f'<b>{", ".join(str(a) + "°" for a in ang(R["x_q"], angles))}</b>. A normal computer checked '
                    f'all {R["n_valid"]} possible plans to confirm it.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict miss">The quantum circuit\'s best plan is {gap:.1f}% above the exact '
                    f'optimum. Try more shots or a deeper circuit.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chance one shot is the best plan", f"{100 * R['p_opt']:.0f}%",
              f"random guess {100 / R['n_valid']:.1f}%", delta_color="off", delta_arrow="off")
    c2.metric("Valid plans among shots", f"{100 * R['valid']:.0f}%", f"exactly {k} beams", delta_color="off", delta_arrow="off")
    c3.metric("Greedy method", "optimal" if R["f_g"] <= R["f_opt"] * (1 + 1e-9)
              else f"+{100 * (R['f_g'] - R['f_opt']) / R['f_opt']:.1f}%", "vs exact answer", delta_color="off", delta_arrow="off")
    c4.metric("Circuit size", f"{R['twoq']:,} gates", f"12 qubits, depth {R['depth']}", delta_color="off", delta_arrow="off")

    m_q, D_q = organ_doses(qubo, R["x_q"])
    D_c, m_c, _ = clinical_plan(anat)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, (title, D, a_, m) in zip(axes, [("QURIE quantum plan", D_q, ang(R["x_q"], angles), m_q),
                                             ("Hospital's angles (same simple model)", D_c, anat.clinical_angles, m_c)]):
        P._draw_slice(ax, anat, D, a_, title,
                      f"heart {100 * m['Heart']['mean']:.0f}%  ·  near lung {100 * m['Lung_ipsi']['mean']:.0f}%  ·  "
                      f"target covered {100 * m['PTV']['V95']:.0f}%")
    fig.subplots_adjust(wspace=0.04)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Orange = radiation dose (darker = more). White outline = target, pink = heart, blue = lungs. "
               "Percentages are mean organ dose relative to the target dose, in our simplified 2D model.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What the quantum computer measured")
        top = R["counts"].most_common(8)
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        labels = ["/".join(str(a) for a in ang(np.array([int(c) for c in b[::-1]]), angles)) for b, _ in top]
        cols = ["#2a78d6" if np.array_equal(np.array([int(c) for c in b[::-1]]), R["x_opt"]) else "#c9c8c2" for b, _ in top]
        ax.bar(range(len(top)), [c for _, c in top], color=cols, width=0.6)
        ax.set_xticks(range(len(top))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"times seen (of {shots})")
        ax.grid(axis="x", visible=False)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Each bar is one beam plan (angles in degrees). Blue = the exact best plan.")
    with right:
        st.subheader("Side by side")
        st.table({
            "Plan": ["QURIE (quantum)", "Exact best (brute force)", "Greedy", "Hospital"],
            "Beam angles": [", ".join(map(str, ang(R["x_q"], angles))), ", ".join(map(str, ang(R["x_opt"], angles))),
                            ", ".join(map(str, ang(R["x_g"], angles))),
                            ", ".join(str(round(a)) for a in anat.clinical_angles)],
        })
        st.caption(f"Quantum run: {R['t_quantum']:.1f} s on a simulator. Brute force: "
                   f"{1000 * R['t_brute']:.0f} ms. No quantum speed-up at this size; brute force is our referee.")

# ------------------------------------------------------------------ tab 2: study
with tab_study:
    S = SUMMARY
    q10 = S["qaoa"]["xy"]["10"]
    a, b, c = st.columns(3)
    a.metric("Patients tested", S["n_patients"], f"{S['n_left']} left, {S['n_right']} right", delta_color="off", delta_arrow="off")
    b.metric("Exact best plan found (p=10, 100 shots)", f"{100 * q10['found_optimum_rate']:.0f}% of runs")
    c.metric("Angles match the hospital's main beams", f"{S['clinical_tangents_within_15deg']} / {S['n_patients']}",
             "within 15°", delta_color="off", delta_arrow="off")
    for f, cap in [("fig4_methods.png", "Every method against the exact answer."),
                   ("fig3_depth.png", "Deeper circuits find better plans. The XY mixer (blue) vastly beats the penalty version (orange)."),
                   ("fig7_angles_vs_hospital.png", "QURIE's angles vs the hospital's, for every patient."),
                   ("fig5_tradeoff.png", "Shifting priority between heart and lung moves the best plan along a trade-off curve.")]:
        st.image(str(ROOT / "figures" / f), caption=cap, width="stretch")

# ------------------------------------------------------------------ tab 3: how
with tab_how:
    st.markdown("""
**The problem.** Radiotherapy fires X-ray beams at the treated area from a few angles. Each extra beam costs machine time,
and every angle is a trade-off: for a left-sided breast, the heart sits right behind the target.

**As an optimisation problem.** 12 candidate angles → 12 on/off switches → **12 qubits**. Rule: exactly *k* beams.
Cost = uneven target coverage + dose to heart and lungs. Because dose adds up linearly, this cost is an exact
**QUBO**, which maps to a qubit Hamiltonian with Z and ZZ terms.

**The quantum part (QAOA).** The circuit starts in an equal mix of *only valid k-beam plans* and alternates
*cost layers* (push probability toward cheap plans) with *XY-mixer layers* (move a beam from one angle to another).
So every measurement is a valid plan. The layer angles follow a simple ramp: 2 numbers tuned once on one patient,
reused for everyone.

**Honest limits.** Noise-free simulation; simplified 2D dose; no quantum speed-up at 12 qubits. The point is a
correct, checkable quantum formulation that can grow to the billions of plans in real planning.
""")
    st.image(str(ROOT / "figures" / "architecture.png"), width="stretch")
    st.caption("Data: Rezaei et al., Data in Brief 13 (2017), doi:10.1016/j.dib.2017.06.002. "
               "Code: Qiskit 2.x. Built for Qiskit Fall Fest 2026.")
