"""Export everything the website needs into website/data.js (run after run_experiment.py)."""
import base64
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qurie.dose import all_beams, candidate_angles
from qurie.qaoa_solver import QAOABeamSolver
from qurie.qubo import BeamQUBO, organ_doses
from qurie.real_data import load_patient

FEATURED = ["P03", "P22", "P17", "P15", "P04", "P39", "P13", "P49"]
DEPTHS = [1, 2, 3, 5, 10, 20]
ramps = json.loads((ROOT / "results/ramp_params.json").read_text())
S = json.loads((ROOT / "results/summary.json").read_text())
PP = json.loads((ROOT / "results/per_patient.json").read_text())


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode()


featured = {}
for pid in FEATURED:
    anat = load_patient(pid)
    ang = candidate_angles(anat)
    D = all_beams(anat, ang)
    q = BeamQUBO(D, anat, 4)
    ny, nx = anat.shape
    ct = np.clip((anat.ct_hu.astype(float) + 500) / 900, 0, 1) * 255 * anat.body
    lab = np.full((ny, nx), 255, np.uint8)
    lab[anat.body] = 0
    for code, name in [(3, "Lung_L"), (4, "Lung_R"), (2, "Heart"), (1, "PTV")]:
        lab[anat.organs[name]] = code
    dmax = float(D.max())
    combos = list(itertools.combinations(range(12), 4))
    masks, cost, heart, lung, v95 = [], [], [], [], []
    for c in combos:
        x = np.zeros(12); x[list(c)] = 1
        m, _ = organ_doses(q, x)
        masks.append(sum(1 << i for i in c)); cost.append(round(q.cost(x), 5))
        heart.append(round(m["Heart"]["mean"], 4)); lung.append(round(m["Lung_ipsi"]["mean"], 4))
        v95.append(round(m["PTV"]["V95"], 3))
    probs = {}
    for p in DEPTHS:
        s = QAOABeamSolver(q, "xy", p).set_ramp(*ramps["xy"][str(p)])
        pr = s.probabilities(s.theta)
        probs[str(p)] = [round(float(pr[mk]), 5) for mk in masks]
    s = QAOABeamSolver(q, "penalty", 10).set_ramp(*ramps["penalty"]["10"])
    pr = s.probabilities(s.theta)
    pen = [round(float(pr[mk]), 5) for mk in masks]
    rec = next(r for r in PP if r["pid"] == pid)
    featured[pid] = dict(
        side=anat.side, w=nx, h=ny, angles=[int(a) for a in ang], clinical=[float(a) for a in anat.clinical_angles],
        target=[round(float(v), 2) for v in anat.centroid("PTV")], ct=b64(ct), lab=b64(lab),
        dose=b64(np.round(D / dmax * 255)), dmax=dmax, masks=masks, cost=cost, heart=heart, lung=lung, v95=v95,
        probs=probs, penalty=pen, penalty_valid=round(float(sum(pen)), 4),
        greedy_gap=round(rec["greedy"]["gap_pct"], 2), greedy=rec["greedy"]["angles"],
        clinical_metrics=dict(heart=rec["clinical"]["heart"], lung=rec["clinical"]["lung_l"], v95=rec["clinical"]["v95"]))
    print(pid, "ok", flush=True)

patients = [dict(pid=r["pid"], side=r["side"], opt=r["optimal_angles"], clin=[round(a) for a in r["clinical"]["angles"]],
                 q=r["qaoa"]["xy"]["10"]["angles"], found=r["qaoa"]["xy"]["10"]["found_optimum_rate"],
                 popt=round(r["qaoa"]["xy"]["10"]["p_optimal"], 4), greedy_gap=round(r["greedy"]["gap_pct"], 2),
                 heart=round(r["optimal_metrics"]["heart"], 4), heart_clin=round(r["clinical"]["heart"], 4),
                 lung=round(r["optimal_metrics"]["lung_l"], 4), lung_clin=round(r["clinical"]["lung_l"], 4))
            for r in PP]
summary = {k: S[k] for k in ["n_patients", "n_left", "n_right", "greedy_found", "sa_found_rate", "random_found_rate",
                             "even_found", "clinical_tangents_within_15deg", "qurie_beats_clinical_angles",
                             "heart_mean_optimal", "heart_mean_clinical_angles", "brute_force_time_s_mean",
                             "greedy_gap_pct_mean", "sa_gap_pct_mean", "random_gap_pct_mean", "even_gap_pct_mean"]}
summary["xy"] = {p: {k: S["qaoa"]["xy"][p][k] for k in ["p_optimal_mean", "found_optimum_rate", "approx_ratio_mean",
                                                          "two_qubit_gates", "best_sampled_gap_pct_mean"]} for p in S["qaoa"]["xy"]}
summary["penalty"] = {p: {k: S["qaoa"]["penalty"][p][k] for k in ["p_optimal_mean", "found_optimum_rate", "p_feasible_mean",
                                                                    "approx_ratio_mean", "best_sampled_gap_pct_mean"]}
                      for p in S["qaoa"]["penalty"]}
out = "window.QURIE = " + json.dumps(dict(featured=featured, patients=patients, summary=summary), separators=(",", ":")) + ";\n"
(ROOT / "website/data.js").write_text(out)
print("data.js", len(out) / 1e6, "MB")
