#!/usr/bin/env python
"""QURIE - QUantum Radiotherapy Inverse-planning Engine.  One command runs everything.

    python run_experiment.py            # full study: 48 real patients (about 12 minutes)
    python run_experiment.py --quick    # one patient, stored circuit angles (about 30 seconds)
    python run_experiment.py --patient P22 --quick

Pipeline:  patient slice -> dose of 12 candidate beams -> QUBO -> QAOA (Qiskit) + classical
references -> scores, tables (results/) and pictures (figures/).
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from qurie import classical as C
from qurie import plots as P
from qurie.dose import CANDIDATES_LEFT, all_beams, candidate_angles
from qurie.qaoa_solver import QAOABeamSolver
from qurie.qubo import DEFAULT_WEIGHTS, BeamQUBO, organ_doses, plan_dose
from qurie.real_data import list_patients, load_patient

# ------------------------------------------------------------------ problem settings
# 12 candidate beam directions = 12 qubits (see qurie/dose.py).  0 = from the front,
# 90 = from the patient's left (IEC gantry convention).  For a right-sided target
# candidate_angles() mirrors the list automatically.
ANGLES = CANDIDATES_LEFT
K = 4                       # beams allowed (limited machine time)
DEPTHS = [1, 2, 3, 5, 10, 20]
TUNE_PATIENT = "P01"        # the only patient used to tune the 2 ramp numbers per depth
SHOWCASE = "P03"
SHOTS, REPEATS = 100, 5     # each quantum "experiment" = 100 measurements, repeated 5 times
TRADEOFF_PATIENT = "P22"
HEART_SHARE = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]   # share of 8 weight points on heart vs left lung

ROOT = Path(__file__).resolve().parent
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
RAMP_FILE = RESULTS / "ramp_params.json"


def build(pid, k=K, weights=None):
    anat = load_patient(pid)
    angles = candidate_angles(anat)            # mirrored automatically for right-sided patients
    qubo = BeamQUBO(all_beams(anat, angles), anat, k, weights)
    qubo.angles = angles
    return anat, qubo


def angles_of(x, qubo):
    return sorted(int(a) for a in qubo.angles[np.asarray(x) == 1])


def clean(d):
    """Drop big arrays / numpy types so a result dict can be written as JSON."""
    out = {}
    for k_, v in d.items():
        if k_ in ("probs", "rank_order"):
            continue
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k_] = v
    return out


# ------------------------------------------------------------------ step 1: tune the ramp once
def tune_ramps(force=False):
    if RAMP_FILE.exists() and not force:
        return json.loads(RAMP_FILE.read_text())
    print(f"\n[1] Tuning the QAOA ramp (2 numbers per depth) on patient {TUNE_PATIENT} only ...")
    _, qubo = build(TUNE_PATIENT)
    ramps = {"tuned_on": TUNE_PATIENT, "xy": {}, "penalty": {}}
    for variant in ("xy", "penalty"):
        for p in DEPTHS:
            s = QAOABeamSolver(qubo, variant, p)
            dg, db = s.tune_ramp()
            ramps[variant][str(p)] = [dg, db]
            print(f"    {variant:8s} p={p:<2d}  d_gamma={dg:7.3f}  d_beta={db:7.4f}  "
                  f"({s.n_evals} circuit evaluations, {s.opt_time:.0f}s)")
    RAMP_FILE.write_text(json.dumps(ramps, indent=2))
    return ramps


# ------------------------------------------------------------------ step 2: one patient
def solve_patient(pid, ramps, depths=DEPTHS, verbose=False):
    anat, qubo = build(pid)
    t0 = time.time()
    x_opt, f_best, all_costs = qubo.brute_force()               # exact answer = the referee
    t_brute = time.time() - t0
    f_worst, n_valid = float(all_costs.max()), len(all_costs)
    gap = lambda f: 100 * (f - f_best) / f_best
    is_opt = lambda f: bool(f <= f_best * (1 + 1e-9))

    m_opt, _ = organ_doses(qubo, x_opt)
    xg, fg = C.greedy(qubo)
    sa = [C.simulated_annealing(qubo, seed=s, steps=SHOTS)[1] for s in range(REPEATS)]
    xe, fe = C.evenly_spaced(qubo)
    rng = np.random.default_rng(0)                               # random search, same budget as QAOA
    rnd = np.array([all_costs[rng.integers(0, n_valid, SHOTS)].min() for _ in range(400)])
    _, m_clin, f_clin = C.clinical_plan(anat)

    rec = {
        "pid": pid, "side": anat.side, "candidate_angles": [int(a) for a in qubo.angles],
        "n_valid_plans": n_valid, "best_cost": f_best, "worst_cost": f_worst,
        "median_cost": float(np.median(all_costs)),
        "random_approx_ratio": float((f_worst - all_costs.mean()) / (f_worst - f_best)),
        "optimal_angles": angles_of(x_opt, qubo), "brute_force_time_s": t_brute,
        "optimal_metrics": {"heart": m_opt["Heart"]["mean"], "lung_l": m_opt["Lung_ipsi"]["mean"],
                            "v95": m_opt["PTV"]["V95"]},
        "greedy": {"angles": angles_of(xg, qubo), "gap_pct": gap(fg), "found": is_opt(fg)},
        "sim_annealing": {"found_rate": float(np.mean([is_opt(f) for f in sa])),
                          "gap_pct": float(np.mean([gap(f) for f in sa])), "cost_evals": SHOTS},
        "evenly_spread": {"angles": angles_of(xe, qubo), "gap_pct": gap(fe), "found": is_opt(fe)},
        "random_search": {"found_rate": float(np.mean(rnd <= f_best * (1 + 1e-9))),
                          "found_rate_theory": 1 - (1 - 1 / n_valid) ** SHOTS,
                          "gap_pct": float(np.mean(gap(rnd))), "draws": SHOTS},
        "clinical": {"angles": [float(a) for a in anat.clinical_angles], "cost": f_clin,
                     "gap_pct": gap(f_clin), "heart": m_clin["Heart"]["mean"],
                     "lung_l": m_clin["Lung_ipsi"]["mean"], "v95": m_clin["PTV"]["V95"]},
        "qaoa": {"xy": {}, "penalty": {}},
    }
    probs = {"xy": {}, "penalty": {}}
    for variant in ("xy", "penalty"):
        for p in depths:
            s = QAOABeamSolver(qubo, variant, p).set_ramp(*ramps[variant][str(p)])
            r = s.evaluate(f_best, f_worst, shots=SHOTS, repeats=REPEATS, seed=p)
            probs[variant][p] = r["probs"][r["rank_order"]]      # probability of plan rank 1, 2, ...
            r = clean(r)
            r["angles"] = angles_of(r["x_best"], qubo) if r["x_best"] else None
            rec["qaoa"][variant][str(p)] = r
            if verbose:
                print(f"    QAOA {variant:8s} p={p:<2d}  P(optimum)={100*r['p_optimal']:5.1f}%  "
                      f"valid shots={100*r['p_feasible']:5.1f}%  approx. ratio={r['approx_ratio']:.3f}  "
                      f"found optimum in {100*r['found_optimum_rate']:.0f}% of {REPEATS}x{SHOTS}-shot runs  "
                      f"[{r['two_qubit_gates']} two-qubit gates]")
    return rec, probs, anat, qubo


# ------------------------------------------------------------------ step 3: extra studies
def tradeoff_study(pid, ramps, p=10):
    """Shift priority from the left lung to the heart and watch the best plan move (competing objectives).

    heart weight = 8*s, left-lung weight = 8*(1-s), s = HEART_SHARE.  Everything else unchanged.
    """
    anat, qubo0 = build(pid)
    pts = []
    import itertools
    qubo = qubo0
    for combo in itertools.combinations(range(qubo0.n), K):
        x = np.zeros(qubo0.n); x[list(combo)] = 1
        m, _ = organ_doses(qubo0, x)
        pts.append([m["Heart"]["mean"], m["Lung_ipsi"]["mean"]])
    pareto, picks, rows = [], [], []
    for w in HEART_SHARE:
        weights = dict(DEFAULT_WEIGHTS, Heart=8 * w, Lung_L=8 * (1 - w))
        qubo = BeamQUBO(qubo0.dose_maps, anat, K, weights)
        qubo.angles = qubo0.angles
        x_opt, f_best, all_costs = qubo.brute_force()
        m, _ = organ_doses(qubo, x_opt)
        s = QAOABeamSolver(qubo, "xy", p).set_ramp(*ramps["xy"][str(p)])
        r = s.evaluate(f_best, all_costs.max(), shots=SHOTS, repeats=REPEATS, seed=1)
        mq, _ = organ_doses(qubo, r["x_best"])
        pareto.append((w, m["Heart"]["mean"], m["Lung_ipsi"]["mean"]))
        picks.append((mq["Heart"]["mean"], mq["Lung_ipsi"]["mean"]))
        rows.append({"heart_share": w, "exact_angles": angles_of(x_opt, qubo), "qaoa_angles": angles_of(r["x_best"], qubo),
                     "heart_mean": m["Heart"]["mean"], "lung_l_mean": m["Lung_ipsi"]["mean"],
                     "p_optimal": r["p_optimal"], "found_optimum_rate": r["found_optimum_rate"]})
    return np.array(pts), pareto, picks, rows


def k_study(pid, ramps, p=10):
    """Same circuit recipe with 2, 3, 4, 5 beams allowed (ramp numbers NOT re-tuned)."""
    rows = []
    for k in (2, 3, 4, 5):
        anat, qubo = build(pid, k=k)
        x_opt, f_best, all_costs = qubo.brute_force()
        m, _ = organ_doses(qubo, x_opt)
        s = QAOABeamSolver(qubo, "xy", p).set_ramp(*ramps["xy"][str(p)])
        r = s.evaluate(f_best, all_costs.max(), shots=SHOTS, repeats=REPEATS, seed=2)
        rows.append({"k": k, "n_valid_plans": len(all_costs), "exact_angles": angles_of(x_opt, qubo),
                     "heart_mean": m["Heart"]["mean"], "lung_l_mean": m["Lung_ipsi"]["mean"], "v95": m["PTV"]["V95"],
                     "p_optimal": r["p_optimal"], "approx_ratio": r["approx_ratio"],
                     "found_optimum_rate": r["found_optimum_rate"]})
    return rows


def ablations(ramps):
    """(a) penalty strength, (b) free COBYLA fine-tuning of all angles - on the tuning patient."""
    _, qubo = build(TUNE_PATIENT)
    _, f_best, all_costs = qubo.brute_force()
    f_worst = all_costs.max()
    out = {"penalty_weight": [], "free_angles": []}
    base = qubo.penalty_lambda()
    for factor in (0.25, 0.5, 1.0, 2.0):
        s = QAOABeamSolver(qubo, "penalty", 10, lam=factor * base)
        s.tune_ramp()
        r = clean(s.evaluate(f_best, f_worst, shots=SHOTS, repeats=REPEATS))
        out["penalty_weight"].append({"lambda_factor": factor, "p_optimal": r["p_optimal"],
                                      "p_feasible": r["p_feasible"], "approx_ratio": r["approx_ratio"]})
    for p in (1, 2, 3):
        s = QAOABeamSolver(qubo, "xy", p).set_ramp(*ramps["xy"][str(p)])
        r0 = s.evaluate(f_best, f_worst, shots=SHOTS, repeats=1)
        s.refine_all_angles(maxiter=200)
        r1 = s.evaluate(f_best, f_worst, shots=SHOTS, repeats=1)
        out["free_angles"].append({"p": p, "ramp_p_optimal": r0["p_optimal"], "ramp_approx_ratio": r0["approx_ratio"],
                                   "cobyla_p_optimal": r1["p_optimal"], "cobyla_approx_ratio": r1["approx_ratio"]})
    return out


# ------------------------------------------------------------------ reporting helpers
def print_patient(rec):
    q = rec["qaoa"]["xy"]
    p_show = "10" if "10" in q else sorted(q, key=int)[-1]
    r = q[p_show]
    print(f"\n  Patient {rec['pid']}: {rec['n_valid_plans']} valid plans "
          f"(choose {K} of {len(ANGLES)} beams)")
    print(f"    exact best plan (brute force) : {rec['optimal_angles']}  cost {rec['best_cost']:.3f}"
          f"   heart {100*rec['optimal_metrics']['heart']:.0f}%  near lung {100*rec['optimal_metrics']['lung_l']:.0f}%")
    print(f"    QURIE quantum pick (p={p_show})     : {r['angles']}  "
          f"{'= exact optimum' if r['best_sampled_gap_pct'] < 1e-6 else '+%.1f%% above optimum' % r['best_sampled_gap_pct']}"
          f"   P(optimum per shot) {100*r['p_optimal']:.0f}%")
    print(f"    greedy                         : {rec['greedy']['angles']}  +{rec['greedy']['gap_pct']:.1f}%")
    print(f"    evenly spread beams            : {rec['evenly_spread']['angles']}  +{rec['evenly_spread']['gap_pct']:.1f}%")
    print(f"    hospital's angles (our model)  : {[round(a) for a in rec['clinical']['angles']]}  "
          f"{rec['clinical']['gap_pct']:+.1f}%   heart {100*rec['clinical']['heart']:.0f}%  "
          f"near lung {100*rec['clinical']['lung_l']:.0f}%")


def summarise(per_patient, depths):
    n = len(per_patient)
    mean = lambda f: float(np.mean([f(r) for r in per_patient]))
    S = {"n_patients": n, "n_left": sum(r["side"] == "L" for r in per_patient),
         "n_right": sum(r["side"] == "R" for r in per_patient),
         "n_qubits": len(ANGLES), "k": K, "candidate_angles_left": [int(a) for a in ANGLES],
         "n_valid_plans": per_patient[0]["n_valid_plans"], "shots": SHOTS, "repeats": REPEATS,
         "greedy_found": int(sum(r["greedy"]["found"] for r in per_patient)),
         "greedy_gap_pct_mean": mean(lambda r: r["greedy"]["gap_pct"]),
         "greedy_gap_pct_max": float(max(r["greedy"]["gap_pct"] for r in per_patient)),
         "sa_found_rate": mean(lambda r: r["sim_annealing"]["found_rate"]),
         "sa_gap_pct_mean": mean(lambda r: r["sim_annealing"]["gap_pct"]),
         "even_found": int(sum(r["evenly_spread"]["found"] for r in per_patient)),
         "even_gap_pct_mean": mean(lambda r: r["evenly_spread"]["gap_pct"]),
         "random_found_rate": mean(lambda r: r["random_search"]["found_rate"]),
         "random_found_rate_theory": per_patient[0]["random_search"]["found_rate_theory"],
         "random_gap_pct_mean": mean(lambda r: r["random_search"]["gap_pct"]),
         "random_approx_ratio": mean(lambda r: r["random_approx_ratio"]),
         "brute_force_time_s_mean": mean(lambda r: r["brute_force_time_s"]),
         "qurie_beats_clinical_angles": int(sum(r["clinical"]["gap_pct"] > 0 for r in per_patient)),
         "heart_mean_optimal": mean(lambda r: r["optimal_metrics"]["heart"]),
         "heart_mean_clinical_angles": mean(lambda r: r["clinical"]["heart"]),
         "lung_mean_optimal": mean(lambda r: r["optimal_metrics"]["lung_l"]),
         "lung_mean_clinical_angles": mean(lambda r: r["clinical"]["lung_l"]),
         "v95_mean_optimal": mean(lambda r: r["optimal_metrics"]["v95"]),
         "v95_mean_clinical_angles": mean(lambda r: r["clinical"]["v95"]),
         "qaoa": {}}
    # how close are the hospital's two tangent beams to one of QURIE's chosen angles?
    dev = []
    for r in per_patient:
        d = lambda a, b: abs((a - b + 180) % 360 - 180)
        ref = (122, 302) if r["side"] == "L" else (238, 58)          # typical tangent pair
        tang = sorted(r["clinical"]["angles"], key=lambda t: min(d(t, ref[0]), d(t, ref[1])))[:2]
        dev.append(max(min(d(t, s) for s in r["optimal_angles"]) for t in tang))
    S["clinical_tangents_within_15deg"] = int(sum(v <= 15 for v in dev))
    S["clinical_tangent_deviation_deg_mean"] = float(np.mean(dev))
    S["by_side"] = {}
    for sd in ("L", "R"):
        grp = [r for r in per_patient if r["side"] == sd]
        if not grp:
            continue
        g = lambda f: float(np.mean([f(r) for r in grp]))
        S["by_side"][sd] = {
            "n": len(grp), "greedy_found": int(sum(r["greedy"]["found"] for r in grp)),
            "greedy_gap_pct_max": float(max(r["greedy"]["gap_pct"] for r in grp)),
            "xy_p10_p_optimal_mean": g(lambda r: r["qaoa"]["xy"]["10"]["p_optimal"]),
            "xy_p10_found_rate": g(lambda r: r["qaoa"]["xy"]["10"]["found_optimum_rate"]),
            "xy_p10_approx_ratio": g(lambda r: r["qaoa"]["xy"]["10"]["approx_ratio"]),
            "penalty_p10_p_optimal_mean": g(lambda r: r["qaoa"]["penalty"]["10"]["p_optimal"]),
            "beats_clinical_angles": int(sum(r["clinical"]["gap_pct"] > 0 for r in grp)),
            "heart_optimal": g(lambda r: r["optimal_metrics"]["heart"]),
            "heart_clinical_angles": g(lambda r: r["clinical"]["heart"]),
            "lung_optimal": g(lambda r: r["optimal_metrics"]["lung_l"]),
            "lung_clinical_angles": g(lambda r: r["clinical"]["lung_l"]),
            "v95_optimal": g(lambda r: r["optimal_metrics"]["v95"]),
            "v95_clinical_angles": g(lambda r: r["clinical"]["v95"]),
            "clinical_tangents_within_15deg": int(sum(v <= 15 for v, r in zip(dev, per_patient) if r["side"] == sd))}
    for variant in ("xy", "penalty"):
        S["qaoa"][variant] = {}
        for p in depths:
            g = lambda key: [r["qaoa"][variant][str(p)][key] for r in per_patient]
            gaps = [v for v in g("best_sampled_gap_pct") if v is not None]
            S["qaoa"][variant][str(p)] = {
                "p_optimal_mean": float(np.mean(g("p_optimal"))), "p_optimal_min": float(np.min(g("p_optimal"))),
                "p_optimal_max": float(np.max(g("p_optimal"))), "p_top5_mean": float(np.mean(g("p_top5"))),
                "p_feasible_mean": float(np.mean(g("p_feasible"))),
                "approx_ratio_mean": float(np.mean(g("approx_ratio"))),
                "approx_ratio_min": float(np.min(g("approx_ratio"))),
                "found_optimum_rate": float(np.mean(g("found_optimum_rate"))),
                "best_sampled_gap_pct_mean": float(np.mean(gaps)) if gaps else None,
                "shots_for_95pct_median": float(np.median([v for v in g("shots_for_95pct") if v])),
                "two_qubit_gates": per_patient[0]["qaoa"][variant][str(p)]["two_qubit_gates"],
                "depth": per_patient[0]["qaoa"][variant][str(p)]["depth"]}
    return S


def print_summary(S, depths):
    n = S["n_patients"]
    print("\n" + "=" * 100)
    print(f" SUMMARY over {n} real patients   ({S['n_qubits']} qubits, choose {S['k']} beams, "
          f"{S['n_valid_plans']} valid plans, quantum budget {S['shots']} shots)")
    print("=" * 100)
    print(f" {'method':42s}{'finds exact optimum':>22s}{'avg. cost above optimum':>26s}")
    row = lambda name, a, b: print(f" {name:42s}{a:>22s}{b:>26s}")
    row("exact brute force (reference)", "100%", "0.0%")
    for p in depths:
        q = S["qaoa"]["xy"][str(p)]
        row(f"QURIE: XY-mixer QAOA, p={p}", f"{100*q['found_optimum_rate']:.0f}% of runs", f"{q['best_sampled_gap_pct_mean']:.2f}%")
    q = S["qaoa"]["penalty"][str(depths[-1])]
    row(f"penalty QAOA, p={depths[-1]}", f"{100*q['found_optimum_rate']:.0f}% of runs", f"{q['best_sampled_gap_pct_mean']:.2f}%")
    row("greedy", f"{S['greedy_found']}/{n} patients", f"{S['greedy_gap_pct_mean']:.2f}%")
    row(f"simulated annealing ({S['shots']} cost evals)", f"{100*S['sa_found_rate']:.0f}% of runs", f"{S['sa_gap_pct_mean']:.2f}%")
    row(f"random search ({S['shots']} draws)", f"{100*S['random_found_rate']:.0f}% of runs", f"{S['random_gap_pct_mean']:.2f}%")
    row("evenly spread beams", f"{S['even_found']}/{n} patients", f"{S['even_gap_pct_mean']:.1f}%")
    print("-" * 100)
    print(f" {'QAOA per-shot statistics (mean)':34s}" + "".join(f"{'p='+str(p):>9s}" for p in depths))
    for variant, name in (("xy", "XY mixer"), ("penalty", "penalty")):
        for key, lab, mult in (("p_optimal_mean", "P(optimum) %", 100), ("p_feasible_mean", "valid shots %", 100),
                               ("approx_ratio_mean", "approx. ratio", 1)):
            print(f" {name + ': ' + lab:34s}" + "".join(
                f"{mult*S['qaoa'][variant][str(p)][key]:9.{1 if mult == 100 else 3}f}" for p in depths))
    print("-" * 100)
    print(f" QURIE's exact-best plan has a lower cost than the hospital's angles (in our dose model) "
          f"in {S['qurie_beats_clinical_angles']}/{n} patients")
    print(f"   mean heart dose: {100*S['heart_mean_optimal']:.1f}% (QURIE)  vs  {100*S['heart_mean_clinical_angles']:.1f}% (hospital angles)"
          f"   | lung on treated side: {100*S['lung_mean_optimal']:.1f}% vs {100*S['lung_mean_clinical_angles']:.1f}%")
    print(f" Hospital's two tangent beams lie within 15 degrees of a QURIE angle in "
          f"{S['clinical_tangents_within_15deg']}/{n} patients")
    for sd, name in (("L", "LEFT "), ("R", "RIGHT")):
        b = S["by_side"].get(sd)
        if b:
            print(f" {name} ({b['n']} patients): QAOA-XY p=10 finds optimum in {100*b['xy_p10_found_rate']:.0f}% of runs, "
                  f"P(opt)/shot {100*b['xy_p10_p_optimal_mean']:.0f}%; greedy {b['greedy_found']}/{b['n']}; "
                  f"heart {100*b['heart_optimal']:.1f}% vs {100*b['heart_clinical_angles']:.1f}% (hospital angles); "
                  f"lung {100*b['lung_optimal']:.1f}% vs {100*b['lung_clinical_angles']:.1f}%")
    print(f" Honest note: brute force takes {1000*S['brute_force_time_s_mean']:.0f} ms here - no quantum speed-up at this size.")
    print("=" * 100)


def write_csv(per_patient, path, depths):
    p_main = "10" if 10 in depths else str(depths[-1])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "exact_best_angles", "best_cost", "heart_pct", "left_lung_pct", "v95_pct",
                    f"qaoa_xy_p{p_main}_angles", "qaoa_xy_p_optimum_pct", "qaoa_xy_found_rate_pct",
                    "qaoa_xy_approx_ratio", "qaoa_penalty_p_optimum_pct", "qaoa_penalty_valid_pct",
                    "greedy_gap_pct", "sim_annealing_gap_pct", "evenly_spread_gap_pct",
                    "hospital_angles", "hospital_angles_gap_pct", "hospital_heart_pct", "hospital_left_lung_pct"])
        for r in per_patient:
            x, pn = r["qaoa"]["xy"][p_main], r["qaoa"]["penalty"][p_main]
            w.writerow([r["pid"], " ".join(map(str, r["optimal_angles"])), f"{r['best_cost']:.4f}",
                        f"{100*r['optimal_metrics']['heart']:.1f}", f"{100*r['optimal_metrics']['lung_l']:.1f}",
                        f"{100*r['optimal_metrics']['v95']:.0f}", " ".join(map(str, x["angles"])),
                        f"{100*x['p_optimal']:.1f}", f"{100*x['found_optimum_rate']:.0f}", f"{x['approx_ratio']:.3f}",
                        f"{100*pn['p_optimal']:.2f}", f"{100*pn['p_feasible']:.0f}",
                        f"{r['greedy']['gap_pct']:.2f}", f"{r['sim_annealing']['gap_pct']:.2f}",
                        f"{r['evenly_spread']['gap_pct']:.1f}",
                        " ".join(f"{a:.0f}" for a in r["clinical"]["angles"]), f"{r['clinical']['gap_pct']:.1f}",
                        f"{100*r['clinical']['heart']:.1f}", f"{100*r['clinical']['lung_l']:.1f}"])


def showcase_figures(rec, probs, anat, qubo):
    pid = rec["pid"]
    depths = sorted(probs["xy"])
    p_main = 10 if 10 in depths else depths[-1]
    q = rec["qaoa"]["xy"][str(p_main)]
    x_q = np.array(q["x_best"])
    m_q, D_q = organ_doses(qubo, x_q)
    xe, _ = C.evenly_spaced(qubo)
    m_e, D_e = organ_doses(qubo, xe)
    D_c, m_c, _ = C.clinical_plan(anat)
    P.plan_maps(anat, [
        (f"QURIE quantum plan (QAOA, p={p_main})", D_q, angles_of(x_q, qubo), m_q),
        ("Evenly spread beams", D_e, angles_of(xe, qubo), m_e),
        ("Hospital's angles, same dose model", D_c, anat.clinical_angles, m_c)],
        FIGS / "fig1_dose_maps.png", f"Patient {pid}: where the radiation goes for three beam plans")
    P.dvh(anat, [("QURIE quantum plan", D_q), ("evenly spread beams", D_e)], FIGS / "fig6_dvh.png")
    show = {"xy": {p: probs["xy"][p] for p in depths if p in (1, 3, 5, 10, 20)},
            "penalty": {depths[-1]: probs["penalty"][depths[-1]]}}
    P.topn_curve(show, FIGS / "fig2_top_n_probability.png", rec["n_valid_plans"])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="one patient only, re-use stored ramp angles")
    ap.add_argument("--patient", default=SHOWCASE, help="patient shown in detail (default %(default)s)")
    ap.add_argument("--retune", action="store_true", help="re-tune the ramp even if results/ramp_params.json exists")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    t_start = time.time()
    print("QURIE - quantum beam-angle selection for left-sided breast radiotherapy")
    print(f"candidate beams (1 qubit each): {[int(a) for a in ANGLES]} degrees,  choose k={K}")

    ramps = tune_ramps(force=args.retune)
    print(f"\n[2] Detailed run on patient {args.patient}")
    rec, probs, anat, qubo = solve_patient(args.patient, ramps, depths=DEPTHS, verbose=True)
    print_patient(rec)
    showcase_figures(rec, probs, anat, qubo)
    P.architecture(FIGS / "architecture.png")
    (RESULTS / f"showcase_{args.patient}.json").write_text(json.dumps(rec, indent=2))
    if args.quick:
        print(f"\nDone in {time.time()-t_start:.0f}s.  Pictures: figures/fig1_dose_maps.png, fig2_top_n_probability.png, fig6_dvh.png")
        return

    print(f"\n[3] All {len(list_patients())} patients (left- and right-sided) (ramp angles re-used unchanged from {ramps['tuned_on']})")
    per_patient = []
    for pid in list_patients():
        r = rec if pid == args.patient else solve_patient(pid, ramps)[0]
        per_patient.append(r)
        x = r["qaoa"]["xy"]["10"]
        print(f"    {pid} {r['side']}: exact {str(r['optimal_angles']):22s} QAOA-XY p=10 {str(x['angles']):22s} "
              f"P(opt)={100*x['p_optimal']:4.0f}%  found {100*x['found_optimum_rate']:3.0f}%  "
              f"greedy +{r['greedy']['gap_pct']:.1f}%  hospital angles {r['clinical']['gap_pct']:+.0f}%")
    S = summarise(per_patient, DEPTHS)

    print("\n[4] Trade-off study (heart weight sweep), beam-count study, ablations")
    pts, pareto, picks, trade_rows = tradeoff_study(TRADEOFF_PATIENT, ramps)
    S["tradeoff"] = trade_rows
    S["k_study"] = k_study(args.patient, ramps)
    S["ablations"] = ablations(ramps)
    S["ramp_params"] = ramps
    S["scaling"] = [{"candidates": n, "beams": k, "valid_plans": math.comb(n, k)}
                    for n, k in ((12, 4), (24, 5), (36, 5), (72, 7), (180, 7), (180, 9))]
    S["runtime_s"] = time.time() - t_start

    print_summary(S, DEPTHS)
    (RESULTS / "summary.json").write_text(json.dumps(S, indent=2))
    (RESULTS / "per_patient.json").write_text(json.dumps(per_patient, indent=2))
    write_csv(per_patient, RESULTS / "per_patient.csv", DEPTHS)

    P.depth_curves(per_patient, DEPTHS, S["n_valid_plans"], FIGS / "fig3_depth.png")
    n = S["n_patients"]
    rows = [(f"QURIE XY-mixer QAOA, p={p}", 100 * S["qaoa"]["xy"][str(p)]["found_optimum_rate"],
             S["qaoa"]["xy"][str(p)]["best_sampled_gap_pct_mean"], "quantum") for p in (20, 10, 5, 3, 1)]
    rows += [(f"Penalty QAOA, p={p}", 100 * S["qaoa"]["penalty"][str(p)]["found_optimum_rate"],
              S["qaoa"]["penalty"][str(p)]["best_sampled_gap_pct_mean"], "penalty") for p in (20, 10)]
    rows += [("Greedy", 100 * S["greedy_found"] / n, S["greedy_gap_pct_mean"], "classical"),
             (f"Simulated annealing ({SHOTS} evals)", 100 * S["sa_found_rate"], S["sa_gap_pct_mean"], "classical"),
             (f"Random search ({SHOTS} draws)", 100 * S["random_found_rate"], S["random_gap_pct_mean"], "classical"),
             ("Evenly spread beams", 100 * S["even_found"] / n, S["even_gap_pct_mean"], "classical")]
    P.method_bars(rows, FIGS / "fig4_methods.png", n, SHOTS)
    r_t = next(r for r in per_patient if r["pid"] == TRADEOFF_PATIENT)
    clin = (r_t["clinical"]["heart"], r_t["clinical"]["lung_l"])
    P.tradeoff(pts, pareto, picks, clin, FIGS / "fig5_tradeoff.png", TRADEOFF_PATIENT)
    P.angle_match(per_patient, FIGS / "fig7_angles_vs_hospital.png")
    print(f"\nAll done in {time.time()-t_start:.0f}s.  Tables: results/   Pictures: figures/")


if __name__ == "__main__":
    main()
