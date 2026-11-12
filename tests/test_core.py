"""Sanity checks.  Run:  python -m pytest -q     (or)     python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qurie.anatomy import make_phantom
from qurie.dose import all_beams, candidate_angles, mirror_anatomy, target_side
from qurie.qaoa_solver import QAOABeamSolver
from qurie.qubo import BeamQUBO, organ_doses, objective_from_dose
from qurie.real_data import list_patients, load_patient

ANGLES = np.array([90, 105, 120, 135, 150, 165, 285, 300, 315, 330, 345, 0])


def _qubo(k=4):
    anat = load_patient("P01")
    return BeamQUBO(all_beams(anat, ANGLES), anat, k)


def test_dataset_loads():
    assert len(list_patients()) == 48
    a = load_patient("P03")
    assert a.organs["PTV"].any() and a.organs["Heart"].any() and a.organs["Lung_L"].any()


def test_qubo_equals_physical_objective():
    """The QUBO polynomial must equal the dose-based objective for any beam choice."""
    q = _qubo()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.zeros(q.n)
        x[rng.choice(q.n, q.k, replace=False)] = 1
        _, D = organ_doses(q, x)
        assert abs(q.cost(x) - objective_from_dose(q.anat, D, q.weights)) < 1e-9


def test_ising_mapping_matches_qubo_on_all_bitstrings():
    """The qubit Hamiltonian must give the same number as the QUBO for all 4096 bitstrings."""
    q = _qubo()
    for lam in (None, q.penalty_lambda()):
        op, offset = q.to_ising(lam=lam)
        diag, _ = q.diagonal(lam=lam)
        ising = np.real(op.to_matrix(sparse=True).diagonal()) + offset
        assert np.abs(ising - diag).max() < 1e-9


def test_penalty_keeps_optimum_valid():
    q = _qubo()
    diag, X = q.diagonal(lam=q.penalty_lambda())
    assert X[np.argmin(diag)].sum() == q.k
    assert np.allclose(X[np.argmin(diag)], q.brute_force()[0])


def test_xy_mixer_never_breaks_the_k_beam_rule():
    q = _qubo()
    s = QAOABeamSolver(q, "xy", reps=2).set_ramp(8.0, -0.1)
    assert abs(s.probabilities(s.theta)[s.feasible].sum() - 1.0) < 1e-9
    assert all(b.count("1") == q.k for b in s.sample(shots=50, seed=1))


def test_qaoa_beats_random_guessing():
    q = _qubo()
    x_opt, f_best, costs = q.brute_force()
    s = QAOABeamSolver(q, "xy", reps=5).set_ramp(15.0, -0.11)
    r = s.evaluate(f_best, costs.max(), shots=100, repeats=1)
    assert r["p_optimal"] > 10 / len(costs)          # at least 10x better than a random guess


def test_right_sided_patient_gives_mirrored_plan():
    """Flip a patient left-to-right: the candidate beams and the best plan must flip too."""
    left = load_patient("P01")
    right = mirror_anatomy(left)
    assert target_side(left) == "L" and target_side(right) == "R"
    ang_l, ang_r = candidate_angles(left), candidate_angles(right)
    assert np.array_equal(ang_r, (360 - ang_l) % 360)
    ql = BeamQUBO(all_beams(left, ang_l), left, 4)
    qr = BeamQUBO(all_beams(right, ang_r), right, 4)
    xl, fl, _ = ql.brute_force()
    xr, fr, _ = qr.brute_force()
    assert abs(fl - fr) / fl < 0.02
    assert sorted((360 - ang_l[xl == 1]) % 360) == sorted(ang_r[xr == 1])


def test_real_right_sided_patient_uses_mirrored_candidates():
    a = load_patient("P04")
    assert a.side == "R" and target_side(a) == "R"
    assert np.array_equal(candidate_angles(a), (360 - candidate_angles(load_patient("P01"))) % 360)


def test_phantom_pipeline_runs():
    anat = make_phantom()
    q = BeamQUBO(all_beams(anat, np.arange(0, 360, 30)), anat, 4)
    x, f, costs = q.brute_force()
    assert x.sum() == 4 and f == costs.min()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
