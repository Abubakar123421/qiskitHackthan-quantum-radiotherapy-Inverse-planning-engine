# QURIE — QUantum Radiotherapy Inverse-planning Engine

**Choosing radiation beam angles that hit the tumour and spare the heart, solved with QAOA in Qiskit.**

Qiskit Fall Fest 2026 Hackathon · Domain: **Healthcare** · Team: FAST-NUCES Islamabad

> In 1898 Marie Curie's discovery of radium gave us radiation therapy. QURIE uses a quantum computer to help aim it.

---

## 1. The problem (plain words)

After surgery, most breast-cancer patients get **radiotherapy**: a machine rotates around the patient and fires X-ray beams at the treated area from a few **angles**.

* **Limited resource:** every extra beam costs machine time, so a plan may use only **k = 4** beams.
* **Competing objectives:** cover the target fully, **and** keep radiation away from the **heart** (long-term heart disease) and the **lungs** (lung damage). For a **left** breast, the heart sits right behind the target, so the angle that avoids the heart often hits the lung, and the other way round.

**Question:** out of 12 candidate angles, which 4 give the best trade-off for *this* patient?

**Who it serves:** radiotherapy planners and medical physicists (angle choice is still largely manual), patients (fewer side effects), and busy hospitals, especially in low-resource settings.

## 2. Formulation

| | |
|---|---|
| **Decision variables** | `x_b ∈ {0,1}` for 12 candidate beams, `x_b = 1` means use beam b. **1 qubit per beam, 12 qubits.** |
| **Constraint** | `Σ x_b = k = 4` (machine time) |
| **Dose model** | each selected beam gives 1/k of the dose: `D_v(x) = Σ_b x_b · a_bv`, where `a_bv` is beam b's dose in pixel v (straight strip aimed at the target, attenuated by CT-measured tissue density) |
| **Objective (minimise)** | `f(x) = w_T · mean_target (D−1)² + Σ_organs w_o · mean_organ D²` (target 10, heart 5, each lung 1.5, other tissue 0.3) |
| **Instance** | **48 real patients** (25 left-sided, 23 right-sided). For each: C(12,4) = **495 valid plans**, 2¹² = 4096 bitstrings |

Because `D` is linear in `x`, `f` is **exactly quadratic**: a genuine QUBO with no approximation. The `x_i·x_j` terms are **beam overlap** (two beams hitting the same healthy tissue). Mapping `x_i = (1 − Z_i)/2` turns it into a 12-qubit Ising Hamiltonian with Z and ZZ terms.

Candidate angles: two 75° arcs facing the treated chest wall, every 15° (left: 90–165° and 285–0°). For right-sided patients the list is mirrored automatically.

## 3. Quantum algorithm: QAOA with a constraint-preserving XY mixer

* **Why QAOA:** the problem is natively a QUBO, and QAOA builds its circuit straight from the problem's Hamiltonian.
* **Our upgrade, the XY mixer (Hadfield et al. 2019):** the circuit starts in an equal superposition of **only the valid 4-beam plans** (a Dicke state). The mixer can only *move* a beam from one angle to another, never add or remove one. So **100% of measurements obey the k-beam rule**, and no probability is wasted on invalid plans.
* **Baseline quantum variant:** textbook QAOA with an X mixer plus a penalty `λ(Σx − k)²`. We run both and compare.
* **Choosing the circuit angles:** a *linear-ramp* schedule (a digitised quantum anneal) has only **2 numbers** for any depth p. They are tuned **once, on one patient (P01)**, then reused **unchanged** on all 48 patients. No per-patient training, and no knowledge of the true optimum is used.
* **Measurement:** `StatevectorSampler` (Qiskit V2 primitive), 100 shots, keep the cheapest valid plan.

Built on **Qiskit 2.x only** (`qaoa_ansatz`, `SparsePauliOp`, `Statevector`, `StatevectorSampler`). It does not depend on `qiskit-optimization`, which IBM archived in July 2026.

## 4. Results

All numbers below come from `python run_experiment.py` (`results/summary.json`). Noise-free statevector simulation; brute force over all 495 plans is the exact referee.

**48 real patients (25 left-sided, 23 right-sided), 12 qubits, choose 4 beams, 495 valid plans each.**

| Method | Finds the exact optimum | Avg. cost above optimum |
|---|---|---|
| Exact brute force (reference) | 100 % | 0 % |
| **QURIE: XY-mixer QAOA, p = 10, best of 100 shots** | **100 % of runs (all 48 patients, 5 runs each)** | **0.00 %** |
| QURIE: XY-mixer QAOA, p = 3 | 100 % of runs | 0.00 % |
| QURIE: XY-mixer QAOA, p = 1 | 93 % of runs | 0.21 % |
| Penalty QAOA (textbook), p = 20 | 41 % of runs | 3.9 % |
| Greedy (add best beam one at a time) | 38 / 48 patients | 0.62 % (worst 8 %) |
| Simulated annealing, 100 cost evaluations | 80 % of runs | 0.75 % |
| Random search, 100 draws | 17 % of runs | 7.5 % |
| Evenly spread beams | 0 / 48 | 83 % |

**Per-shot quality of the quantum circuit (mean over 48 patients)**

| depth p | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| XY mixer: chance one shot = exact optimum | 2.6 % | 8.6 % | 13.4 % | 28.4 % | **46.0 %** | 62.0 % |
| XY mixer: valid shots (exactly 4 beams) | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % |
| XY mixer: approximation ratio | 0.829 | 0.887 | 0.920 | 0.946 | **0.973** | 0.987 |
| Penalty: chance one shot = optimum | 0.1 % | 0.1 % | 0.1 % | 0.2 % | 0.3 % | 0.5 % |
| Penalty: valid shots | 40 % | 46 % | 57 % | 58 % | 76 % | 89 % |
| Shots needed for 95 % chance of the optimum (XY, median) | 117 | 36 | 22 | 9 | **5** | 3 |

A random guess hits the optimum 0.2 % of the time (1 in 495). At p = 10 the XY circuit hits it **46 % of the time on average (between 20 % and 76 % per patient), about 230× better**, with the angles tuned on one patient and reused on 47 unseen ones. The penalty version wastes 11–60 % of shots on invalid plans and never gets past 0.5 %: **putting the constraint inside the circuit is what makes QAOA work here.**

**Clinical sanity check against the hospital's real plans** (scored in *our* simplified dose model, equal beam weights):

* The hospital's two main (tangent) beams lie within 15° of an angle QURIE chose in **47 of 48 patients**, although the optimiser was never told the clinical angles.
* QURIE's plan has a lower objective than the hospital's angles in 41 / 48 patients. Mean heart dose: **13.2 % vs 17.4 %** of the prescription (left-sided: 19.4 % vs 25.9 %); treated-side lung 34.0 % vs 33.3 %; target coverage (V95) 73.5 % vs 72.1 %. This uses an extra two beams' worth of freedom (4 vs 2–3 beams) and a much simpler physics model, so it is a sanity check, **not** a claim to beat clinicians.

**Trade-off (competing objectives):** shifting priority from the left lung to the heart moves the optimal plan along a heart-vs-lung front (patient P22: heart 16 % / lung 39 % → heart 3 % / lung 51 %), and QAOA tracks the exact optimum at every setting (`figures/fig5_tradeoff.png`).

**Scaling:** brute force is instant here (2 ms). The search space explodes with finer angles: 36 candidates × 5 beams = 377k plans, 72 × 7 = 1.5 billion, 180 × 9 = 4.5 × 10¹⁴. That regime is where a scalable optimiser is needed.

| Figure | Shows |
|---|---|
| `fig1_dose_maps.png` | where the radiation goes: QURIE plan vs evenly spread vs hospital angles |
| `fig2_top_n_probability.png` | chance one quantum measurement lands in the top-N plans, by depth |
| `fig3_depth.png` | P(optimum) and approximation ratio vs circuit depth, XY vs penalty |
| `fig4_methods.png` | every method against the exact answer |
| `fig5_tradeoff.png` | heart-vs-lung trade-off and QAOA tracking it |
| `fig6_dvh.png` | dose-volume histogram |
| `fig7_angles_vs_hospital.png` | QURIE's angles vs the angles the hospital used, 48 patients |
| `architecture.png` | system architecture |


## 5. How to run (judges: start here)

**Interactive website (for everyone):** **https://abubakar123421.github.io/qiskitHackthan-quantum-radiotherapy-Inverse-planning-engine/** (source in `website/`). Plan a patient's beams yourself, then race the quantum computer.

**Live Qiskit web app:** **https://qurie-app.streamlit.app**. Pick any of the 48 patients, change beams, circuit depth and heart-vs-lung priority, and the QAOA circuit runs live.

Run the web app locally: `streamlit run app.py`

Requires **Python 3.10–3.13**.

```bash
git clone <this repo>
cd qurie
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python demo.py                        # ~10 s: LIVE DEMO, every Qiskit step printed (start here)
python run_experiment.py --quick      # ~30 s: one patient in full detail + figures
python run_experiment.py              # ~15 min: all 48 patients, every table and figure
python -m pytest -q                   # 9 sanity checks (optional: pip install pytest)
```

Other options: `--patient P04` (any ID from `data/clean_slices/patients.csv`), `--retune` (re-tune the 2 ramp numbers instead of using `results/ramp_params.json`).

**Outputs**

* `results/summary.json`: every number in this README
* `results/per_patient.csv`: one row per patient (exact plan, quantum plan, baselines, hospital angles)
* `figures/`: dose maps, dose-volume histogram, QAOA success curves, method comparison, trade-off curve, hospital-angle match, architecture diagram

## 6. What the tests check (`tests/test_core.py`)

1. The QUBO polynomial equals the physical dose objective for random plans.
2. The Ising Hamiltonian equals the QUBO on **all 4096 bitstrings** (mapping correctness).
3. The penalty weight keeps the penalised optimum a valid 4-beam plan.
4. The XY-mixer circuit never produces an invalid plan (exact probabilities and sampled shots).
5. QAOA beats random guessing by more than 10×.
6. Mirroring a patient left↔right mirrors the candidate angles and the optimal plan.
7. Real right-sided patients get the mirrored candidate list.
8. Plus checks that the dataset and the phantom pipeline load and run.

## 7. Repository layout

```
website/                 interactive website (static, Vercel-ready)
tools/export_web.py      exports results for the website
app.py                   Streamlit web app (live demo)
demo.py                  terminal demo, every Qiskit step printed
run_experiment.py        one command: data -> dose -> QUBO -> QAOA + baselines -> results + figures
qurie/
  real_data.py           loads the 48 real patient slices
  anatomy.py             simple 2D chest phantom (teaching / fallback)
  dose.py                beam dose model, candidate angles, left/right handling
  qubo.py                QUBO builder, brute force, Ising mapping, dose metrics
  qaoa_solver.py         QAOA (XY mixer and penalty variants), ramp tuning, sampling, scoring
  classical.py           greedy, simulated annealing, random search, evenly spread, hospital angles
  plots.py               all figures
data/clean_slices/       anonymised 2D slices (npz) + patients.csv + data README
tests/test_core.py       sanity checks
results/  figures/       generated outputs
```

## 8. Honest limitations

* **No quantum speed-up.** Brute force over 495 plans takes milliseconds; we use it as the referee. The quantum part is a correct, verifiable *method*, not a faster one. The argument for quantum is scaling: 36 candidate angles with 5 beams already gives 377k plans, and couch rotations push it to billions.
* **Noise-free simulation.** p = 10 needs about 2,000 two-qubit gates, far beyond today's noisy hardware without error mitigation. Low depth (p = 1–3) is closer to hardware-ready but less accurate.
* **The Dicke start state** is loaded directly in simulation. On hardware it needs a dedicated O(n·k) circuit (Bärtschi & Eidenbenz 2019).
* **Simplified 2D dose model.** Straight beams, density-weighted attenuation, no scatter, equal beam weights. Real planning is 3D, with optimised beam weights, wedges and multi-leaf collimators (IMRT/VMAT). Dose percentages compare *plans within our model*. They are **not** clinical doses.
* **Hospital comparison caveat.** The hospital's angles are re-scored *in our simple model with equal weights*; their real plans used wedges, different weights and 3D information. "Lower cost than the hospital's angles" means "in our model", not "better than the doctors".
* **Targets** are the post-mastectomy chest wall (CTV), not a tumour in an intact breast. There are no spinal-cord or other-breast outlines in the data.
* **Objective weights** were chosen by us. Clinicians would tune them, and the trade-off figure shows how the answer moves when they do.
* **One slice per patient** (the one with the strongest heart–target overlap), not the full 3D volume.

## 9. Data and credits

Patient data: Rezaei M., Mohammadbeigi A., Khoshgard K., Haghparast A. *CT images and radiotherapy treatment planning of patients with breast cancer: A dataset.* Data in Brief 13 (2017) 390–395, doi:10.1016/j.dib.2017.06.002. Data: https://data.mendeley.com/datasets/vj83jj6fvt (doi:10.17632/vj83jj6fvt.3). Check the dataset licence before redistributing. This repo contains only anonymised, down-sampled 2D slices; **raw DICOM files (which contain patient names) are never committed.**

Methods: QAOA (Farhi et al. 2014); Quantum Alternating Operator Ansatz / XY mixer (Hadfield et al. 2019); Dicke-state preparation (Bärtschi & Eidenbenz 2019); linear-ramp QAOA schedules (e.g. Montañez-Barrera & Michielsen 2024); least-squares plan objective and beam-angle optimisation background (Craft et al., CORT dataset, 2014).
