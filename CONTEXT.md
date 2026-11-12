# CONTEXT: read this first (for teammates and AI assistants)

This file gives anyone (a person, Claude, ChatGPT, Copilot, Cursor...) the full picture of the project so they can continue the work without the original chat. It is the single source of truth for decisions. `README.md` is the judge-facing summary; this file explains the *why*.

## 1. The hackathon
* **Event:** Qiskit Fall Fest 2026 Hackathon. **Deadline: 12 Nov 2026, 11:59 pm Pakistan time (UTC+5).**
* **Brief:** "How can quantum computing be used to address a meaningful real-world **optimization** problem with competing objectives or limited resources?" Pick one of 6 domains. **We chose Healthcare.**
* **Required:** identify the problem and its users; formulate it (variables, objective, constraints, small instance); justify the quantum algorithm; implement it in **Qiskit**; compare with a classical or exact baseline; reflect on limitations.
* **Deliverables:** GitHub repo (this one, reproducible); report `.docx` with architecture diagram, results, baseline comparison and limitations (`QURIE_Report.docx`); 3-minute video (script in `VIDEO_SCRIPT.md`), shared as "anyone with the link".
* **Judging rubric (100):**
  * A. Problem validity: 20
  * B. Meaningful quantum use: 25 (mapping 10, algorithm justification 10, real use plus awareness of limits 5)
  * C. Implementation and validation: 25 (working Qiskit 10, interpretable outputs 5, baseline 5, reproducibility 5)
  * D. Originality: 15
  * E. Presentation: 15
* Simulator only is fine; quantum speed-up is NOT required; honesty about limitations is rewarded.

## 2. The team and how to talk to them
Team of 3–4 CS students, FAST-NUCES Islamabad. They know CS and AI, but **little about medicine or quantum physics. Explain in plain words.**

## 3. How we got here (decisions and why)
1. The team first liked GitHub projects that **classify** breast cancer with quantum ML (VQC/QSVM on the Wisconsin dataset). We analysed them and found they are classification, not optimization (so they don't fit the brief), and in each one a classical model beat the quantum one. One repo also tests its quantum model on its own training samples. **Decision:** keep the breast-cancer theme but solve a real optimization problem.
2. Ideas considered: cheapest diagnostic-test subset, **radiotherapy beam-angle selection**, chemo drug mix, mobile mammography van routing, gene panel. **Chosen: beam angles.** It is a real hard planning problem, very visual, and none of the reference repos did it.
3. Breast rather than brain: for a left-breast target the heart sits right behind it, which gives a clear trade-off that works in 2D.
4. Name: **QURIE = QUantum Radiotherapy Inverse-planning Engine**, after Marie Curie. ("Inverse planning" is the real clinical term.)
5. Data: we started with a simulated chest (`qurie/anatomy.py::make_phantom`, still kept as a fallback), then switched to **real patients**: Rezaei et al. 2017 (Data in Brief; Mendeley doi:10.17632/vj83jj6fvt.3). A teammate converted it to anonymised 2D slices: **48 patients, 25 left-sided and 23 right-sided** (`data/clean_slices/`, see its README). Overfitting is not a concern: nothing is "trained" per patient, and each patient is an independent test case.
6. Quantum method: plain penalty-QAOA performed badly (most shots invalid, about 0.3% chance of the optimum). **Final choice: XY-mixer QAOA** (Dicke start state, which keeps exactly k beams) with a **linear-ramp angle schedule**: 2 numbers per depth, tuned once on patient P01 and reused on all others.
7. We do **not** use `qiskit-optimization`, which IBM archived in July 2026. The QUBO and Ising mapping are written by hand.
8. Honesty rule: **never claim quantum speed-up.** At this size brute force takes 2 ms and is used as the referee.
9. QURIE does **not detect** cancer. The patient is already diagnosed; QURIE chooses radiation beam directions.

## 4. The model (exact definitions)
* 12 candidate angles (1 qubit each), two 75° arcs facing the treated side, every 15°. Left: `[90,105,120,135,150,165,285,300,315,330,345,0]`. Right: mirrored automatically (`dose.py::candidate_angles`). IEC convention: 0° = from the front, 90° = from the patient's left.
* Choose **k = 4** → C(12,4) = 495 valid plans.
* Dose: straight strip aimed at the target centre, width = target extent + margin, `exp(-0.05·radiological depth)` using CT density, normalised so the target mean = 1. Each chosen beam gives 1/k.
* Objective (`qubo.py`): `10·mean_PTV(D−1)² + 5·mean_heart D² + 1.5·mean_lungL D² + 1.5·mean_lungR D² + 0.3·mean_other D²`. This is **exactly quadratic**, so it is a QUBO. Ising via x = (1−Z)/2.
* QAOA (`qaoa_solver.py`): `qaoa_ansatz(cost_op, reps=p, mixer_operator=XY all-pairs, initial_state=empty)`. The Dicke state is loaded into the Statevector (on hardware it would need the Bärtschi–Eidenbenz circuit). Ramp: `γ_l = Δγ·l/p`, `β_l = Δβ·(1−(l−1)/p)`. Ramp numbers are stored in `results/ramp_params.json`.
* Sampling: `StatevectorSampler`, 100 shots, keep the cheapest valid plan; 5 repeats per patient.

## 5. Key results (from `results/summary.json`, 48 patients)
* XY-QAOA p=10: exact optimum found in **100% of runs**; per-shot P(optimum) **46%** (range 20–76%) vs 0.2% random; approximation ratio 0.973; 100% valid shots.
* Penalty-QAOA p=10: 25% found, 0.3% per shot, 24% of shots invalid.
* Greedy 38/48; simulated annealing (100 evaluations) 80%; random search (100 draws) 17%; evenly spread beams 0/48.
* QURIE's angles are within 15° of the hospital's main beams in **47/48** patients. QURIE's plan has lower cost than the hospital's angles in our model in 41/48 (heart 13.2% vs 17.4%). This is a sanity check only, not a claim to beat doctors.
* Trade-off (patient P22): shifting priority from lung to heart moves the plan from heart 16% / lung 39% to heart 3% / lung 51%; QAOA tracks the exact optimum each time.

## 6. Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (`streamlit run app.py`); deploy on share.streamlit.io, main file `app.py`, Python 3.12 |
| `demo.py` | live demo, about 10 s, prints every Qiskit step (use this for judges) |
| `run_experiment.py` | full study (`--quick` = one patient) |
| `qurie/*.py` | real_data, anatomy (phantom), dose, qubo, qaoa_solver, classical, plots |
| `tests/test_core.py` | 9 checks (QUBO equals physics, Ising equals QUBO on all 4096 bitstrings, XY keeps k, mirror test...) |
| `results/` | summary.json, per_patient.json/csv, ramp_params.json |
| `figures/` | fig1–fig7, architecture.png, demo_*.png |
| `QURIE_Report.docx` | submission report (fill in team names and links on the title page) |
| `VIDEO_SCRIPT.md` | 3-min script plus likely judge questions |

## 7. Status and what is left (as of 12 Nov, ~18:30 PKT)
- [x] Code, real data, results, figures, tests, README, report, video script, clean-install test
- [ ] Check the dataset licence on the Mendeley page before making the repo public (otherwise remove `data/clean_slices/slices` and add a download note)
- [ ] Add team names, GitHub link and video link to `QURIE_Report.docx`
- [ ] Deploy `app.py` on Streamlit Community Cloud and add the link to README and the report
- [ ] Record and upload the video; open every link in incognito; submit before 11:59 pm

## 8. Rules for anyone editing
* Never commit raw DICOM files (they contain patient names). `.gitignore` blocks `*.dcm`.
* If you change the model, re-run `python run_experiment.py` and update the numbers in README, the report and the video script. The report generator is not in the repo; edit the .docx directly.
* Keep the honest framing: no speed-up claims, and the hospital comparison is "in our simplified model".
