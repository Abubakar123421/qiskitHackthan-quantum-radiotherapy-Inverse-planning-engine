# QURIE: 3-minute video script

About 420 spoken words, which fits 3:00 at a calm pace. **[Brackets]** = what is on screen. Rehearse once with a timer.

---

### 0:00–0:25 · Hook and problem  *(Speaker 1)*
**[Slide: title "QURIE: QUantum Radiotherapy Inverse-planning Engine", then a photo-free diagram of a radiotherapy machine]**

> In 1898, Marie Curie discovered radium and gave us radiation therapy. Today, most breast-cancer patients get radiotherapy after surgery. A machine rotates around the patient and fires beams from a few angles. But for a left-sided cancer, the **heart sits right behind the target**. Every angle is a trade-off between hitting the target and hurting the heart or the lung, and every extra beam costs machine time.

### 0:25–0:55 · Formulation  *(Speaker 1)*
**[Slide: 12 angles around a chest slice, "12 switches = 12 qubits", the cost formula, "exactly 4 beams"]**

> We turned this into an optimisation problem. There are 12 candidate angles, and each one is a switch: use the beam or not. That's 12 variables, so 12 qubits. The rule: exactly 4 beams. The cost rewards covering the target evenly and penalises dose to the heart and lungs. Because dose adds up linearly, this cost is *exactly* a QUBO, with no approximation, and it maps directly to a 12-qubit Ising Hamiltonian.

### 0:55–1:30 · Quantum method  *(Speaker 2)*
**[Slide: `architecture.png`, then a simple picture: "Dicke state → [cost layer → XY mixer] × p → measure"]**

> We solve it with QAOA in Qiskit 2. Our key design choice: instead of the textbook version, which allows any number of beams and punishes mistakes with a penalty, we start in a superposition of **only valid 4-beam plans** and use an **XY mixer**, which can only move a beam from one angle to another. So every single measurement is a valid plan. The circuit angles follow a simple ramp with just two numbers. We tuned them on **one** patient and reused them on all the others.

### 1:30–2:20 · Demo and results  *(Speaker 2 runs it, Speaker 3 explains)*
**[Screen recording: terminal, `python run_experiment.py --quick` printing P03's result: "QURIE quantum pick = exact optimum". Then `fig1_dose_maps.png`, then `fig4_methods.png`, then `fig3_depth.png`]**

> Here it is running on a real patient's CT scan. We tested on **48 real patients**, 25 left-sided and 23 right-sided, from a public hospital dataset. For every patient, a normal computer checked all 495 possible plans to find the true best one: that's our referee.
>
> *(dose maps)* Evenly spread beams go straight through the heart. QURIE's plan skims the chest wall.
>
> *(methods chart)* With 100 measurements, QURIE found the **exact best plan in 100% of runs, on all 48 patients**. Greedy missed on 10 patients. The textbook penalty QAOA found it only 25% of the time, so keeping the rule inside the circuit is what makes it work.
>
> *(depth chart)* One single shot gives the exact optimum 46% of the time, versus 0.2% for a random guess.

### 2:20–2:40 · Clinical check  *(Speaker 3)*
**[`fig7_angles_vs_hospital.png`, then `fig5_tradeoff.png`]**

> We never told the optimiser what the doctors did, yet in 47 of 48 patients it chose angles within 15 degrees of the hospital's main beams. And if a doctor shifts priority from lung to heart, the plan moves along the trade-off curve, with QAOA tracking the exact optimum every time.

### 2:40–3:00 · Honesty and future  *(Speaker 1)*
**[Slide: Limitations, plus the scaling table: 495 → 377k → 1.5 billion plans]**

> To be honest: at 12 qubits a normal computer is instant, so there's no quantum speed-up here. That's exactly why we can prove our answers are correct. Our model is 2D and simplified, and deep circuits are still too noisy for today's hardware. But real planning has hundreds of angles and billions of combinations. QURIE shows a verified, physically meaningful quantum formulation, ready to scale as the hardware does. Thank you.

---

## Recording checklist
- [ ] Before recording, run `python run_experiment.py --quick` once so it is fast on camera (about 15–30 s).
- [ ] Zoom the terminal font to at least 18 pt.
- [ ] Show `figures/` images full-screen, one at a time.
- [ ] Keep it under 3:00. If long, cut the trade-off sentence first.
- [ ] Upload to Google Drive → Share → **"Anyone with the link" → Viewer**, then open the link in an incognito window to check it works.

## Likely judge questions (short answers)
- **Why not just brute force?** At 12 angles we do, and we use it as the referee. With 36–180 angles plus couch rotations there are hundreds of thousands to trillions of plans; that's the regime a quantum optimiser targets.
- **Is it faster than classical?** No, and we don't claim it. Simulated annealing with more evaluations also reaches the optimum here.
- **Why the XY mixer?** It keeps every state a valid 4-beam plan. Penalty QAOA wasted 11–60% of shots on invalid plans and stayed at or below 0.5% optimum per shot.
- **Would it run on real hardware?** Not well at p = 10 (about 2,000 two-qubit gates). p = 1–3 is closer and still found the optimum in 93–100% of runs in simulation; noise would lower that.
- **Did you overfit the angles?** Only 2 numbers per depth, tuned on 1 patient (P01), then reused unchanged on 47 others, including 23 right-sided.
- **Is the dose realistic?** It's simplified: 2D, straight beams, density-weighted attenuation. Good for comparing plans, not for clinical dose values.
