# QURIE: 3-minute pitch video script (YC style)

**Speakers:** Abubakar (story), Zeeshan (how it works), Muntazir (demo and results)
**Total:** 2:55. About 390 spoken words plus about 20 seconds of B-roll with no talking.
**Rule:** while a person is talking, cut to the B-roll listed at least every 4–5 seconds. Show faces on the key lines (marked 👤).

---

## 0:00–0:08 · Cold open (no talking, 8 s)
**B-roll:** black screen. Text types on: *"Every year, about 2 million women are diagnosed with breast cancer."* Then cut to a CT slice from `figures/fig1_dose_maps.png` (left panel), slowly zooming in.
**Music:** low, calm beat starts.

---

## 0:08–0:40 · The problem: ABUBAKAR

👤 **ABUBAKAR:**
> "In 1898, Marie Curie discovered radium, and gave us radiation therapy.
> Today, most breast-cancer patients get radiation after surgery. A machine fires beams at the chest from a few angles."

**B-roll (0:16):** `fig1_dose_maps.png`, middle panel ("Evenly spread beams"). Zoom into the beam lines going through the pink heart outline. Caption overlay: **"The heart sits right behind the left breast"**

> "But for a left-breast cancer, the heart is right behind the target. Pick the wrong angles, and you cure the cancer but damage the heart. Pick too many beams, and you waste machine time that other patients are waiting for."

👤 *(back to face)*
> "So every plan has to answer one question: **which few angles hit the cancer and spare the heart?**"

**Overlay text (big):** *"Which 4 angles?"*

---

## 0:40–1:15 · The idea: ZEESHAN

👤 **ZEESHAN:**
> "We're Abubakar, Zeeshan and Muntazir, and we built **QURIE**, a quantum engine that plans those angles."

**B-roll (0:45):** animated overlay (make in Canva/CapCut): a chest outline with 12 dashed beam lines around it, then 4 of them light up blue. Text: **"12 angles = 12 qubits · choose 4 · 495 possible plans"**

> "We give the machine 12 possible angles. Each angle is a switch, on or off, and each switch is one qubit. The rule: exactly four beams. The score: cover the target evenly, keep heart and lung dose low."

**B-roll (0:58):** `figures/architecture.png`, slowly panning from box 1 to box 5.

> "Because radiation dose adds up, this score becomes an exact quantum optimisation problem. We solve it with QAOA in Qiskit."

👤 *(face)*
> "And here's our key idea: our circuit **only ever holds valid four-beam plans**. Every single measurement is a real, usable plan, with nothing wasted."

**B-roll (1:10):** `figures/demo_circuit.png`, zooming in on the "XY MIXER" block.

---

## 1:15–1:23 · Transition (no talking, 8 s)
**B-roll:** screen recording: someone types `streamlit run app.py` (or opens the live app link), and the QURIE app loads. Text overlay: **"Live demo · real patient CT scans"**

---

## 1:23–2:15 · Demo and results: MUNTAZIR

👤 **MUNTAZIR:**
> "This is QURIE running on a real patient."

**B-roll (1:26):** screen recording of the Streamlit app. Pick **P03 · left breast** in the sidebar. The blue box appears: *"The quantum circuit found the exact best plan."* Highlight it with a box/arrow overlay.

> "The quantum circuit runs, and finds the **exact best plan**. A normal computer double-checks all 495 options, and it's a perfect match."

**B-roll (1:36):** in the app, drag the **"Protect the lung ↔ protect the heart"** slider to the heart side. The plan and the dose map change. Then switch the patient to **P04 · right breast**.

> "Doctors can shift the priority between heart and lung, and QURIE re-plans instantly. It works for left and right breast."

**B-roll (1:48):** `figures/fig4_methods.png`. Animate a highlight on the blue "100%" bars, then the orange "25%" bar.

> "We tested it on **48 real patients** from a public hospital dataset. QURIE found the exact best plan in **100% of runs**. The standard quantum approach managed only 25%. That's the difference our constraint design makes."

👤 *(face)*
> "And the part we're proudest of..."

**B-roll (2:05):** `figures/fig7_angles_vs_hospital.png`. Overlay text: **"47 / 48"**

> "We never told QURIE what the doctors did. Yet in **47 out of 48 patients**, it chose the same angles the real radiotherapists used."

---

## 2:15–2:52 · Honesty and vision: ABUBAKAR

👤 **ABUBAKAR:**
> "We want to be honest. At 12 qubits, a normal computer is still fast, and that's exactly why we could prove every answer is correct."

**B-roll (2:23):** animated scaling overlay, numbers counting up:
**"12 angles → 495 plans"**
**"36 angles → 377,000 plans"**
**"72 angles → 1.5 billion plans"**

> "But real clinics choose from many more angles and couch positions, which means billions of combinations. That's where quantum optimisers are expected to matter, and QURIE is already built for it: a real clinical problem, real patients, verified quantum results."

👤 *(face, all three in frame if possible, or quick cuts of each)*
> "Marie Curie gave us radiation. QURIE helps aim it. Thank you."

---

## 2:52–2:58 · End card (no talking, 6 s)
**On screen:**
**QURIE: QUantum Radiotherapy Inverse-planning Engine**
Abubakar · Zeeshan · Muntazir · FAST-NUCES Islamabad
GitHub: `github.com/Abubakar123421/qiskitHackthan-quantum-radiotherapy-Inverse-planning-engine`
Live app: https://qurie-app.streamlit.app  ·  Website: https://abubakar123421.github.io/qiskitHackthan-quantum-radiotherapy-Inverse-planning-engine/
Qiskit Fall Fest 2026

---

## B-roll checklist (record or export these first)
| # | File / clip | Used at |
|---|---|---|
| 1 | `figures/fig1_dose_maps.png` | 0:08, 0:16 |
| 2 | Animated 12-beam chest overlay (Canva/CapCut) | 0:45 |
| 3 | `figures/architecture.png` | 0:58 |
| 4 | `figures/demo_circuit.png` | 1:10 |
| 5 | Screen recording: app loading | 1:15 |
| 6 | Screen recording: P03 result, priority slider, switch to P04 | 1:26–1:48 |
| 7 | `figures/fig4_methods.png` | 1:48 |
| 8 | `figures/fig7_angles_vs_hospital.png` | 2:05 |
| 9 | Scaling numbers overlay | 2:23 |
| 10 | End card | 2:52 |

Optional extra: a 2-second shot of the terminal running `python demo.py` showing **"MATCH: the quantum circuit found the true optimum"**. Drop it in at 1:30 if you have time.

## Filming tips
- Phone in landscape, at eye level, facing a window, plain wall behind. Use earphones with a mic for audio.
- Learn the lines in chunks and talk naturally; small wording changes are fine. **Keep the numbers exact:** 48 patients, 100%, 25%, 47 of 48, 495 plans.
- 2–3 takes per section; keep the best.
- Big captions on every number. Auto-captions in CapCut.
- If the edit runs over 3:00, cut the 0:00 cold open text first, then the P04 switch.

## If a judge asks...
- **"Is it faster than a normal computer?"** No, not at 12 qubits, and we say so. The normal computer is our referee; the value is a verified formulation built to scale.
- **"Would it run on real quantum hardware?"** The deep version is too big for today's noisy machines (about 2,000 two-qubit gates). Shallow versions (1–3 layers) are closer to hardware-ready and still found the best plan in 93–100% of runs in simulation.
- **"Did you overfit?"** The circuit has only 2 tuned numbers per depth, set on one patient and reused unchanged on 47 others, including 23 right-sided.
- **"Is the dose realistic?"** It's a simplified 2D model. It's good for comparing plans, not for clinical dose numbers. Real planning is 3D.
- **"Why not just use the penalty method?"** It wastes up to 60% of shots on invalid plans and found the best plan only 25% of the time; ours keeps every shot valid.
