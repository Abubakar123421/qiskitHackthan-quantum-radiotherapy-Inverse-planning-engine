# QURIE website (interactive demo for judges)

A static site, with no build step and no server. `index.html` + `data.js`.

* **Open locally:** run `python -m http.server 8000` in this folder, then open http://localhost:8000
* **Deploy on Vercel:** New Project → import the GitHub repo → **Root Directory: `website`** → Framework preset **Other** → Deploy.
* **Or GitHub Pages / Netlify:** publish the `website/` folder as-is.

`data.js` is generated from the real results by `python tools/export_web.py` (run after `run_experiment.py`).
The "Measure 100 times" button draws shots in the browser from the exact output probabilities of the Qiskit QAOA circuit (XY mixer) for that patient and depth. The live Qiskit run itself is in `app.py` (Streamlit) and `demo.py`.
