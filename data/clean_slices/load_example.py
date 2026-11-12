"""Minimal example: load one patient slice and show it. Needs only numpy + matplotlib.

    python load_example.py            # shows P01
    python load_example.py P18        # any id from patients.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

pid = sys.argv[1] if len(sys.argv) > 1 else "P01"
d = np.load(Path(__file__).parent / "slices" / f"{pid}.npz")

ct = d["ct_hu"]                                  # (rows, cols) int16, Hounsfield units, 4 mm pixels
organs = {k: d[k].astype(bool) for k in ["body", "ctv", "heart", "lung_l", "lung_r"]}
print(pid, "grid", ct.shape, "pixel", float(d["pixel_mm"]), "mm",
      "| clinical beams", d["clinical_gantry_deg"], "deg |",
      {k: int(v.sum()) for k, v in organs.items()}, "pixels")

plt.imshow(ct, cmap="gray", vmin=-400, vmax=400)
for name, color in [("ctv", "red"), ("heart", "magenta"), ("lung_l", "cyan"), ("lung_r", "lime"), ("body", "yellow")]:
    plt.contour(organs[name], [0.5], colors=color, linewidths=1)
plt.title(f"{pid}: top = anterior, image right = patient LEFT")
plt.axis("off")
plt.show()
