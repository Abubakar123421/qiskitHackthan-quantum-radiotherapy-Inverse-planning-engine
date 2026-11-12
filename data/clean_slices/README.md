# QURIE clean dataset: real breast radiotherapy slices

This folder has **48 real breast-cancer patients**:
- **25 left-sided:** this is the **main case**, because the heart sits behind the target.
- **23 right-sided:** a **contrast case**. Here the heart is far from the target, and the main organ at risk is the right lung.

The `side` column in `patients.csv` tells you which is which.

For each patient there is one 2D axial CT slice, already prepared for the QURIE pipeline (dose model → QUBO → QAOA). It is meant as a drop-in replacement for the phantom (`make_phantom()`). Load an `.npz` and you get the same kind of masks, on the same 4 mm grid.

Everything is anonymous. It contains no names, patient IDs, dates or DICOM files.

## Contents
```
README.md          this file
patients.csv       one row per patient (side, clinical info, slice info, clinical plan)
slices/PXX.npz     one compressed numpy file per patient   (np.load)
previews/PXX.png   CT + outlines + clinical dose + clinical beam lines (for checking / slides)
load_example.py    10-line example (numpy + matplotlib only)
```

## What is in each `slices/PXX.npz`
| Key | Shape / type | Meaning |
|---|---|---|
| `ct_hu` | (rows, cols) int16 | CT in Hounsfield units, **4 mm pixels** |
| `body` | (rows, cols) uint8 0/1 | outline of the patient (everything inside the skin) |
| `ctv` | uint8 0/1 | **target** (CTV): the chest wall / breast region to be treated. This replaces the "PTV" of the phantom. |
| `heart` | uint8 0/1 | heart |
| `lung_l`, `lung_r` | uint8 0/1 | left and right lung |
| `clinical_dose_gy` | float32 | the dose the hospital actually planned on this slice, in Gy (total for that plan). **Use it only for comparison.** |
| `ctv_centroid_rc` | (2,) float | (row, col) of the CTV centre on the 4 mm grid. This is a good point to aim the beams at (isocentre). |
| `clinical_gantry_deg` | (n,) float | beam angles the hospital used (see "Angles" below) |
| `fractions` | int | number of treatment sessions in that plan |
| `pixel_mm` | float | 4.0 |
| `slice_z_mm` | float | position of this slice along the body (head–foot) axis |
| `*_native` | 512×512 | the same arrays at the original CT resolution (`native_pixel_mm`, about 0.7–1.0 mm). Ignore these unless you need finer detail. |

The 4 mm grids are cropped to the body plus a 2 cm border, so they are about 58–80 × 82–109 pixels, much like the phantom's 90×120.

```python
import numpy as np, pandas as pd
meta = pd.read_csv("patients.csv")
left_ids = meta.loc[meta.side == "L", "id"]          # main case
d = np.load("slices/P01.npz")
body, ptv, heart = d["body"].astype(bool), d["ctv"].astype(bool), d["heart"].astype(bool)
lung_l, lung_r = d["lung_l"].astype(bool), d["lung_r"].astype(bool)
iso = d["ctv_centroid_rc"]          # aim beams here
```

## Orientation and angles (important)
- **Row 0 is the front of the patient** (anterior). Rows increase towards the back.
- **The image's right side is the patient's LEFT.** This is the standard CT convention and matches the phantom.
  - The heart is always towards the image's right.
  - On left-sided patients the target is also on the image's right.
  - On right-sided patients the target is on the image's left.
- **Never flip right-sided patients to make extra "left" cases.** Flipping moves the heart to the wrong side, so the anatomy stops being realistic.
- **Angles:** all patients were scanned lying on their back, head first (HFS). The clinical gantry angles use the standard IEC convention: **0° = beam from the front, 90° = beam from the patient's left.** This is the same as the QURIE convention, so you can compare the clinical angles with QAOA's chosen angles directly.
  - Unit vector from the isocentre towards the beam source, in (row, col): `(-cos θ, sin θ)`.
  - In `previews/`, the dashed lines are the clinical beams drawn with this formula; they lie along the treated chest wall, which confirms the convention.

## patients.csv columns
| Column | Meaning |
|---|---|
| `id` | patient ID (P01, P02, …) |
| `side` | `L` or `R` |
| `side_note` | empty, unless the spreadsheet's side label was corrected (see below) |
| `age`, `T_stage`, `N_stage`, `energy_MV`, `surgery` | clinical info from the original spreadsheet (MRM = modified radical mastectomy; `mas` as in the source). P14 has no clinical info. |
| `fractions`, `clinical_angles_deg` | the hospital's plan |
| `slice_index`, `slice_z_mm`, `native_pixel_mm`, `grid_shape` | which slice was taken, and its size |
| `ctv_area_cm2`, `heart_area_cm2`, `heart_ctv_min_distance_mm` | sizes on this slice, and how close the heart is to the target |
| `clinical_ctv_mean_gy`, `clinical_heart_mean_gy` | mean planned dose to the target and the heart on this slice |
| `qa_body_match` | fraction of pixels where the body outline agrees with the CT. Every patient here also passed a whole-scan slice-alignment check. |

## How the data was made
- **Source:** the public dataset *"CT images and radiotherapy treatment planning of patients with breast cancer"* (Data in Brief, 2017). It has 52 patients, with outlines drawn by doctors and real treatment plans. **Cite it in the report** (take the exact reference and DOI from the download page) and check its licence before publishing.
- **Kept:** 48 of the 52 patients.
- **Side labels corrected (P31 and P50):** the spreadsheet says right, but **both** the outlines and the hospital's beam angles (about 120°/300°, the left tangent pair) say left. They are included as left-sided, and this is noted in `side_note`.
- **Excluded:**
  - **P12**: one CT slice in the source is an empty or corrupt file, so the slices can't be lined up with the outlines.
  - **P25**: its only plan is a single 0° beam that puts about 0 Gy on the target, so it is not the breast plan and the clinical comparison would be meaningless.
  - **P46**: the spreadsheet says left-sided, but the outlines put the target on the patient's right, and its single 0° beam can't settle which is correct.
  - **P52**: no organ outlines in the source.
- **Slice choice:** for each patient, the slice where **(target area × heart area)** is largest, i.e. where the heart–target trade-off is strongest.
- **Checks:**
  - The outline slices line up with the CT slices: this was verified against the original RTSTRUCT z-positions, and on every patient the best match over the whole scan is at zero shift.
  - The clinical dose is about **1.9 Gy per session** on every plan, which shows the dose map is correctly placed and in Gy.
- **Resampling:** masks were averaged over each 4 mm block and thresholded at 0.5. CT and dose were averaged over the same blocks.

## Things to know (put these in the report's limitations)
- The target is the **chest wall / breast after surgery (CTV)**, not a tumour inside an intact breast.
- **There are no outlines for the spinal cord or the other breast.**
  - Either drop those terms from the objective, or approximate them yourself.
  - Example for the spinal cord: bone pixels (`ct_hu > 200`) near the midline at the back of the body.
- `clinical_dose_gy` is the **total** dose of the hospital plan, **not** a separate dose per beam. You still need the QURIE dose model to build the per-beam matrix for the QUBO.
- The clinical plans are mostly **2 opposed tangent beams**:
  - **left-sided:** about 300° + 120°;
  - **right-sided:** about 60° + 240°.

  Some plans add a third, front-facing beam: about 330–342° on the left, about 19–25° on the right. It passes near the heart, which is why those patients show a higher heart dose.
- **Plans with 10–13 sessions** show about 13–24 Gy on the target, and **plans with 25 sessions** show about 37–49 Gy. Both are about 1.9 Gy per session.
- These are 2D slices taken from a 3D treatment. They are fine for comparing beam angles, but they are not a full clinical dose evaluation.

The build script that made this folder from the raw data is kept separately, in the team's repo as `tools/build_clean_dataset.py`. The raw data contains real patient names, so **never** commit or share the raw DICOM files.
