"""Load the real breast radiotherapy slices (data/clean_slices, 48 patients: 25 left, 23 right).

Source: Rezaei et al., "CT images and radiotherapy treatment planning of patients with
breast cancer: A dataset", Data in Brief 13 (2017) 390-395, doi:10.1016/j.dib.2017.06.002,
data: https://data.mendeley.com/datasets/vj83jj6fvt  (doi:10.17632/vj83jj6fvt.3)
"""
import csv
from pathlib import Path
import numpy as np
from .anatomy import Anatomy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_slices"


def list_patients():
    return sorted(p.stem for p in (DATA_DIR / "slices").glob("P*.npz"))


def patient_table():
    with open(DATA_DIR / "patients.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def load_patient(pid):
    d = np.load(DATA_DIR / "slices" / f"{pid}.npz")
    ctv = d["ctv"].astype(bool)
    body = d["body"].astype(bool) | ctv | d["heart"].astype(bool) \
        | d["lung_l"].astype(bool) | d["lung_r"].astype(bool)
    organs = {
        "PTV": ctv,                                   # target (chest wall after surgery)
        "Heart": d["heart"].astype(bool) & ~ctv,
        "Lung_L": d["lung_l"].astype(bool) & ~ctv,
        "Lung_R": d["lung_r"].astype(bool) & ~ctv,
    }
    side = patient_table()[pid]["side"]                # "L" or "R" = which breast is treated
    other = body.copy()
    for m in organs.values():
        other &= ~m
    organs["OtherTissue"] = other
    # alias (not in the cost weights, so not double counted): the lung on the treated side
    organs["Lung_ipsi"] = organs["Lung_L"] if side == "L" else organs["Lung_R"]
    # tissue density relative to water, straight from the CT numbers (Hounsfield units)
    density = np.clip(1.0 + d["ct_hu"].astype(float) / 1000.0, 0.0, 1.6) * body
    anat = Anatomy(body=body, organs=organs, pixel_cm=float(d["pixel_mm"]) / 10, source=pid,
                   density=density)
    anat.side = side
    anat.ct_hu = d["ct_hu"]
    anat.clinical_dose_gy = d["clinical_dose_gy"]
    anat.clinical_angles = np.array(d["clinical_gantry_deg"], float)
    anat.fractions = int(d["fractions"])
    return anat
