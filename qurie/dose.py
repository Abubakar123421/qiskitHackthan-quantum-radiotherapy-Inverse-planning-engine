"""Very simple photon-beam dose model (teaching-level, not clinical).

Each candidate beam:
  - comes in from gantry angle theta (0 deg = from the front / anterior,
    90 deg = from the patient's left, clockwise in the image),
  - is a straight strip aimed at the tumour centre, as wide as the tumour + margin,
  - loses intensity with depth inside the body: exp(-MU * depth), where depth is weighted
    by tissue density taken from the CT scan (lung is mostly air, so it attenuates less).
A beam's dose map is scaled so that its average dose inside the tumour is 1.
"""
import numpy as np

MU_PER_CM = 0.05  # effective attenuation of a ~6 MV photon beam in water

# 12 candidate beam directions (= 12 qubits) for a LEFT-sided target: two 75-degree arcs that
# face the left chest wall, every 15 degrees.  Beams entering from the far side would have to
# cross the whole chest first, so they are not offered.
CANDIDATES_LEFT = np.array([90, 105, 120, 135, 150, 165, 285, 300, 315, 330, 345, 0])


def target_side(anat):
    """"L" if the target lies on the patient's left (= right half of the image), else "R"."""
    xs_body = np.nonzero(anat.body)[1]
    return "L" if anat.centroid("PTV")[0] > xs_body.mean() else "R"


def candidate_angles(anat):
    """Candidate beam directions for this patient; mirrored automatically for a right-sided target."""
    if target_side(anat) == "L":
        return CANDIDATES_LEFT.copy()
    return (360 - CANDIDATES_LEFT) % 360


def mirror_anatomy(anat):
    """Flip a patient left-to-right (used to test right-sided planning)."""
    from .anatomy import Anatomy
    organs = {k: v[:, ::-1].copy() for k, v in anat.organs.items() if k != "Lung_ipsi"}
    if "Lung_L" in organs and "Lung_R" in organs:
        organs["Lung_L"], organs["Lung_R"] = organs["Lung_R"], organs["Lung_L"]
    out = Anatomy(body=anat.body[:, ::-1].copy(), organs=organs, pixel_cm=anat.pixel_cm,
                  source=str(anat.source) + "-mirrored", density=anat.density[:, ::-1].copy())
    if getattr(anat, "ct_hu", None) is not None:
        out.ct_hu = anat.ct_hu[:, ::-1].copy()
    return out


def beam_direction(theta_deg):
    """Unit vector the radiation TRAVELS along (source -> patient), image coords (x, y)."""
    t = np.deg2rad(theta_deg)
    # source at angle theta measured clockwise from "front" (top of image)
    src = np.array([np.sin(t), -np.cos(t)])   # where the source sits relative to target
    return -src


def beam_dose(anat, theta_deg, margin_px=1.5, mu=MU_PER_CM):
    ny, nx = anat.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    target = anat.centroid("PTV")
    u = beam_direction(theta_deg)
    perp = np.array([-u[1], u[0]])

    # half-width of the beam = tumour extent across the beam + margin
    ys, xs = np.nonzero(anat.organs["PTV"])
    across = (xs - target[0]) * perp[0] + (ys - target[1]) * perp[1]
    half_w = np.abs(across).max() + margin_px

    dist_perp = np.abs((xx - target[0]) * perp[0] + (yy - target[1]) * perp[1])
    profile = np.clip(half_w + 0.5 - dist_perp, 0.0, 1.0)  # soft 1-pixel edge (penumbra)

    # "radiological depth": walk back towards the source and add up tissue density
    # (so air-filled lung attenuates the beam much less than muscle or bone)
    step = 0.5
    n_steps = int(np.hypot(ny, nx) / step) + 2
    depth_px = np.zeros_like(xx)
    density = anat.density
    for s in range(1, n_steps):
        px = np.rint(xx - s * step * u[0]).astype(int)
        py = np.rint(yy - s * step * u[1]).astype(int)
        inside = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        hit = np.zeros_like(xx)
        hit[inside] = density[py[inside], px[inside]]
        depth_px += hit * step
    depth_cm = depth_px * anat.pixel_cm

    dose = profile * np.exp(-mu * depth_cm) * anat.body
    dose /= dose[anat.organs["PTV"]].mean()
    return dose


def all_beams(anat, angles):
    return np.stack([beam_dose(anat, a) for a in angles])  # (n_beams, ny, nx)
