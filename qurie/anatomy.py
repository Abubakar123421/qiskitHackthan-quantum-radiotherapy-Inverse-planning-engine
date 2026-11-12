"""2D chest model (one axial CT-like slice) for left-sided breast cancer.

Image convention (same as a CT viewer, "radiological view"):
  - top of the image  = front of the patient (anterior)
  - right of the image = patient's LEFT side
So a LEFT breast tumour sits top-right, and the heart sits just behind it.

Every structure is a boolean mask on a regular grid.  One pixel = PIXEL_CM cm.
"""
from dataclasses import dataclass, field
import numpy as np

PIXEL_CM = 0.4  # 1 pixel = 4 mm


@dataclass
class Anatomy:
    body: np.ndarray
    organs: dict = field(default_factory=dict)  # name -> bool mask (tumour = "PTV")
    pixel_cm: float = PIXEL_CM
    source: str = "phantom"
    density: np.ndarray = None   # relative to water (1.0); lungs ~0.3. Used for beam attenuation.

    def __post_init__(self):
        if self.density is None:
            self.density = self.body.astype(float)

    @property
    def shape(self):
        return self.body.shape

    def centroid(self, name):
        ys, xs = np.nonzero(self.organs[name])
        return np.array([xs.mean(), ys.mean()])


def _ellipse(shape, cx, cy, ax, ay):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0


def make_phantom(ny=90, nx=120):
    """Simplified adult female chest slice (~48 cm wide).

    Sizes/positions are rough textbook proportions, NOT a real patient.
    """
    s = (ny, nx)
    torso = _ellipse(s, 60, 50, 54, 32)
    breast_L = _ellipse(s, 86, 24, 17, 12)   # patient's left breast (image right)
    breast_R = _ellipse(s, 34, 24, 17, 12)
    body = torso | breast_L | breast_R

    heart = _ellipse(s, 68, 50, 13, 11)
    lung_L = _ellipse(s, 88, 54, 13, 19) & ~heart
    lung_R = _ellipse(s, 32, 54, 14, 20) & ~heart
    cord = _ellipse(s, 60, 75, 3, 3)
    ptv = _ellipse(s, 88, 22, 5, 4.5)          # tumour + safety margin (PTV)

    organs = {
        "PTV": ptv,
        "Heart": heart & ~ptv,
        "Lung_L": lung_L & ~ptv,
        "Lung_R": lung_R & ~ptv,
        "SpinalCord": cord,
        "Breast_R": breast_R & ~torso,
    }
    healthy = body.copy()
    for m in organs.values():
        healthy &= ~m
    organs["OtherTissue"] = healthy
    density = body.astype(float)
    density[lung_L | lung_R] = 0.3
    return Anatomy(body=body, organs=organs, source="phantom", density=density)

