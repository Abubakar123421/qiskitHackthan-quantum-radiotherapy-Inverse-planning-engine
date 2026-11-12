"""All figures for the report / video.  Static PNGs, light background."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # light -> dark
MAGENTA = "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 150,
})
DOSE_CMAP = LinearSegmentedColormap.from_list(
    "dose", [(1, 0.93, 0.85, 0.0), (0.98, 0.66, 0.42, 0.55), (0.92, 0.41, 0.20, 0.8), (0.55, 0.16, 0.05, 0.92)])


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- anatomy + dose
def _draw_slice(ax, anat, D, angles, title, subtitle):
    if getattr(anat, "ct_hu", None) is not None:
        ax.imshow(anat.ct_hu, cmap="gray", vmin=-600, vmax=400)
    else:
        ax.imshow(anat.density, cmap="gray", vmin=0, vmax=1.6)
    ax.imshow(np.ma.masked_less(D, 0.03), cmap=DOSE_CMAP, vmin=0, vmax=1.1)
    for name, col, ls in [("PTV", "#ffffff", "-"), ("Heart", MAGENTA, "-"),
                          ("Lung_L", "#6da7ec", "-"), ("Lung_R", "#6da7ec", "-")]:
        if name in anat.organs and anat.organs[name].any():
            ax.contour(anat.organs[name], [0.5], colors=col, linewidths=1.4, linestyles=ls)
    cx, cy = anat.centroid("PTV")
    ny, nx = anat.shape
    for a in angles:
        t = np.deg2rad(a)
        ux, uy = np.sin(t), -np.cos(t)                           # towards the beam source
        # stop the line 4 px before the picture edge
        lims = [((nx - 5 - cx) / ux) if ux > 1e-9 else ((4 - cx) / ux) if ux < -1e-9 else np.inf,
                ((ny - 5 - cy) / uy) if uy > 1e-9 else ((4 - cy) / uy) if uy < -1e-9 else np.inf]
        L = max(min(lims), 6)
        ax.plot([cx, cx + L * ux], [cy, cy + L * uy], color="#fff6c2", lw=1.1, ls=(0, (4, 3)))
        ax.text(cx + (L - 3) * ux, cy + (L - 3) * uy, f"{a:.0f}°", color="#1a1a19", fontsize=7.5,
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="#fff6c2", ec="none"))
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)
    ax.set_title(title, color=INK)
    ax.text(0, -0.03, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_visible(False)


def plan_maps(anat, plans, path, suptitle):
    """plans = [(title, dose_map, angles, metrics_dict), ...]"""
    fig, axes = plt.subplots(1, len(plans), figsize=(4.4 * len(plans), 4.3))
    for ax, (title, D, angles, m) in zip(np.atleast_1d(axes), plans):
        sub = (f"heart {100*m['Heart']['mean']:.0f}%  ·  near lung {100*m['Lung_ipsi']['mean']:.0f}%  ·  "
               f"target covered (V95) {100*m['PTV']['V95']:.0f}%")
        _draw_slice(ax, anat, D, angles, title, sub)
    handles = [plt.Line2D([], [], color="#b9b8b2", lw=2, label="target (white outline)"),
               plt.Line2D([], [], color=MAGENTA, lw=2, label="heart"),
               plt.Line2D([], [], color="#6da7ec", lw=2, label="lungs"),
               plt.Line2D([], [], color="#d9c96a", lw=1.2, ls="--", label="beam directions"),
               plt.Rectangle((0, 0), 1, 1, color="#eb6834", alpha=0.8, label="radiation dose (darker = more)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.9, "Mean organ dose as % of the prescribed target dose. Top of image = front of patient; "
             "image right = patient's left.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.82, wspace=0.05)
    _save(fig, path)


def dvh(anat, plans, path):
    """Dose-volume histogram: for each organ, share of the organ receiving at least dose d."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    d = np.linspace(0, 1.2, 241)
    organs = [("PTV", "target", BLUE), ("Heart", "heart", ORANGE), ("Lung_ipsi", "lung on treated side", AQUA)]
    styles = ["-", (0, (4, 2)), (0, (1, 1.5))]
    for (pname, D), ls in zip(plans, styles):
        for key, _, col in organs:
            v = D[anat.organs[key]]
            ax.plot(100 * d, [100 * (v >= t).mean() for t in d], color=col, ls=ls, lw=2)
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=l) for _, l, c in organs]
               + [plt.Line2D([], [], color=INK2, ls=ls, lw=2, label=p[0]) for p, ls in zip(plans, styles)],
               fontsize=8.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.13))
    ax.set_xlabel("dose (% of prescription)")
    ax.set_ylabel("% of organ receiving at least this dose")
    ax.set_xlim(0, 120); ax.set_ylim(0, 102)
    ax.set_title("Dose-volume histogram: target stays covered, heart curve drops")
    _save(fig, path)


# ----------------------------------------------------------------------------- quantum results
def topn_curve(show, path, n_valid):
    """Chance that ONE measurement lands within the best N plans."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ranks = np.arange(1, n_valid + 1)
    ax.plot(ranks, 100 * ranks / n_valid, color=GRAY, ls=(0, (4, 2)), lw=1.6)
    ax.annotate("random guessing", (300, 100 * 300 / n_valid), xytext=(-4, -30), textcoords="offset points",
                ha="right", fontsize=9, color=INK2)
    depths = sorted(show["xy"])
    cols = BLUE_RAMP[-len(depths):]
    for p_, col in zip(depths, cols):
        ax.plot(ranks, 100 * np.cumsum(show["xy"][p_]), color=col, lw=2)
        ax.annotate(f"p={p_}", (1, 100 * show["xy"][p_][0]), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK)
    pen_p = max(show["penalty"])
    pen = 100 * np.cumsum(show["penalty"][pen_p])
    ax.plot(ranks, pen, color=ORANGE, lw=2)
    ax.annotate(f"penalty QAOA, p={pen_p}", (30, pen[29]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.55, n_valid); ax.set_ylim(0, 102)
    ax.set_xlabel(f"N  (plans ranked best to worst, out of {n_valid} valid plans)")
    ax.set_ylabel("chance one measurement is in the top N (%)")
    fig.suptitle("One shot of the quantum circuit usually returns a near-best plan", x=0.02, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.02, 0.905, "Blue: QURIE (XY-mixer QAOA) at circuit depth p.  Orange: standard penalty QAOA.  "
             "Dashed: random guessing.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.86)
    _save(fig, path)


def depth_curves(per_patient, depths, n_valid, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, key, ylab, title in [
            (axes[0], "p_optimal", "chance one shot = exact best plan (%)", "Hitting the exact optimum"),
            (axes[1], "approx_ratio", "approximation ratio (1.0 = perfect)", "Average quality of measured plans")]:
        for variant, col, lab in [("xy", BLUE, "XY mixer (QURIE)"), ("penalty", ORANGE, "penalty QAOA")]:
            M = np.array([[r["qaoa"][variant][str(p_)][key] for p_ in depths] for r in per_patient])
            if key == "p_optimal":
                M = 100 * M
            med, lo, hi = np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0)
            ax.fill_between(depths, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(depths, med, color=col, marker="o", ms=5, mec=SURFACE, mew=1.5)
            ax.annotate(lab, (depths[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        if key == "p_optimal":
            ax.axhline(100 / n_valid, color=GRAY, ls=(0, (4, 2)), lw=1.4)
            ax.annotate("random guess", (depths[0], 100 / n_valid), xytext=(0, -11),
                        textcoords="offset points", fontsize=8.5, color=INK2)
            ax.set_yscale("log")
            ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"])
        else:
            rnd = np.median([r["random_approx_ratio"] for r in per_patient])
            ax.axhline(rnd, color=GRAY, ls=(0, (4, 2)), lw=1.4)
            ax.annotate("random guess", (depths[0], rnd), xytext=(0, -11), textcoords="offset points",
                        fontsize=8.5, color=INK2)
            ax.set_ylim(0.6, 1.005)
        ax.set_xscale("log"); ax.set_xticks(depths); ax.set_xticklabels([str(d) for d in depths])
        ax.minorticks_off()
        ax.set_xlim(depths[0] * 0.9, depths[-1] * 2.6)
        ax.set_xlabel("circuit depth p (number of QAOA layers)")
        ax.set_ylabel(ylab); ax.set_title(title)
    fig.suptitle(f"Deeper circuits find better plans; keeping the k-beam rule inside the circuit matters most  "
                 f"(median and middle 50% of {len(per_patient)} patients)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.42)
    _save(fig, path)


def method_bars(rows, path, n_patients, shots):
    """rows = [(label, success_pct, gap_pct, kind)], kind in {quantum, penalty, classical}"""
    col = {"quantum": BLUE, "penalty": ORANGE, "classical": GRAY}
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, idx, xlabel, title, fmt in [
            (axes[0], 1, "patients / runs where the exact best plan was found (%)", "Finds the true optimum", "{:.0f}%"),
            (axes[1], 2, "average cost above the optimum (%)", "How far off when it misses", "{:.1f}%")]:
        vals = [r[idx] for r in rows]
        ax.barh(y, vals, height=0.56, color=[col[r[3]] for r in rows])
        for yi, v in zip(y, vals):
            ax.text(v + 0.012 * max(max(vals), 1e-9), yi, fmt.format(v), va="center", fontsize=9, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel(xlabel); ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.16 + 1e-9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="QURIE quantum (XY mixer)"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="quantum, penalty method"),
               plt.Rectangle((0, 0), 1, 1, color=GRAY, label="classical")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"All methods against the exact answer on {n_patients} real patients "
                 f"(quantum: best of {shots} shots)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.86, wspace=0.08)
    _save(fig, path)


def tradeoff(all_pts, pareto, picks, clinical, path, pid):
    """Heart dose vs left-lung dose for every valid plan, and how the best plan moves with the heart weight."""
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.scatter(100 * all_pts[:, 0], 100 * all_pts[:, 1], s=9, color="#c9c8c2", lw=0, label="every valid 4-beam plan")
    P = np.array([[h, l] for _, h, l in pareto])
    ax.plot(100 * P[:, 0], 100 * P[:, 1], color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
            label="exact best plan as priority shifts lung → heart")
    Q = np.array(picks)
    ax.scatter(100 * Q[:, 0], 100 * Q[:, 1], s=110, facecolors="none", edgecolors=ORANGE, lw=1.8,
               label="QURIE quantum pick (same weights)", zorder=5)
    w_lo, w_hi = pareto[0][0], pareto[-1][0]
    ax.annotate("protect lung most", (100 * P[0, 0], 100 * P[0, 1]), xytext=(8, -10),
                textcoords="offset points", fontsize=9)
    ax.annotate("protect heart most", (100 * P[-1, 0], 100 * P[-1, 1]), xytext=(8, 2),
                textcoords="offset points", fontsize=9)
    if clinical is not None:
        ax.scatter([100 * clinical[0]], [100 * clinical[1]], marker="X", s=90, color=INK, zorder=6,
                   label="hospital's angles (in our dose model)")
    ax.set_xlabel("mean heart dose (% of prescription)")
    ax.set_ylabel("mean left-lung dose (% of prescription)")
    ax.set_title(f"Competing objectives: sparing the heart costs lung dose (patient {pid})")
    ax.legend(fontsize=8.5, loc="upper right")
    _save(fig, path)


def angle_match(per_patient, path):
    sides = [("L", "Left-sided patients (heart behind the target)"), ("R", "Right-sided patients")]
    groups = [[r for r in per_patient if r["side"] == sd] for sd, _ in sides]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True,
                             gridspec_kw={"width_ratios": [max(len(g), 1) for g in groups]})
    for ax, (sd, title), grp in zip(axes, sides, groups):
        for x0, r in enumerate(grp):
            ax.scatter([x0] * len(r["optimal_angles"]), r["optimal_angles"], s=36, color=BLUE, zorder=3,
                       edgecolors=SURFACE, lw=1.2)
            ax.scatter([x0] * len(r["clinical"]["angles"]), r["clinical"]["angles"], marker="x", s=50,
                       color=INK, lw=1.6, zorder=4)
        ax.set_xticks(range(len(grp))); ax.set_xticklabels([r["pid"] for r in grp], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(0, 361, 45)); ax.set_ylim(-12, 372)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("gantry angle (degrees)")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="QURIE's best 4 angles"),
                        plt.Line2D([], [], marker="x", ls="", color=INK, ms=7, mew=1.8,
                                   label="angles the hospital actually used")],
               fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Without being told, the optimiser lands on the same angle groups the doctors use",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.05)
    _save(fig, path)


# ----------------------------------------------------------------------------- architecture
def architecture(path):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec="#b9b8b2", lw=1))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.35)

    def arrow(x1, y1, x2, y2, lab=None, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     color="#6d6c68", lw=1.3, connectionstyle=f"arc3,rad={rad}"))
        if lab:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, lab, ha="center", fontsize=7.5, color=INK2)

    C1, C2, C3 = "#e9f1fc", "#fdeee6", "#e8f6f0"
    box(2, 44, 21, 16, "1  Patient data", "real CT slice + doctors'\noutlines (target, heart,\nlungs), 48 patients\nreal_data.py", C3)
    box(27, 44, 21, 16, "2  Dose model", "12 candidate beams\n→ dose map of each beam\n(CT-density attenuation)\ndose.py", C3)
    box(52, 44, 21, 16, "3  QUBO builder", "cost = target coverage\n+ heart / lung dose,\nrule: exactly k beams\nqubo.py", C3)
    box(77, 44, 21, 16, "4  Ising Hamiltonian", "x = (1 − Z)/2\n12 qubits, Z and ZZ terms\n(SparsePauliOp)\nqubo.py", C1)
    arrow(23.6, 53, 26.4, 53); arrow(48.6, 53, 51.4, 53); arrow(73.6, 53, 76.4, 53)

    box(44, 20, 54, 20, "5  Quantum solver: QAOA in Qiskit 2.x   (qaoa_solver.py)",
        "start: superposition of all valid k-beam plans (Dicke state)\n"
        "p layers of [cost layer] + [XY mixer: swaps one beam for another]\n"
        "angles: linear ramp, 2 numbers tuned on one patient, reused on all\n"
        "StatevectorSampler: 100 shots, keep the cheapest plan\n"
        "comparison variant: standard X mixer + penalty term", C1)
    arrow(87.5, 43.4, 87.5, 40.6)

    box(2, 20, 38, 20, "6  Classical references   (classical.py)",
        "exact brute force over all 495\nvalid plans (the referee)\n"
        "greedy · simulated annealing · random search\n"
        "evenly spread beams · hospital's own angles", C2)
    arrow(60, 43.4, 30, 40.6, rad=0.0)

    box(18, 2, 64, 14, "7  Evaluation and outputs   (run_experiment.py, plots.py)",
        "quantum vs exact: approximation ratio, chance of the optimum, valid-shot rate, shots needed\n"
        "clinical view: heart / lung dose, target coverage, dose maps, dose-volume histogram\n"
        "trade-off curve (heart vs lung)  ·  results/*.json, *.csv  ·  figures/*.png", "#f3f2ee")
    arrow(72, 19.4, 62, 16.6); arrow(24, 19.4, 38, 16.6)
    ax.text(50, 61.2, "QURIE architecture: from a patient's CT slice to a verified beam plan",
            ha="center", fontsize=11.5, fontweight="bold")
    _save(fig, path)
