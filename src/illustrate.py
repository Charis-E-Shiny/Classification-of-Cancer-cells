"""Schematic illustrations for the README.  Run:  python -m src.illustrate

IMPORTANT: the dataset contains NO images, only 30 numbers per sample that were measured from
microscope images of cell nuclei. The nuclei drawn here are SCHEMATIC reconstructions driven by
those measurements (radius -> size, concavity -> outline irregularity, symmetry -> elongation,
texture -> speckle). They are illustrations, not real micrographs.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.data_loader import ROOT, SEED, load_data

FIG = ROOT / "reports" / "figures"
BG, NUC, EDGE = "#f3d9e4", "#7a4fa3", "#3d1f63"      # H&E-style pink background / purple nucleus
BEN, MAL = "#2a7ab9", "#c0392b"


def nucleus_outline(row, rng, n=360):
    th = np.linspace(0, 2 * np.pi, n)
    R = row["mean radius"] / 12.0
    amp = 0.7 * row["mean concavity"] + 0.4 * row["mean concave points"]
    r = np.ones_like(th)
    for k in range(3, 10):                                  # irregular bumps and dents
        r += (amp / k ** 0.5) * rng.uniform(0.4, 1) * np.sin(k * th + rng.uniform(0, 2 * np.pi))
    stretch = 1 + 3 * max(row["mean symmetry"] - 0.15, 0)   # more asymmetric -> more elongated
    ang = rng.uniform(0, np.pi)
    x, y = R * r * np.cos(th) * stretch, R * r * np.sin(th) / stretch
    return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang), th, r


def draw_nucleus(ax, row, seed, title=None, color=None):
    rng = np.random.default_rng(seed)
    x, y, _, _ = nucleus_outline(row, rng)
    ax.set_facecolor(BG); ax.fill(x, y, color=NUC, alpha=0.85, ec=EDGE, lw=2)
    n_dots = int(row["mean texture"] * 6)                    # texture -> chromatin speckle
    px, py = rng.normal(0, 0.35 * row["mean radius"] / 12, (2, n_dots))
    from matplotlib.path import Path
    inside = Path(np.c_[x, y]).contains_points(np.c_[px, py])
    ax.scatter(px[inside], py[inside], s=6, c="#2b1245", alpha=0.6)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, color=color)


def gallery(X, y):
    rng = np.random.default_rng(SEED)
    picks = {0: rng.choice(X.index[y == 0], 5, replace=False), 1: rng.choice(X.index[y == 1], 5, replace=False)}
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.2))
    for r, cls in enumerate([0, 1]):
        for c, idx in enumerate(picks[cls]):
            row = X.loc[idx]
            draw_nucleus(axes[r, c], row, int(idx),
                         f"sample {idx}\nradius {row['mean radius']:.1f} | concavity {row['mean concavity']:.2f}",
                         BEN if cls == 0 else MAL)
        axes[r, 0].set_ylabel(["BENIGN", "MALIGNANT"][r], fontsize=13, fontweight="bold", color=[BEN, MAL][r])
    fig.suptitle("Schematic nuclei drawn from real dataset measurements (illustrations, not micrographs)\n"
                 "Malignant nuclei tend to be larger, more irregular and more variable", fontsize=12)
    plt.tight_layout(h_pad=2.2); plt.savefig(FIG / "nuclei_gallery.png", dpi=140); plt.close()


def measurements(X, y):
    row = X[y == 1].median(); rng = np.random.default_rng(3)
    x, yy, th, r = nucleus_outline(row, rng)
    fig, ax = plt.subplots(figsize=(11, 7)); ax.set_facecolor(BG)
    ax.fill(x, yy, color=NUC, alpha=0.85, ec=EDGE, lw=2.5)
    ax.plot(x, yy, color="#ffd54f", lw=1.8, ls="--")
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        i = np.argmin(abs(th - a)); ax.plot([0, x[i]], [0, yy[i]], color="white", lw=0.8, alpha=0.8)
    dents = [i for i in range(1, len(r) - 1) if r[i] < r[i - 1] and r[i] < r[i + 1] and r[i] < 0.97]
    ax.scatter(x[dents], yy[dents], s=80, c="#ff5252", zorder=5, ec="white")
    ax.set_xlim(-5.4, 5.4); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    box = dict(boxstyle="round", fc="white", ec="#888")
    def note(text, target, pos, ha):
        ax.annotate(text, xy=target, xytext=pos, ha=ha, va="center", fontsize=9, bbox=box, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2, shrinkA=0, shrinkB=2))
    pick = lambda score, ids=None: max(ids if ids is not None else range(len(x)), key=lambda i: score(x[i], yy[i]))
    i_rad = pick(lambda a, b: -a + b)                       # boundary point towards the top-left
    i_per = pick(lambda a, b: a + b)                        # top-right
    i_sym = pick(lambda a, b: -a - 0.1 * abs(b))            # far left
    d_left = pick(lambda a, b: -a - b, dents); d_right = pick(lambda a, b: a - b, dents)
    note("RADIUS: average distance from the\ncentre to the outline (white lines)", (x[i_rad] / 2, yy[i_rad] / 2), (-5.3, 3.0), "left")
    note("PERIMETER: length of the outline\nAREA: size of the purple region", (x[i_per], yy[i_per]), (5.3, 3.0), "right")
    note("SYMMETRY, COMPACTNESS,\nFRACTAL DIMENSION: shape\nregularity and outline roughness", (x[i_sym], yy[i_sym]), (-5.3, 0.2), "left")
    note("TEXTURE: variation of grey\nvalues inside the nucleus", (0.7, 0.1), (5.3, 0.2), "right")
    note("CONCAVE POINTS: number of\ninward dents (red dots)", (x[d_left], yy[d_left]), (-5.3, -2.9), "left")
    note("CONCAVITY: how deep the dents go\nSMOOTHNESS: local radius variation", (x[d_right], yy[d_right]), (5.3, -2.9), "right")
    ax.set_title("The 10 measurements taken per nucleus (schematic)\n"
                 "Each is summarised as mean, standard error and worst value = 30 features", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "feature_measurements.png", dpi=140); plt.close()


def three_cells(X, y):
    """Normal reference (conceptual) vs benign and malignant medians (from the data)."""
    normal = X[y == 0].median().copy()
    normal[["mean radius", "mean concavity", "mean concave points", "mean symmetry", "mean texture"]] = [9.0, 0.01, 0.01, 0.15, 12.0]
    cells = [("NORMAL breast cell", "reference illustration, not from the dataset", normal, "#2e8b57",
              ["Small, round nucleus", "Smooth, even outline", "Evenly stained chromatin"]),
             ("BENIGN tumour cell", f"median of {int((y == 0).sum())} benign samples", X[y == 0].median(), BEN,
              ["Slightly larger, still regular", "Mostly smooth outline", "Uniform appearance"]),
             ("MALIGNANT cancer cell", f"median of {int((y == 1).sum())} malignant samples", X[y == 1].median(), MAL,
              ["Larger nucleus", "Irregular outline with dents", "Uneven, coarse chromatin"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4))
    for ax, (t, sub, row, c, bullets) in zip(axes, cells):
        draw_nucleus(ax, row, 11); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
        ax.set_title(t, fontsize=13, fontweight="bold", color=c, pad=22)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=8.5, color="#444")
        vals = f"radius {row['mean radius']:.1f} | concavity {row['mean concavity']:.2f}"
        ax.set_xlabel(vals + "\n\n" + "\n".join("- " + b for b in bullets), fontsize=9.5, labelpad=8)
    fig.suptitle("From normal tissue to cancer: how the nucleus changes (schematic, drawn from measurements)", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.25, wspace=0.06)
    plt.savefig(FIG / "normal_benign_malignant.png", dpi=140); plt.close()


def separation(X, y):
    feats = ["worst concave points", "worst perimeter", "mean concavity", "mean texture"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, f in zip(axes, feats):
        for cls, c, l in [(0, BEN, "benign"), (1, MAL, "malignant")]:
            ax.hist(X.loc[y == cls, f], bins=25, alpha=0.6, color=c, label=l)
        ax.set_title(f, fontsize=10); ax.set_yticks([])
    axes[0].legend(); fig.suptitle("Malignant and benign samples differ clearly in these measurements "
                                   "(overlap = where errors come from)", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "feature_separation.png", dpi=140); plt.close()


def pipeline_diagram():
    steps = ["569 samples\n30 measurements", "Stratified split\n80% train / 20% test", "Pipeline:\nscale -> classifier",
             "5-fold CV on train\npick best model", "Pick threshold\n(98% recall, CV only)",
             "Final test\n(used once)", "Explain +\nfalse-negative review"]
    fig, ax = plt.subplots(figsize=(15, 2.8)); ax.axis("off"); ax.set_xlim(0, 7.6); ax.set_ylim(0, 1)
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i * 1.08 + 0.02, 0.25), 0.95, 0.5, boxstyle="round,pad=0.02",
                                    fc="#e8f1fa" if i != 5 else "#fde8e5", ec="#355c7d"))
        ax.text(i * 1.08 + 0.495, 0.5, s, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i * 1.08 + 0.99, 0.5), ((i + 1) * 1.08 + 0.01, 0.5), arrowstyle="-|>", mutation_scale=14))
    ax.text(3.8, 0.9, "The test set stays untouched until the last step, so reported results are not inflated",
            ha="center", fontsize=10, style="italic")
    plt.savefig(FIG / "pipeline_diagram.png", dpi=140, bbox_inches="tight"); plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    gallery(X, y); three_cells(X, y); measurements(X, y); separation(X, y); pipeline_diagram()
    print("Wrote nuclei_gallery, normal_benign_malignant, feature_measurements, feature_separation, pipeline_diagram")


if __name__ == "__main__":
    main()
