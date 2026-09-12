"""Paper figures from the stored results (decoder2.json, decoder.json, symmetry.json, preds/).
Writes PNG + PDF to microns_ambiguity/outputs/paper/ and an HTML page with captions.
Run: python -m microns_ambiguity.paper_figures
"""
import base64
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "microns_ambiguity", "outputs", "paper")
TEX = os.environ.get("PAPER_TEX") == "1"          # PAPER_TEX=1: PDFs without in-figure titles, for the LaTeX paper
if TEX:
    OUT = os.path.join(OUT, "tex")
    matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
os.makedirs(OUT, exist_ok=True)
M = json.load(open(os.path.join(ROOT, "microns_ambiguity", "outputs", "decoder2.json")))
A = json.load(open(os.path.join(ROOT, "allen", "outputs", "decoder.json")))
SYM = json.load(open(os.path.join(ROOT, "microns_ambiguity", "outputs", "symmetry.json")))
PREDS = os.path.join(ROOT, "allen", "outputs", "preds")
MPREDS = os.path.join(ROOT, "microns_ambiguity", "outputs", "preds")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False, "figure.constrained_layout.use": True})
C = {"iv": "#1f5f8b", "twin": "#e8a33d", "base": "#b8b8b8", "ref": "#444444", "mix": "#2a9d8f", "single": "#b5533c",
     "s0": "#1f5f8b", "s1": "#e8a33d", "s2": "#2a9d8f"}
FIGS = []  # (filename, title, caption, section)


def save(fig, name, title, caption, section):
    if not TEX:
        fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig); FIGS.append((name, title, caption, section)); print("wrote", name)


def chance(ax, x_text=1.42, allen=False):
    """45° chance line for angular error (errors of an uninformed decoder are uniform on 0–90°);
    on Allen also the 7.5° mean disagreement a perfect decoder shows against labels on a 30° grid."""
    ax.axhline(45, color="#888", lw=0.9, zorder=0)
    if x_text is not None: ax.text(x_text, 45.8, "chance", fontsize=6.5, color="#666", va="bottom")
    if allen:
        ax.axhline(7.5, color="#888", lw=0.8, ls="--", zorder=0)
        if x_text is not None: ax.text(x_text, 8.3, "label grid", fontsize=6.5, color="#666", va="bottom")


def have(src, tags):
    """Tags present in a results dict (runs still in flight are skipped)."""
    return [t for t in tags if t in src]


def scatter_tag(d, tag):
    p = os.path.join(d, tag + ".npz"); return p if os.path.exists(p) else None


def ori_scatter(ax, path, title, allen=False):
    """Decoded vs true preferred orientation for every scored neuron."""
    z = np.load(path); P, y = z["P"], z["y"]
    if y.ndim == 2: y = y[:, 0]
    ok = np.isfinite(y); P, y = P[ok], y[ok]
    th = (np.rad2deg(np.arctan2(P[:, 1], P[:, 0])) / 2) % 180; y = y % 180
    x = y + (np.random.default_rng(0).uniform(-4, 4, len(y)) if allen else 0)
    ax.scatter(x, th, s=3, color="#1f5f8b" if not allen else C["mix"], alpha=0.35 if allen else 0.25, lw=0, rasterized=True)
    ax.plot([0, 180], [0, 180], "-", color="#999", lw=0.8, zorder=0)
    ax.set_xlim(-8, 188); ax.set_ylim(0, 180); ax.set_xticks([0, 45, 90, 135, 180]); ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("true preferred orientation (°)"); ax.set_ylabel("decoded (°)"); ax.set_title(title, fontsize=8)
    err = np.abs((th - y + 90) % 180 - 90).mean(); ax.text(0.03, 0.97, f"mean error {err:.1f}°\n{len(y):,} neurons", transform=ax.transAxes, va="top", fontsize=6.8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))


def ms(tags, key):
    v = np.array([M[t][key] for t in tags]); return v.mean(), v.std(), v



# ---------------------------------------------------------------- Fig 1: MICrONS data and pipeline
def fig_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
    from microns_ambiguity.data import Dataset
    from microns_ambiguity.decoder2 import normalize_features
    ds = Dataset(); rng = np.random.default_rng(4)
    INK, MUTED, TR, TE = "#1c2230", "#6a7380", "#1f5f8b", "#e8a33d"
    NC = ["#1f5f8b", "#2a9d8f", "#e8a33d", "#b5533c", "#6a4c93", "#3a7d44"]
    ok = ds.ori_ok & ds.rf_ok; cand = np.flatnonzero(ok); six = rng.choice(cand, 6, replace=False)
    F = normalize_features(ds.func_iv); Gsix = F[six] @ F[six].T
    fig = plt.figure(figsize=(10.8, 6.5))
    TOP, FOOT = 0.905, 0.59                                                # bands: panel titles / footnotes of the top row
    def label(ax, txt, y=None):
        x0 = ax.get_position().x0; fig.text(x0, TOP if y is None else y, txt, fontsize=8.5, weight="bold", va="bottom", color=INK)
    def foot(x0, x1, txt): fig.text((x0 + x1) / 2, FOOT, txt, fontsize=6.5, ha="center", va="top", color="#444")
    def arrow(ax, x0, y0, x1, y1, txt=None, dy=0.05, fs=6.5):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1, transform=ax.transAxes, clip_on=False))
        if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, transform=ax.transAxes, fontsize=fs, ha="center", color=MUTED)
    # ---- a: the volume
    ax = fig.add_axes([0.03, 0.62, 0.24, 0.285]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); label(ax, "a  one mouse, 1 mm³ of cortex, 13 scans")
    dx, dy = 2.2, 1.4; X0, Y0, S = 1.2, 1.2, 5.2
    front = [(X0, Y0), (X0 + S, Y0), (X0 + S, Y0 + S), (X0, Y0 + S)]
    top = [(X0, Y0 + S), (X0 + S, Y0 + S), (X0 + S + dx, Y0 + S + dy), (X0 + dx, Y0 + S + dy)]
    side = [(X0 + S, Y0), (X0 + S + dx, Y0 + dy), (X0 + S + dx, Y0 + S + dy), (X0 + S, Y0 + S)]
    for poly, fc in [(top, "#e6eae8"), (side, "#d5dbd8"), (front, "#f2f4f3")]:
        ax.add_patch(Polygon(poly, closed=True, fc=fc, ec="#8a939c", lw=0.9))
    areas = ds.area.astype(str); amap = {"V1": TR, "RL": "#2a9d8f", "AL": TE, "LM": "#b5533c"}
    for z in np.linspace(0.15, 0.85, 4):                                    # four imaging planes with neurons coloured by area
        yy = Y0 + z * S; ax.plot([X0 + 0.15, X0 + S - 0.15], [yy, yy], color="#8a939c", lw=0.5, alpha=0.6)
        idx = rng.choice(len(areas), 22); xs = rng.uniform(X0 + 0.3, X0 + S - 0.3, 22); ys = yy + rng.uniform(-0.28, 0.28, 22)
        ax.scatter(xs, ys, s=7, c=[amap.get(a_, "#999") for a_ in areas[idx]], ec="none", zorder=3)
    for i, (k, v) in enumerate(amap.items()):
        ax.scatter([0.9 + i * 2.3], [0.35], s=12, c=v, ec="none"); ax.text(1.2 + i * 2.3, 0.35, k, fontsize=6.5, va="center", color="#444")
    foot(0.03, 0.27, "12,894 neurons, co-registered to the EM volume")
    # ---- b: responses -> correlations (real data)
    axb = fig.add_axes([0.31, 0.62, 0.40, 0.285]); axb.axis("off"); label(axb, "b  responses to a shared stimulus → correlations")
    axt = fig.add_axes([0.315, 0.663, 0.20, 0.19])
    T = F[six][:, :120]; T = (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
    for i in range(6): axt.plot(np.arange(120), T[i] * 0.32 + (5 - i), color=NC[i], lw=0.9)
    axt.set_xlim(0, 119); axt.set_ylim(-0.9, 5.9); axt.set_yticks([]); axt.set_xticks([0, 60, 119]); axt.set_xticklabels(["0", "60", "120"], fontsize=6); axt.tick_params(length=2)
    axt.set_xlabel("stimulus bin (oracle movie clips)", fontsize=6.5, labelpad=1); axt.spines["left"].set_visible(False); axt.set_title("six neurons, trial-averaged response", fontsize=6.8, pad=2)
    axg = fig.add_axes([0.585, 0.653, 0.115, 0.195])
    Gm = Gsix.copy(); np.fill_diagonal(Gm, np.nan); axg.imshow(Gm, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    for i in range(6):
        for j in range(6):
            if i != j: axg.text(j, i, f"{Gsix[i, j]:.2f}", ha="center", va="center", fontsize=5, color="k" if abs(Gsix[i, j]) < 0.4 else "w")
    axg.set_xticks(range(6)); axg.set_yticks(range(6)); axg.set_xticklabels([]); axg.set_yticklabels([]); axg.tick_params(length=0)
    for i in range(6):
        axg.add_patch(Rectangle((-1.05, i - 0.35), 0.4, 0.7, fc=NC[i], ec="none", clip_on=False)); axg.add_patch(Rectangle((i - 0.35, -1.05), 0.7, 0.4, fc=NC[i], ec="none", clip_on=False))
    axg.set_title("correlation (Gram)", fontsize=6.8, pad=14)
    arrow(axb, 0.545, 0.5, 0.64, 0.5)
    foot(0.31, 0.71, "full matrix 12,894 × 12,894; the decoder sees only such matrices, never the stimulus")
    # ---- c: contents (real labels of the same six neurons)
    axc = fig.add_axes([0.74, 0.62, 0.24, 0.285]); axc.axis("off"); label(axc, "c  contents (training targets only)")
    axo = fig.add_axes([0.745, 0.643, 0.10, 0.215]); axo.set_xlim(-1.3, 1.3); axo.set_ylim(-1.3, 1.3); axo.set_aspect("equal"); axo.axis("off")
    axo.add_patch(Circle((0, 0), 1.0, fill=False, ec="#c9cfcc", lw=0.8))
    for i, n in enumerate(six):
        th = np.deg2rad(ds.ori[n]); axo.plot([-np.cos(th), np.cos(th)], [-np.sin(th), np.sin(th)], color=NC[i], lw=2.2, solid_capstyle="round")
    for k in range(8):
        th = np.deg2rad(k * 22.5); axo.text(1.18 * np.cos(th), 1.18 * np.sin(th), f"{int(k * 22.5)}°", fontsize=4.8, ha="center", va="center", color="#777")
    axo.set_title("preferred orientation\n(8 classes)", fontsize=6.8, pad=2)
    axr = fig.add_axes([0.865, 0.653, 0.11, 0.195]); axr.set_xlim(-1.05, 1.05); axr.set_ylim(-1.05, 1.05); axr.set_aspect("equal")
    axr.add_patch(Rectangle((-1, -1), 2, 2, fc="#f2f4f3", ec="#8a939c", lw=0.8))
    for i, n in enumerate(six): axr.scatter(ds.rf[n, 0], ds.rf[n, 1], s=28, c=NC[i], ec="white", lw=0.6, zorder=3)
    axr.set_xticks([]); axr.set_yticks([]); [sp.set_visible(False) for sp in axr.spines.values()]; axr.set_title("receptive-field centre\n(screen)", fontsize=6.8, pad=2)
    foot(0.74, 0.98, "5,287 neurons with orientation (gOSI ≥ 0.25) · 11,326 with RF")
    # ---- d: protocol. Full matrix ordered training-first; a population Gram is a diagonal block; two paths through the
    # same decoder: training blocks meet labels in a loss, test blocks are only scored.
    axd = fig.add_axes([0.03, 0.02, 0.95, 0.46]); axd.set_xlim(0, 18.2); axd.set_ylim(0.4, 5.4); axd.set_aspect("equal"); axd.axis("off")
    label(axd, "d  protocol: the decoder input is the Gram of a 512-neuron sample from one half; train with samples of the training half, score samples of the test half", y=0.5)
    bx, by, bw, bh = 0.03, 0.02, 0.95, 0.46; ux = bw / 18.2; uy = ux * 10.8 / 6.5
    y_off = by + (bh - 5.0 * uy) / 2
    def rect(x, y, w, h): return [bx + x * ux, y_off + (y - 0.4) * uy, w * ux, h * uy]
    def arr(x0, y0, x1, y1, col=MUTED, lw=1.0, rad=0.0):
        axd.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=col, lw=lw, connectionstyle=f"arc3,rad={rad}"))
    ORANGE_T = "#b07a1e"
    # (i) the neurons, halved
    for i in range(36):
        cx, cy = 0.5 + (i % 6) * 0.4, 4.4 - (i // 6) * 0.4
        if i % 6 < 3: axd.add_patch(Circle((cx, cy), 0.13, fc=TR, ec="none"))
        else: axd.add_patch(Circle((cx, cy), 0.13, fc="white", ec=TE, lw=1))
    axd.text(0.9, 2.05, "training\nhalf", fontsize=6.2, color=TR, ha="center", va="top"); axd.text(2.1, 2.05, "test\nhalf", fontsize=6.2, color=ORANGE_T, ha="center", va="top")
    axd.text(1.5, 1.3, "all 12,894 neurons,\nsplit into two halves", fontsize=6.0, ha="center", va="top", color="#444")
    arr(2.85, 3.4, 3.35, 3.4)
    # (ii) the full correlation matrix (real 200-neuron excerpt), ordered training-first
    half = rng.permutation(np.flatnonzero(ds.ori_ok & ds.rf_ok)); trn, tst = half[: len(half) // 2], half[len(half) // 2:]
    ex = np.concatenate([rng.choice(trn, 100, replace=False), rng.choice(tst, 100, replace=False)])
    Gf = F[ex] @ F[ex].T; np.fill_diagonal(Gf, np.nan)
    axf = fig.add_axes(rect(3.5, 1.7, 3.3, 3.3)); axf.imshow(Gf, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest"); axf.set_xticks([]); axf.set_yticks([])
    axf.add_patch(Rectangle((-0.5, -0.5), 100, 100, fc=TR, ec=TR, alpha=0.10, lw=1.2)); axf.add_patch(Rectangle((99.5, 99.5), 100, 100, fc=TE, ec=TE, alpha=0.14, lw=1.2))
    axf.add_patch(Rectangle((99.5, -0.5), 100, 100, fc="white", ec="none", alpha=0.78)); axf.add_patch(Rectangle((-0.5, 99.5), 100, 100, fc="white", ec="none", alpha=0.78))
    axf.text(150, 50, "never\nused", ha="center", va="center", fontsize=5.8, color="#777"); axf.text(50, 150, "never\nused", ha="center", va="center", fontsize=5.8, color="#777")
    axf.text(50, 4, "training × training", ha="center", va="top", fontsize=5.8, color=TR, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axf.text(150, 104, "test × test", ha="center", va="top", fontsize=5.8, color=ORANGE_T, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    TB, SB = (40, 20), (140, 150)
    axf.add_patch(Rectangle((TB[1] - 0.5, TB[0] - 0.5), 34, 34, fill=False, ec=TR, lw=1.3, ls=(0, (3, 1.5)))); axf.add_patch(Rectangle((SB[1] - 0.5, SB[0] - 0.5), 34, 34, fill=False, ec=TE, lw=1.3, ls=(0, (3, 1.5))))
    axf.set_title("full correlation matrix, 12,894 × 12,894\n(200 neurons shown, ordered training first)", fontsize=6.3, pad=2)
    axd.text(5.15, 1.45, "one input sample = 512 neurons drawn from one half;\nits Gram is the dashed diagonal block of the full matrix", fontsize=6.0, ha="center", va="top", color="#444")
    # (iii) two paths: training (top) and test (bottom), same decoder
    def sub(rc): r0, c0 = rc; return Gf[r0:r0 + 34, c0:c0 + 34]
    YT, YS = 4.25, 2.15
    for y, blk, col, name in [(YT, TB, TR, "train"), (YS, SB, TE, "score")]:
        axg_ = fig.add_axes(rect(8.0, y - 0.6, 1.2, 1.2)); axg_.imshow(sub(blk), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest"); axg_.set_xticks([]); axg_.set_yticks([]); [sp.set(color=col, lw=1.3) for sp in axg_.spines.values()]
        arr(9.3, y, 9.9, y, col=col, lw=1.1)
        axd.add_patch(FancyBboxPatch((9.9, y - 0.55), 1.9, 1.1, boxstyle="round,pad=0.03,rounding_size=0.18", fc="#2b3038", ec="none", zorder=2))
        axd.text(10.85, y + 0.12, "decoder", fontsize=7, ha="center", va="center", weight="bold", color="white", zorder=9); axd.text(10.85, y - 0.22, "rows = tokens", fontsize=5.6, ha="center", va="center", color="#c9ced6", zorder=9)
        arr(11.9, y, 12.5, y, col=col, lw=1.1)
        for k in range(5): axd.add_patch(Rectangle((12.55, y - 0.45 + k * 0.2), 0.55, 0.12, fc=col, ec="none", alpha=0.45 + 0.12 * k))   # per-row predictions
        axd.text(12.82, y - 0.62, "one prediction per\nneuron of the sample", fontsize=5.4, ha="center", va="top", color="#666")
    arr(6.05, 3.55, 7.95, YT + 0.1, col=TR, lw=0.9, rad=-0.25); arr(6.95, 2.5, 7.95, YS - 0.1, col=TE, lw=0.9, rad=0.15)
    axd.text(8.6, YT + 0.72, "Gram of a sample from\nthe training half (512 × 512)", fontsize=5.8, ha="center", va="bottom", color=TR); axd.text(8.6, YS - 0.72, "Gram of a sample from\nthe test half (512 × 512)", fontsize=5.8, ha="center", va="top", color=ORANGE_T)
    axd.add_patch(FancyArrowPatch((10.85, YT - 0.6), (10.85, YS + 0.62), arrowstyle="-|>", mutation_scale=9, color="#6a7380", lw=1.1)); axd.text(11.0, (YT + YS) / 2, "trained\nweights", fontsize=5.6, ha="left", va="center", color="#555")
    # training path: predictions meet the labels in a loss
    arr(13.2, YT, 13.8, YT, col=TR, lw=1.1)
    axd.add_patch(FancyBboxPatch((13.85, YT - 0.55), 1.55, 1.1, boxstyle="round,pad=0.03,rounding_size=0.18", fc="white", ec=TR, lw=1.1))
    axd.text(14.62, YT + 0.2, "loss", fontsize=7, ha="center", va="center", weight="bold", color=TR); axd.text(14.62, YT - 0.2, "vs. labels of the\nlabelled neurons in it", fontsize=5.4, ha="center", va="center", color="#444")
    axd.text(16.1, YT, "gradient updates\nthe decoder;\nthousands of samples\nper epoch", fontsize=5.6, ha="left", va="center", color="#444")
    # test path: predictions are only scored
    arr(13.2, YS, 13.8, YS, col=TE, lw=1.1)
    axd.add_patch(FancyBboxPatch((13.85, YS - 0.55), 1.55, 1.1, boxstyle="round,pad=0.03,rounding_size=0.18", fc="white", ec=TE, lw=1.1))
    axd.text(14.62, YS + 0.2, "score", fontsize=7, ha="center", va="center", weight="bold", color=ORANGE_T); axd.text(14.62, YS - 0.2, "accuracy / R² vs.\nheld-out labels", fontsize=5.4, ha="center", va="center", color="#444")
    axd.text(16.1, YS, "no gradient;\nthe decoder never\nsees a test label", fontsize=5.6, ha="left", va="center", color="#444")
    axd.text(18.0, 0.95, "epoch chosen on a held-out slice of the training half.  One brain, samples drawn\nwithin one animal, new neurons: the `within` cell of Fig. 4", fontsize=5.8, ha="right", va="top", color="#444")
    fig.suptitle("Fig. 1  MICrONS: from one imaged cortical volume to a label-free per-neuron decoding task", fontsize=9, y=0.985)
    save(fig, "fig1_microns_pipeline", "How the MICrONS task is built",
         "(a) The MICrONS functional-connectomics release (Ding, Fahey, Papadopoulos et al. 2025): one mouse, 13 two-photon scans of a cubic millimetre of visual cortex "
         "(areas V1, RL, AL, LM), 12,894 neurons co-registered to the electron-microscopy volume. (b) The relational substrate. Each neuron is its trial-averaged "
         "response vector to a stimulus all neurons saw (in vivo: 120 bins of the oracle natural-movie clips; digital twin: 4,999 bins compressed to 512 principal "
         "components); shown are six real neurons and their correlation matrix. The oracle clips were shown in all 13 scans, so the matrix is defined across scans (same-scan pairs correlate ~50 % more than cross-scan pairs at matched RF distance; the twin has no such effect). The decoder only ever sees such correlations, never the stimulus. (c) The contents "
         "of the same six neurons: preferred orientation (in vivo, 8 classes, 5,287 neurons with gOSI ≥ 0.25) and receptive-field centre (digital-twin fit, 11,326 "
         "neurons). (d) The protocol, with two terms kept apart. The <i>halves</i>: all 12,894 neurons are split at random into a training half and a test half, "
         "and the full 12,894 × 12,894 correlation matrix, ordered training-first (a real 200-neuron excerpt is shown), is used only in its training × training "
         "and test × test blocks. A <i>sample</i>: 512 neurons drawn at random from one half; its Gram is the corresponding diagonal sub-block (dashed), and that "
         "512 × 512 matrix is the decoder's entire input, one row per neuron, one prediction per row. Training draws thousands of samples per epoch from the "
         "training half; their predictions meet the labels of the labelled neurons in the sample in a loss that updates the decoder. Scoring draws samples from "
         "the test half and compares predictions with held-out labels; no gradient. Labels never enter the input on either path, and the halves share no neuron. "
         "The epoch is chosen on a held-out slice of the training half. One animal, samples drawn within one animal, new neurons: the `within` cell of Fig. 4.", "main")


# ---------------------------------------------------------------- Fig 2: MICrONS label-free decoding
def fig1():
    seeds = {("iv", "ori", "2.2M"): ["c_iv_2M_s0", "c_iv_2M_s1", "c_iv_2M_s2"], ("twin", "ori", "2.2M"): ["c_is_2M_s0", "c_is_2M_s1", "c_is_2M_s2"],
             ("iv", "ori", "17M"): ["c_iv_17M_s0", "c_iv_17M_s1", "c_iv_17M_s2"], ("twin", "ori", "17M"): ["c_is_17M_s0", "c_is_17M_s1", "c_is_17M_s2"],
             ("iv", "rf", "2.2M"): ["es_iv_rf_s0", "es_iv_rf_s1", "es_iv_rf_s2"], ("twin", "rf", "2.2M"): ["es_twin_rf_s0", "es_twin_rf_s1", "es_twin_rf_s2"],
             ("iv", "rf", "17M"): ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"], ("twin", "rf", "17M"): ["g3_is_rf_n512_d512L8", "g9_is_rf_n512_d512L8_s1", "g9_is_rf_n512_d512L8_s2"]}
    single = {("iv", "ori", "0.3M"): "c_iv_03M_s0", ("twin", "ori", "0.3M"): "c_is_03M_s0", ("iv", "rf", "0.3M"): "n512_rel_rf", ("twin", "rf", "0.3M"): "twin_n512_rf"}
    sc = scatter_tag(MPREDS, "c_iv_17M_sp1")
    fig, axes = plt.subplots(1, 3 if sc else 2, figsize=(10.2 if sc else 8, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.9] if sc else [1, 1]))
    for ax, con, key, ylab in [(axes[0], "ori", "err", "orientation error (°), lower is better"), (axes[1], "rf", "r2", "receptive-field position R²")]:
        levels = ["0.3M", "2.2M", "17M"]; w = 0.2
        for gi, sub in enumerate(["iv", "twin"]):
            x0 = gi * 1.0
            for li, lv in enumerate(levels):
                x = x0 + (li - 1) * w
                if lv == "0.3M": v, e = M[single[(sub, con, lv)]][key], 0.0
                else: v, e, _ = ms(seeds[(sub, con, lv)], key)
                ax.bar(x, v, w * 0.9, color=C[sub], alpha=0.45 + 0.27 * li, yerr=e if e > 0 else None, capsize=2, ecolor="k")
                ax.text(x, v + (1.0 if con == "ori" else 0.012), lv, ha="center", va="bottom", fontsize=6.5, rotation=90)
        if con == "ori": chance(ax, x_text=1.42)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["in vivo", "digital twin"]); ax.set_ylabel(ylab)
        ax.set_xlim(-0.55, 1.75); ax.set_ylim(0, {"ori": 50, "rf": 0.6}[con])
    if sc: ori_scatter(axes[2], sc, "in vivo, 17M, held-out neurons")
    fig.suptitle("Fig. 2  MICrONS: per-neuron content decoded from the population correlation matrix alone (512 neurons, no labels, no reference)", fontsize=9)
    save(fig, "fig2_microns_decoder", "MICrONS label-free per-neuron decoding",
         "Left: mean absolute error of the decoded preferred orientation (degrees, orientation is defined modulo 180°) on held-out neurons of the same animal, "
         "for decoders of 0.3M, 2.2M and 17M parameters; error bars are the s.d. over 3 seeds. A decoder that knows nothing has errors spread evenly over 0–90°, mean 45° (grey line). "
         "Middle: R² of the decoded receptive-field centre. The decoder sees only the standardised correlation matrix of 512 sampled neurons: neither labels nor a reference "
         "population enter the input, and the test neurons never influenced model selection. Right: decoded against true orientation for every held-out in vivo neuron (17M decoder). "
         "Orientation is read to 25° in vivo and 20° on the twin; receptive-field position at R² 0.23 in vivo and 0.48 on the twin.", "main")


# ---------------------------------------------------------------- Fig 2: symmetry
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    v = SYM["func_iv"]; G = np.array(v["class_gram"]); Gc = np.array(v["circulant"]); K = v["K"]
    lab = [f"{int(i * 180 / K)}" for i in range(K)]
    for ax, mat, ttl in [(axes[0], G, "in-vivo class-Gram (8 × 8)"), (axes[1], Gc, "its circulant projection")]:
        mm = mat.copy(); np.fill_diagonal(mm, np.nan)
        im = ax.imshow(mm, cmap="viridis"); ax.set_title(ttl); ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(lab, fontsize=6); ax.set_yticklabels(lab, fontsize=6)
        ax.set_xlabel("preferred orientation (°)")
    ax = axes[2]; w = 0.35
    for i, sub in enumerate(["func_iv", "func_is"]):
        s = SYM[sub]
        ax.bar(i - w / 2, s["acc_raw"], w, color=C["iv" if sub == "func_iv" else "twin"], label="raw class-Gram" if i == 0 else None)
        ax.bar(i + w / 2, s["acc_circulant"], w, color=C["iv" if sub == "func_iv" else "twin"], alpha=0.4, label="circulant projection" if i == 0 else None)
        ax.plot([i - w / 2, i + w / 2], [s["acc_mod_raw"], s["acc_mod_circulant"]], "k.", ms=7, label="modulo D8" if i == 0 else None)
        ax.text(i, 1.12, f"circulant var. {s['frac_var_circulant']:.2f}", ha="center", fontsize=7)
    ax.axhline(1 / 8, color=C["base"], lw=1); ax.text(1.4, 1 / 8 + 0.02, "chance", fontsize=7, color="#666")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["in vivo", "digital twin"]); ax.set_ylim(0, 1.24); ax.set_ylabel("class matching accuracy"); ax.legend(loc="center right", fontsize=7, bbox_to_anchor=(1.0, 0.55))
    ax.set_title("matching over all 8! relabellings", fontsize=9)
    fig.suptitle("Fig. 3  What fixes the absolute orientation frame: the non-circulant part of the class relations", fontsize=9)
    save(fig, "fig3_symmetry", "Symmetry: the anisotropy fixes the frame",
         "Left: mean correlation between orientation classes in vivo (diagonal masked); the structure is close to circulant (71 % of the variance in vivo, "
         "81 % in the twin): neighbouring orientations correlate, orthogonal ones anti-correlate. Middle: the circulant projection of the same matrix. "
         "Right: matching a held-out half's class-Gram to the reference over all 8! relabellings. With the raw matrix the true labelling wins (accuracy 1.0). "
         "After the circulant projection, absolute accuracy collapses to chance while accuracy modulo the dihedral group D8 (dots) stays at 1.0: the circulant "
         "part fixes the structure up to rotation and reflection, and only the residual anisotropy (a cardinal bias that differs between areas) pins which class is 0°. "
         "This is what a label-free decoder has to use to output absolute orientation.", "main")



# ---------------------------------------------------------------- Fig 3: regime schematic
def fig_regimes():
    from matplotlib.patches import FancyBboxPatch
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2))
    TR, TE, MUTED = "#1f5f8b", "#e8a33d", "#8a939c"
    MX = [0.3, 2.75, 5.2, 7.65]; W = 2.2; Y0, Y1 = 1.2, 4.9            # four mice per panel
    ROWS = [1.75, 2.5, 3.6, 4.35]                                       # two test rows (lower), two training rows (upper), with a gap
    cells = [((0, 0), "within", "training animals · single-animal populations", "train"),
             ((0, 1), "pooledwithin", "training animals · mixed populations", "train"),
             ((1, 0), "cross", "held-out animals · single-animal populations", "heldout"),
             ((1, 1), "pooledcross", "held-out animals · mixed populations", "heldout")]
    for (r, c), name, desc, split in cells:
        ax = axes[r, c]; ax.set_xlim(0, 10.2); ax.set_ylim(0.1, 6.4); ax.axis("off"); P = {}
        for mi, x0 in enumerate(MX):
            ax.add_patch(FancyBboxPatch((x0, Y0), W, Y1 - Y0, boxstyle="round,pad=0.04,rounding_size=0.3", fc="#f2f4f3", ec=MUTED, lw=1))
            ax.text(x0 + W / 2, Y1 + 0.18, f"mouse {mi + 1}", ha="center", fontsize=6.8, color="#555")
            gx, gy = np.meshgrid(np.linspace(x0 + 0.45, x0 + W - 0.45, 3), ROWS)
            xy = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.07, 0.07, (12, 2))
            is_test = (xy[:, 1] < 3.05) if split == "train" else np.full(12, mi >= 2)
            P[mi] = (xy, is_test)
            ax.scatter(xy[~is_test, 0], xy[~is_test, 1], s=15, c=TR, ec="none", zorder=3)
            ax.scatter(xy[is_test, 0], xy[is_test, 1], s=15, fc="white", ec=TE, lw=1.1, zorder=3)
        def loop(pts, col, lab, lab_y):
            pts = np.concatenate(pts); pad = 0.22
            x0, y0 = pts.min(0) - pad; x1, y1 = pts.max(0) + pad
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.35", fill=False, ec=col, lw=1.4, ls=(0, (4, 2)), zorder=4))
            ax.text((x0 + x1) / 2, lab_y, lab, ha="center", fontsize=6.8, color=col)
        tr = lambda m: P[m][0][~P[m][1]]; te = lambda m: P[m][0][P[m][1]]
        if name == "within":
            loop([tr(0)], TR, "training population", 0.75); loop([te(0)], TE, "test population", 0.4)
        elif name == "pooledwithin":
            loop([tr(m) for m in range(4)], TR, "training population (all mice, training neurons)", 0.75)
            loop([te(m) for m in range(4)], TE, "test population (all mice, test neurons)", 0.4)
        elif name == "cross":
            loop([tr(0)], TR, "training population", 0.6); loop([te(2)], TE, "test population", 0.6)
        else:
            loop([tr(0), tr(1)], TR, "training population (training mice)", 0.6); loop([te(2), te(3)], TE, "test population (held-out mice)", 0.6)
        ax.text(5.1, 6.35, f"`{name}`", ha="center", fontsize=9, family="monospace", weight="bold", va="top")
        ax.text(5.1, 5.85, desc, ha="center", fontsize=7.5, color="#444", va="top")
    fig.text(0.5, 0.005, "filled = training neurons (labels are targets)   ·   hollow = test neurons (scored; never used for training or model selection)   ·   dashed box = one sampled population, the unit the Gram is computed on", ha="center", fontsize=6.8, color="#444")
    fig.suptitle("Fig. 4  Allen decoder regimes: where the test neurons come from × what one population contains", fontsize=9)
    save(fig, "fig4_regimes", "The four Allen decoder regimes",
         "Two independent choices define a run. The split (rows) decides where test neurons come from: each animal's cells halved into training and test neurons "
         "(top; tests generalisation to new neurons of seen brains) or whole animals held out of training and model selection (bottom; new brains). The population "
         "(columns) decides what one sample is: the Gram is computed on a sampled population of 128–1024 neurons drawn from one animal (left; a within-circuit "
         "correlation matrix) or from several animals (right; most entries are between-animal correlations, which exist only because all mice saw the same stimuli). "
         "Training populations are drawn from training neurons and test populations from test neurons in every regime. MICrONS is one animal, so all MICrONS results "
         "are in the top-left cell.", "main")


# ---------------------------------------------------------------- Fig 5: Allen 2x2 orientation
def fig3():
    sc = scatter_tag(PREDS, "c_pc_2M_plain")
    fig, axes = plt.subplots(1, 2 if sc else 1, figsize=(10.2 if sc else 7.4, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]) if sc else None)
    ax = axes[0] if sc else axes
    cells = [("within", "training animals\nsingle-animal pop.", have(A, ["c_wi_2M_sp0", "c_wi_2M_sp1", "c_wi_2M_sp2"]), C["single"]),
             ("pooledwithin", "training animals\nmixed pop.", have(A, ["c_pw_2M_sp0", "c_pw_2M_sp1", "c_pw_2M_sp2"]), C["mix"]),
             ("cross", "held-out animals\nsingle-animal pop.", have(A, ["c_cr_2M_sp0", "c_cr_2M_sp1", "c_cr_2M_sp2"]), C["single"]),
             ("pooledcross", "held-out animals\nmixed pop.", have(A, ["c_pc_2M_plain", "c_pc_2M_plain_sp1", "c_pc_2M_plain_sp2"]), C["mix"])]
    for i, (reg, lab, tags, col) in enumerate(cells):
        vals = [A[t]["err"] for t in tags]
        ax.bar(i, np.mean(vals), 0.6, color=col, alpha=0.8)
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
        ax.plot(i + np.array(jitter), vals, "o", color="k", ms=4, mfc="white")
        ax.text(i, 2.5, f"`{reg}`", ha="center", fontsize=5.8, family="monospace", color="white")
    chance(ax, x_text=3.45, allen=True)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in cells], fontsize=7.5); ax.set_ylabel("orientation error (°), lower is better"); ax.set_ylim(0, 50); ax.set_xlim(-0.6, 4.4)
    if sc: ori_scatter(axes[1], sc, "`pooledcross`, 2M, held-out mice", allen=True)
    fig.suptitle("Fig. 5  Allen (33 mice, grating relations): orientation is readable only from populations that mix animals", fontsize=9)
    save(fig, "fig5_allen_2x2_orientation", "Allen 2 × 2: split × population, orientation",
         "Left: mean angular error of the label-free decoder (2M parameters, 256 neurons per population) on labelled test cells for the four regimes. Split: test neurons from the training animals "
         "(each animal's cells halved) or from 9 held-out animals. Population: each sampled population (the unit the Gram is computed on) drawn from one animal or from several. "
         "Points: the three splits (of neurons within each animal for the top row, of animals for the bottom row). Grey line: 45° chance; dashed line: the 7.5° mean disagreement "
         "a perfect decoder would show against labels that sit on a 30° grid. Single-animal populations are within 3° of chance whether the animal was seen in training or not; "
         "mixed populations are 8–10° below it in both splits, so transfer to unseen brains has no detectable cost. Right: decoded against true orientation for the held-out mice; "
         "the six columns are the six grating orientations.", "main")


# ---------------------------------------------------------------- Fig 4: Allen RF mouse-level
def fig4():
    from allen.data import Allen
    ds = Allen(session="C", movie="both")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    runs = {"single-animal populations (`cross`)": ["g8_rf_cross_fine", "g11_rf_cross_fine_sp1", "g11_rf_cross_fine_sp2"],
            "mixed populations (`pooledcross`)": ["g4_rf_n256_d256L4", "g11_rf_pooledcross_sp1", "g11_rf_pooledcross_sp2"]}
    for ax, (name, tags) in zip(axes[:2], runs.items()):
        offs, rel = [], []
        for si, tag in enumerate(tags):
            z = np.load(os.path.join(PREDS, tag + ".npz")); idx, P, y = z["idx"], z["P"], z["y"]; ok = np.isfinite(y).all(1); idx, P, y = idx[ok], P[ok], y[ok]
            trm = ~np.isin(ds.mouse.astype(str), A[tag]["test_mice"]) & ds.rf_ok; mu, sd = np.nanmean(ds.rf[trm], 0), np.nanstd(ds.rf[trm], 0); pred = P * sd + mu
            m = ds.mouse[idx]
            for mm in np.unique(m):
                k = m == mm; offs.append((pred[k].mean(0), y[k].mean(0), si)); rel.append(np.c_[pred[k] - pred[k].mean(0), y[k] - y[k].mean(0)])
        offs = np.array([(a[0][0], a[0][1], a[1][0], a[1][1], a[2]) for a in offs]); rel = np.concatenate(rel)
        for si in range(3):
            k = offs[:, 4] == si
            ax.plot(offs[k, 3], offs[k, 1], "o", color=C[f"s{si}"], ms=5, label=f"split {si}")
            ax.plot(offs[k, 2], offs[k, 0], "s", color=C[f"s{si}"], ms=4, mfc="none")
        rx = np.corrcoef(offs[:, 2], offs[:, 0])[0, 1]; ry = np.corrcoef(offs[:, 3], offs[:, 1])[0, 1]
        rrx = np.corrcoef(rel[:, 0], rel[:, 2])[0, 1]; rry = np.corrcoef(rel[:, 1], rel[:, 3])[0, 1]
        lo, hi = min(offs[:, :4].min(), 0) - 5, offs[:, :4].max() + 5; ax.plot([lo, hi], [lo, hi], "-", color="#bbb", lw=0.8, zorder=0)
        ax.set_title(name, fontsize=8.5); ax.set_xlabel("true mean RF position (°)"); ax.set_ylabel("predicted mean (°)"); ax.set_ylim(lo, hi + 12)
        ax.text(0.03, 0.97, f"per-mouse mean, 27 mice: r = {rx:.2f} (x), {ry:.2f} (y)\nwithin mouse, {len(rel):,} cells: r = {rrx:.2f}, {rry:.2f}", transform=ax.transAxes, va="top", fontsize=6.8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        if ax is axes[0]: ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Fig. 6  Allen RF: what crosses animals is each animal's screen position, not the retinotopic layout within it", fontsize=9)
    save(fig, "fig6_allen_rf_mouse_level", "Allen RF: a population-level readout",
         "Predicted versus true mean receptive-field position of each held-out mouse (9 mice × 3 mouse splits = 27 points; x squares, y dots), from "
         "decoders trained on single-animal or mixed populations. The per-mouse mean is recovered (r 0.36–0.71; permutation p ≤ 0.03), the position of a neuron "
         "relative to its mouse-mates is not (r ≈ 0.1 over 1,259 cells after centring per mouse). R² on absolute coordinates is therefore not a stable summary "
         "here: it depends on whether the between-mouse spread of a given test set is reproduced at the right scale. With 23–112 RF-labelled cells per mouse the "
         "decoder learns the population-level signal and not the fine one; MICrONS (one animal, 11k labels) shows the same decoder recovers neuron-level RF at R² 0.48.", "main")


# ---------------------------------------------------------------- Fig A1: scaling
def figA1():
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7))
    curves = {("iv", "ori"): [(0.3, "c_iv_03M_s0"), (2.2, ["c_iv_2M_s0", "c_iv_2M_s1", "c_iv_2M_s2"]), (17, ["c_iv_17M_s0", "c_iv_17M_s1", "c_iv_17M_s2"])],
              ("twin", "ori"): [(0.3, "c_is_03M_s0"), (2.2, ["c_is_2M_s0", "c_is_2M_s1", "c_is_2M_s2"]), (17, ["c_is_17M_s0", "c_is_17M_s1", "c_is_17M_s2"]), (57, "c_is_57M_s0")]}
    curves_rf = {("iv", "rf"): [(0.3, "n512_rel_rf"), (2.2, ["es_iv_rf_s0", "es_iv_rf_s1", "es_iv_rf_s2"]), (17, ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"])],
                 ("twin", "rf"): [(0.3, "twin_n512_rf"), (2.2, ["es_twin_rf_s0", "es_twin_rf_s1", "es_twin_rf_s2"]), (7.3, "huge_twin_n512_rf_long"), (17, ["g3_is_rf_n512_d512L8", "g9_is_rf_n512_d512L8_s1", "g9_is_rf_n512_d512L8_s2"]), (57, "g9_is_rf_n512_d768L12")]}
    for ax, cv, key, ttl in [(axes[0], curves, "err", "MICrONS orientation"), (axes[1], curves_rf, "r2", "MICrONS RF (x, y)")]:
        for (sub, con), pts in cv.items():
            xs, ys, es = [], [], []
            for p, t in pts:
                if isinstance(t, list): m, e, _ = ms(t, key)
                else: m, e = M[t][key], 0
                xs.append(p); ys.append(m); es.append(e)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, color=C[sub], label={"iv": "in vivo", "twin": "twin"}[sub], capsize=2)
        ax.set_xscale("log"); ax.set_xticks([0.3, 2.2, 7.3, 17, 57]); ax.set_xticklabels(["0.3", "2.2", "7", "17", "57"]); ax.minorticks_off(); ax.set_xlabel("decoder parameters (M)"); ax.set_ylabel({"err": "orientation error (°)", "r2": "R²"}[key]); ax.set_title(ttl); ax.legend()
        if key == "err": chance(ax, x_text=None); ax.set_ylim(0, 50)
        else: ax.axhline(0, color=C["base"], lw=1, zorder=0)
    ax = axes[2]
    pts = [(2.2, "c_pc_2M_plain"), (17, "c_pc_17M_plain"), (57, "c_pc_57M_plain")]; pts = [(p, t) for p, t in pts if t in A]
    ax.plot([p for p, _ in pts], [A[t]["err"] for _, t in pts], "o-", color=C["mix"], label="mixed pop., 256 neurons")
    if all(t in A for t in ["c_pc_2M_plain_s1", "c_pc_2M_plain_s2"]):
        v = [A[t]["err"] for t in ["c_pc_2M_plain", "c_pc_2M_plain_s1", "c_pc_2M_plain_s2"]]; ax.errorbar([2.2], [np.mean(v)], yerr=[np.std(v)], fmt="none", ecolor="k", capsize=3)
    chance(ax, x_text=None, allen=True); ax.set_xscale("log"); ax.set_xticks([2.2, 17, 57]); ax.set_xticklabels(["2.2", "17", "57"]); ax.minorticks_off(); ax.set_xlabel("decoder parameters (M)"); ax.set_ylabel("orientation error (°)"); ax.set_title("Allen orientation, held-out mice (split 0)"); ax.legend(fontsize=7, loc="lower left"); ax.set_ylim(0, 50)
    fig.suptitle("Fig. A1  Capacity: RF keeps gaining with model size, orientation does not (label counts 11,326 vs 5,287 in MICrONS; 5,529 in Allen)", fontsize=9)
    save(fig, "figA1_scaling_params", "Appendix: scaling with decoder size",
         "Test metric versus number of decoder parameters at fixed population size (512 neurons MICrONS, 256 Allen). Error bars: s.d. over 3 seeds where run. "
         "Every model above 2M memorises its training neurons within 4–17 epochs and is early-stopped on a held-out selection set, so the curve is the gain from stopping early "
         "on a larger model. On MICrONS RF the gain continues to 57M; on orientation the error is flat from 2M upward on both substrates, and on Allen from 2M to 57M.", "appendix")


# ---------------------------------------------------------------- Fig A2: population size
def figA2():
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.6))
    ax = axes[0]
    for tags, lab, col in [(["c_iv_03M_n128", "c_iv_03M_n256", "c_iv_03M_s0", "c_iv_03M_n1024"], "in vivo, 0.3M", C["iv"]), (["c_is_2M_n256", "c_is_2M_s0", "c_is_2M_n1024"], "twin, 2.2M", C["twin"])]:
        tags = have(M, tags); xs = [M[t]["n"] for t in tags]; ys = [M[t]["err"] for t in tags]; ax.plot(xs, ys, "o-", color=col, label=lab)
    chance(ax, x_text=None); ax.set_xscale("log", base=2); ax.set_xlabel("neurons per population"); ax.set_ylabel("orientation error (°)"); ax.set_title("MICrONS orientation"); ax.legend(fontsize=6.5); ax.set_ylim(0, 50)
    ax = axes[1]
    for tags, lab, col, src in [(["n128_rel_rf", "n256_rel_rf", "n512_rel_rf", "n1024_rel_rf"], "MICrONS in vivo, 0.3M", C["iv"], M), (["twin_n256_rf", "big_twin_n512_rf", "big_twin_n1024_rf"], "MICrONS twin, 2.2M", C["twin"], M),
                                (["g4_rf_n256_d256L4_cf50", "g4_rf_n512_d512L8_cf50", "g4_rf_n1024_d512L8_cf50"], "Allen mixed pop., held-out mice (split 0)", C["mix"], A)]:
        xs = [src[t]["n"] for t in tags]; ys = [src[t]["r2"] for t in tags]; ax.plot(xs, ys, "s--", color=col, mfc="white", label=lab)
    ax.axhline(0, color=C["base"], lw=1, zorder=0); ax.set_xscale("log", base=2); ax.set_xlabel("neurons per population"); ax.set_ylabel("R²"); ax.set_title("receptive field"); ax.legend(fontsize=6.5)
    ax = axes[2]
    tags = have(A, ["c_pc_17M_plain", "c_pc_17M_n512", "c_pc_17M_n1024"]); ax.plot([A[t]["n"] for t in tags], [A[t]["err"] for t in tags], "o-", color=C["mix"], label="17M, mixed pop.")
    chance(ax, x_text=None, allen=True); ax.set_xscale("log", base=2); ax.set_xlabel("neurons per population"); ax.set_ylabel("orientation error (°)"); ax.set_title("Allen orientation, held-out mice (split 0)"); ax.legend(fontsize=6.5); ax.set_ylim(0, 50)
    fig.suptitle("Fig. A2  Population size saturates by 512 neurons (MICrONS) and 256 (Allen)", fontsize=9)
    save(fig, "figA2_population_size", "Appendix: population size",
         "Test metric versus the number of neurons in each sampled population (the size of the Gram the decoder sees). Beyond a few hundred neurons the additional companions "
         "add redundant context; on Allen RF, 1024-neuron mixed populations are worse (0.09).", "appendix")


# ---------------------------------------------------------------- Fig A3: training dynamics
def figA4():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    panels = [("c_pc_17M_plain", A, "err", "Allen orientation, 17M, mixed pop."), ("c_is_17M_s0", M, "err", "MICrONS twin orientation, 17M"), ("g3_is_rf_n512_d512L8", M, "r2", "MICrONS twin RF, 17M")]
    for ax, (tag, src, key, ttl) in zip(axes, panels):
        h = src[tag]["history"]; ep = np.arange(len(h)); sgn = -1 if key == "err" else 1
        ax.plot(ep, [x[key] for x in h], "-", color=C["mix"] if src is A else C["twin"], label="test metric")
        ax.plot(ep, [sgn * x["sel"] for x in h], ":", color="k", lw=1, label="selection set")
        b = src[tag]["best_epoch"]; ax.axvline(b, color="#999", lw=0.8); lo_, hi_ = ax.get_ylim(); top = h[b][key] < (lo_ + hi_) / 2
        ax.text(b + 0.8, hi_ if top else lo_ + 0.02 * (hi_ - lo_), f"selected\nepoch {b}", fontsize=6.5, color="#666", va="top" if top else "bottom")
        ax2 = ax.twinx(); ax2.plot(ep, [x["loss"] for x in h], "-", color="#c44", lw=1, alpha=0.7); ax2.set_ylabel("training loss", color="#c44", fontsize=7); ax2.tick_params(axis="y", labelsize=7, colors="#c44"); ax2.spines["right"].set_visible(True)
        ax.set_xlabel("epoch"); ax.set_ylabel({"err": "orientation error (°)", "r2": "R²"}[key]); ax.set_title(ttl, fontsize=8.5)
        if ax is axes[0]: h1, l1 = ax.get_legend_handles_labels(); h1.append(ax2.get_lines()[0]); l1.append("training loss")
    fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7)
    fig.suptitle("Fig. A3  Every large run memorises its training neurons; the reported score is the epoch chosen on a held-out selection set", fontsize=9)
    save(fig, "figA3_training_dynamics", "Appendix: training dynamics and early stopping",
         "Test metric (solid), selection-set metric (dotted; held-out mice on Allen, a held-out 20 % of training neurons on MICrONS) and training loss (red, right axis) "
         "per epoch. The selection set, never the test set, picks the epoch (grey line) and its weights are restored before evaluation. The training loss collapses within "
         "10–20 epochs on every large run while the test error keeps drifting up, so the selected epoch matters.", "appendix")


# ---------------------------------------------------------------- Fig A4: Allen split variance + class-level
def figA5():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    groups = {"17M, 50 % conditions\n(previous standard)": [["c_pc_17M_cf50_sp0_s0", "c_pc_17M_cf50_sp0_s1", "c_pc_17M_cf50_sp0_s2"], ["c_pc_17M_cf50_sp1"], ["c_pc_17M_cf50_sp2"]],
              "2M, plain\n(standard)": [["c_pc_2M_plain", "c_pc_2M_plain_s1", "c_pc_2M_plain_s2"], ["c_pc_2M_plain_sp1"], ["c_pc_2M_plain_sp2"]]}
    for gi, (g, splits) in enumerate(groups.items()):
        for si, tags in enumerate(splits):
            tags = have(A, tags)
            if not tags: continue
            v = [A[t]["err"] for t in tags]; ax.bar(gi + (si - 1) * 0.25, np.mean(v), 0.22, color=C[f"s{si}"], label=f"split {si}" if gi == 0 else None)
            if len(v) > 1: ax.plot(gi + (si - 1) * 0.25 + np.linspace(-0.05, 0.05, len(v)), v, "o", color="k", ms=3, mfc="white")
    chance(ax, x_text=None, allen=True); ax.set_xticks([0, 1]); ax.set_xticklabels(list(groups.keys()), fontsize=8); ax.set_ylabel("orientation error (°), held-out mice"); ax.set_ylim(0, 56); ax.legend(fontsize=7, loc="upper center", ncol=3); ax.set_title("Allen orientation, `pooledcross`, by mouse split", fontsize=8.5)
    ax = axes[1]
    G = json.load(open(os.path.join(ROOT, "allen", "outputs", "geometric_DG_nm1_sg.json")))["summary"]; Gm = json.load(open(os.path.join(ROOT, "allen", "outputs", "geometric_both_sg.json")))["summary"]
    labs = ["within-mouse\nsplit-half", "one mouse →\nanother", "pooled reference →\nheld-out mouse"]
    for i, k in enumerate(["within_mean", "cross_mean", "pooled_mean"]):
        ax.bar(i - 0.18, Gm[k], 0.34, color="#999", label="movie relations" if i == 0 else None); ax.bar(i + 0.18, G[k], 0.34, color=C["mix"], label="grating relations" if i == 0 else None)
    ax.axhline(1 / 6, color="k", lw=0.8); ax.text(2.4, 1 / 6 + 0.005, "chance", fontsize=7); ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel("class-level matching accuracy (K = 6)"); ax.legend(fontsize=7); ax.set_title("Allen class-level test", fontsize=8.5); ax.set_ylim(0.1, 0.3)
    fig.suptitle("Fig. A4  Allen: mouse-split variance of the decoder, and the class-level matching test by relation type", fontsize=9)
    save(fig, "figA4_allen_splits_classlevel", "Appendix: Allen split variance and class-level test",
         "Left: `pooledcross` orientation error on the same three held-out-mouse splits for the previous standard configuration (17M, Gram from 50 % of the conditions) and the "
         "final one (2M, plain); split 0 carries three seeds (points), and the spread between splits exceeds the spread between seeds. Right: the class-level test (6 × 6 class-Gram "
         "matched over all 720 relabellings) with relations from natural movies versus drifting gratings; only the grating relations move every test off chance.", "appendix")


# ---------------------------------------------------------------- Fig A5: error by true orientation
def figA6():
    from microns_ambiguity.balance_check import ang_err
    panels = [(os.path.join(MPREDS, "c_is_17M_sp2.npz"), 8, C["twin"], "MICrONS twin, 17M"), (os.path.join(MPREDS, "c_iv_17M_sp2.npz"), 8, C["iv"], "MICrONS in vivo, 17M"), (os.path.join(PREDS, "c_pc_2M_plain.npz"), 6, C["mix"], "Allen `pooledcross`, 2M")]
    panels = [p for p in panels if os.path.exists(p[0])]
    if not panels: return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.6))
    for ax, (path, K, col, ttl) in zip(np.atleast_1d(axes), panels):
        z = np.load(path); P, y = z["P"], z["y"]
        if y.ndim == 2: y = y[:, 0]
        ok = np.isfinite(y); P, y = P[ok], y[ok] % 180
        th = (np.rad2deg(np.arctan2(P[:, 1], P[:, 0])) / 2) % 180; d = ang_err(th, y); w = 180 / K
        yb = ((y + w / 2) // w % K).astype(int); tb = ((th + w / 2) // w % K).astype(int)
        per = [d[yb == k].mean() for k in range(K)]; frac = np.bincount(yb, minlength=K) / len(y); pf = np.bincount(tb, minlength=K) / len(y)
        x = np.arange(K) * w
        ax.bar(x, per, w * 0.8, color=col, alpha=0.85, label="error of neurons in bin")
        ax.axhline(45, color="#888", lw=0.9, zorder=0); ax.set_ylim(0, 50); ax.set_xticks(x); ax.set_xticklabels([f"{v:.0f}" for v in x], fontsize=7); ax.set_xlabel("true preferred orientation (°)"); ax.set_ylabel("mean error (°)")
        ax2 = ax.twinx(); ax2.plot(x, frac, "o-", color="k", ms=3, lw=0.8, label="fraction of labels"); ax2.plot(x, pf, "s--", color="#c44", ms=3, lw=0.8, label="fraction of predictions"); ax2.set_ylim(0, 0.5); ax2.set_ylabel("fraction", fontsize=7); ax2.tick_params(axis="y", labelsize=7); ax2.spines["right"].set_visible(True)
        ax.set_title(ttl, fontsize=8.5)
        if ax is np.atleast_1d(axes)[0]: h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7)
    fig.suptitle("Fig. A5  The readout is cardinal: neurons near 0° and 90° are decoded well, obliques near chance, and predictions pile up on the cardinal axes", fontsize=9)
    save(fig, "figA5_error_by_orientation", "Appendix: error by true orientation",
         "Mean angular error of the neurons in each true-orientation bin (bars; grey line: 45° chance), with the fraction of labels (black) and of predictions (red) in each bin. "
         "Cardinal orientations are decoded to 11–12° on the twin and 25° on Allen; oblique orientations sit near chance; 63–72 % of predictions fall on 0° or 90°. "
         "The uneven label distribution alone buys little: the best constant prediction scores 41° (MICrONS) and 44° (Allen).", "appendix")


def html():
    css = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f4f6f5;--surface:#ffffff;--ink:#1c2230;--muted:#5b6470;--rule:#d6dbd9;--accent:#1f7f74;--accent2:#c9862a;--plate:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141821;--surface:#1c2230;--ink:#e6e8ec;--muted:#a2a9b4;--rule:#2c3340;--accent:#4fc3b5;--accent2:#f0b45a;--plate:#f4f6f5}}
:root[data-theme="dark"]{--bg:#141821;--surface:#1c2230;--ink:#e6e8ec;--muted:#a2a9b4;--rule:#2c3340;--accent:#4fc3b5;--accent2:#f0b45a;--plate:#f4f6f5}
body{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:16px;line-height:1.5;margin:0}
main{max-width:960px;margin:0 auto;padding:3rem 1.25rem 5rem}
h1{font-family:Fraunces,Georgia,"Times New Roman",serif;font-weight:600;font-size:2.1rem;line-height:1.15;margin:0 0 .6rem;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:Fraunces,Georgia,serif;font-weight:500;font-size:1.35rem;margin:3.5rem 0 1.25rem;padding-bottom:.4rem;border-bottom:1px solid var(--rule);text-wrap:balance}
.stand{max-width:66ch;color:var(--muted);font-size:1.05rem;margin:0 0 2rem}
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem 1.5rem;margin:0 0 1rem;padding:0;list-style:none}
.glance li{border-top:2px solid var(--accent);padding-top:.6rem;font-size:.95rem}
.glance b{display:block;font-family:"IBM Plex Mono",Menlo,monospace;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:.3rem;font-weight:500}
.glance .n{font-family:"IBM Plex Mono",Menlo,monospace;font-variant-numeric:tabular-nums}
figure{display:grid;grid-template-columns:5.5rem 1fr;gap:0 1.25rem;margin:0 0 3rem;align-items:start}
figure .lab{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);padding-top:.35rem;font-weight:500}
figure .lab.ap{color:var(--accent2)}
figure .body{min-width:0}
figure img{display:block;width:100%;background:var(--plate);border:1px solid var(--rule);padding:.5rem;box-sizing:border-box}
figcaption{max-width:70ch;font-size:.95rem;color:var(--ink);margin-top:.75rem}
figcaption b{font-weight:600}
figcaption code,p code{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.84em;background:var(--surface);border:1px solid var(--rule);padding:0 .25em;border-radius:2px}
.meta{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.75rem;color:var(--muted);margin:2.5rem 0 0}
@media (max-width:640px){figure{grid-template-columns:1fr}figure .lab{padding-top:0;margin-bottom:.4rem}}
@media (prefers-reduced-motion:no-preference){figure img{transition:border-color .2s}figure img:hover{border-color:var(--accent)}}
</style>"""
    parts = ["<title>Relational Decoding Figures</title>" + css + "<main>",
             "<h1>Neural content from relational structure alone</h1>",
             "<p class=\"stand\">Figures for the paper draft: a transformer that sees only the correlation matrix of a few hundred neurons, never a label and never a reference population, "
             "assigns each neuron its preferred orientation and receptive-field position. One animal (MICrONS) at scale, thirty-three animals (Allen) for what transfers between brains. "
             "Every number is on held-out neurons or held-out animals; the test set never selected a model.</p>",
             "<ul class=\"glance\">",
             "<li><b>MICrONS, in vivo</b>orientation to <span class=\"n\">25°</span> mean error (chance <span class=\"n\">45°</span>), <span class=\"n\">20°</span> on the twin; receptive field R² <span class=\"n\">0.23</span> in vivo and <span class=\"n\">0.48</span> on the twin; 17M decoder, 3 seeds</li>",
             "<li><b>Why it is possible</b>the circulant part of the orientation relations fixes structure only up to rotation and reflection; a small anisotropy pins the frame</li>",
             "<li><b>Allen, 33 mice</b>orientation transfers to unseen brains (<span class=\"n\">35–39°</span> vs <span class=\"n\">45°</span>) only through populations that mix animals; single-animal populations stay within 3° of chance</li>",
             "<li><b>Allen, receptive fields</b>correlations carry where each animal looks on the screen (r <span class=\"n\">0.4–0.7</span>), not the layout within the animal</li>",
             "</ul>"]
    for section, head, intro in [("main", "Main text", "Four figures, in the order the argument runs: the result, the reason it can work, and the two cross-animal tests."),
                                 ("appendix", "Appendix", "What the sweeps established about capacity, population size, augmentation and training dynamics, so the main-text configuration reads as chosen rather than picked.")]:
        parts.append(f"<h2>{head}</h2><p class=\"stand\" style=\"margin-bottom:1.5rem\">{intro}</p>")
        for name, title, cap, sec in FIGS:
            if sec != section: continue
            b64 = base64.b64encode(open(os.path.join(OUT, name + ".png"), "rb").read()).decode()
            num = name.split("_")[0].replace("fig", "").replace("A", "A")
            lab = ("Fig. " + num) if section == "main" else ("Fig. " + num)
            parts.append(f'<figure><div class="lab{" ap" if section == "appendix" else ""}">{lab}</div><div class="body"><img src="data:image/png;base64,{b64}" alt="{title}"><figcaption><b>{title}.</b> {cap}</figcaption></div></figure>')
    parts.append("<p class=\"meta\">generated by microns_ambiguity/paper_figures.py from decoder2.json, decoder.json, symmetry.json and allen/outputs/preds · PNG and PDF in microns_ambiguity/outputs/paper/</p></main>")
    open(os.path.join(OUT, "figures.html"), "w").write("\n".join(parts)); print("wrote figures.html")


if __name__ == "__main__":
    fig_pipeline(); fig1(); fig2(); fig_regimes(); fig3(); fig4(); figA1(); figA2(); figA4(); figA5(); figA6()
    if not TEX: html()
