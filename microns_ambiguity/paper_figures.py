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
os.makedirs(OUT, exist_ok=True)
M = json.load(open(os.path.join(ROOT, "microns_ambiguity", "outputs", "decoder2.json")))
A = json.load(open(os.path.join(ROOT, "allen", "outputs", "decoder.json")))
SYM = json.load(open(os.path.join(ROOT, "microns_ambiguity", "outputs", "symmetry.json")))
PREDS = os.path.join(ROOT, "allen", "outputs", "preds")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False, "figure.constrained_layout.use": True})
C = {"iv": "#1f5f8b", "twin": "#e8a33d", "base": "#b8b8b8", "ref": "#444444", "mix": "#2a9d8f", "single": "#b5533c",
     "s0": "#1f5f8b", "s1": "#e8a33d", "s2": "#2a9d8f"}
FIGS = []  # (filename, title, caption, section)


def save(fig, name, title, caption, section):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig); FIGS.append((name, title, caption, section)); print("wrote", name)


def baseline(ax):
    """Allen majority-class baseline: thin line at the mean over splits, faint band for the range."""
    ax.axhspan(0.187, 0.201, color=C["base"], alpha=0.18, lw=0); ax.axhline(0.194, color="#888", lw=0.9, zorder=0)


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
    fig = plt.figure(figsize=(10.8, 5.4))
    def label(ax, txt):
        ax.text(0, 1.02, txt, transform=ax.transAxes, fontsize=8.5, weight="bold", va="bottom", color=INK)
    def arrow(ax, x0, y0, x1, y1, txt=None, dy=0.05, fs=6.5):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1, transform=ax.transAxes, clip_on=False))
        if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, transform=ax.transAxes, fontsize=fs, ha="center", color=MUTED)
    # ---- a: the volume
    ax = fig.add_axes([0.03, 0.60, 0.24, 0.32]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); label(ax, "a  one mouse, 1 mm³ of cortex, 13 scans")
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
    ax.text(X0 + S / 2 + dx / 2, Y0 + S + dy + 0.5, "12,894 neurons, co-registered to the EM volume", fontsize=6.5, ha="center", color="#444")
    # ---- b: responses -> correlations (real data)
    axb = fig.add_axes([0.31, 0.60, 0.40, 0.32]); axb.axis("off"); label(axb, "b  responses to a shared stimulus → correlations")
    axt = fig.add_axes([0.315, 0.66, 0.20, 0.24])
    T = F[six][:, :120]; T = (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
    for i in range(6): axt.plot(np.arange(120), T[i] * 0.32 + (5 - i), color=NC[i], lw=0.9)
    axt.set_xlim(0, 119); axt.set_ylim(-0.9, 5.9); axt.set_yticks([]); axt.set_xticks([0, 60, 119]); axt.set_xticklabels(["0", "60", "120"], fontsize=6); axt.tick_params(length=2)
    axt.set_xlabel("stimulus bin (oracle movie clips)", fontsize=6.5, labelpad=1); axt.spines["left"].set_visible(False); axt.set_title("six neurons, trial-averaged response", fontsize=6.8, pad=3)
    axg = fig.add_axes([0.575, 0.64, 0.125, 0.24])
    Gm = Gsix.copy(); np.fill_diagonal(Gm, np.nan); axg.imshow(Gm, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    for i in range(6):
        for j in range(6):
            if i != j: axg.text(j, i, f"{Gsix[i, j]:.2f}", ha="center", va="center", fontsize=5, color="k" if abs(Gsix[i, j]) < 0.4 else "w")
    axg.set_xticks(range(6)); axg.set_yticks(range(6)); axg.set_xticklabels([]); axg.set_yticklabels([]); axg.tick_params(length=0)
    for i in range(6):
        axg.add_patch(Rectangle((-1.05, i - 0.35), 0.4, 0.7, fc=NC[i], ec="none", clip_on=False)); axg.add_patch(Rectangle((i - 0.35, -1.05), 0.7, 0.4, fc=NC[i], ec="none", clip_on=False))
    axg.set_title("correlation (Gram)", fontsize=6.8, pad=8)
    arrow(axb, 0.54, 0.45, 0.63, 0.45)
    axb.text(0.5, -0.04, "full matrix 12,894 × 12,894; the decoder sees only such matrices, never the stimulus", transform=axb.transAxes, fontsize=6.5, ha="center", color="#444")
    # ---- c: contents (real labels of the same six neurons)
    axc = fig.add_axes([0.74, 0.60, 0.24, 0.32]); axc.axis("off"); label(axc, "c  contents (training targets only)")
    axo = fig.add_axes([0.745, 0.64, 0.10, 0.25]); axo.set_xlim(-1.3, 1.3); axo.set_ylim(-1.3, 1.3); axo.set_aspect("equal"); axo.axis("off")
    axo.add_patch(Circle((0, 0), 1.0, fill=False, ec="#c9cfcc", lw=0.8))
    for i, n in enumerate(six):
        th = np.deg2rad(ds.ori[n]); axo.plot([-np.cos(th), np.cos(th)], [-np.sin(th), np.sin(th)], color=NC[i], lw=2.2, solid_capstyle="round")
    for k in range(8):
        th = np.deg2rad(k * 22.5); axo.text(1.18 * np.cos(th), 1.18 * np.sin(th), f"{int(k * 22.5)}°", fontsize=4.8, ha="center", va="center", color="#777")
    axo.set_title("preferred orientation\n(8 classes)", fontsize=6.8, pad=2)
    axr = fig.add_axes([0.865, 0.65, 0.11, 0.23]); axr.set_xlim(-1.05, 1.05); axr.set_ylim(-1.05, 1.05); axr.set_aspect("equal")
    axr.add_patch(Rectangle((-1, -1), 2, 2, fc="#f2f4f3", ec="#8a939c", lw=0.8))
    for i, n in enumerate(six): axr.scatter(ds.rf[n, 0], ds.rf[n, 1], s=28, c=NC[i], ec="white", lw=0.6, zorder=3)
    axr.set_xticks([]); axr.set_yticks([]); [sp.set_visible(False) for sp in axr.spines.values()]; axr.set_title("receptive-field centre\n(screen)", fontsize=6.8, pad=2)
    axc.text(0.5, -0.04, "5,287 neurons with orientation (gOSI ≥ 0.25) · 11,326 with RF", transform=axc.transAxes, fontsize=6.5, ha="center", color="#444")
    # ---- d: protocol (equal aspect so the neurons are round)
    axd = fig.add_axes([0.03, 0.04, 0.95, 0.46]); axd.set_xlim(0, 20); axd.set_ylim(0, 4.9); axd.set_aspect("equal"); axd.axis("off")
    label(axd, "d  protocol: halve the neurons, sample a population inside one half, decode every neuron of the population from its row of the Gram")
    for i in range(72):
        cx, cy = 0.5 + (i % 12) * 0.38, 4.1 - (i // 12) * 0.38
        if i % 12 < 6: axd.add_patch(Circle((cx, cy), 0.13, fc=TR, ec="none"))
        else: axd.add_patch(Circle((cx, cy), 0.13, fc="white", ec=TE, lw=1))
    axd.text(1.45, 1.75, "training half", fontsize=6.5, color=TR, ha="center"); axd.text(3.75, 1.75, "test half", fontsize=6.5, color=TE, ha="center")
    axd.text(2.6, 1.25, "random split of the 12,894 neurons", fontsize=6.3, ha="center", color="#444")
    sel = rng.choice(np.arange(36), 9, replace=False)
    for j in sel:
        cx, cy = 0.5 + (j % 6) * 0.38, 4.1 - (j // 6) * 0.38; axd.add_patch(Circle((cx, cy), 0.2, fill=False, ec=INK, lw=0.9, ls=(0, (2, 1.5))))
    axd.add_patch(FancyArrowPatch((5.3, 3.1), (6.3, 3.1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1)); axd.text(5.8, 3.3, "sample\n512", fontsize=6.3, ha="center", color=MUTED)
    pop = rng.choice(np.flatnonzero(ds.ori_ok), 40, replace=False); Gp = F[pop] @ F[pop].T; np.fill_diagonal(Gp, np.nan)
    bx, by, bw, bh = 0.03, 0.04, 0.95, 0.46; ux, uy = bw / 20, bh / 4.9 * (4.9 / 4.9)
    def rect(x, y, w, h): return [bx + x * ux, by + y * (bh / 4.9), w * ux, h * (bh / 4.9)]
    axp = fig.add_axes(rect(6.5, 1.9, 2.4, 2.4)); axp.imshow(Gp, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto"); axp.set_xticks([]); axp.set_yticks([])
    axp.add_patch(Rectangle((-0.5, 8.5), 40, 1, fill=False, ec=INK, lw=1.4)); axp.set_title("population Gram (512 × 512, 40 shown)", fontsize=6.5, pad=2)
    axd.text(7.7, 1.55, "one row = one neuron's token", fontsize=6.3, ha="center", color="#444")
    axd.add_patch(FancyArrowPatch((9.1, 3.1), (10.1, 3.1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    for k in range(4):
        axd.add_patch(FancyBboxPatch((10.2 + k * 0.1, 2.15 + k * 0.12), 2.6, 1.8, boxstyle="round,pad=0.03,rounding_size=0.2", fc="white", ec="#8a939c", lw=0.9, zorder=2 + k))
    axd.text(11.8, 3.45, "transformer", fontsize=7.5, ha="center", va="center", weight="bold", color=INK, zorder=9); axd.text(11.8, 2.95, "rows as tokens,\nattention over the population", fontsize=6.2, ha="center", va="center", color="#444", zorder=9)
    axd.text(11.8, 1.55, "trained on labelled training neurons;\nepoch chosen on a held-out slice of the training half", fontsize=6.0, ha="center", va="top", color="#444")
    axd.add_patch(FancyArrowPatch((13.3, 3.1), (14.3, 3.1), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
    axd.text(16.9, 4.35, "per-neuron prediction, scored on the test half", fontsize=6.8, ha="center", color=INK)
    axo2 = fig.add_axes(rect(14.6, 2.0, 1.9, 2.0)); axo2.set_xlim(-1.3, 1.3); axo2.set_ylim(-1.3, 1.3); axo2.set_aspect("equal"); axo2.axis("off")
    axo2.add_patch(Circle((0, 0), 1.0, fill=False, ec="#c9cfcc", lw=0.8)); th = np.deg2rad(ds.ori[six[0]]); axo2.plot([-np.cos(th), np.cos(th)], [-np.sin(th), np.sin(th)], color=NC[0], lw=2.2, solid_capstyle="round"); axo2.set_title("orientation class", fontsize=6.3, pad=1)
    axr2 = fig.add_axes(rect(17.0, 2.05, 1.9, 1.9)); axr2.set_xlim(-1.05, 1.05); axr2.set_ylim(-1.05, 1.05); axr2.set_aspect("equal")
    axr2.add_patch(Rectangle((-1, -1), 2, 2, fc="#f2f4f3", ec="#8a939c", lw=0.8)); axr2.scatter(ds.rf[six[0], 0], ds.rf[six[0], 1], s=28, c=NC[0], ec="white", lw=0.6, zorder=3); axr2.set_xticks([]); axr2.set_yticks([]); [sp.set_visible(False) for sp in axr2.spines.values()]; axr2.set_title("RF centre (x, y)", fontsize=6.3, pad=1)
    axd.text(16.9, 1.55, "no labels, no reference population in the input;\nsame brain, single-animal populations: the `within` cell of Fig. 4", fontsize=6.0, ha="center", va="top", color="#444")
    fig.suptitle("Fig. 1  MICrONS: from one imaged cortical volume to a label-free per-neuron decoding task", fontsize=9, y=0.995)
    save(fig, "fig1_microns_pipeline", "How the MICrONS task is built",
         "(a) The MICrONS functional-connectomics release (Ding, Fahey, Papadopoulos et al. 2025): one mouse, 13 two-photon scans of a cubic millimetre of visual cortex "
         "(areas V1, RL, AL, LM), 12,894 neurons co-registered to the electron-microscopy volume. (b) The relational substrate. Each neuron is its trial-averaged "
         "response vector to a stimulus all neurons saw (in vivo: 120 bins of the oracle natural-movie clips; digital twin: 4,999 bins compressed to 512 principal "
         "components); shown are six real neurons and their correlation matrix. The full matrix is 12,894 × 12,894, and the decoder only ever sees such correlations, "
         "never the stimulus. (c) The contents of the same six neurons: preferred orientation (in vivo, 8 classes, 5,287 neurons with gOSI ≥ 0.25) and receptive-field "
         "centre (digital-twin fit, 11,326 neurons). These are training targets only; they never enter the input. (d) The protocol: neurons are halved at random; "
         "populations of 512 are sampled inside one half; the population's standardised Gram (a real 40-neuron excerpt is shown) gives one row per neuron, which "
         "becomes that neuron's token; a transformer attends over the population and predicts every token's content. Training uses labelled training neurons, the "
         "epoch is chosen on a held-out slice of the training half, and the score is taken on the test half. One animal, single-animal populations, new neurons: the "
         "`within` cell of the regime figure (Fig. 4).", "main")


# ---------------------------------------------------------------- Fig 2: MICrONS label-free decoding
def fig1():
    REF = {("iv", "ori"): 0.41, ("twin", "ori"): 0.49, ("iv", "rf"): 0.32, ("twin", "rf"): 0.67}
    BASE = {"ori": 0.255, "rf": 0.0}
    seeds = {("iv", "ori", "2.2M"): ["es_iv_ori_s0", "es_iv_ori_s1", "es_iv_ori_s2"],
             ("iv", "rf", "2.2M"): ["es_iv_rf_s0", "es_iv_rf_s1", "es_iv_rf_s2"],
             ("twin", "ori", "2.2M"): ["es_twin_ori_s0", "es_twin_ori_s1", "es_twin_ori_s2"],
             ("twin", "rf", "2.2M"): ["es_twin_rf_s0", "es_twin_rf_s1", "es_twin_rf_s2"],
             ("iv", "ori", "17M"): ["g3_iv_ori_n512_d512L8", "g12_iv_ori_s1", "g12_iv_ori_s2"],
             ("iv", "rf", "17M"): ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"],
             ("twin", "ori", "17M"): ["g3_is_ori_n512_d512L8", "g12_is_ori_s1", "g12_is_ori_s2"],
             ("twin", "rf", "17M"): ["g3_is_rf_n512_d512L8", "g9_is_rf_n512_d512L8_s1", "g9_is_rf_n512_d512L8_s2"]}
    single = {("iv", "ori", "0.3M"): "n512_rel", ("iv", "rf", "0.3M"): "n512_rel_rf", ("twin", "ori", "0.3M"): "twin_n512", ("twin", "rf", "0.3M"): "twin_n512_rf"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    for ax, con, key, ylab in [(axes[0], "ori", "acc", "orientation accuracy (8 classes)"), (axes[1], "rf", "r2", "receptive-field position R²")]:
        levels = ["0.3M", "2.2M", "17M"]; w = 0.2
        for gi, sub in enumerate(["iv", "twin"]):
            x0 = gi * 1.0
            for li, lv in enumerate(levels):
                x = x0 + (li - 1) * w
                if lv == "0.3M":
                    v, e = M[single[(sub, con, lv)]][key], 0.0
                else:
                    v, e, _ = ms(seeds[(sub, con, lv)], key)
                ax.bar(x, v, w * 0.9, color=C[sub], alpha=0.45 + 0.27 * li, yerr=e if e > 0 else None, capsize=2, ecolor="k")
                ax.text(x, v + 0.012, lv, ha="center", va="bottom", fontsize=6.5, rotation=90)
        if BASE[con] > 0:
            ax.axhline(BASE[con], color=C["base"], lw=1, zorder=0); ax.text(1.42, BASE[con] - 0.03, "majority class", fontsize=6.5, color="#666")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["in vivo", "digital twin"]); ax.set_ylabel(ylab)
        ax.set_xlim(-0.55, 1.75); ax.set_ylim(0, {"ori": 0.5, "rf": 0.6}[con])
    fig.suptitle("Fig. 2  MICrONS: per-neuron content decoded from the population correlation matrix alone (512 neurons, no labels, no reference)", fontsize=9)
    save(fig, "fig2_microns_decoder", "MICrONS label-free per-neuron decoding",
         "Bars: label-free decoder accuracy (left, preferred orientation, 8 classes) and R² (right, receptive-field centre) on held-out neurons of the same animal, "
         "for decoders of 0.3M, 2.2M and 17M parameters; error bars are the s.d. over 3 seeds (2.2M and 17M). The decoder sees only the standardised "
         "correlation matrix of 512 sampled neurons: neither labels nor a reference population enter the input, and the test neurons never influenced model "
         "selection. Grey line: majority class. Orientation is read at 0.38 in vivo (8 classes, chance 0.125, majority 0.255); receptive-field position at R² 0.23 in vivo and 0.48 on the twin.", "main")


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
        ax.text(i, 1.06, f"circulant var. {s['frac_var_circulant']:.2f}", ha="center", fontsize=7)
    ax.axhline(1 / 8, color=C["base"], lw=1); ax.text(1.4, 1 / 8 + 0.02, "chance", fontsize=7, color="#666")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["in vivo", "digital twin"]); ax.set_ylim(0, 1.15); ax.set_ylabel("class matching accuracy"); ax.legend(loc="center right", fontsize=7)
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


# ---------------------------------------------------------------- Fig 4: Allen 2x2 orientation
def fig3():
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    cells = [("within", "training animals\nsingle-animal pop.", [A["g11_ori_within_n128"]["acc"]], C["single"]),
             ("pooledwithin", "training animals\nmixed pop.", [A["g7_ori_pooledwithin_n256"]["acc"], A["g7_ori_pooledwithin_n128"]["acc"]], C["mix"]),
             ("cross", "held-out animals\nsingle-animal pop.", [A["dg3_cross_128"]["acc"], A["g11_ori_cross_sp1"]["acc"], A["g11_ori_cross_sp2"]["acc"]], C["single"]),
             ("pooledcross", "held-out animals\nmixed pop.", [np.mean([A[t]["acc"] for t in ["g2_n256_d512L8_cf50", "g10_d512L8_cf50_s1", "g10_d512L8_cf50_s2"]]), A["g10_d512L8_cf50_sp1"]["acc"], A["g10_d512L8_cf50_sp2"]["acc"]], C["mix"])]
    for i, (reg, lab, vals, col) in enumerate(cells):
        ax.bar(i, np.mean(vals), 0.6, color=col, alpha=0.8)
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
        ax.plot(i + np.array(jitter), vals, "o", color="k", ms=4, mfc="white")
        ax.text(i, 0.322, f"`{reg}`", ha="center", fontsize=7, family="monospace")
    baseline(ax); ax.text(3.45, 0.194, "majority-class\nbaseline", fontsize=7, va="center", color="#666")
    ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in cells], fontsize=7.5); ax.set_ylabel("orientation accuracy (6 classes)"); ax.set_ylim(0.1, 0.34); ax.set_xlim(-0.6, 4.4)
    ax.set_title("Fig. 5  Allen (33 mice, grating relations): orientation is readable only from populations that mix animals", fontsize=9, pad=10)
    save(fig, "fig5_allen_2x2_orientation", "Allen 2 × 2: split × population, orientation",
         "Label-free decoder accuracy on labelled test cells for the four regimes. Split: test neurons from the training animals (each animal's cells halved) "
         "or from 9 held-out animals. Population: each sampled population (the unit the Gram is computed on) drawn from one animal or from several. "
         "Points: individual mouse splits (held-out regimes; the `pooledcross` split-0 point is the mean of 3 seeds) or population sizes (128 / 256). "
         "Single-animal populations are at the majority baseline whether the animal was seen in training or not; mixed populations are above it in both splits "
         "at the same level, so transfer to unseen brains costs nothing. Mixed populations contain between-animal correlations, which exist only because all mice "
         "saw the same 40 grating conditions and measure tuning similarity; the decoder still never sees condition identities.", "main")


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
    ax = axes[0]
    curves = {("iv", "ori"): [(0.3, "n512_rel"), (2.2, ["es_iv_ori_s0", "es_iv_ori_s1", "es_iv_ori_s2"]), (17, ["g3_iv_ori_n512_d512L8", "g12_iv_ori_s1", "g12_iv_ori_s2"])],
              ("twin", "ori"): [(0.3, "twin_n512"), (2.2, ["es_twin_ori_s0", "es_twin_ori_s1", "es_twin_ori_s2"]), (7.3, "huge_twin_n512"), (17, ["g3_is_ori_n512_d512L8", "g12_is_ori_s1", "g12_is_ori_s2"]), (57, "g9_is_ori_n512_d768L12")]}
    curves_rf = {("iv", "rf"): [(0.3, "n512_rel_rf"), (2.2, ["es_iv_rf_s0", "es_iv_rf_s1", "es_iv_rf_s2"]), (17, ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"])],
                 ("twin", "rf"): [(0.3, "twin_n512_rf"), (2.2, ["es_twin_rf_s0", "es_twin_rf_s1", "es_twin_rf_s2"]), (7.3, "huge_twin_n512_rf_long"), (17, ["g3_is_rf_n512_d512L8", "g9_is_rf_n512_d512L8_s1", "g9_is_rf_n512_d512L8_s2"]), (57, "g9_is_rf_n512_d768L12")]}
    for ax, cv, key, ttl in [(axes[0], curves, "acc", "MICrONS orientation"), (axes[1], curves_rf, "r2", "MICrONS RF (x, y)")]:
        for (sub, con), pts in cv.items():
            xs, ys, es = [], [], []
            for p, t in pts:
                if isinstance(t, list): m, e, _ = ms(t, key)
                else: m, e = M[t][key], 0
                xs.append(p); ys.append(m); es.append(e)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, color=C[sub], label={"iv": "in vivo", "twin": "twin"}[sub], capsize=2)
        ax.set_xscale("log"); ax.set_xticks([0.3, 2.2, 7.3, 17, 57]); ax.set_xticklabels(["0.3", "2.2", "7", "17", "57"]); ax.minorticks_off(); ax.set_xlabel("decoder parameters (M)"); ax.set_ylabel({"acc": "accuracy", "r2": "R²"}[key]); ax.set_title(ttl); ax.legend()
        ax.axhline({"acc": 0.255, "r2": 0}[key], color=C["base"], lw=1, zorder=0)
    ax = axes[2]
    xs = [2.2, 17, 57]; ys = [A["g1_n256_d256L4_e60"]["acc"], A["g1_n256_d512L8_e60"]["acc"], A["g1_n256_d768L12_e60"]["acc"]]
    ax.plot(xs, ys, "o-", color=C["mix"], label="plain, 256 neurons")
    ax.plot([2.2, 17, 57], [A["g2_n256_d256L4_cf50"]["acc"], A["g2_n256_d512L8_cf50"]["acc"], A["g10_d768L12_cf50"]["acc"]], "s--", color=C["mix"], mfc="white", label="Gram from 50 % of conditions")
    ax.errorbar([17], [np.mean([A[t]["acc"] for t in ["g2_n256_d512L8_cf50", "g10_d512L8_cf50_s1", "g10_d512L8_cf50_s2"]])], yerr=[np.std([A[t]["acc"] for t in ["g2_n256_d512L8_cf50", "g10_d512L8_cf50_s1", "g10_d512L8_cf50_s2"]])], fmt="none", ecolor="k", capsize=3)
    baseline(ax); ax.set_xscale("log"); ax.set_xticks([2.2, 17, 57]); ax.set_xticklabels(["2.2", "17", "57"]); ax.minorticks_off(); ax.set_xlabel("decoder parameters (M)"); ax.set_ylabel("accuracy"); ax.set_title("Allen orientation, held-out mice (split 0)"); ax.legend(fontsize=7); ax.set_ylim(0.15, 0.35)
    fig.suptitle("Fig. A1  Capacity: RF keeps gaining with model size, orientation does not (label counts 11,326 vs 5,287 in MICrONS; 5,529 in Allen)", fontsize=9)
    save(fig, "figA1_scaling_params", "Appendix: scaling with decoder size",
         "Test metric versus number of decoder parameters at fixed population size (512 neurons MICrONS, 256 Allen). Error bars: s.d. over 3 seeds where run. "
         "Every model above 2M memorises its training neurons within 4–17 epochs (training loss 0.001–0.06) and is early-stopped on a held-out selection set, so "
         "the curve is the gain from stopping early on a larger model. On MICrONS RF the gain continues to 57M; on orientation it stops at 7M. On Allen the plain "
         "curve rises slowly and the augmented 17M model equals the 2M standard configuration within seed noise (error bar); augmentation on the 57M model hurts.", "appendix")


# ---------------------------------------------------------------- Fig A2: population size
def figA2():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    for sub, tags, key, lab, col in [("iv", ["n128_rel", "n256_rel", "n512_rel", "n1024_rel"], "acc", "in vivo, orientation, 0.3M", C["iv"]),
                                     ("iv", ["n128_rel_rf", "n256_rel_rf", "n512_rel_rf", "n1024_rel_rf"], "r2", "in vivo, RF, 0.3M", C["iv"]),
                                     ("twin", ["big_twin_n256", "big_twin_n512", "big_twin_n1024"], "acc", "twin, orientation, 2.2M", C["twin"]),
                                     ("twin", ["twin_n256_rf", "big_twin_n512_rf", "big_twin_n1024_rf"], "r2", "twin, RF, 2.2M", C["twin"])]:
        xs = [M[t]["n"] for t in tags]; ys = [M[t][key] for t in tags]
        ax.plot(xs, ys, "o-" if key == "acc" else "s--", color=col, label=lab, mfc="white" if key == "r2" else col)
    ax.set_xscale("log", base=2); ax.set_xlabel("neurons per population"); ax.set_ylabel("accuracy / R²"); ax.set_title("MICrONS"); ax.legend(fontsize=6.5)
    ax = axes[1]
    ax.plot([256, 512, 1024], [A["g1_n256_d512L8_e60"]["acc"], A["g1_n512_d512L8_e60"]["acc"], A["g1_n1024_d512L8_e60"]["acc"]], "o-", color=C["mix"], label="orientation, 17M, mixed pop.")
    ax.plot([128, 256], [A["dg3_128_sp0_s0"]["acc"], A["dg3_256_sp0_s0"]["acc"]], "o-", color=C["mix"], alpha=0.5, label="orientation, 2M, mixed pop.")
    ax.plot([256, 512, 1024], [A["g4_rf_n256_d256L4_cf50"]["r2"], A["g4_rf_n512_d512L8_cf50"]["r2"], A["g4_rf_n1024_d512L8_cf50"]["r2"]], "s--", color=C["mix"], mfc="white", label="RF R², mixed pop. (split 0)")
    ax.set_xscale("log", base=2); ax.set_xlabel("neurons per population"); ax.set_title("Allen, held-out mice"); ax.legend(fontsize=6.5); baseline(ax)
    fig.suptitle("Fig. A2  Population size saturates by 512 neurons (MICrONS) and 256 (Allen)", fontsize=9)
    save(fig, "figA2_population_size", "Appendix: population size",
         "Test metric versus the number of neurons in each sampled population (the size of the Gram the decoder sees). The 0.3M in-vivo curves are the original "
         "sweep; the 2.2M twin curves and all Allen points use the fixed selection protocol. Beyond a few hundred neurons the additional companions add redundant "
         "context; on Allen RF, 1024-neuron mixed populations are worse (0.09).", "appendix")


# ---------------------------------------------------------------- Fig A3: augmentation
def figA3():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    ax.plot([1.0, 0.75, 0.5, 0.3], [A["g1_n256_d256L4_e60"]["acc"], A["g2_n256_d256L4_cf75"]["acc"], A["g2_n256_d256L4_cf50"]["acc"], A["g2_n256_d256L4_cf30"]["acc"]], "o-", color=C["mix"], alpha=0.55, label="2M")
    ax.plot([1.0, 0.85, 0.5], [A["g1_n256_d512L8_e60"]["acc"], A["g5_d512L8_cf85"]["acc"], A["g2_n256_d512L8_cf50"]["acc"]], "o-", color=C["mix"], label="17M")
    ax.plot([1.0, 0.5], [A["g1_n256_d768L12_e60"]["acc"], A["g10_d768L12_cf50"]["acc"]], "o-", color="#1b6b60", label="57M")
    ax.plot([1.0, 1.0], [A["g2_n256_d256L4_gd20"]["acc"], A["g1_n256_d512L8_e60_do2"]["acc"]], "x", color="k", label="Gram dropout 20 % (2M) / dropout 0.2 (17M)")
    ax.invert_xaxis(); ax.set_xlabel("fraction of the 40 conditions kept"); ax.set_ylabel("accuracy, held-out mice (split 0)"); ax.set_title("Allen orientation"); ax.legend(fontsize=6.5); baseline(ax)
    ax = axes[1]
    ax.errorbar([1.0], [np.mean([M[t]["r2"] for t in ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"]])], yerr=[np.std([M[t]["r2"] for t in ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"]])], fmt="o", color=C["iv"], capsize=2)
    ax.plot([1.0, 0.75, 0.5], [np.mean([M[t]["r2"] for t in ["g9_iv_rf_n512_d512L8", "g12_iv_rf_s1", "g12_iv_rf_s2"]]), M["g9_iv_rf_n512_d512L8_cf75"]["r2"], M["g3_iv_rf_n512_d512L8_cf50"]["r2"]], "o-", color=C["iv"], label="in vivo RF, 17M (120 stimulus bins)")
    ax.plot([1.0, 0.5], [M["g3_iv_ori_n512_d512L8"]["acc"], M["g3_iv_ori_n512_d512L8_cf50"]["acc"]], "s--", color=C["iv"], mfc="white", label="in vivo orientation, 17M")
    ax.plot([1.0, 0.5], [M["g3_is_rf_n512_d512L8"]["r2"], M["g3_is_rf_n512_d512L8_cf50"]["r2"]], "s--", color=C["twin"], mfc="white", label="twin RF, 17M (512 PCs)")
    ax.plot([1.0, 0.5], [M["g3_is_ori_n512_d512L8"]["acc"], M["g3_is_ori_n512_d512L8_cf50"]["acc"]], "o-", color=C["twin"], label="twin orientation, 17M")
    ax.invert_xaxis(); ax.set_xlabel("fraction of feature dimensions kept"); ax.set_ylabel("accuracy / R²"); ax.set_title("MICrONS"); ax.legend(fontsize=6.5)
    fig.suptitle("Fig. A3  Label-free relation augmentation: each training population's Gram from a random subset of stimulus conditions", fontsize=9)
    save(fig, "figA3_augmentation", "Appendix: relation augmentation",
         "Training populations get a Gram computed from a random subset of the response dimensions (rows re-standardised); test Grams use all dimensions. "
         "On Allen the augmentation helps only with capacity: the 2M model is hurt by anything below 75 %, the 17M model peaks at 50–85 %, the 57M model is hurt. "
         "On MICrONS it helps only in-vivo RF (75 % of the 120 stimulus bins), costs a point on in-vivo orientation, and destroys the twin Gram, whose 512 "
         "dimensions are principal components and not exchangeable conditions.", "appendix")


# ---------------------------------------------------------------- Fig A4: training dynamics
def figA4():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for ax, tag, src, key, ttl in [(axes[0], "g1_n256_d512L8_e60", A, "acc", "Allen orientation, 17M, plain"), (axes[1], "g2_n256_d512L8_cf50", A, "acc", "Allen orientation, 17M, 50 % conditions"), (axes[2], "g3_is_rf_n512_d512L8", M, "r2", "MICrONS twin RF, 17M")]:
        h = src[tag]["history"]; ep = np.arange(len(h))
        ax.plot(ep, [x[key] for x in h], "-", color=C["mix"] if src is A else C["twin"], label="test metric")
        ax.plot(ep, [x["sel"] for x in h], ":", color="k", lw=1, label="selection set")
        b = src[tag]["best_epoch"]; ax.axvline(b, color="#999", lw=0.8); ax.text(b + 0.5, ax.get_ylim()[0] + 0.01, f"selected\nepoch {b}", fontsize=6.5, color="#666")
        ax2 = ax.twinx(); ax2.plot(ep, [x["loss"] for x in h], "-", color="#c44", lw=1, alpha=0.7); ax2.set_ylabel("training loss", color="#c44", fontsize=7); ax2.tick_params(axis="y", labelsize=7, colors="#c44"); ax2.spines["right"].set_visible(True)
        ax.set_xlabel("epoch"); ax.set_ylabel({"acc": "accuracy", "r2": "R²"}[key]); ax.set_title(ttl, fontsize=8.5)
        if ax is axes[0]: ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Fig. A4  Every large run memorises its training neurons; the reported score is the epoch chosen on a held-out selection set", fontsize=9)
    save(fig, "figA4_training_dynamics", "Appendix: training dynamics and early stopping",
         "Test metric (solid), selection-set metric (dotted; held-out mice on Allen, a held-out 20 % of training neurons on MICrONS) and training loss (red, right axis) "
         "per epoch. The selection set, never the test set, picks the epoch (grey line) and its weights are restored before evaluation. Without augmentation the "
         "training loss collapses within ~15 epochs; the condition-subsampling augmentation keeps it up and moves the selected epoch from 13 to 35 on Allen.", "appendix")


# ---------------------------------------------------------------- Fig A5: Allen split variance + class-level
def figA5():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    groups = {"first protocol\n128 n, 12 ep": ["dg_ori_pooledcross_nm1", "dg_pc_s1", "dg_pc_s2"], "17M + 50 %\n256 n, 60 ep": ["g2_n256_d512L8_cf50", "g10_d512L8_cf50_sp1", "g10_d512L8_cf50_sp2"]}
    for gi, (g, tags) in enumerate(groups.items()):
        for si, t in enumerate(tags): ax.bar(gi + (si - 1) * 0.25, A[t]["acc"], 0.22, color=C[f"s{si}"], label=f"split {si}" if gi == 0 else None)
    baseline(ax); ax.set_xticks([0, 1]); ax.set_xticklabels(list(groups.keys()), fontsize=8); ax.set_ylabel("accuracy, held-out mice"); ax.set_ylim(0.15, 0.34); ax.legend(fontsize=7, loc="upper left", ncol=3); ax.set_title("Allen orientation, `pooledcross`, by mouse split", fontsize=8.5)
    ax = axes[1]
    G = json.load(open(os.path.join(ROOT, "allen", "outputs", "geometric_DG_nm1_sg.json")))["summary"]; Gm = json.load(open(os.path.join(ROOT, "allen", "outputs", "geometric_both_sg.json")))["summary"]
    labs = ["within-mouse\nsplit-half", "one mouse →\nanother", "pooled reference →\nheld-out mouse"]
    for i, k in enumerate(["within_mean", "cross_mean", "pooled_mean"]):
        ax.bar(i - 0.18, Gm[k], 0.34, color="#999", label="movie relations" if i == 0 else None); ax.bar(i + 0.18, G[k], 0.34, color=C["mix"], label="grating relations" if i == 0 else None)
    ax.axhline(1 / 6, color="k", lw=0.8); ax.text(2.4, 1 / 6 + 0.005, "chance", fontsize=7); ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel("class-level matching accuracy (K = 6)"); ax.legend(fontsize=7); ax.set_title("Allen class-level test", fontsize=8.5); ax.set_ylim(0.1, 0.3)
    fig.suptitle("Fig. A5  Allen: mouse-split variance of the decoder, and the class-level matching test by relation type", fontsize=9)
    save(fig, "figA5_allen_splits_classlevel", "Appendix: Allen split variance and class-level test",
         "Left: `pooledcross` orientation accuracy on the same three held-out-mouse splits for the first protocol and the final configuration; split 0 is the easy "
         "set of test mice for both. Right: the class-level test (6 × 6 class-Gram matched over all 720 relabellings) with relations from natural movies versus "
         "drifting gratings; only the grating relations move every test off chance.", "appendix")


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
             "<li><b>MICrONS, in vivo</b>orientation <span class=\"n\">0.38</span> (majority class <span class=\"n\">0.26</span>), receptive field R² <span class=\"n\">0.23</span> in vivo and <span class=\"n\">0.48</span> on the twin; 17M decoder, 3 seeds</li>",
             "<li><b>Why it is possible</b>the circulant part of the orientation relations fixes structure only up to rotation and reflection; a small anisotropy pins the frame</li>",
             "<li><b>Allen, 33 mice</b>orientation transfers to unseen brains (<span class=\"n\">0.26</span> vs <span class=\"n\">0.19</span>) only through populations that mix animals</li>",
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
    fig_pipeline(); fig1(); fig2(); fig_regimes(); fig3(); fig4(); figA1(); figA2(); figA3(); figA4(); figA5(); html()
