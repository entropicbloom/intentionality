"""Assemble the cached Allen session-A experiments into one population.

Per neuron: response vector (trial-averaged natural_movie_one, 900 bins, and
natural_movie_three, 3600 bins; both movies shared across all mice), preferred
orientation (pref_dir_dg mod 180), OSI, DG significance, mouse (container) id."""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "allen_data", "cache")


class Allen:
    """ori_source: 'sg' = static gratings, 6 orientations (0..150 deg step 30),
    labels joined from the cell table (cells matched across sessions);
    'dg' = drifting gratings, 4 orientations (directions mod 180)."""
    def __init__(self, osi_min=0.25, p_max=0.05, movie="both", ori_source="sg", remove_pcs=0, demean=False, session="A"):
        """remove_pcs: project out the top-k principal components of each session's
        (cells x time) response matrix before use; demean: subtract the session's
        mean population response per time bin.  Both remove shared fluctuations
        that would otherwise dominate pairwise correlations."""
        import pandas as pd
        cache = CACHE if session == "A" else CACHE + "_" + session
        files = sorted(f for f in os.listdir(cache) if f.endswith(".npz"))
        tab = pd.read_csv(os.path.join(os.path.dirname(CACHE), "cell_specimens.csv")).set_index("cell_specimen_id")
        R, ids, mouse, cre, depth, exp = [], [], [], [], [], []
        for f in files:
            z = np.load(os.path.join(cache, f), allow_pickle=True)
            # for session "DG": r_nm1 = condition means (40), r_nm3 = condition time courses (40 x 60)
            r = {"nm1": z["r_nm1"], "nm3": z["r_nm3"], "both": np.concatenate([z["r_nm1"], z["r_nm3"]], 1)}[movie].astype(np.float64)
            r = r - r.mean(1, keepdims=True)
            if demean:
                r = r - r.mean(0, keepdims=True)
            if remove_pcs:
                U, S, Vt = np.linalg.svd(r, full_matrices=False)
                r = r - (U[:, :remove_pcs] * S[:remove_pcs]) @ Vt[:remove_pcs]
            n = len(z["cell_ids"]); R.append(r); ids.append(z["cell_ids"])
            mouse += [str(z["donor"])] * n; cre += [str(z["cre"])] * n; depth += [int(z["depth"])] * n; exp += [int(z["experiment"])] * n
        self.R = np.concatenate(R).astype(np.float32); self.ids = np.concatenate(ids)
        lab = tab.reindex(self.ids)
        self.ori_source = ori_source
        if ori_source == "sg":
            self.ori = lab.pref_ori_sg.values.astype(float); self.osi = lab.g_osi_sg.values.astype(float); self.p = lab.p_sg.values.astype(float); self.K = 6; self.step = 30.0
        else:
            self.ori = lab.pref_dir_dg.values.astype(float) % 180; self.osi = lab.g_osi_dg.values.astype(float); self.p = lab.p_dg.values.astype(float); self.K = 4; self.step = 45.0
        self.mouse = np.array(mouse); self.cre = np.array(cre); self.depth = np.array(depth); self.exp = np.array(exp)
        self.n = len(self.ori)
        self.ori_ok = (self.osi >= osi_min) & (self.p < p_max) & np.isfinite(self.ori)
        self.mice = sorted(set(self.mouse))
        self.ori_class = np.where(np.isfinite(self.ori), np.round(np.nan_to_num(self.ori) / self.step).astype(int) % self.K, -1)
        # receptive fields (locally sparse noise, session C): ON-subfield centre in degrees of visual angle
        self.rf = lab[["rf_center_on_x_lsn", "rf_center_on_y_lsn"]].values.astype(float)
        self.rf_ok = np.isfinite(self.rf).all(1) & (lab.rf_chi2_lsn.values.astype(float) < 0.05)
        # relative RF: centred on the mouse's own RF-cloud median (each field of view covers a different patch)
        self.rf_rel = np.full_like(self.rf, np.nan)
        for m in self.mice:
            sel = (self.mouse == m) & self.rf_ok
            if sel.sum() >= 5:
                self.rf_rel[sel] = self.rf[sel] - np.median(self.rf[sel], 0)
        self.rf_rel_ok = self.rf_ok & np.isfinite(self.rf_rel).all(1)
        self.rf_dist = np.linalg.norm(np.nan_to_num(self.rf_rel), axis=1); self.rf_dist[~self.rf_rel_ok] = np.nan

    def summary(self):
        per = {m: int(((self.mouse == m) & self.ori_ok).sum()) for m in self.mice}
        return dict(n=self.n, n_ori_ok=int(self.ori_ok.sum()), n_rf_ok=int(self.rf_ok.sum()), mice=len(self.mice), per_mouse_ori_ok=per,
                    per_mouse_rf_ok={m: int(((self.mouse == m) & self.rf_ok).sum()) for m in self.mice},
                    response_dim=self.R.shape[1], K=self.K, ori_source=self.ori_source)
