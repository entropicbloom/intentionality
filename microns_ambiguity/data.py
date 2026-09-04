"""Load the Ding et al. 2025 MICrONS functional-connectomics release and build
relational substrates (neuron x feature matrices) and content labels.

Substrates
----------
struct_out      148 proofread presynaptic neurons x 12,894 postsynaptic targets,
                entries = number of synapses.  Relations between two axons =
                overlap of their synaptic target sets.
struct_out_adp  same rows; entries = 1 where the axon passes within reach of the
                target's dendrite (axon-dendrite proximity, ADP) OR is connected.
                Proximity-only ("potential") connectivity: the wiring null.
struct_in       postsynaptic neurons with >= 1 synapse from a proofread axon
                x 148 axons, entries = number of synapses.  Relations between two
                neurons = shared presynaptic inputs.
struct_in_adp   same rows; potential-connectivity version.
func_iv         all neurons x 120 trial-averaged in-vivo responses to the shared
                oracle natural-movie clips (z-scored -> cosine = signal correlation).
func_is         all neurons x 4,999 digital-twin responses to a shared movie.
soma            all neurons x 3 soma coordinates (um); used through a Gaussian
                distance kernel as the purely spatial null substrate.

Contents
--------
ori      in-vivo preferred orientation (deg, period 180), gated on gOSI
rf       receptive-field centre (STA fit, stimulus coordinates in [-1, 1])
soma_xz  tangential cortical position (um)
depth    cortical depth (um)
layer    L2/3, L4, L5
area     V1, RL, AL (LM dropped, n = 26)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import CACHE, DATA, GOSI_MIN, CC_ABS_MIN


def load_tables():
    nodes = pd.read_pickle(DATA / "node_data_v1.pkl").reset_index(drop=True)
    edges = pd.read_pickle(DATA / "edge_data_v1.pkl")
    return nodes, edges


class Dataset:
    def __init__(self):
        nodes, edges = load_tables()
        self.nodes = nodes
        self.n = len(nodes)
        self.nid = nodes.nucleus_id.values
        self.idx_of = {int(v): i for i, v in enumerate(self.nid)}

        # ---- content labels -------------------------------------------------
        self.ori = nodes.pref_ori_iv.values.astype(float)            # deg [0,180)
        self.gosi = nodes.gosi_iv.values.astype(float)
        self.ori_ok = self.gosi >= GOSI_MIN
        self.rf = np.stack(nodes.sta_mu_comb.values).astype(float)   # (n, 2)
        self.rf_cvt = np.stack(nodes.position_stim_cvt.values).astype(float)
        self.rf_ok = nodes.cc_abs_cvt.values >= CC_ABS_MIN
        self.soma = nodes[["nucleus_x", "nucleus_y", "nucleus_z"]].values / 1000.0  # um
        self.soma_xz = self.soma[:, [0, 2]]
        self.depth = self.soma[:, 1]
        self.layer = nodes.layer.astype(str).values
        self.layer_ok = np.isin(self.layer, ["L2/3", "L4", "L5"])
        self.area = nodes.brain_area.astype(str).values
        self.area_ok = np.isin(self.area, ["V1", "RL", "AL"])
        self.scan = (nodes.scan_session.astype(int).astype(str) + "-" +
                     nodes.scan_idx.astype(int).astype(str)).values

        # ---- functional substrates -----------------------------------------
        self.func_iv = np.stack(nodes.in_vivo_mean_resp.values).astype(np.float32)
        self.func_is = np.stack(nodes.in_silico_resp.values).astype(np.float32)

        # ---- structural substrates -----------------------------------------
        pre_ids = np.sort(edges.pre_nucleus_id.unique())
        self.pre_idx = np.array([self.idx_of[int(p)] for p in pre_ids])
        pre_pos = {int(p): k for k, p in enumerate(pre_ids)}
        conn = edges[edges.population == "Connected"]
        adp = edges[edges.population.isin(["Connected", "ADP"])]

        def bip(df, val):
            r = df.pre_nucleus_id.map(pre_pos).values
            c = df.post_nucleus_id.map(self.idx_of).values
            return sp.csr_matrix((val, (r, c)), shape=(len(pre_ids), self.n), dtype=np.float32)

        self.W_syn = bip(conn, conn.n_synapses.values.astype(np.float32))
        self.W_adp = bip(adp, np.ones(len(adp), np.float32))
        self.W_adp.data[:] = 1.0
        self.post_idx = np.flatnonzero(np.asarray(self.W_syn.sum(0)).ravel() > 0)
        self.post_adp_idx = np.flatnonzero(np.asarray(self.W_adp.sum(0)).ravel() > 0)

    # ---- substrate accessors ----------------------------------------------
    def substrate(self, name: str, W_syn=None):
        """Return (row_indices_into_nodes, feature_matrix) for a substrate.
        W_syn may be overridden with a rewired null connectome."""
        W = self.W_syn if W_syn is None else W_syn
        if name == "struct_out":
            return self.pre_idx, W.toarray()
        if name == "struct_out_adp":
            return self.pre_idx, self.W_adp.toarray()
        if name == "struct_in":
            return self.post_idx, W.T.toarray()[self.post_idx]
        if name == "struct_in_adp":
            return self.post_idx, self.W_adp.T.toarray()[self.post_idx]
        if name == "struct_in_adp_all":
            return self.post_adp_idx, self.W_adp.T.toarray()[self.post_adp_idx]
        if name == "func_iv":
            return np.arange(self.n), zscore_rows(self.func_iv)
        if name == "func_is":
            return np.arange(self.n), zscore_rows(self.func_is)
        if name == "soma":
            return np.arange(self.n), self.soma.astype(np.float32)
        raise KeyError(name)

    def content(self, name: str):
        """Return (values, ok_mask) over all nodes."""
        if name == "ori":
            return self.ori, self.ori_ok
        if name == "rf":
            return self.rf, self.rf_ok
        if name == "rf_resid":
            # RF centre minus the local retinotopic map: for each neuron, subtract
            # the mean RF of its 50 nearest somata (same area, itself excluded).
            # What remains is the local scatter of retinotopy, which cortical
            # location cannot predict by construction.
            from scipy.spatial import cKDTree
            resid = np.full_like(self.rf, np.nan); ok = self.rf_ok
            for a in np.unique(self.area):
                idx = np.flatnonzero((self.area == a) & ok)
                if len(idx) < 60: continue
                tree = cKDTree(self.soma[idx]); _, nn = tree.query(self.soma[idx], k=51)
                resid[idx] = self.rf[idx] - self.rf[idx][nn[:, 1:]].mean(1)
            return resid, ok & ~np.isnan(resid[:, 0])
        if name == "soma_xz":
            return self.soma_xz, np.ones(self.n, bool)
        if name == "depth":
            return self.depth, np.ones(self.n, bool)
        if name == "layer":
            return self.layer, self.layer_ok
        if name == "area":
            return self.area, self.area_ok
        raise KeyError(name)


def zscore_rows(X):
    X = X - X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True)
    s[s == 0] = 1
    return (X / s).astype(np.float32)


def rewire_within_adp(ds: Dataset, rng: np.random.Generator):
    """Proximity-constrained null connectome: each axon keeps its number of
    synaptic partners (and its multiset of synapse counts) but the partners are
    redrawn uniformly from its potential (ADP or connected) targets."""
    W = ds.W_syn.tolil()
    A = ds.W_adp.tocsr()
    rows, cols, vals = [], [], []
    for r in range(W.shape[0]):
        counts = np.array(W.data[r], dtype=np.float32)
        if len(counts) == 0:
            continue
        cand = A.indices[A.indptr[r]:A.indptr[r + 1]]
        pick = rng.choice(cand, size=min(len(counts), len(cand)), replace=False)
        rows += [r] * len(pick)
        cols += list(pick)
        vals += list(rng.permutation(counts)[: len(pick)])
    return sp.csr_matrix((vals, (rows, cols)), shape=W.shape, dtype=np.float32)
