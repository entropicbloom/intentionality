"""Download session-A NWB files for VISp excitatory containers with >= MIN_CELLS
orientation-labelled cells, and cache per-experiment arrays:
  responses to natural_movie_one (trial-averaged over 10 repeats, 900 frames)
  responses to natural_movie_three (10 repeats, 3600 frames)
  labels: pref_dir_dg, osi_dg, g_osi_dg, p_dg, dsi_dg, reliability_dg; cell ids; container / donor.
Both movies are shared across every session A in the dataset, so response
vectors are comparable across mice."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADATA = os.path.join(ROOT, "allen_data")
MIN_CELLS = 150
EXC = "Slc17a7|Cux2|Emx1|Rorb|Scnn1a|Nr5a1|Rbp4|Fezf2|Tlx3|Ntsr1"


def trial_average(dff, table, n_frames):
    """mean dF/F per movie frame across repeats -> (cells, n_frames)."""
    out = np.zeros((dff.shape[0], n_frames), np.float32); cnt = np.zeros(n_frames)
    for fr, s, e in zip(table.frame.values, table.start.values, table.end.values):
        if e <= s:                      # 31 Hz imaging vs 30 Hz movie: some frames have no sample
            continue
        v = np.nanmean(dff[:, s:e], 1); good = np.isfinite(v)
        out[good, fr] += v[good]; cnt[fr] += 1
    return out / np.maximum(cnt, 1)


SESSIONS = {"A": ("three_session_A", [("natural_movie_one", 900), ("natural_movie_three", 3600)]),
            "C": (("three_session_C", "three_session_C2"), [("natural_movie_one", 900), ("natural_movie_two", 900)]),
            # drifting gratings (session A): condition-averaged responses, 8 directions x 5 temporal frequencies
            "DG": ("three_session_A", None)}


def grating_responses(dff, table):
    """mean dF/F per (direction, temporal frequency) condition over trials, minus the
    blank-sweep mean -> (cells, 40); also the per-condition time course (cells, 40, 60)
    at 30 Hz over the 2 s presentation."""
    t = table[(table.blank_sweep == 0) & table.orientation.notna()]
    dirs, tfs = sorted(t.orientation.unique()), sorted(t.temporal_frequency.unique())
    blank = table[table.blank_sweep == 1]
    b = np.mean([dff[:, s:e].mean(1) for s, e in zip(blank.start.values, blank.end.values)], 0) if len(blank) else 0
    mean = np.zeros((dff.shape[0], len(dirs), len(tfs)), np.float32); course = np.zeros((dff.shape[0], len(dirs), len(tfs), 60), np.float32)
    for i, d in enumerate(dirs):
        for j, f in enumerate(tfs):
            tt = t[(t.orientation == d) & (t.temporal_frequency == f)]
            segs = [dff[:, s:s + 60] for s, e in zip(tt.start.values, tt.end.values) if s + 60 <= dff.shape[1]]
            course[:, i, j] = np.mean(segs, 0); mean[:, i, j] = np.mean([sg.mean(1) for sg in segs], 0) - b
    return mean.reshape(dff.shape[0], -1), course.reshape(dff.shape[0], -1), np.array(dirs), np.array(tfs)


def main(max_containers=None, session="A"):
    stype, movies = SESSIONS[session]
    stype = (stype,) if isinstance(stype, str) else stype
    boc = BrainObservatoryCache(manifest_file=os.path.join(ADATA, "manifest.json"))
    ecs = pd.DataFrame(boc.get_experiment_containers())
    exps = pd.DataFrame(boc.get_ophys_experiments())
    cells = pd.DataFrame(boc.get_cell_specimens())
    exc = ecs[ecs.cre_line.str.contains(EXC, regex=True) & (ecs.targeted_structure == "VISp")]
    sig = cells[(cells.area == "VISp") & (cells.p_dg < 0.05) & cells.osi_dg.notna() & cells.experiment_container_id.isin(exc.id)]
    counts = sig.groupby("experiment_container_id").size().sort_values(ascending=False)
    keep = counts[counts >= MIN_CELLS].index.tolist()[: (max_containers or 10 ** 6)]
    meta = []
    for cid in keep:
        row = exps[exps.session_type.isin(stype) & (exps.experiment_container_id == cid)]
        if len(row) == 0:
            continue
        eid = int(row.id.iloc[0]); out = os.path.join(ADATA, "cache" if session == "A" else f"cache_{session}", f"{eid}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            meta.append(dict(container=int(cid), experiment=eid)); continue
        t = time.time(); nwb = os.path.join(ADATA, "ophys_experiment_data", f"{eid}.nwb")
        for attempt in range(2):
            try:
                ds = boc.get_ophys_experiment_data(eid); break
            except Exception as e:                      # truncated / corrupt download: remove and retry once
                print(f"{eid}: open failed ({type(e).__name__}), retrying", flush=True)
                if os.path.exists(nwb): os.remove(nwb)
                if attempt == 1: raise
        ids = np.array(ds.get_cell_specimen_ids()); ts, dff = ds.get_dff_traces()
        if session == "DG":
            r1, r3, dirs, tfs = grating_responses(dff, ds.get_stimulus_table("drifting_gratings"))
        else:
            r1 = trial_average(dff, ds.get_stimulus_table(movies[0][0]), movies[0][1])
            r3 = trial_average(dff, ds.get_stimulus_table(movies[1][0]), movies[1][1])
        lab = cells.set_index("cell_specimen_id").reindex(ids)
        cells.to_csv(os.path.join(ADATA, "cell_specimens.csv"), index=False)
        np.savez(out, cell_ids=ids, r_nm1=r1, r_nm3=r3,
                 pref_dir_dg=lab.pref_dir_dg.values.astype(float), osi_dg=lab.osi_dg.values.astype(float),
                 g_osi_dg=lab.g_osi_dg.values.astype(float), p_dg=lab.p_dg.values.astype(float),
                 dsi_dg=lab.dsi_dg.values.astype(float), reliability_dg=lab.reliability_dg.values.astype(float),
                 container=cid, experiment=eid, donor=str(ecs.set_index("id").loc[cid, "donor_name"]),
                 cre=str(ecs.set_index("id").loc[cid, "cre_line"]), depth=int(ecs.set_index("id").loc[cid, "imaging_depth"]))
        meta.append(dict(container=int(cid), experiment=eid))
        print(f"{eid}: {len(ids)} cells, {time.time()-t:.0f}s", flush=True)
        os.remove(os.path.join(ADATA, "ophys_experiment_data", f"{eid}.nwb"))   # keep only the cache
    json.dump(meta, open(os.path.join(ADATA, "cache" if session == "A" else f"cache_{session}", "index.json"), "w"))
    print("done", len(meta))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None, session=sys.argv[2] if len(sys.argv) > 2 else "A")
