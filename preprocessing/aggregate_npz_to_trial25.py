# aggregate_npz_to_trial25.py
# Aggregate per-subject 25D .npz features (W, 62, 25) into per-trial 25D features and export to .mat.
import os, re
import numpy as np
import scipy.io as sio
import pandas as pd

EEG_DIR   = r"\SEED-VII\EEG_features"
NPZ_DIR   = r"\SEED-VI\features_25dim"
OUT_DIR   = r"\SEED-VII\Aux25_features"
LABEL_CSV = r".\labels.csv"
SF = 200; WIN_SEC = 4; STEP_SEC = 4

os.makedirs(OUT_DIR, exist_ok=True)

def windows_in_trial(eeg_mat, trial):
    """Compute the number of windows for a trial using (N - win) / step + 1 based on the EEG signal length."""
    key = f"de_LDS_{trial}"
    for k in (key, f"de_{trial}", f"psd_{trial}"):
        if k in eeg_mat:
            arr = np.array(eeg_mat[k])     # [T, 5, 62]
            if arr.ndim != 3: raise ValueError(f"{k} shape={arr.shape}")
            return int(arr.shape[0])
    raise KeyError(f"trial {trial} not found in EEG mat")

def aggregate_trial(block_3d):
    if block_3d.size == 0:
        return np.zeros((25,), dtype=np.float32)
    aux = np.nanmean(block_3d, axis=(0, 1))   # (25,)
    aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)
    return aux.astype(np.float32)

def process_subject(file_id, labels_df):
    eeg_path = os.path.join(EEG_DIR, f"{file_id}.mat")
    npz_path = os.path.join(NPZ_DIR, f"{file_id}.npz")
    if not os.path.isfile(eeg_path):
        print(f"[WARN] skip {file_id}: EEG mat not found"); return
    if not os.path.isfile(npz_path):
        print(f"[WARN] skip {file_id}: NPZ not found"); return

    eeg = sio.loadmat(eeg_path)
    npz = np.load(npz_path, allow_pickle=True)
    X = np.array(npz["features"])  # (W,62,25)

    sub = labels_df[labels_df["file_id"].astype(int) == int(file_id)]
    trials = sorted(set(sub["trial"].astype(int).tolist())) if not sub.empty else list(range(1,81))

    ptr = 0; aux_dict = {}; issues = []
    total_W_expected = 0
    Ws = []
    for t in trials:
        Wt = windows_in_trial(eeg, t)
        Ws.append(Wt); total_W_expected += Wt

    if X.shape[0] < total_W_expected:
        issues.append(f"NPZ windows {X.shape[0]} < expected {total_W_expected} (will pad zeros)")
    elif X.shape[0] > total_W_expected:
        issues.append(f"NPZ windows {X.shape[0]} > expected {total_W_expected} (tail will be dropped)")

    for t, Wt in zip(trials, Ws):
        end = min(ptr + Wt, X.shape[0])
        blk = X[ptr:end]                  # (<=Wt,62,25)
        if blk.shape[0] < Wt:
            pad = np.zeros((Wt - blk.shape[0], X.shape[1], X.shape[2]), dtype=X.dtype)
            blk = np.concatenate([blk, pad], axis=0)
        aux25 = aggregate_trial(blk)
        aux_dict[f"aux_{t}"] = aux25.reshape(1, 25)
        ptr += Wt

    out_path = os.path.join(OUT_DIR, f"{file_id}.mat")
    sio.savemat(out_path, aux_dict)
    print(f"[OK] saved {out_path} with {len(aux_dict)} trials; issues={len(issues)}")
    for s in issues: print("  -", s)

def main():
    if not os.path.isfile(LABEL_CSV):
        raise FileNotFoundError("labels.csv not found")
    df = pd.read_csv(LABEL_CSV)

    subj_ids = []
    for name in os.listdir(NPZ_DIR):
        m = re.match(r"^\s*(\d+)\.npz\s*$", name)
        if m: subj_ids.append(int(m.group(1)))
    subj_ids = sorted(set(subj_ids))
    print(f"[INFO] found {len(subj_ids)} npz subjects: {subj_ids}")

    for sid in subj_ids:
        process_subject(sid, df)

if __name__ == "__main__":
    main()
