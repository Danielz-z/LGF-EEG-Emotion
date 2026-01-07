import os, glob
import numpy as np
import scipy.io as sio

EEG_TRIAL_MAT_DIR = r"/SEED-VII/EEG_raw_trials_mat"
OUT_DIR            = r"/SEED-VII/GraphPerChannel_features"
os.makedirs(OUT_DIR, exist_ok=True)

def _degree_w(C):
    return C.sum(axis=1)          # [62]

def _clustering_unw(C):
    A = (C > 0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    k = A.sum(1)                  # [62]
    A2 = A @ A
    A3_diag = (A2 @ A).diagonal()
    tri = A3_diag / 2.0
    denom = k * (k - 1) / 2.0
    denom[denom == 0] = 1.0
    return (tri / denom).astype(np.float32)

def _betweenness_approx(C, topk=10):
    d = _degree_w(C)
    d_sorted = np.sort(d)[::-1]
    thr = d_sorted[min(topk, len(d_sorted)-1)]
    return (d >= thr).astype(np.float32)

def _pagerank_like(C, iters=20, damping=0.85):
    W = C.copy()
    col_sum = W.sum(0, keepdims=True) + 1e-8
    P = W / col_sum
    n = P.shape[0]
    pr = np.ones((n,), dtype=np.float32) / n
    for _ in range(iters):
        pr = damping * (P @ pr) + (1 - damping) / n
    return pr

def corr_relu(window_62_t: np.ndarray) -> np.ndarray:
    """window_62_t: [62, T] -> corrcoef -> ReLU(>0) & diag=0"""
    C = np.corrcoef(window_62_t)  # [-1,1], [62,62]
    np.fill_diagonal(C, 0.0)
    C = np.maximum(C, 0.0)
    return C.astype(np.float32)

def features_from_window(C: np.ndarray) -> np.ndarray:
    deg = _degree_w(C)
    clu = _clustering_unw(C)
    bc  = _betweenness_approx(C)
    pr  = _pagerank_like(C)
    return np.concatenate(
        [deg.reshape(-1,1), clu.reshape(-1,1), bc.reshape(-1,1), pr.reshape(-1,1)],
        axis=1
    ).astype(np.float32)                  # [62,4]

def process_subject(mat_path: str):
    subj = os.path.splitext(os.path.basename(mat_path))[0]
    d = sio.loadmat(mat_path)
    trial_keys = sorted([int(k) for k in d.keys() if str(k).isdigit()])

    out = {}
    for t in trial_keys:
        X = np.array(d[str(t)])
        if X.ndim != 2 or X.shape[0] != 62:
            print(f"[WARN] {subj} trial {t} unexpected shape {X.shape}, skip")
            continue

        C = corr_relu(X)                 # [62,62]
        feat_62x4 = features_from_window(C)   # [62,4]

        out[f"graph_ch_{t}"] = feat_62x4

    save_path = os.path.join(OUT_DIR, f"{subj}.mat")
    sio.savemat(save_path, out)
    print(f"[OK] saved {save_path} with {len(out)} trials")

def main():
    for p in glob.glob(os.path.join(EEG_TRIAL_MAT_DIR, "*.mat")):
        process_subject(p)

if __name__ == "__main__":
    main()
