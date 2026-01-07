import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib

def _norm_fid(x):
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
    except Exception:
        pass
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _loadmat_any(path: Path):
    try:
        return loadmat(path)
    except OSError:
        try:
            import h5py
            out = {}
            with h5py.File(str(path), "r") as f:
                def _read(name):
                    obj = f[name]
                    if hasattr(obj, "dtype"):  # dataset
                        arr = obj[()]
                        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                            return np.array(arr).T
                        return np.array(arr)
                    # group
                    d = {}
                    for k in obj.keys():
                        d[k] = _read(f"{name}/{k}")
                    return d
                for k in f.keys():
                    out[k] = _read(k)
            return out
        except Exception as e:
            raise e


def _first_existing(*cands: Path) -> Path:
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    return cands[0]


def load_one_trial(eeg_dir, graph_dir, aux_dir, file_id, trial):
    feats = {}
    fid = _norm_fid(file_id)

    # EEG: de_LDS_{t} -> [T,5,62] -> mean over T -> flatten 310D
    if eeg_dir:
        eeg_p1 = Path(eeg_dir) / f"{fid}.mat"
        eeg_p2 = Path(eeg_dir) / f"{file_id}.mat"
        eeg_mat_path = _first_existing(eeg_p1, eeg_p2)
        mat_eeg = _loadmat_any(eeg_mat_path)
        key = f"de_LDS_{trial}"
        if key not in mat_eeg:
            raise KeyError(f"[EEG] missing key '{key}' in {eeg_mat_path.name}")
        x = np.array(mat_eeg[key])
        if x.ndim == 3:        # [T,5,62]
            x = x.mean(axis=0) # [5,62]
        elif x.ndim == 2:      # [5,62] or [62,5]
            if x.shape == (62, 5):
                x = x.T
        else:
            raise ValueError(f"[EEG] unexpected shape for {key}: {x.shape}")
        feats["eeg"] = x.reshape(-1).astype(np.float32)  # 5*62=310

    # Graph: graph_ch_{t} -> [62,4] (or with time) -> flatten 248D
    if graph_dir:
        g_p1 = Path(graph_dir) / f"{fid}.mat"
        g_p2 = Path(graph_dir) / f"{file_id}.mat"
        g_mat_path = _first_existing(g_p1, g_p2)
        mat_g = _loadmat_any(g_mat_path)
        key = f"graph_ch_{trial}"
        if key not in mat_g:
            raise KeyError(f"[Graph] missing key '{key}' in {g_mat_path.name}")
        g = np.array(mat_g[key])
        if g.ndim == 3:
            if g.shape[0] not in (62, 4):
                g = g.mean(axis=0)
            elif g.shape[-1] not in (62, 4):
                g = g.mean(axis=-1)
        if g.ndim != 2:
            raise ValueError(f"[Graph] unexpected shape for {key}: {g.shape}")
        if g.shape == (4, 62):
            g = g.T
        feats["graph"] = g.reshape(-1).astype(np.float32)  # 62*4=248

    # Aux25: aux_{t} -> (25,) / (1,25) / (25,1)
    if aux_dir:
        a_p1 = Path(aux_dir) / f"{fid}.mat"
        a_p2 = Path(aux_dir) / f"{file_id}.mat"
        a_mat_path = _first_existing(a_p1, a_p2)
        mat_a = _loadmat_any(a_mat_path)
        key = f"aux_{trial}"
        if key not in mat_a:
            raise KeyError(f"[Aux] missing key '{key}' in {a_mat_path.name}")
        a = np.array(mat_a[key]).reshape(-1).astype(np.float32)
        feats["aux"] = a  # 25D

    return feats


def assemble_feature(feats, which: str):
    if which == "all":
        keys = ["eeg", "graph", "aux"]
    else:
        keys = which.split("+")  # e.g., "eeg+graph" -> ["eeg","graph"]

    parts = []
    missing = []
    for k in keys:
        if k in feats:
            parts.append(feats[k])
        else:
            missing.append(k)

    if not parts:
        raise ValueError(f"No features selected or available for feat_set='{which}'")

    if missing:
        raise KeyError(f"Requested feature(s) missing for feat_set='{which}': {missing}")

    return np.concatenate(parts, axis=0)


def run_fold(train_df, test_df, args, fold_idx, test_subject, save_dir):
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    # build train
    for _, r in train_df.iterrows():
        feats = load_one_trial(args.eeg_dir, args.graph_ch_dir, args.aux25_dir,
                               r.file_id, int(r.trial))
        x = assemble_feature(feats, args.feat_set)
        X_train.append(x); y_train.append(int(r.label))

    # build test
    for _, r in test_df.iterrows():
        feats = load_one_trial(args.eeg_dir, args.graph_ch_dir, args.aux25_dir,
                               r.file_id, int(r.trial))
        x = assemble_feature(feats, args.feat_set)
        X_test.append(x); y_test.append(int(r.label))

    X_train, X_test = np.vstack(X_train), np.vstack(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    if args.model == "svm_rbf":
        clf = SVC(C=2.0, gamma="scale", kernel="rbf",
                  class_weight="balanced", probability=False, random_state=42)
    elif args.model == "svm_linear":
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    elif args.model == "logreg":
        clf = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1,
                                 class_weight="balanced", penalty="l2", C=1.0,
                                 random_state=42)
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                     class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=42)
    elif args.model == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    elif args.model == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256, 128),
                            activation="relu", solver="adam",
                            alpha=1e-4, batch_size="auto",
                            max_iter=200, early_stopping=True,
                            n_iter_no_change=15, validation_fraction=0.1,
                            random_state=42, verbose=False)
    else:
        raise ValueError(f"Unknown models: {args.model}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm  = confusion_matrix(y_test, y_pred)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    sj_str = f"{test_subject:02d}" if isinstance(test_subject, (int, np.integer)) or \
             (isinstance(test_subject, str) and test_subject.isdigit()) else str(test_subject)
    stem = f"{args.feat_set.upper()}_{args.model}_testS_{sj_str}_ACC{acc:.4f}_F1{f1m:.4f}"
    model_path = Path(save_dir) / f"{stem}.joblib"
    meta_path  = Path(save_dir) / f"{stem}.json"

    bundle = {
        "scaler": scaler,
        "models": clf,
        "config": {
            "feat_set": args.feat_set,
            "model_name": args.model,
            "test_subject": test_subject,
            "intensity_threshold": args.intensity_threshold,
            "eeg_dir": args.eeg_dir,
            "graph_ch_dir": args.graph_ch_dir,
            "aux25_dir": args.aux25_dir,
        },
        "metrics": {
            "acc": float(acc),
            "f1_macro": float(f1m),
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
    }
    joblib.dump(bundle, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"] | bundle["config"], f, ensure_ascii=False, indent=2)

    return acc, f1m, cm, str(model_path)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Path to labels.csv")
    p.add_argument("--eeg_dir", default=None)
    p.add_argument("--graph_ch_dir", default=None)
    p.add_argument("--aux25_dir", default=None)
    p.add_argument("--feat_set", default="all",
                   choices=["eeg", "graph", "aux",
                            "eeg+graph", "eeg+aux", "graph+aux", "all"])
    p.add_argument("--models", default="svm_rbf",
                   choices=["svm_rbf", "svm_linear", "logreg", "rf", "knn", "mlp"])
    p.add_argument("--intensity_threshold", type=float, default=0.20,
                   help="Keep samples with intensity >= threshold; set <0 to disable")
    p.add_argument("--save", default="./baseline_results.json",
                   help="Summary JSON path")
    p.add_argument("--save_models_dir", default="./baseline_models",
                   help="Directory to save per-fold best models")
    args = p.parse_args()

    df = pd.read_csv(args.labels)

    needed = {"file_id", "trial", "label", "intensity", "domain"}
    if not needed.issubset(df.columns):
        raise ValueError(f"labels.csv must contain columns: {needed}")

    df["file_id"] = df["file_id"].apply(_norm_fid)
    df["trial"]   = df["trial"].astype(int)

    if args.intensity_threshold >= 0:
        df = df[df["intensity"] >= args.intensity_threshold].reset_index(drop=True)

    # LOSO by domain/subject
    subjects = sorted(df["domain"].unique().tolist())
    all_acc, all_f1, cms, saved_paths = [], [], [], []

    for i, s in enumerate(subjects):
        train_df = df[df["domain"] != s].reset_index(drop=True)
        test_df  = df[df["domain"] == s].reset_index(drop=True)
        acc, f1m, cm, mpath = run_fold(train_df, test_df, args, i, s, args.save_models_dir)
        all_acc.append(acc); all_f1.append(f1m); cms.append(cm.tolist()); saved_paths.append(mpath)
        print(f"[LOSO] test_subject={str(s):>3}  ACC={acc:.4f}  F1m={f1m:.4f}  | saved: {mpath}")

    summary = {
        "feat_set": args.feat_set,
        "models": args.model,
        "intensity_threshold": args.intensity_threshold,
        "mean_acc": float(np.mean(all_acc)),
        "std_acc": float(np.std(all_acc)),
        "mean_f1m": float(np.mean(all_f1)),
        "std_f1m": float(np.std(all_f1)),
        "per_fold_acc": [float(x) for x in all_acc],
        "per_fold_f1m": [float(x) for x in all_f1],
        "confusion_matrices": cms,
        "subjects": subjects,
        "saved_model_paths": saved_paths,
    }
    Path(os.path.dirname(args.save) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Summary saved to {args.save}")
    print(f"ACC mean±std = {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f} | "
          f"F1m mean±std = {summary['mean_f1m']:.4f} ± {summary['std_f1m']:.4f}")


if __name__ == "__main__":
    main()
