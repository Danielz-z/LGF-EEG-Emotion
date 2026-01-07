import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

from datasets.dataset_MAET import MAETFeatureDataset
from train_MAET_amplified_LOSO import (
    build_model,
    stack_and_normalize_bimodal,
)

CHECKPOINT_DIR = r"/checkpoints"
LABEL_CSV = r"./labels.csv"
EEG_DIR = r"/SEED-VII/EEG_features"
GRAPH_DIR = r"/SEED-VII/GraphPerChannel_features"
AUX25_DIR = r"/SEED-VII/Aux25_features"

BATCH_SIZE = 128
NUM_CLASSES = 7
CLASS_NAMES = ["Disgust", "Fear", "Sad", "Neutral", "Happy", "Anger", "Surprise"]
EYE_DIM = 25


def collate(batch):
    eegs, eyes, labels, domains = [], [], [], []

    for b in batch:
        eegs.append(b["eeg"])

        eye = b["eye"]
        if eye is None:
            eye = torch.zeros(EYE_DIM, dtype=torch.float32)
        eyes.append(eye)

        labels.append(b["label"])
        domains.append(b["domain"])

    return {
        "eeg": torch.stack(eegs, 0),
        "eye": torch.stack(eyes, 0),
        "label": torch.stack(labels, 0),
        "domain": torch.stack(domains, 0),
    }


def load_dataset():
    ds = MAETFeatureDataset(
        eeg_dir=EEG_DIR,
        graph_ch_dir=GRAPH_DIR,
        aux25_dir=AUX25_DIR,
        label_csv=LABEL_CSV,
        eeg_key_prefix="de_LDS_",
        graph_key_prefix="graph_ch_",
    )

    if hasattr(ds, "meta") and "intensity" in ds.meta.columns:
        thr = 0.20   #intensity threshold
        before = len(ds.meta)
        ds.meta = ds.meta[(ds.meta["intensity"].fillna(0.0) >= thr)].reset_index(drop=True)
        after = len(ds.meta)
        print(f"[INFO] intensity >= {thr:.2f}: kept {after}/{before} samples")

    d0_eeg = int(ds[0]["eeg"].numel())

    eye0 = ds[0]["eye"]
    if eye0 is None:
        d0_eye = EYE_DIM
        print("[WARN] eye features missing for sample 0 → using zeros as fallback")
    else:
        d0_eye = int(eye0.numel())

    print(f"[CONF] sample dims: eeg={d0_eeg}, eye={d0_eye}")
    return ds


def get_subject_to_ckpt():
    pattern = os.path.join(CHECKPOINT_DIR, "maet.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoint found with pattern: {pattern}")

    subj2ckpt = {}
    for p in paths:
        m = re.search(r"_loso_S(\d+)\.pt$", os.path.basename(p))
        if m:
            sid = int(m.group(1))
            subj2ckpt[sid] = p

    if not subj2ckpt:
        raise RuntimeError("No 'loso_S#.pt' checkpoint parsed. Check filename pattern.")

    print("[INFO] Found checkpoints for subjects:", sorted(subj2ckpt.keys()))
    return subj2ckpt


def build_model_from_ckpt(ckpt_path, num_domains, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args_dict = ckpt.get("args", {})

    class DummyArgs:
        pass

    args = DummyArgs()
    setattr(args, "embed_dim", args_dict.get("embed_dim", 60))
    setattr(args, "depth", args_dict.get("depth", 3))
    setattr(args, "heads", args_dict.get("heads", 6))
    setattr(args, "drop_path", args_dict.get("drop_path", 0.1))

    model = build_model(args, num_domains=num_domains)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def compute_subject_stats_bimodal(ds_subset):
    from collections import defaultdict

    if isinstance(ds_subset, Subset):
        indices = ds_subset.indices
        base_ds = ds_subset.dataset
    else:
        indices = range(len(ds_subset))
        base_ds = ds_subset

    acc_eeg, acc_eye = defaultdict(list), defaultdict(list)

    for i in indices:
        s = base_ds[i]
        sid = int(s["domain"])
        eeg = s["eeg"]
        eye = s["eye"]

        acc_eeg[sid].append(eeg.numpy())

        if eye is not None:
            acc_eye[sid].append(eye.numpy())

    stats = {}
    for sid in set(list(acc_eeg.keys()) + list(acc_eye.keys())):
        # eeg
        if sid in acc_eeg and len(acc_eeg[sid]) > 0:
            Xe = np.stack(acc_eeg[sid], 0)
            me, se = Xe.mean(0), Xe.std(0)
            se[se == 0] = 1.0
            eeg_pair = (
                torch.tensor(me, dtype=torch.float32),
                torch.tensor(se, dtype=torch.float32),
            )
        else:
            eeg_pair = (None, None)

        # eye
        if sid in acc_eye and len(acc_eye[sid]) > 0:
            Xv = np.stack(acc_eye[sid], 0)
            mv, sv = Xv.mean(0), Xv.std(0)
            sv[sv == 0] = 1.0
            eye_pair = (
                torch.tensor(mv, dtype=torch.float32),
                torch.tensor(sv, dtype=torch.float32),
            )
        else:
            eye_pair = (None, None)

        stats[sid] = (eeg_pair, eye_pair)

    return stats


def eval_one_subject(subj_id, model, ds, device):
    meta = ds.meta
    idx = np.where(meta["file_id"].values == subj_id)[0]
    if len(idx) == 0:
        raise RuntimeError(f"No samples found for subject {subj_id} in dataset.")

    subset = Subset(ds, idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    subj_stats = compute_subject_stats_bimodal(subset)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for b in loader:
            x = b["eeg"].to(device, non_blocking=True)
            v = b["eye"].to(device, non_blocking=True)
            sid = b["domain"].to(device, non_blocking=True)

            x, v = stack_and_normalize_bimodal(
                x, v, sid,
                subj_stats,
                device,
                fallback_eeg=None,
                fallback_eye=None,
            )

            out = model(eeg=x, eye=v, alpha_=0.0)
            if isinstance(out, tuple):
                out = out[0]

            preds = out.argmax(dim=-1).cpu().numpy()
            labels = b["label"].cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    print(f"[SUBJ {subj_id}] samples = {len(y_true)}")
    return y_true, y_pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = load_dataset()
    subj2ckpt = get_subject_to_ckpt()

    n_domains = len(np.unique(ds.meta["domain"].values))

    all_cm_norm = []
    all_prec, all_rec, all_f1 = [], [], []

    for subj_id, ckpt_path in sorted(subj2ckpt.items()):
        print(f"\n========== Subject {subj_id} | ckpt = {ckpt_path} ==========")
        model = build_model_from_ckpt(ckpt_path, num_domains=n_domains, device=device)
        y_true, y_pred = eval_one_subject(subj_id, model, ds, device)

        cm_norm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            normalize="true",
        )
        all_cm_norm.append(cm_norm)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

    mean_cm = np.mean(np.stack(all_cm_norm, axis=0), axis=0)
    mean_prec = np.mean(np.stack(all_prec, axis=0), axis=0)
    mean_rec = np.mean(np.stack(all_rec, axis=0), axis=0)
    mean_f1 = np.mean(np.stack(all_f1, axis=0), axis=0)

    print("\n===== Mean per-class metrics over all LOSO folds =====")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{i} - {name:9s} | "
            f"P={mean_prec[i]:.3f}  R={mean_rec[i]:.3f}  F1={mean_f1[i]:.3f}"
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_cm, interpolation="nearest")
    ax.set_title("Mean Confusion Matrix (normalized)")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = mean_cm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig("mean_confusion_matrix.png", dpi=300)
    print("[SAVE] mean_confusion_matrix.png")

    x = np.arange(NUM_CLASSES)
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, mean_prec, width, label="Precision")
    plt.bar(x, mean_rec, width, label="Recall")
    plt.bar(x + width, mean_f1, width, label="F1-score")

    plt.xticks(x, CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Per-class Precision / Recall / F1 (mean over LOSO folds)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("class_prf_bars.png", dpi=300)
    print("[SAVE] class_prf_bars.png")


if __name__ == "__main__":
    main()
