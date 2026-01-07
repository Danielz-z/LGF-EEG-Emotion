import os, re, glob, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from models.MAET_model import MAET
from datasets.dataset_MAET import MAETFeatureDataset

def compute_subject_stats_eeg(ds_subset):
    from collections import defaultdict
    acc = defaultdict(list)
    for i in ds_subset.indices:
        s = ds_subset.dataset[i]
        acc[int(s["domain"])].append(s["eeg"].numpy())
    stats = {}
    for sid, xs in acc.items():
        X = np.stack(xs, 0)
        m, s = X.mean(0), X.std(0)
        s[s == 0] = 1.0
        stats[sid] = (
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
        )
    return stats


def compute_subject_stats_bimodal(ds_subset):
    from collections import defaultdict
    acc_eeg, acc_eye = defaultdict(list), defaultdict(list)
    for i in ds_subset.indices:
        s = ds_subset.dataset[i]
        sid = int(s["domain"])
        acc_eeg[sid].append(s["eeg"].numpy())
        if s["eye"] is not None:
            acc_eye[sid].append(s["eye"].numpy())
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


def _normalize_vec(x, ms, device):
    if ms is None or ms[0] is None:
        return x
    m, s = ms
    return (x - m.to(device)) / s.to(device)


def stack_and_normalize_eeg(x_b, sid_b, subj_stats, device, fallback=None):
    xs = []
    for i in range(x_b.size(0)):
        sid = int(sid_b[i].item())
        ms = subj_stats[sid] if sid in subj_stats else fallback
        xs.append(_normalize_vec(x_b[i], ms, device))
    return torch.stack(xs, 0)


def stack_and_normalize_bimodal(
    x_b, v_b, sid_b, subj_stats, device,
    fallback_eeg=None, fallback_eye=None
):
    xs, vs = [], []
    B = sid_b.size(0)
    for i in range(B):
        sid = int(sid_b[i].item())
        eeg_ms = subj_stats.get(sid, ((None, None), (None, None)))[0] or fallback_eeg
        eye_ms = subj_stats.get(sid, ((None, None), (None, None)))[1] or fallback_eye
        xs.append(_normalize_vec(x_b[i], eeg_ms, device))
        vs.append(_normalize_vec(v_b[i], eye_ms, device))
    return torch.stack(xs, 0), torch.stack(vs, 0)

def collate_unimodal(batch):
    eegs, labels, domains = [], [], []
    for b in batch:
        eegs.append(b["eeg"])
        labels.append(b["label"])
        domains.append(b["domain"])
    return {
        "eeg": torch.stack(eegs, 0),
        "label": torch.stack(labels, 0),
        "domain": torch.stack(domains, 0),
    }

def collate_bimodal(batch):
    eegs, eyes, labels, domains = [], [], [], []
    for b in batch:
        eegs.append(b["eeg"])
        labels.append(b["label"])
        domains.append(b["domain"])
        assert b["eye"] is not None, "eye=None, please check"
        eyes.append(b["eye"])
    return {
        "eeg": torch.stack(eegs, 0),
        "eye": torch.stack(eyes, 0),
        "label": torch.stack(labels, 0),
        "domain": torch.stack(domains, 0),
    }


def _match_dim(x, target_dim):
    if target_dim is None:
        return x
    D = x.size(1)
    if D == target_dim:
        return x
    if D < target_dim:
        pad = target_dim - D
        return F.pad(x, (0, pad))
    else:
        return x[:, :target_dim]

@torch.no_grad()
def infer_one_fold_unimodal(
    model, loader, device, subj_stats_val,
    fallback=None, amp=False, eeg_dim_model=None
):
    model.eval()
    ys, ps = [], []
    for b in loader:
        x = b["eeg"].to(device, non_blocking=True)
        sid = b["domain"].to(device, non_blocking=True)
        x = stack_and_normalize_eeg(x, sid, subj_stats_val, device, fallback)
        x = _match_dim(x, eeg_dim_model)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(eeg=x, eye=None, alpha_=0.0)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(-1).cpu().numpy()
        ys.extend(b["label"].cpu().numpy())
        ps.extend(pred)
    return np.array(ys), np.array(ps)

@torch.no_grad()
def infer_one_fold_bimodal(
    model, loader, device, subj_stats_val,
    fallback_eeg=None, fallback_eye=None, amp=False,
    eeg_dim_model=None, eye_dim_model=None
):
    model.eval()
    ys, ps = [], []
    for b in loader:
        x = b["eeg"].to(device, non_blocking=True)
        v = b["eye"].to(device, non_blocking=True)
        sid = b["domain"].to(device, non_blocking=True)
        x, v = stack_and_normalize_bimodal(
            x, v, sid, subj_stats_val, device,
            fallback_eeg=fallback_eeg, fallback_eye=fallback_eye
        )
        x = _match_dim(x, eeg_dim_model)
        v = _match_dim(v, eye_dim_model)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(eeg=x, eye=v, alpha_=0.0)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(-1).cpu().numpy()
        ys.extend(b["label"].cpu().numpy())
        ps.extend(pred)
    return np.array(ys), np.array(ps)


def parse_left_subjects_from_name(name: str):
    m = re.search(r"loso_S([0-9\-]+)", name)
    if not m:
        return None
    subs = [int(x) for x in m.group(1).split("-") if x.isdigit()]
    return set(subs) if subs else None


def infer_config_from_ckpt(state_dict, args_ck, defaults):
    def pick_path(arg_path, default_path):
        if arg_path is None:
            return default_path
        if not isinstance(arg_path, str):
            return default_path
        if os.path.exists(arg_path):
            return arg_path
        return default_path

    eeg_in = state_dict["eeg_transform.transform1.weight"].shape[1]
    eye_w = state_dict.get("eye_transform.transform1.weight", None)
    eye_in = int(eye_w.shape[1]) if eye_w is not None else 0

    labels_csv = pick_path(args_ck.get("labels"),    defaults["labels"])
    eeg_dir    = pick_path(args_ck.get("eeg_dir"),   defaults["eeg_dir"])
    graph_ch_dir_default = pick_path(args_ck.get("graph_ch_dir"), defaults["graph_ch_dir"])
    aux25_dir_default    = pick_path(args_ck.get("aux25_dir"),    defaults["aux25_dir"])

    if eeg_in >= 558:
        graph_ch_dir = graph_ch_dir_default
    else:
        graph_ch_dir = None

    if eye_in == 25:
        aux25_dir = aux25_dir_default
    else:
        aux25_dir = None

    eeg_key_prefix = args_ck.get("eeg_key_prefix", "de_LDS_")
    graph_key_prefix = args_ck.get("graph_key_prefix", "graph_ch_")
    num_classes = args_ck.get("num_classes", 7)

    return {
        "eeg_in": eeg_in,
        "eye_in": eye_in,
        "labels_csv": labels_csv,
        "eeg_dir": eeg_dir,
        "graph_ch_dir": graph_ch_dir,
        "aux25_dir": aux25_dir,
        "eeg_key_prefix": eeg_key_prefix,
        "graph_key_prefix": graph_key_prefix,
        "num_classes": num_classes,
    }


def build_model_from_ckpt(state_dict, args_ckpt, num_domains: int):
    eeg_in = state_dict["eeg_transform.transform1.weight"].shape[1]
    eye_w = state_dict.get("eye_transform.transform1.weight", None)
    eye_in = int(eye_w.shape[1]) if eye_w is not None else 0

    embed_dim = args_ckpt.get("embed_dim", 60)
    depth = args_ckpt.get("depth", 3)
    heads = args_ckpt.get("heads", 6)
    drop_path = args_ckpt.get("drop_path", 0.1)
    num_classes = args_ckpt.get("num_classes", 7)

    model = MAET(
        embed_dim=embed_dim,
        num_classes=num_classes,
        eeg_seq_len=args_ckpt.get("eeg_seq_len", 5),
        eye_seq_len=args_ckpt.get("eye_seq_len", 5),
        eeg_dim=eeg_in,
        eye_dim=eye_in,
        depth=depth,
        num_heads=heads,
        qkv_bias=True,
        mixffn_start_layer_index=args_ckpt.get("mixffn_start_layer_index", 2),
        prompt=args_ckpt.get("prompt", False),
        prompt_len=args_ckpt.get("prompt_len", 2),
        drop_path_rate=drop_path,
        domain_generalization=args_ckpt.get("domain_generalization", True),
        num_domains=num_domains,
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default="./checkpoints", help="weight directory")
    ap.add_argument("--pattern", default="best_maet0_loso_S*", help="Wildcard for matching weights")
    ap.add_argument("--csv_out", default="./metrics_loso_0.csv")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--intensity_threshold", type=float, default=0.20)
    args = ap.parse_args()

    defaults = dict(
        labels="./labels.csv",
        eeg_dir=r"/SEED-VII/EEG_features",
        graph_ch_dir=r"/SEED-VII/GraphPerChannel_features",
        aux25_dir=r"/SEED-VII/Aux25_features",
    )

    ckpts = sorted(glob.glob(os.path.join(args.weights_dir, args.pattern)))
    assert ckpts, f"No checkpoints matched {args.weights_dir}/{args.pattern}"

    device = torch.device(args.device)
    rows, y_all, p_all = [], [], []

    for path in ckpts:
        name = os.path.basename(path)
        left = parse_left_subjects_from_name(name)
        if not left:
            print(f"[WARN] skip (no loso_S*) -> {name}")
            continue

        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        args_ck = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

        cfg = infer_config_from_ckpt(state, args_ck, defaults)

        ds = MAETFeatureDataset(
            eeg_dir=cfg["eeg_dir"],
            graph_ch_dir=cfg["graph_ch_dir"],
            aux25_dir=cfg["aux25_dir"],
            label_csv=cfg["labels_csv"],
            eeg_key_prefix=cfg["eeg_key_prefix"],
            graph_key_prefix=cfg["graph_key_prefix"],
        )

        if hasattr(ds, "meta") and "intensity" in ds.meta.columns:
            thr = args_ck.get("intensity_threshold", args.intensity_threshold)
            if thr is not None:
                before = len(ds.meta)
                ds.meta = ds.meta[
                    (ds.meta["intensity"].fillna(0.0) >= float(thr))
                ].reset_index(drop=True)
                after = len(ds.meta)
                print(f"[{name}] [INFO] intensity >= {thr:.2f}: kept {after}/{before} samples")

        s0 = ds[0]
        data_eeg_dim = int(s0["eeg"].numel())
        data_eye_dim = int(s0["eye"].numel()) if s0["eye"] is not None else 0

        raw_use_eye = data_eye_dim > 0
        use_eye = raw_use_eye and (cfg["eye_in"] == data_eye_dim)

        print(
            f"[CONF] {name}: model_eeg_dim={cfg['eeg_in']}, data_eeg_dim={data_eeg_dim}; "
            f"model_eye_dim={cfg['eye_in']}, data_eye_dim={data_eye_dim}; "
            f"raw_use_eye={raw_use_eye}, use_eye={use_eye}"
        )

        if data_eeg_dim != cfg["eeg_in"]:
            print(f"[WARN] EEG dim mismatch: model={cfg['eeg_in']} vs data={data_eeg_dim}")
        if raw_use_eye and cfg["eye_in"] != data_eye_dim:
            print(f"[INFO] eye dim mismatch: model={cfg['eye_in']} vs data={data_eye_dim} -> fallback to EEG only")

        # LOSO
        groups = ds.meta["file_id"].values
        val_idx = [i for i, g in enumerate(groups) if int(g) in left]
        if not val_idx:
            print(f"[WARN] no samples for {name}, left={left}")
            continue
        train_idx = [i for i in range(len(ds)) if i not in set(val_idx)]

        ds_tr = Subset(ds, train_idx)
        ds_va = Subset(ds, val_idx)

        if use_eye:
            stats_tr = compute_subject_stats_bimodal(ds_tr)
            stats_va = compute_subject_stats_bimodal(ds_va)

            def _mean_pair(d, idx):
                ms = [v[idx][0] for v in d.values() if v[idx][0] is not None]
                ss = [v[idx][1] for v in d.values() if v[idx][1] is not None]
                if ms and ss:
                    return (torch.stack(ms).mean(0), torch.stack(ss).mean(0))
                return (None, None)

            fallback_eeg = _mean_pair(stats_tr, 0)
            fallback_eye = _mean_pair(stats_tr, 1)

            ld_va = DataLoader(
                ds_va,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_bimodal,
            )
        else:
            stats_tr = compute_subject_stats_eeg(ds_tr)
            stats_va = compute_subject_stats_eeg(ds_va)

            fallback = None
            if stats_tr:
                fm = torch.stack([v[0] for v in stats_tr.values()]).mean(0)
                fs = torch.stack([v[1] for v in stats_tr.values()]).mean(0)
                fallback = (fm, fs)

            ld_va = DataLoader(
                ds_va,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_unimodal,
            )

        n_domains = len(np.unique(groups))
        model = build_model_from_ckpt(state, args_ck, num_domains=n_domains).to(device)
        model.load_state_dict(state, strict=True)

        if use_eye:
            y_true, y_pred = infer_one_fold_bimodal(
                model,
                ld_va,
                device,
                stats_va,
                fallback_eeg=fallback_eeg,
                fallback_eye=fallback_eye,
                amp=args.amp,
                eeg_dim_model=cfg["eeg_in"],
                eye_dim_model=cfg["eye_in"],
            )
        else:
            y_true, y_pred = infer_one_fold_unimodal(
                model,
                ld_va,
                device,
                stats_va,
                fallback=fallback,
                amp=args.amp,
                eeg_dim_model=cfg["eeg_in"],
            )

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        f1w = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        per_cls = {
            f"f1_class_{k}": v["f1-score"]
            for k, v in report.items()
            if k.isdigit()
        }

        eeg_in = cfg["eeg_in"]
        eye_in_model = cfg["eye_in"]
        eye_in_data = data_eye_dim if use_eye else 0
        mode = []
        if eeg_in == 310 and eye_in_data == 0:
            mode.append("EEG(310)")
        elif eeg_in == 558 and eye_in_data == 0:
            mode.append("EEG+Graph(558)")
        elif eeg_in == 310 and eye_in_data > 0:
            mode.append(f"EEG(310)+Eye({eye_in_data})")
        elif eeg_in == 558 and eye_in_data > 0:
            mode.append(f"EEG+Graph(558)+Eye({eye_in_data})")
        else:
            if eye_in_data > 0:
                mode.append(f"EEG({eeg_in})+Eye({eye_in_data})")
            else:
                mode.append(f"EEG({eeg_in})")

        rows.append(
            {
                "checkpoint": name,
                "left_subjects": "-".join(map(str, sorted(left))),
                "eeg_in": eeg_in,
                "eye_in_model": eye_in_model,
                "eye_in_data": eye_in_data,
                "use_eye": int(use_eye),
                "mode": "+".join(mode),
                "acc": acc,
                "f1_macro": f1m,
                "f1_weighted": f1w,
                **per_cls,
            }
        )
        y_all.append(y_true)
        p_all.append(y_pred)
        print(
            f"[{name}] {rows[-1]['mode']}  ACC={acc:.4f}  F1m={f1m:.4f}  F1w={f1w:.4f}"
        )

    df = pd.DataFrame(rows).sort_values(["mode", "checkpoint"])
    if y_all:
        y_all = np.concatenate(y_all)
        p_all = np.concatenate(p_all)
        overall = {
            "checkpoint": "ALL",
            "left_subjects": "ALL",
            "eeg_in": -1,
            "eye_in_model": -1,
            "eye_in_data": -1,
            "use_eye": -1,
            "mode": "ALL",
            "acc": accuracy_score(y_all, p_all),
            "f1_macro": f1_score(y_all, p_all, average="macro"),
            "f1_weighted": f1_score(y_all, p_all, average="weighted"),
        }
        df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    df.to_csv(args.csv_out, index=False)

    g = (
        df[df["mode"] != "ALL"]
        .groupby("mode")[["acc", "f1_macro", "f1_weighted"]]
        .mean()
        .reset_index()
    )
    g.to_csv(
        os.path.splitext(args.csv_out)[0] + "_groupmean_0.csv", index=False
    )

    print(f"\nSaved -> {args.csv_out}")
    print(
        f"Saved group means -> {os.path.splitext(args.csv_out)[0]}_groupmean_0.csv"
    )


if __name__ == "__main__":
    main()
