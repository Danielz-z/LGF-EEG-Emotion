# train_MAET_LOSO.py — EEG(558) -> EEG branch + Aux25(25) -> EYE branch, LOSO protocol
import os
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from models.MAET_model import MAET
from datasets.dataset_MAET import MAETFeatureDataset

try:
    from torch.amp import autocast, GradScaler  # PyTorch ≥ 1.13
    AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW = False



def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.0, mode="max"):
        self.patience = patience; self.min_delta = min_delta; self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf; self.counter = 0
    def step(self, v):
        imp = (v > self.best + self.min_delta) if self.mode=="max" else (v < self.best - self.min_delta)
        if imp: self.best=v; self.counter=0; return False
        self.counter += 1; return self.counter > self.patience



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=str, default=r"/data/labels.csv")
    ap.add_argument("--eeg_dir", type=str, default=r"/data/seedVII/EEG_features")
    ap.add_argument("--graph_ch_dir", type=str, default=r"/data/GraphPerChannel_features")
    ap.add_argument("--aux25_dir", type=str, default=r"/data/seedVII/Aux25_features")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save", type=str, default=r"/data/checkpoints/maet_558eeg_25eye.pt")
    ap.add_argument("--seed", type=int, default=42)

    # models
    ap.add_argument("--embed_dim", type=int, default=60)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--drop_path", type=float, default=0.1)

    # domain
    ap.add_argument("--lambda_dom_max", type=float, default=0.00)
    ap.add_argument("--feat_dropout", type=float, default=0.05)

    # eval
    ap.add_argument("--eval_mode", type=str, choices=["loso", "group_split"], default="loso")
    ap.add_argument("--test_size", type=float, default=0.2)

    # filter
    ap.add_argument("--intensity_threshold", type=float, default=0.20)

    # AMP
    ap.add_argument("--amp", action="store_true")

    return ap.parse_args()



def collate(batch):
    eegs, eyes, labels, domains = [], [], [], []
    for b in batch:
        eegs.append(b["eeg"]); eyes.append(b["eye"])
        labels.append(b["label"]); domains.append(b["domain"])
    return {
        "eeg": torch.stack(eegs, 0),      # [B,558]
        "eye": torch.stack(eyes, 0),      # [B,25]
        "label": torch.stack(labels, 0),
        "domain": torch.stack(domains, 0)
    }



def compute_subject_stats_bimodal(ds_subset):
    from collections import defaultdict
    acc_eeg, acc_eye = defaultdict(list), defaultdict(list)
    for i in ds_subset.indices:
        s = ds_subset.dataset[i]
        sid = int(s["domain"])
        acc_eeg[sid].append(s["eeg"].numpy())   # 558
        acc_eye[sid].append(s["eye"].numpy())   # 25
    stats = {}
    for sid in set(list(acc_eeg.keys()) + list(acc_eye.keys())):
        # eeg
        if sid in acc_eeg and len(acc_eeg[sid])>0:
            Xe = np.stack(acc_eeg[sid], 0); me, se = Xe.mean(0), Xe.std(0); se[se==0]=1.0
            eeg_pair = (torch.tensor(me, dtype=torch.float32), torch.tensor(se, dtype=torch.float32))
        else:
            eeg_pair = (None, None)
        # eye
        if sid in acc_eye and len(acc_eye[sid])>0:
            Xv = np.stack(acc_eye[sid], 0); mv, sv = Xv.mean(0), Xv.std(0); sv[sv==0]=1.0
            eye_pair = (torch.tensor(mv, dtype=torch.float32), torch.tensor(sv, dtype=torch.float32))
        else:
            eye_pair = (None, None)
        stats[sid] = (eeg_pair, eye_pair)
    return stats


def _normalize_vec(x, ms, device):
    if ms is None or ms[0] is None: return x
    m, s = ms; return (x - m.to(device)) / s.to(device)


def stack_and_normalize_bimodal(x_b, v_b, sid_b, subj_stats, device,
                                fallback_eeg=None, fallback_eye=None):
    xs, vs = [], []
    B = sid_b.size(0)
    for i in range(B):
        sid = int(sid_b[i].item())
        eeg_ms = subj_stats.get(sid, ((None,None),(None,None)))[0] or fallback_eeg
        eye_ms = subj_stats.get(sid, ((None,None),(None,None)))[1] or fallback_eye
        xs.append(_normalize_vec(x_b[i], eeg_ms, device))
        vs.append(_normalize_vec(v_b[i], eye_ms, device))
    return torch.stack(xs,0), torch.stack(vs,0)


def evaluate(model, loader, device, subj_stats_val, fallback_eeg=None, fallback_eye=None, amp=False):
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for b in loader:
            x = b["eeg"].to(device, non_blocking=True)   # [B,558]
            v = b["eye"].to(device, non_blocking=True)   # [B,25]
            sid = b["domain"].to(device, non_blocking=True)
            x, v = stack_and_normalize_bimodal(x, v, sid, subj_stats_val, device,
                                               fallback_eeg=fallback_eeg, fallback_eye=fallback_eye)
            with autocast(device_type="cuda", enabled=amp):
                out = model(eeg=x, eye=v, alpha_=0.0)
            if isinstance(out, tuple): out = out[0]
            preds.append(out.argmax(-1).cpu())
            gts.append(b["label"].cpu())
    return accuracy_score(torch.cat(gts).numpy(), torch.cat(preds).numpy())


def compute_class_weights(ds_subset):
    labels = [int(ds_subset.dataset[i]["label"]) for i in ds_subset.indices]
    cnt = np.bincount(np.array(labels), minlength=7).astype(np.float32)
    w = cnt.sum()/np.maximum(cnt,1.0); w = w/w.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_model(args, num_domains):
    model = MAET(
        embed_dim=args.embed_dim, num_classes=7,
        eeg_seq_len=5, eye_seq_len=5,
        eeg_dim=558, eye_dim=25,            # EEG=558, Aux25=25
        depth=args.depth, num_heads=args.heads, qkv_bias=True,
        mixffn_start_layer_index=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        prompt=False, prompt_len=2,
        drop_path_rate=args.drop_path,
        domain_generalization=True, num_domains=num_domains
    )
    return model


def train_one_split(args, ds_train, ds_val, device, fold_name, n_domains):
    loader_tr = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, collate_fn=collate)
    loader_va = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True, collate_fn=collate)

    stats_tr = compute_subject_stats_bimodal(ds_train)
    stats_va = compute_subject_stats_bimodal(ds_val)

    def _mean_pair(d, idx):
        ms = [v[idx][0] for v in d.values() if v[idx][0] is not None]
        ss = [v[idx][1] for v in d.values() if v[idx][1] is not None]
        if ms and ss:
            return (torch.stack(ms).mean(0), torch.stack(ss).mean(0))
        return (None, None)
    fallback_eeg = _mean_pair(stats_tr, 0)
    fallback_eye = _mean_pair(stats_tr, 1)

    model = build_model(args, n_domains).to(device)
    class_w = compute_class_weights(ds_train).to(device)

    try:
        cls_criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    except TypeError:
        def ls_ce(logits, target, eps=0.1, classes=7):
            with torch.no_grad():
                td = torch.zeros_like(logits); td.fill_(eps/(classes-1))
                td.scatter_(1, target.unsqueeze(1), 1-eps)
            return -(td * torch.log_softmax(logits,1)).sum(1).mean()
        cls_criterion = ls_ce

    dom_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(loader_tr))
    warm = max(1, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warm: return float(step)/float(max(1,warm))
        prog = float(step - warm)/float(max(1,total_steps - warm))
        return 0.5*(1.0 + math.cos(math.pi*prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    stopper = EarlyStopper(patience=20, min_delta=0.0, mode="max")
    feat_dropout = nn.Dropout(p=args.feat_dropout).to(device)

    try:
        scaler = GradScaler(device_type="cuda", enabled=args.amp) if AMP_NEW else GradScaler(enabled=args.amp)
    except TypeError:
        scaler = GradScaler(enabled=args.amp)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        run_loss = run_hit = run_tot = 0
        alpha = 2/(1+math.exp(-10*ep/max(1,args.epochs))) - 1
        lambda_dom = args.lambda_dom_max * float(alpha)

        for b in loader_tr:
            x = b["eeg"].to(device, non_blocking=True)   # [B,558]
            v = b["eye"].to(device, non_blocking=True)   # [B,25]
            y = b["label"].to(device, non_blocking=True)
            sid = b["domain"].to(device, non_blocking=True)
            if y.dtype != torch.long: y = y.long()
            sid0 = (sid - 1).long()

            x, v = stack_and_normalize_bimodal(x, v, sid, stats_tr, device,
                                               fallback_eeg=fallback_eeg, fallback_eye=fallback_eye)
            x = feat_dropout(x)

            with autocast(device_type="cuda", enabled=args.amp):
                logits, dom_logits = model(eeg=x, eye=v, alpha_=alpha)
                loss_cls = cls_criterion(logits, y) if callable(cls_criterion) else cls_criterion(logits, y)
                loss_dom = dom_criterion(dom_logits, sid0)
                loss = loss_cls + lambda_dom * loss_dom

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            sched.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                run_hit += (pred == y).sum().item()
                run_tot += y.size(0)
                run_loss += loss.item() * y.size(0)

        tr_acc = run_hit / max(1, run_tot)
        tr_loss = run_loss / max(1, run_tot)
        va_acc = evaluate(model, loader_va, device, stats_va,
                          fallback_eeg=fallback_eeg, fallback_eye=fallback_eye, amp=args.amp)

        if va_acc > best:
            best = va_acc
            save_path = insert_suffix(args.save, f"_{fold_name}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "best_acc": best, "args": vars(args)}, save_path)

        print(f"[{fold_name}] [Epoch {ep+1}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_acc={va_acc:.4f} best={best:.4f} | alpha={alpha:.3f} lambda_dom={lambda_dom:.3f}")

        if stopper.step(va_acc):
            print(f"[Early Stop {fold_name}] epoch={ep}, best_acc={best:.4f}")
            break
    return best


def insert_suffix(path, suffix):
    b, e = os.path.splitext(path); return f"{b}{suffix}{e}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("Using device:", device)

    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    ds = MAETFeatureDataset(
        eeg_dir=args.eeg_dir,
        graph_ch_dir=args.graph_ch_dir,
        aux25_dir=args.aux25_dir,
        label_csv=args.labels,
        eeg_key_prefix="de_LDS_", graph_key_prefix="graph_ch_"
    )

    # dimension confirmation
    d0_eeg = int(ds[0]["eeg"].numel()); d0_eye = int(ds[0]["eye"].numel())
    print(f"[CONF] sample dims: eeg={d0_eeg} (expect 558), eye={d0_eye} (expect 25)")
    assert d0_eeg == 558 and d0_eye == 25

    # intensity threshold
    if hasattr(ds, "meta") and "intensity" in ds.meta.columns and args.intensity_threshold is not None:
        before = len(ds.meta)
        ds.meta = ds.meta[(ds.meta["intensity"].fillna(0.0) >= float(args.intensity_threshold))].reset_index(drop=True)
        after = len(ds.meta)
        print(f"[INFO] Intensity filter >= {args.intensity_threshold:.2f}: kept {after}/{before} samples")

    meta = ds.meta
    groups = meta["file_id"].values
    labels = meta["label"].values
    n_domains = len(np.unique(groups))

    if args.eval_mode == "group_split":
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, val_idx = next(gss.split(np.zeros(len(meta)), labels, groups))
        ds_tr, ds_va = Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())
        best = train_one_split(args, ds_tr, ds_va, device, fold_name="grp", n_domains=n_domains)
        print(f"[GroupSplit] best val acc = {best:.4f}")
    else:
        logo = LeaveOneGroupOut()
        bests = []
        for fold, (tr_idx, va_idx) in enumerate(logo.split(np.zeros(len(meta)), labels, groups), start=1):
            left = set(groups[va_idx])
            fold_name = f"loso_S{'-'.join(map(str, sorted(left)))}"
            ds_tr, ds_va = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist())
            best = train_one_split(args, ds_tr, ds_va, device, fold_name=fold_name, n_domains=n_domains)
            bests.append(best)
            print(f"[{fold_name}] fold best acc = {best:.4f}")
        print(f"[LOSO] mean acc over {len(bests)} folds = {float(np.mean(bests)):.4f}")


if __name__ == "__main__":
    main()
