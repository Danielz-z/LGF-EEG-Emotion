import os
import glob
import numpy as np
import scipy.io as sio
import pandas as pd
import torch
from torch.utils.data import Dataset


class MAETFeatureDataset(Dataset):
    def __init__(self,
                 eeg_dir=r"data/EEG_features",
                 graph_ch_dir=r"data/GraphPerChannel_features",
                 aux25_dir=r"data/Aux25_features",
                 label_csv=r"data/labels.csv",
                 eeg_key_prefix="de_LDS_",      # [T,5,62]
                 graph_key_prefix="graph_ch_"): # [62,4]
        self.eeg_dir = eeg_dir
        self.graph_ch_dir = graph_ch_dir
        self.aux25_dir = aux25_dir
        self.eeg_key_prefix = eeg_key_prefix
        self.graph_key_prefix = graph_key_prefix

        if not os.path.isfile(label_csv):
            raise FileNotFoundError(label_csv)
        df = pd.read_csv(label_csv)
        need = {"file_id", "trial", "label", "domain"}
        if not need.issubset(df.columns):
            raise ValueError(f"labels.csv missing {need - set(df.columns)}")
        self.meta = df.reset_index(drop=True)

        self.eeg_paths = {os.path.splitext(os.path.basename(p))[0]: p
                          for p in glob.glob(os.path.join(eeg_dir, "*.mat"))}
        self.graph_paths = {os.path.splitext(os.path.basename(p))[0]: p
                            for p in glob.glob(os.path.join(graph_ch_dir, "*.mat"))}
        self.aux_paths = {os.path.splitext(os.path.basename(p))[0]: p
                          for p in glob.glob(os.path.join(aux25_dir, "*.mat"))}

        if not self.eeg_paths:
            raise FileNotFoundError(f"No EEG .mat in {eeg_dir}")
        if not self.graph_paths:
            raise FileNotFoundError(f"No GraphPerChannel .mat in {graph_ch_dir}")
        if not self.aux_paths:
            raise FileNotFoundError(f"No Aux25 .mat in {aux25_dir}")

        self._eeg_cache, self._graph_cache, self._aux_cache = {}, {}, {}

    def __len__(self):
        return len(self.meta)

    @staticmethod
    def _load(cache, paths, fid):
        if fid in cache:
            return cache[fid]
        if fid not in paths:
            raise FileNotFoundError(f"{fid}.mat not found")
        cache[fid] = sio.loadmat(paths[fid])
        return cache[fid]

    def __getitem__(self, idx: int):
        r = self.meta.iloc[idx]
        fid = str(int(r["file_id"]))
        t = int(r["trial"])
        y = int(r["label"])
        sid = int(r["domain"])

        # EEG: [T,5,62] -> [62,5]
        emat = self._load(self._eeg_cache, self.eeg_paths, fid)
        ekey = f"{self.eeg_key_prefix}{t}"
        if ekey not in emat:
            cand = [k for k in emat.keys()
                    if k.startswith(self.eeg_key_prefix) and k.split("_")[-1] == str(t)]
            if not cand:
                raise KeyError(f"{fid}.mat missing {ekey}")
            ekey = cand[0]
        eeg_3d = np.array(emat[ekey])  # [T,5,62]
        if eeg_3d.ndim != 3 or eeg_3d.shape[1] != 5 or eeg_3d.shape[2] != 62:
            raise ValueError(f"EEG shape expect [T,5,62], got {eeg_3d.shape}")
        eeg_fc = eeg_3d.mean(axis=0)                  # [5,62]
        eeg_per_ch = eeg_fc.T.astype(np.float32)      # [62,5]

        # Graph per-channel: [62,4]
        gmat = self._load(self._graph_cache, self.graph_paths, fid)
        gkey = f"{self.graph_key_prefix}{t}"
        if gkey not in gmat:
            raise KeyError(f"{fid}.mat missing {gkey}")
        graph_per_ch = np.array(gmat[gkey]).astype(np.float32)
        if graph_per_ch.shape != (62, 4):
            raise ValueError(f"Graph per-channel expect [62,4], got {graph_per_ch.shape}")

        # Channel-wise concatenation -> [62, 9] -> flatten to 558-D EEG feature
        feat_per_ch = np.concatenate([eeg_per_ch, graph_per_ch], axis=1)  # [62,9]
        eeg_vec = feat_per_ch.reshape(-1).astype(np.float32)              # 558

        # Aux25 -> eye
        amat = self._load(self._aux_cache, self.aux_paths, fid)
        akey = f"aux_{t}"
        if akey not in amat:
            raise KeyError(f"[Aux25] {fid}.mat missing key '{akey}'")
        eye_vec = np.array(amat[akey]).reshape(-1).astype(np.float32)     # (25,)
        if eye_vec.size != 25:
            raise ValueError(f"[Aux25] expect 25 dims, got {eye_vec.shape}")

        return {
            "eeg": torch.from_numpy(eeg_vec),          # 558
            "eye": torch.from_numpy(eye_vec),          # 25
            "label": torch.tensor(y, dtype=torch.long),
            "domain": torch.tensor(sid, dtype=torch.long),
        }