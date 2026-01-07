import os, glob, re
import numpy as np
import pandas as pd
import scipy.io as sio

EXCEL_PATH = r"../seedVII/emotion_label_and_stimuli_order.xlsx"
CONT_LABEL_DIR = r"../seedVII/continuous_labels"
OUT_CSV = "labels.csv"

EMO2ID = {
    "Disgust":0, "Fear":1, "Sad":2, "Neutral":3,
    "Happy":4, "Anger":5, "Surprise":6
}
EMO_SET = set(EMO2ID.keys())
is_int = lambda x: isinstance(x,(int,np.integer)) or (isinstance(x,str) and re.fullmatch(r"\s*\d+\s*", x))

xls = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)

pairs = []  # (trial,label_id)
for i in range(0, len(xls), 2):
    if i+1 >= len(xls):
        break
    emo_row = xls.iloc[i,   1:]  # emotion
    idx_row = xls.iloc[i+1, 1:]  # trial index

    for j, (maybe_idx, maybe_emo) in enumerate(zip(idx_row, emo_row), start=2):
        if pd.isna(maybe_idx) and pd.isna(maybe_emo):
            continue
        a = None if pd.isna(maybe_idx) else str(maybe_idx).strip()
        b = None if pd.isna(maybe_emo) else str(maybe_emo).strip()

        trial_idx = None
        emo_str   = None


        if a is not None and is_int(a) and b in EMO_SET:
            trial_idx = int(a)
            emo_str = b

        elif b is not None and is_int(b) and a in EMO_SET:
            trial_idx = int(b)
            emo_str = a
        else:
            print(f"[WARN] 跳过 (row_group={i}/{i+1}, col={j}) -> idx='{a}', emo='{b}'")
            continue

        pairs.append((trial_idx, EMO2ID[emo_str]))

df_labels = pd.DataFrame(pairs, columns=["trial","label"]).drop_duplicates(subset=["trial"]).sort_values("trial")


def to_float(x):
    arr = np.array(x).squeeze()
    if arr.size == 0: return np.nan
    if arr.ndim == 0: return float(arr)
    return float(np.mean(arr))

rows = []
for mat_path in glob.glob(os.path.join(CONT_LABEL_DIR, "*.mat")):
    subj = int(os.path.splitext(os.path.basename(mat_path))[0])
    d = sio.loadmat(mat_path)
   
    trial_keys = sorted(int(k) for k in d.keys() if str(k).isdigit())
   
    
    for t in trial_keys:
        intensity = to_float(d[str(t)])
        # calculating the mean
        rows.append({"file_id": subj, "trial": t, "domain": subj, "intensity": intensity})

df_intensity = pd.DataFrame(rows)

print( df_labels ) 
df_final = df_intensity.merge(df_labels, on="trial", how="left").sort_values(["file_id","trial"]).reset_index(drop=True)
df_final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
df_final["file_id"] = df_final["file_id"].astype(int)
df_final["domain"] = df_final["domain"].astype(int)
df_final["trial"] = df_final["trial"].astype(int)

print("Saved:", OUT_CSV, "shape=", df_final.shape)
print(df_final.head(12))