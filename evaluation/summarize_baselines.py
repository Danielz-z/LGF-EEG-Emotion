import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory that contains *.summary.json files")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("*.summary.json"))
    if not files:
        raise SystemExit(f"No *.summary.json found in {root}")

    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            js = json.load(fh)
        rows.append({
            "feat_set": js.get("feat_set"),
            "models": js.get("models"),
            "intensity_threshold": js.get("intensity_threshold"),
            "mean_acc": js.get("mean_acc"),
            "std_acc": js.get("std_acc"),
            "mean_f1m": js.get("mean_f1m"),
            "std_f1m": js.get("std_f1m"),
            "save_json": str(f),
        })

    df = pd.DataFrame(rows).sort_values(
        by=["mean_f1m", "mean_acc"], ascending=[False, False]
    ).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[DONE] Wrote summary table: {args.out}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
