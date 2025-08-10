from pathlib import Path
import sys
import pandas as pd
import joblib

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "processed"
MODELS = ROOT / "models"

sys.path.append(str(ROOT / "src"))
try:
    from feature_engineering import FeatureEngineer  # noqa: F401
except Exception:
    pass

df_full = pd.read_csv(DATA / "data_01.csv")

X_full = df_full.drop(columns=["Attrition"]) if "Attrition" in df_full.columns else df_full.copy()

candidates = [
    MODELS / "final_pipeline.joblib",
    MODELS / "final_pipeline.pkl",
    ROOT   / "final_pipeline.joblib",
    ROOT   / "final_pipeline.pkl",
]
pipe_path = next((p for p in candidates if p.exists()), None)
if pipe_path is None:
    raise FileNotFoundError("Couldn't find final pipeline. Looked for:\n" + "\n".join(map(str, candidates)))

pipe = joblib.load(pipe_path)

if not hasattr(pipe, "predict_proba"):
    raise AttributeError(
        f"Loaded object at {pipe_path} has no predict_proba(). "
        "Make sure the last step of final_pipeline is your classifier."
    )

pred_prob = pipe.predict_proba(X_full)[:, 1]

df_out = df_full.copy()
df_out["pred_prob"]  = pred_prob
df_out["pred_label"] = (df_out["pred_prob"] >= 0.50).astype(int)
if "Attrition" in df_out.columns:
    df_out["Attrited"] = (df_out["Attrition"] == "Yes").astype(int)

out_path = DATA / "full_with_predictions.csv"
df_out.to_csv(out_path, index=False)
print(f"Wrote {out_path}  shape={df_out.shape}  "
      f"pred_prob∈[{df_out['pred_prob'].min():.4f},{df_out['pred_prob'].max():.4f}]")

import numpy as np
df_out.insert(0, "row_id", np.arange(1, len(df_out)+1))
df_out.to_csv(DATA / "full_with_predictions.csv", index=False)
