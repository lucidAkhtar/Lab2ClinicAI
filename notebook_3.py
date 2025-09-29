# Databricks notebook source
# 1. Imports and Seed

import os, math, json, random, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

SEED = 42
random.seed(SEED); np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")


# COMMAND ----------

# 2. Registry + resilient CSV loader

DATASET_REGISTRY = {
    # Regression: ESOL solubility (logS)
    "esol": {
        "path": "data/delaney_processed.csv",
        "type": "regression",
        # Common column names seen in ESOL variants
        "smiles_candidates": ["smiles", "SMILES"],
        "target_candidates": ["logS", "measured log solubility in mols per litre", "Solubility", "solubility"],
        "rename_to": {"y": "logS"}
    },
    # Regression: Lipophilicity (experimental LogD/LogP-like)
    "lipo": {
        "path": "data/lipo.csv",
        "type": "regression",
        "smiles_candidates": ["smiles", "SMILES", "mol"],
        "target_candidates": ["exp", "LogD", "logD", "LogD7.4", "y"],
        "rename_to": {"y": "logD"}
    },
    # Classification: BBBP
    "bbbp": {
        "path": "data/bbbp.csv",
        "type": "classification",
        "smiles_candidates": ["smiles", "SMILES"],
        # 1/0 or P/NP labels in various versions
        "target_candidates": ["p_np", "BBBP", "label", "Class"],
        "rename_to": {"y": "BBBP"}
    },
}

def _find_col(candidates, cols):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None

def load_dataset(cfg):
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path} — skipping.")
        return None
    df = pd.read_csv(path)
    smi_col = _find_col(cfg["smiles_candidates"], df.columns) or _find_col(["smiles"], df.columns)
    if smi_col is None:
        raise ValueError(f"No SMILES column found in {path}")
    y_col = _find_col(cfg["target_candidates"], df.columns)
    if y_col is None:
        raise ValueError(f"No target column found in {path}. Checked {cfg['target_candidates']}")
    df = df.rename(columns={smi_col: "smiles", y_col: "y"})
    # Normalize target (for BBBP, coerce to 0/1)
    if cfg["type"] == "classification":
        # Map text labels (P/NP, True/False) to 1/0
        df["y"] = df["y"].map({"P": 1, "NP": 0, "p":1, "np":0}).fillna(df["y"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
    else:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
    # Clean
    df = df.dropna(subset=["smiles", "y"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    # Friendly task name
    task_name = list(cfg["rename_to"].values())[0]
    df["task"] = task_name
    return df[["smiles", "y", "task"]]


# COMMAND ----------

# 3. Load available datasets

bundles = {}
for name, cfg in DATASET_REGISTRY.items():
    df = load_dataset(cfg)
    if df is not None and len(df):
        bundles[name] = {"cfg": cfg, "df": df}
        print(f"✅ {name}: {len(df):,} rows | task={df['task'].iloc[0]} | type={cfg['type']}")
    else:
        print(f"⏭️  {name}: not loaded.")
if not bundles:
    raise SystemExit("No ADME datasets loaded. Place CSVs in ./data and rerun.")


# COMMAND ----------

# 4. Compute Bemis–Murcko scaffolds + Global scaffold split

def smiles_to_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc, isomericSmiles=True) if sc else None

# Union of all molecules across ADME datasets
union = pd.concat([b["df"][["smiles"]] for b in bundles.values()], axis=0)\
          .drop_duplicates().reset_index(drop=True)
union["scaffold"] = union["smiles"].apply(smiles_to_scaffold)
union["scaffold"] = union["scaffold"].fillna(union["smiles"])

def scaffold_split_index(df_union, frac_valid=0.1, frac_test=0.1, seed=SEED):
    groups = df_union.groupby("scaffold").indices
    scaffolds = list(groups.keys())
    rng = np.random.default_rng(seed); rng.shuffle(scaffolds)
    n = len(df_union); nV = int(math.floor(frac_valid*n)); nT = int(math.floor(frac_test*n))
    split = pd.Series(index=df_union.index, dtype="category").cat.set_categories(["train","valid","test"])
    cV=cT=0
    for sc in scaffolds:
        idxs = groups[sc]
        if cV < nV: split.iloc[idxs] = "valid"; cV += len(idxs)
        elif cT < nT: split.iloc[idxs] = "test"; cT += len(idxs)
        else: split.iloc[idxs] = "train"
    return split

union["split_global"] = scaffold_split_index(union, frac_valid=0.1, frac_test=0.1, seed=SEED)

# Attach split to each dataset
for name, b in bundles.items():
    b["df"] = b["df"].merge(union[["smiles","split_global"]], on="smiles", how="left")
    print(f"{name} split:", b["df"]["split_global"].value_counts().to_dict())


# COMMAND ----------

# 5. Persist the prepared tables

os.makedirs("artifacts_nb3_step1", exist_ok=True)
for name, b in bundles.items():
    outp = os.path.join("artifacts_nb3_step1", f"{name}_prepared.parquet")
    b["df"].to_parquet(outp, index=False)
    print("Saved:", outp)


# COMMAND ----------

# MAGIC %md
# MAGIC Step 1 (Notebook 3) — Dataset ingestion & global scaffold split
# MAGIC
# MAGIC Goal: Bring in new ADME datasets (ESOL for solubility, Lipophilicity for logD/logP, BBBP for brain penetration).
# MAGIC
# MAGIC What we did:
# MAGIC
# MAGIC Standardized them into a consistent schema: smiles, y (target), task, split_global.
# MAGIC
# MAGIC Built a union of scaffolds across all datasets, then performed a global Murcko scaffold split (train/valid/test).
# MAGIC
# MAGIC Ensures train/test molecules don’t share scaffolds, giving more realistic generalization — just like real drug discovery.
# MAGIC
# MAGIC 👉 Outcome: Clean, harmonized ADME datasets, each molecule assigned to train/valid/test based on scaffold, ready for featurization.

# COMMAND ----------

# 6. Featurizer and dataset-to-mix

# Featurizer: 2048-bit Morgan FP (ECFP4) + 5 descriptors
from rdkit.Chem import AllChem, Descriptors
import numpy as np

def featurize_fp5(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    arr = np.array(fp, dtype=float)
    desc = np.array([
        Descriptors.MolWt(m), Descriptors.MolLogP(m),
        Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m),
        Descriptors.TPSA(m)
    ], dtype=float)
    return np.concatenate([arr, desc])  # 2053 features

# Build X/y/split for each loaded dataset from Step 1 (bundles[name]["df"])
for name, b in bundles.items():
    df = b["df"].copy()
    feats = [featurize_fp5(s) for s in df["smiles"]]
    mask = [(x is not None) and np.isfinite(y) for x, y in zip(feats, df["y"])]
    X = np.vstack([x for x, m in zip(feats, mask) if m])
    y = df.loc[mask, "y"].to_numpy()
    spl = df.loc[mask, "split_global"].astype("category").reset_index(drop=True)
    b["X"], b["y"], b["SPL"] = X, y, spl
    print(f"{name}: X={X.shape}, y={y.shape}, split counts={spl.value_counts().to_dict()}")


# COMMAND ----------

# 7. Training utilities (GBM)

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score

def _split_xy(X, y, spl, part):
    idx = (spl == part).to_numpy()
    return X[idx], y[idx]

def train_regressor_gbm(X, y, spl, n_estimators=300, max_depth=5, learning_rate=0.05, seed=42):
    Xtr, ytr = _split_xy(X, y, spl, "train")
    Xva, yva = _split_xy(X, y, spl, "valid")
    Xte, yte = _split_xy(X, y, spl, "test")
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=learning_rate, random_state=seed)
    model.fit(Xtr, ytr)
    def eval_block(Xb, yb):
        p = model.predict(Xb)
        return dict(RMSE=mean_squared_error(yb, p, squared=False),
                    MAE=mean_absolute_error(yb, p),
                    R2=r2_score(yb, p))
    return model, {"valid": eval_block(Xva, yva), "test": eval_block(Xte, yte)}

def train_classifier_gbm(X, y, spl, n_estimators=300, max_depth=5, learning_rate=0.05, seed=42):
    Xtr, ytr = _split_xy(X, y, spl, "train")
    Xva, yva = _split_xy(X, y, spl, "valid")
    Xte, yte = _split_xy(X, y, spl, "test")
    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       learning_rate=learning_rate, random_state=seed)
    model.fit(Xtr, ytr.astype(int))
    def eval_block(Xb, yb):
        if len(np.unique(yb)) < 2:
            return dict(ROC_AUC=np.nan, PR_AUC=np.nan)
        p = model.predict_proba(Xb)[:,1]
        return dict(ROC_AUC=roc_auc_score(yb, p), PR_AUC=average_precision_score(yb, p))
    return model, {"valid": eval_block(Xva, yva), "test": eval_block(Xte, yte)}


# COMMAND ----------

# 8. Trains models for each dataset

metrics_summary = {}

for name, b in bundles.items():
    X, y, spl = b["X"], b["y"], b["SPL"]
    task_type = b["cfg"]["type"]
    print(f"\nTraining {name} ({task_type}) ...")
    if task_type == "regression":
        model, m = train_regressor_gbm(X, y, spl)
    else:
        model, m = train_classifier_gbm(X, y, spl)
    b["model"] = model
    b["metrics"] = m
    metrics_summary[name] = {
        **{f"valid_{k}": v for k, v in m["valid"].items()},
        **{f"test_{k}":  v for k, v in m["test"].items()},
        "n_train": int((spl=="train").sum()),
        "n_valid": int((spl=="valid").sum()),
        "n_test":  int((spl=="test").sum())
    }

pd.DataFrame(metrics_summary).T


# COMMAND ----------

# 9. Save models for Streamlit ADMET tab

import os, joblib, json
EXPORT_DIR = "export_nb3_admet"
os.makedirs(EXPORT_DIR, exist_ok=True)

manifest = {"feature_config": {
                "featurization": "fp+5desc",
                "fp_bits": 2048,
                "fp_radius": 2,
                "descriptors": ["MolWt","MolLogP","NumHAcceptors","NumHDonors","TPSA"]
            },
            "datasets": []}

for name, b in bundles.items():
    outp = os.path.join(EXPORT_DIR, f"{name}.joblib")
    joblib.dump(b["model"], outp)
    manifest["datasets"].append({
        "name": name,
        "type": b["cfg"]["type"],
        "model_file": f"{name}.joblib",
        "task": b["df"]["task"].iloc[0]
    })

with open(os.path.join(EXPORT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved {len(manifest['datasets'])} ADME models to {EXPORT_DIR}")


# COMMAND ----------

# MAGIC %md
# MAGIC Step 2 (Notebook 3) — Featurization & model training
# MAGIC
# MAGIC Goal: Train predictive models on the ADME datasets.
# MAGIC
# MAGIC What we did:
# MAGIC
# MAGIC Featurized each molecule with 2048-bit Morgan fingerprints + 5 descriptors (same feature recipe as toxicity tasks).
# MAGIC
# MAGIC Trained Gradient Boosting models:
# MAGIC
# MAGIC Regression → ESOL (logS) & Lipophilicity (logD).
# MAGIC
# MAGIC Classification → BBBP (brain penetration).
# MAGIC
# MAGIC Evaluated on scaffold-split valid/test sets using appropriate metrics:
# MAGIC
# MAGIC RMSE/MAE/R² for regression.
# MAGIC
# MAGIC ROC-AUC/PR-AUC for classification.
# MAGIC
# MAGIC Saved trained models + a manifest.json for integration into your Streamlit app.
# MAGIC
# MAGIC 👉 Outcome: You now have working ADME predictors (solubility, logD, BBBP) that extend your pipeline beyond toxicity.

# COMMAND ----------

# MAGIC %md
# MAGIC Regression metrics to watch: lower RMSE/MAE and higher R² are better.
# MAGIC
# MAGIC BBBP metrics: higher ROC-AUC/PR-AUC better.
# MAGIC
# MAGIC Uses the global Murcko split from Step 1 (realistic generalization).
# MAGIC
# MAGIC Next (Step 3): add an ADMET tab in Streamlit that loads export_nb3_admet/manifest.json, predicts logS, logD, BBBP, and shows a Developability score + Copilot bullets.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

