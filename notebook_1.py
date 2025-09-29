# Databricks notebook source
# Notebook 01 — Data cleaning, featurization, baseline modeling, evaluation, and MLflow packaging
# Includes LogisticRegression, GradientBoosting, full metrics,
# calibration, scaling of descriptors, ablation, scaffold vs random split, and MLflow pyfunc packaging)

# %%
"""
Cell 0 — Environment & install notes
- If RDKit is not available on your cluster, install with:
    %pip install rdkit-pypi
  or (preferred on conda-enabled environments):
    conda install -c conda-forge rdkit
- Required Python packages for this notebook:
    pandas, numpy, scikit-learn, joblib, matplotlib, mlflow
    %pip install pandas numpy scikit-learn joblib matplotlib mlflow rdkit-pypi

Reasoning: Keep environment setup explicit so Notebook works reproducibly on Databricks free edition.
"""

import os
import random
import math
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import QED

# sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

# Optional MLflow (Databricks typically has mlflow available)
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Feature constants
FINGERPRINT_SIZE = 2048
# Keep descriptors fixed and ordered. This list MUST match what is used at training and inference.
DESCRIPTOR_NAMES = [
    'MolWt',      # molecular weight — often predictive of ADME/toxicity trends
    'MolLogP',    # lipophilicity — affects membrane permeability
    'NumHDonors', # hydrogen bond donors
    'NumHAcceptors', # hydrogen bond acceptors
    'TPSA'        # topological polar surface area
]

#print('Notebook 01 environment initialized. RDKit version:', Chem.__version__)



# COMMAND ----------

# %%
"""
Cell 1 — SMILES standardization & filtering helpers
Reasoning / comments:
- Standardization (largest fragment, sanitize, canonicalize) reduces noise and duplicate representations.
- Basic filters (atom types, size) remove unreasonable molecules uncommon for drug-like chemistry.
"""

from rdkit.Chem import SaltRemover
remover = SaltRemover.SaltRemover()

ALLOWED_ATOMS = set(['H','C','N','O','F','P','S','Cl','Br','I','B','Si'])


def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if frags:
            return max(frags, key=lambda m: m.GetNumAtoms())
    except Exception:
        pass
    return mol


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical sanitized SMILES or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = keep_largest_fragment(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def has_allowed_elements(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            return False
    return True


def basic_filters(smiles: str, min_atoms:int=5, max_atoms:int=120) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        n_atoms = mol.GetNumAtoms()
        if n_atoms < min_atoms or n_atoms > max_atoms:
            return False
        if not has_allowed_elements(mol):
            return False
        return True
    except Exception:
        return False


def standardize_series(smiles_series: pd.Series) -> pd.Series:
    return smiles_series.astype(str).map(canonicalize_smiles)



# COMMAND ----------

# %%
"""
Cell 2 — Scaffold helpers and scaffold split
Reasoning: Scaffold (Bemis–Murcko) split tests generalization to *new* chemotypes.
"""

def get_murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold is not None else None
    except Exception:
        return None


def scaffold_split(df: pd.DataFrame, smiles_col: str = 'smiles', frac_train:float=0.8, frac_valid:float=0.1, frac_test:float=0.1, seed:int=SEED) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    df = df.copy()
    df['scaffold'] = df[smiles_col].map(get_murcko_scaffold)
    df['scaffold'] = df['scaffold'].fillna(df[smiles_col])

    scaff_groups = df.groupby('scaffold').apply(lambda d: d.index.tolist()).to_dict()
    scaffolds = list(scaff_groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)

    n = len(df)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)

    train_idx, valid_idx, test_idx = [], [], []
    cum = 0
    for s in scaffolds:
        idxs = scaff_groups[s]
        if cum < n_train:
            train_idx.extend(idxs)
        elif cum < n_train + n_valid:
            valid_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
        cum += len(idxs)

    train_df = df.loc[train_idx].reset_index(drop=True)
    valid_df = df.loc[valid_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df



# COMMAND ----------

# %%
"""
Cell 3 — Featurization utilities (consistent between training & inference)
Reasoning: centralize feature construction so the same code is reused for both training and inference, preventing dimension mismatches.
"""


def featurize_single(smiles: str) -> np.ndarray:
    """Return a 1-D numpy array of length FINGERPRINT_SIZE + len(DESCRIPTOR_NAMES) or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Try to continue even if sanitize is mildy problematic
        pass

    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FINGERPRINT_SIZE)
    fp_arr = np.zeros((FINGERPRINT_SIZE,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_vect, fp_arr)

    desc_vals = []
    for name in DESCRIPTOR_NAMES:
        func = getattr(Descriptors, name)
        desc_vals.append(float(func(mol)))

    desc_arr = np.array(desc_vals, dtype=np.float32)
    return np.concatenate([fp_arr.astype(np.float32), desc_arr])


def featurize_smiles_list(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    feats = []
    valids = []
    for s in smiles_list:
        v = featurize_single(s)
        if v is not None:
            feats.append(v)
            valids.append(s)
    if len(feats) == 0:
        return np.zeros((0, FINGERPRINT_SIZE + len(DESCRIPTOR_NAMES))), []
    return np.vstack(feats), valids



# COMMAND ----------

# %%
"""
Cell 4 — Model training / evaluation helpers
Reasoning: keep training logic concise and consistent across models; scale descriptor portion only.
"""


def _scale_descriptor_portion(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    fp_dim = FINGERPRINT_SIZE
    scaler = StandardScaler()
    if X_train.shape[0] > 0:
        X_train[:, fp_dim:] = scaler.fit_transform(X_train[:, fp_dim:])
    if X_test.shape[0] > 0:
        X_test[:, fp_dim:] = scaler.transform(X_test[:, fp_dim:])
    return X_train, X_test, scaler


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        'auroc': float(roc_auc_score(y_test, y_prob)),
        'ap': float(average_precision_score(y_test, y_prob)),
        'brier': float(brier_score_loss(y_test, y_prob))
    }
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    metrics['calib_frac_pos'] = frac_pos
    metrics['calib_mean_pred'] = mean_pred
    return metrics


def train_and_eval_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, model_type:str) -> Tuple[Any, Dict[str, Any], Any]:
    """Featurize, scale, train a model, return trained model, metrics, and scaler used.
    model_type: 'logreg' or 'gbm'
    """
    # Featurize
    X_train, smiles_train = featurize_smiles_list(train_df['smiles'].tolist())
    X_test,  smiles_test  = featurize_smiles_list(test_df['smiles'].tolist())

    # Align labels to valid featurized smiles
    y_train = train_df.loc[[s in smiles_train for s in train_df['smiles']], 'label'].values
    y_test  = test_df.loc[[s in smiles_test  for s in test_df['smiles']], 'label'].values

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError('Empty feature matrix after featurization; check your dataset and filters.')

    # Scale descriptor portion only
    X_train, X_test, scaler = _scale_descriptor_portion(X_train, X_test)

    if model_type == 'logreg':
        clf = LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced', n_jobs=-1)
    else:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3)

    clf.fit(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)
    return clf, metrics, scaler



# COMMAND ----------

import requests
from io import BytesIO
import os
import pandas as pd
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_data(data_url):

    # download the dataset from the URL
    try:
        response = requests.get(data_url,verify=False)
        response.raise_for_status() #raise an exception if the request fails
        raw_df = pd.read_csv(BytesIO(response.content),compression='gzip')

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset from data url {data_url} : {e}")
        return pd.DataFrame()

    print(f"Initial size of the raw dataframe - {raw_df.shape}")

    return raw_df

tox21_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
#toxcast_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
tox21_df_v1 = load_data(tox21_url) 
tox21_df_v1

# COMMAND ----------

# %%
"""
Cell 5 — Load dataset, standardize, filter, deduplicate
Reasoning: This cell demonstrates preparation of a real dataset (Tox21) and produces a clean df with columns ['smiles','label'] ready for splitting.
"""

#DATA_PATH = '/dbfs/FileStore/tables/tox21.csv'  # update path as needed

#if os.path.exists(DATA_PATH):

df = tox21_df_v1
# Example: Tox21 has multiple assay columns — create an aggregate label for toxicity (conservative)
assay_cols = [c for c in df.columns if c.lower() not in ['smiles','id','mol_id','label']]
if len(assay_cols) == 0 and 'label' not in df.columns:
    raise ValueError('No assay columns found. Ensure your CSV contains SMILES and assay columns or a label column.')
if 'label' not in df.columns:
    df[assay_cols] = df[assay_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['label'] = (df[assay_cols].sum(axis=1) > 0).astype(int)
df = df[['smiles','label']]
# else:
#     # Minimal synthetic example to allow step-by-step runs
#     sample_smiles = ['CCO','CC(=O)Oc1ccccc1C(=O)O','c1ccccc1','C1CCCCC1','CCN(CC)CC']
#     sample_labels = [0,1,0,0,0]
#     df = pd.DataFrame({'smiles': sample_smiles, 'label': sample_labels})

print('Loaded rows:', len(df))

# Standardize
print('Standardizing SMILES (this may take a moment)...')
df['smiles'] = standardize_series(df['smiles'])
df = df.dropna(subset=['smiles']).reset_index(drop=True)

# Basic filtering
mask = df['smiles'].map(lambda s: basic_filters(s))
df = df[mask].reset_index(drop=True)
print('After basic filters:', len(df))

# Deduplicate
before = len(df)
df = df.drop_duplicates(subset=['smiles']).reset_index(drop=True)
after = len(df)
print(f'Removed {before-after} duplicate SMILES')

print(df.head())



# COMMAND ----------

# %%
"""
Cell 6 — Splits: scaffold split vs random split (for comparison)
Reasoning: Show both splits and document metric differences. Scaffold split is stronger evidence of generalization.
"""

train_s, valid_s, test_s = scaffold_split(df, smiles_col='smiles', frac_train=0.8, frac_valid=0.1, frac_test=0.1)
train_scaffold = pd.concat([train_s, valid_s]).reset_index(drop=True)
print('Scaffold split sizes (train,valid,test):', len(train_scaffold), len(valid_s), len(test_s))

# Random split with stratification
train_r_full, test_r = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
train_r, valid_r = train_test_split(train_r_full, test_size=0.11111, random_state=SEED, stratify=train_r_full['label'])
train_random = pd.concat([train_r, valid_r]).reset_index(drop=True)
print('Random split sizes (train,valid,test):', len(train_random), len(valid_r), len(test_r))



# COMMAND ----------

# %%
"""
Cell 7 — Train baselines on scaffold split and random split; compute metrics and calibration
Reasoning: Evaluate both model classes (LogReg & GBM) on both splits. Log results to MLflow optionally.
"""

results = {}

# Scaffold split - LogisticRegression
clf_s_log, metrics_s_log, scaler_s_log = train_and_eval_pipeline(train_scaffold, test_s, model_type='logreg')
results['scaffold_logreg'] = metrics_s_log
print('Scaffold LogReg metrics:', {k:metrics_s_log[k] for k in ['auroc','ap','brier']})

# Scaffold split - GradientBoosting
clf_s_gbm, metrics_s_gbm, scaler_s_gbm = train_and_eval_pipeline(train_scaffold, test_s, model_type='gbm')
results['scaffold_gbm'] = metrics_s_gbm
print('Scaffold GBM metrics:', {k:metrics_s_gbm[k] for k in ['auroc','ap','brier']})

# Random split - LogisticRegression
clf_r_log, metrics_r_log, scaler_r_log = train_and_eval_pipeline(train_random, test_r, model_type='logreg')
results['random_logreg'] = metrics_r_log
print('Random LogReg metrics:', {k:metrics_r_log[k] for k in ['auroc','ap','brier']})

# Random split - GradientBoosting
clf_r_gbm, metrics_r_gbm, scaler_r_gbm = train_and_eval_pipeline(train_random, test_r, model_type='gbm')
results['random_gbm'] = metrics_r_gbm
print('Random GBM metrics:', {k:metrics_r_gbm[k] for k in ['auroc','ap','brier']})

# Calibration plots helper

def plot_calibration_pair(metrics1, metrics2, title1='Model A', title2='Model B'):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(metrics1['calib_mean_pred'], metrics1['calib_frac_pos'], marker='o')
    plt.plot([0,1],[0,1], '--')
    plt.title(title1)
    plt.xlabel('Mean pred prob'); plt.ylabel('Fraction positives')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(metrics2['calib_mean_pred'], metrics2['calib_frac_pos'], marker='o')
    plt.plot([0,1],[0,1], '--')
    plt.title(title2)
    plt.xlabel('Mean pred prob'); plt.ylabel('Fraction positives')
    plt.grid(True)
    plt.show()

# Example: compare scaffold LogReg vs random LogReg
plot_calibration_pair(metrics_s_log, metrics_r_log, title1='Scaffold LogReg', title2='Random LogReg')




# COMMAND ----------

plot_calibration_pair(metrics_s_gbm, metrics_r_gbm, title1='Scaffold GBM', title2='Random GBM')

# COMMAND ----------

# %%
"""
Cell 8 — Ablation: FP-only vs FP + descriptors
Reasoning: Show whether descriptors add predictive value beyond fingerprints.
"""

def train_fp_only(train_df, test_df, model_type):
    # Featurize then zero-out the descriptor portion
    X_train, _ = featurize_smiles_list(train_df['smiles'].tolist())
    X_test, _  = featurize_smiles_list(test_df['smiles'].tolist())
    # zero descriptors
    if X_train.shape[0] > 0:
        X_train[:, FINGERPRINT_SIZE:] = 0
    if X_test.shape[0] > 0:
        X_test[:, FINGERPRINT_SIZE:] = 0

    # labels
    y_train = train_df.loc[[s in (train_df['smiles'].tolist()) for s in train_df['smiles']], 'label'].values
    y_test  = test_df.loc[[s in (test_df['smiles'].tolist())  for s in test_df['smiles']], 'label'].values

    if model_type == 'logreg':
        clf = LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced', n_jobs=-1)
    else:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3)

    # Align y's more robustly by using featurized smiles (safer approach below)
    X_train, smiles_train = featurize_smiles_list(train_df['smiles'].tolist())
    X_test,  smiles_test  = featurize_smiles_list(test_df['smiles'].tolist())
    y_train = train_df.loc[[s in smiles_train for s in train_df['smiles']], 'label'].values
    y_test  = test_df.loc[[s in smiles_test  for s in test_df['smiles']], 'label'].values

    X_train[:, FINGERPRINT_SIZE:] = 0
    X_test[:, FINGERPRINT_SIZE:] = 0

    clf.fit(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)
    return clf, metrics

clf_s_fp, metrics_s_fp = train_fp_only(train_scaffold, test_s,'gbm')
clf_s_full, metrics_s_full, _ = train_and_eval_pipeline(train_scaffold, test_s,'gbm')

print('Scaffold split - FP only:', {k:metrics_s_fp[k] for k in ['auroc','ap','brier']})
print('Scaffold split - FP+desc:', {k:metrics_s_full[k] for k in ['auroc','ap','brier']})



# COMMAND ----------

import joblib
joblib.dump(clf_s_gbm,"tox21_model_v1.pkl")

# COMMAND ----------

# In your Tox21 binary classifiers, 0 = non-toxic (inactive) and 1 = toxic (active), so toxic compounds should give prediction = 1.

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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

