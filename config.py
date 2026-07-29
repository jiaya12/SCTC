"""
Central configuration for the SCTC experiments: data locations, severity-strata
definition, cross-fitting settings, sensitivity floors, and the feature list.
"""
import os

import numpy as np

# --- data locations (produced by the MIMIC-Sepsis benchmark; see README) ---
DATA_DIR = os.environ.get("SCTC_DATA_DIR", "./data")
TIMESERIES_CSV = os.path.join(DATA_DIR, "timeseries.csv")
PRED_CSV = os.environ.get("SCTC_PRED_CSV", os.path.join(DATA_DIR, "predictions.csv"))

# --- severity strata: pre-window non-cardiovascular SOFA, cut at <=3 / 4-6 / >=7 ---
SOFA_CUTS = (3, 6)
GROUPS = [0, 1, 2]
GROUP_NAMES = ["low", "moderate", "high"]

# --- stay-level cross-fitting ---
N_FOLDS = 5
N_SPLITS = 5

# --- sensitivity floors and evaluation budget used in the paper ---
FLOOR_VASO = 0.80
FLOOR_SEPTIC = 0.70
BUDGET_EVAL = 0.30

# --- finite-sample certificate ---
DELTA = 0.10

# --- per-window physiology and laboratory features ---
FEATS = ['map', 'heart_rate', 'sbp_arterial', 'dbp_arterial', 'respiratory_rate', 'spo2',
         'lactate', 'creatinine', 'wbc', 'urea_nitrogen', 'platelets', 'gcs', 'age',
         'shock_index', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cns', 'sofa_renal']


def severity_stratum(noncv_sofa):
    """Map a non-cardiovascular SOFA score to a severity stratum (0/1/2)."""
    x = np.asarray(noncv_sofa)
    return np.where(x <= SOFA_CUTS[0], 0, np.where(x <= SOFA_CUTS[1], 1, 2))
