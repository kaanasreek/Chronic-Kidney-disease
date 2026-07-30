"""
CKD DATASET PREPROCESSING PIPELINE — SELECTED FEATURES ONLY
=============================================================
Combines 4 datasets into one clean, analysis-ready CSV.
Features retained (14 selected):
  serum_creatinine, blood_urea, albumin, specific_gravity,
  hemoglobin, packed_cell_volume, red_blood_cell_count,
  white_blood_cell_count, sodium, potassium, blood_pressure,
  diabetes_mellitus, hypertension, age

Dataset summary:
  DS1 - ALU_DCS_Santhosh_Research_CKDDataset_1.csv  →  40 rows  | CKD only (no target col)
  DS2 - chronic_kidney_disease.csv                   → 400 rows  | ckd / notckd target
  DS3 - ckd-dataset-v2.csv                           → 200 rows  | range-valued cols, affected=0/1
  DS4 - kidney_disease_dataset.csv                   → 20538 rows| multi-class target, has real eGFR
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Selected feature columns (14 features + target + source)
# ─────────────────────────────────────────────
SELECTED_FEATURES = [
    'serum_creatinine',
    'blood_urea',
    'albumin',
    'specific_gravity',
    'hemoglobin',
    'packed_cell_volume',
    'red_blood_cell_count',
    'white_blood_cell_count',
    'sodium',
    'potassium',
    'blood_pressure',
    'diabetes_mellitus',
    'hypertension',
    'age',
]


# ─────────────────────────────────────────────
# STEP 1 — LOAD EACH DATASET INDIVIDUALLY
# ─────────────────────────────────────────────

def load_ds1(path):
    df = pd.read_csv(path, skiprows=3)
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna(how='all')
    df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]

    df = df.rename(columns={
        'age': 'age',
        'bu': 'blood_urea',
        'sc': 'serum_creatinine',
        'al': 'albumin',
        'sod': 'sodium',
        'pot': 'potassium',
        'hemo': 'hemoglobin',
        'pcv': 'packed_cell_volume',
        'wc': 'white_blood_cell_count',
        'rc': 'red_blood_cell_count',
        'htn': 'hypertension',
        'dm': 'diabetes_mellitus',
    })

    df['target'] = 1        # all records are confirmed CKD patients
    df['source'] = 'ds1'
    return df


def load_ds2(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.drop(columns=['id'], errors='ignore')

    df = df.rename(columns={
        'bp': 'blood_pressure',
        'sg': 'specific_gravity',
        'al': 'albumin',
        'bu': 'blood_urea',
        'sc': 'serum_creatinine',
        'sod': 'sodium',
        'pot': 'potassium',
        'hemo': 'hemoglobin',
        'pcv': 'packed_cell_volume',
        'wc': 'white_blood_cell_count',
        'rc': 'red_blood_cell_count',
        'htn': 'hypertension',
        'dm': 'diabetes_mellitus',
        'classification': 'target',
    })

    df['source'] = 'ds2'
    return df


def load_ds3(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Remove junk metadata rows
    junk_mask = df.apply(
        lambda col: col.astype(str).str.contains(
            r'discrete|^class$|^meta$', case=False, na=False)
    ).any(axis=1)
    df = df[~junk_mask].reset_index(drop=True)

    df = df.rename(columns={
        'bp (diastolic)': 'blood_pressure',
        'sg': 'specific_gravity',
        'al': 'albumin',
        'bu': 'blood_urea',
        'sc': 'serum_creatinine',
        'sod': 'sodium',
        'pot': 'potassium',
        'hemo': 'hemoglobin',
        'pcv': 'packed_cell_volume',
        'rbcc': 'red_blood_cell_count',
        'wbcc': 'white_blood_cell_count',
        'htn': 'hypertension',
        'dm': 'diabetes_mellitus',
        'affected': 'target',
    })

    df['source'] = 'ds3'
    return df


def load_ds4(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    df = df.rename(columns={
        'age of the patient': 'age',
        'blood pressure (mm/hg)': 'blood_pressure',
        'specific gravity of urine': 'specific_gravity',
        'albumin in urine': 'albumin',
        'blood urea (mg/dl)': 'blood_urea',
        'serum creatinine (mg/dl)': 'serum_creatinine',
        'sodium level (meq/l)': 'sodium',
        'potassium level (meq/l)': 'potassium',
        'hemoglobin level (gms)': 'hemoglobin',
        'packed cell volume (%)': 'packed_cell_volume',
        'white blood cell count (cells/cumm)': 'white_blood_cell_count',
        'red blood cell count (millions/cumm)': 'red_blood_cell_count',
        'hypertension (yes/no)': 'hypertension',
        'diabetes mellitus (yes/no)': 'diabetes_mellitus',
        'target': 'target',
    })

    df['source'] = 'ds4'
    return df


# ─────────────────────────────────────────────
# STEP 2 — KEEP ONLY SELECTED FEATURES
# ─────────────────────────────────────────────

def keep_selected_features(df):
    """Keep only the 14 selected features + target + source."""
    keep_cols = SELECTED_FEATURES + ['target', 'source']
    present = [c for c in keep_cols if c in df.columns]
    return df[present]


# ─────────────────────────────────────────────
# STEP 3 — CLEAN CATEGORICAL VALUES → 0/1
# ─────────────────────────────────────────────

def clean_categoricals(df):
    yes_no_map = {
        'yes': 1, 'y': 1, 'present': 1, 'abnormal': 1,
        'no': 0,  'n': 0, 'notpresent': 0, 'not present': 0, 'normal': 0,
        '?': np.nan, 'nan': np.nan, '': np.nan, 'none': np.nan,
    }

    binary_cols = ['hypertension', 'diabetes_mellitus']
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.lower().str.strip()
                       .str.replace(r'\s+', '', regex=True)
                       .map(lambda x: yes_no_map.get(x, np.nan)))

    if 'target' in df.columns:
        target_map = {
            'ckd': 1, 'ckd\t': 1, 'notckd': 0,
            'no_disease': 0,
            'low_risk': 1, 'moderate_risk': 1, 'high_risk': 1, 'severe_disease': 1,
            '1': 1, '1.0': 1, '0': 0, '0.0': 0,
            '?': np.nan, 'nan': np.nan,
        }
        df['target'] = (df['target'].astype(str).str.lower().str.strip()
                        .map(lambda x: target_map.get(x, np.nan)))
        df['target'] = pd.to_numeric(df['target'], errors='coerce')

    return df


# ─────────────────────────────────────────────
# STEP 4 — CONVERT ALL FEATURES TO NUMERIC
# ─────────────────────────────────────────────

def convert_to_numeric(df):
    skip = {'source', 'target'}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ─────────────────────────────────────────────
# STEP 5 — IMPUTE MISSING VALUES
# ─────────────────────────────────────────────

def impute_features(df):
    skip = {'target', 'source'}
    impute_cols = [c for c in df.select_dtypes(include=np.number).columns
                   if c not in skip]

    missing = df[impute_cols].isnull().sum()
    has_missing = missing[missing > 0]
    if not has_missing.empty:
        print(f"\n  Missing values being imputed (median):")
        for col, cnt in has_missing.items():
            print(f"    {col:35s}: {cnt:5,}  ({cnt/len(df)*100:.1f}%)")

    imputer = SimpleImputer(strategy='median')
    df[impute_cols] = imputer.fit_transform(df[impute_cols])
    return df


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CKD PREPROCESSING PIPELINE — SELECTED FEATURES ONLY")
    print("=" * 65)

    # ── 1. Load ─────────────────────────────────────────────────
    print("\n[1/9] Loading datasets...")
    ds1 = load_ds1("ALU_DCS_Santhosh_Research_CKDDataset_1.csv")
    ds2 = load_ds2("chronic_kidney_disease.csv")
    ds3 = load_ds3("ckd-dataset-v2.csv")
    ds4 = load_ds4("kidney_disease_dataset.csv")
    print(f"  DS1: {ds1.shape[0]:>6,} rows")
    print(f"  DS2: {ds2.shape[0]:>6,} rows")
    print(f"  DS3: {ds3.shape[0]:>6,} rows")
    print(f"  DS4: {ds4.shape[0]:>6,} rows")

    # ── 2. Keep only selected features ──────────────────────────
    print("\n[2/9] Keeping only 14 selected features + target + source...")
    ds1 = keep_selected_features(ds1)
    ds2 = keep_selected_features(ds2)
    ds3 = keep_selected_features(ds3)
    ds4 = keep_selected_features(ds4)
    print("  Done.")

    # ── 3. Clean categoricals ────────────────────────────────────
    print("\n[3/9] Cleaning categorical values (yes/no → 1/0)...")
    ds1 = clean_categoricals(ds1)
    ds2 = clean_categoricals(ds2)
    ds3 = clean_categoricals(ds3)
    ds4 = clean_categoricals(ds4)
    print("  Done.")

    # ── 4. Combine ───────────────────────────────────────────────
    print("\n[4/9] Combining all datasets...")
    df = pd.concat([ds1, ds2, ds3, ds4], ignore_index=True)
    print(f"  Combined: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 5. Convert to numeric ────────────────────────────────────
    print("\n[5/9] Converting to numeric...")
    df = convert_to_numeric(df)
    print("  Done.")

    # ── 6. Remove duplicate rows ─────────────────────────────────
    print("\n[6/9] Removing duplicate rows...")
    before = len(df)
    feature_cols = [c for c in df.columns if c != 'source']
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    print(f"  Removed {before - len(df):,} duplicates → {len(df):,} rows remain")

    # ── 7. Drop rows with missing target ─────────────────────────
    print("\n[7/9] Dropping rows with missing target...")
    before = len(df)
    df = df.dropna(subset=['target']).reset_index(drop=True)
    print(f"  Removed {before - len(df):,} rows → {len(df):,} rows remain")

    # ── 8. Impute missing values ─────────────────────────────────
    print("\n[8/9] Imputing missing feature values with median...")
    df = impute_features(df)

    # ── 9. Assign clean sequential IDs ──────────────────────────
    print("\n[9/9] Assigning sequential IDs (1 → N)...")
    df.insert(0, 'id', range(1, len(df) + 1))

    # Final column order
    col_order = ['id'] + SELECTED_FEATURES + ['source', 'target']
    present = [c for c in col_order if c in df.columns]
    df = df[present]
    df['target'] = df['target'].astype(int)

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL DATASET SUMMARY")
    print("=" * 65)
    print(f"\n  Total rows    : {df.shape[0]:,}")
    print(f"  Total columns : {df.shape[1]}")

    print(f"\n  Selected features (14):")
    for i, col in enumerate(SELECTED_FEATURES, 1):
        missing = df[col].isnull().sum() if col in df.columns else 'N/A'
        status = f"  ← ⚠ {missing} missing" if isinstance(missing, int) and missing > 0 else ""
        print(f"    {i:2d}. {col}{status}")

    print(f"\n  Target distribution:")
    for val, cnt in df['target'].value_counts().sort_index().items():
        label = "CKD / At Risk      (1)" if val == 1 else "Healthy / No Disease (0)"
        print(f"    {label}: {cnt:6,}  ({cnt/len(df)*100:.1f}%)")

    print(f"\n  Source breakdown:")
    for src, cnt in df['source'].value_counts().sort_index().items():
        print(f"    {src}: {cnt:,} rows")

    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        print("\n  ✅ ZERO missing values in all feature columns!")
    else:
        print(f"\n  ⚠ {total_missing} missing values remain")

    # ── SAVE ─────────────────────────────────────────────────────
    output_file = "combined_ckd_selected_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  💾 Saved → {output_file}")
    print("=" * 65)

    return df


if __name__ == "__main__":
    df = main()