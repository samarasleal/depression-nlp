# Standardized depression outcomes
import numpy as np
import pandas as pd

HDRS_T0 = "Questionnary1_HDRS_T0_TOT21_Score"
HDRS_T1 = "Questionnary1_HDRS_T1_TOT21_Score"
CDI_T0 = "Questionnary2_CDI_T0_Score"
CDI_T1 = "Questionnary2_CDI_T1_Score"

# Reduction = T0 - T1, so positive values indicate symptom improvement.
def build_depression_outcomes(df: pd.DataFrame, max_hdrs: float, max_cdi: float, min_cdi: float) -> pd.DataFrame:
    """Create continuous and directional binary depression outcomes."""
    clinical_columns = [HDRS_T0, HDRS_T1, CDI_T0, CDI_T1]
    missing = {"Patient_ID", *clinical_columns}.difference(df.columns)

    if missing:
        raise ValueError(f"Clinical data are missing columns: {sorted(missing)}")
    if max_hdrs <= 0 or max_cdi <= min_cdi:
        raise ValueError("Invalid HDRS/CDI normalization limits.")

    clinical = df[["Patient_ID", *clinical_columns]].copy()
    clinical["Patient_ID"] = clinical["Patient_ID"].astype("string").str.strip()

    if clinical["Patient_ID"].isna().any() or clinical["Patient_ID"].eq("").any():
        raise ValueError("Clinical data contain missing Patient_ID values.")

    clinical[clinical_columns] = clinical[clinical_columns].apply(pd.to_numeric, errors="coerce")
    clinical = clinical.groupby("Patient_ID", as_index=False).first()

    rows = []
    for _, row in clinical.iterrows():
        instrument, raw_t0, raw_t1 = None, np.nan, np.nan
        norm_t0, norm_t1 = np.nan, np.nan

        if pd.notna(row[HDRS_T0]) and pd.notna(row[HDRS_T1]):
            instrument, raw_t0, raw_t1 = "HDRS", float(row[HDRS_T0]), float(row[HDRS_T1])
            norm_t0, norm_t1 = raw_t0 / max_hdrs, raw_t1 / max_hdrs
        elif pd.notna(row[CDI_T0]) and pd.notna(row[CDI_T1]):
            instrument, raw_t0, raw_t1 = "CDI", float(row[CDI_T0]), float(row[CDI_T1])
            norm_t0 = (raw_t0 - min_cdi) / (max_cdi - min_cdi)
            norm_t1 = (raw_t1 - min_cdi) / (max_cdi - min_cdi)

        raw_reduction = raw_t0 - raw_t1 if instrument else np.nan
        reduction = norm_t0 - norm_t1 if instrument else np.nan

        if pd.isna(raw_reduction):
            direction, binary = None, np.nan
        elif raw_reduction > 0:
            direction, binary = "Improved", 1
        elif raw_reduction == 0:
            direction, binary = "Stable", 0
        else:
            direction, binary = "Worsened", 0

        rows.append({
            "Patient_ID": row["Patient_ID"],
            "depression_instrument": instrument,
            "depression_baseline_raw": raw_t0,
            "depression_final_raw": raw_t1,
            "depression_raw_reduction": raw_reduction,
            "depression_baseline": norm_t0,
            "depression_final": norm_t1,
            "depression_reduction": reduction,
            "change_direction": direction,
            "binary_outcome": binary,
            "outcome_available": int(pd.notna(reduction)),
            "binary_outcome_available": int(pd.notna(binary)),
        })

    return pd.DataFrame(rows).sort_values("Patient_ID").reset_index(drop=True)