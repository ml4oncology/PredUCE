from tqdm import tqdm

import pandas as pd

def test_date_is_ordered(df: pd.DataFrame, date_col: str, patient_col: str) -> None:
    """Ensure that for each patient, the date is in ascending order"""
    for mrn, group in tqdm(df.groupby(patient_col)):
        assert all(group[date_col].to_numpy() == group[date_col].sort_values().to_numpy())