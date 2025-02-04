import pandas as pd
import pytest

from preduce.summarize import get_patient_characteristics


def test_patient_characteristics():
    data = {
        "mrn": [1, 1, 1, 2, 2, 3],
        "age": [50] * 6,
        "height": [170] * 6,
        "weight": [70] * 6,
        "female": [1] * 6,
    }
    df = pd.DataFrame(data)
    result = get_patient_characteristics(df)
    assert result["Number of Treatments, Median (IQR)"] == "2 (1-2)"
    assert result["Age (years), Median (IQR)"] == "50 (50-50)"
    assert result["Height (cm), Median (IQR)"] == "170.0 (170.0-170.0)"
    assert result["Weight (kg), Median (IQR)"] == "70.0 (70.0-70.0)"
    assert result["Female, No. (%)"] == "6 (100.0)"
