import pandas as pd
import pytest

from preduce.summarize import get_patient_characteristics
import ml_common


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "mrn": [1, 1, 2, 2, 2, 3],
            "age": [60, 61, 70, 72, 71, 65],
            "height": [170.2, 171.0, 168.5, 169.0, 168.8, 172.3],
            "weight": [70.5, 71.0, 80.2, 79.5, 79.8, 75.3],
            "female": [1, 1, 0, 0, 0, 1],
            "regimen": ["A", "B", "A", "A", "B", "C"],
            "cancer_C25": [1, 0, 0, 1, 1, 0],
            "target1": [0, 1, 0, 0, 1, 1],
        }
    )


def test_patient_characteristics(sample_dataframe):
    result = get_patient_characteristics(
        sample_dataframe,
        top_regimens=["B"],
        top_cancers=["cancer_C25"],
        targets=["target1"],
    )
    assert result["Age (years), Median (IQR)"] == "67 (62-70)"
    assert result["Height (cm), Median (IQR)"] == "169.6 (168.9-170.8)"
    assert result["Female, No. (%)"] == "3 (50.0)"
    assert result["Cancer Site Pancreas, No. (%)"] == "3 (50.0)"
    assert result["Regimen B, No. (%)"] == "2 (33.3)"
    assert result["TARGET1, No. (%)"] == "3 (50.0)"
