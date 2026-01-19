import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from make_clinical_dataset.shared.constants import ROOT_DIR
from preduce.acu.pipeline import prepare, train_and_eval

load_dotenv()

DATE = "2025-03-29"
DATA_PATH = f"{ROOT_DIR}/data/final/data_{DATE}/processed/clinic_centered_data.parquet"
DATES_PATH = (
    f"{ROOT_DIR}/data/final/data_{DATE}/processed/clinic_centered_dates.parquet"
)
SAVE_PATH = os.getenv("SAVE_PATH")


def main(first_visit_only: bool):
    df = pd.read_parquet(DATA_PATH)
    if first_visit_only:
        dates = pd.read_parquet(DATES_PATH)
        df = df[dates["treatment_date"].isna()]
    out = prepare(df)
    targ_cols = ["target_ED_30d", "target_ED_60d", "target_ED_90d"]
    res = train_and_eval(out, targets=targ_cols, save_path=SAVE_PATH, load_model=False)
    res["val"].to_csv("val_score.csv", index=False)
    res["test"].to_csv("test_score.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-visit-only",
        action="store_true",
        help="If set, only use first visit data.",
    )
    args = parser.parse_args()
    main(args.first_visit_only)
