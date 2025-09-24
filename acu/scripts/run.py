import os

import pandas as pd
from dotenv import load_dotenv
from make_clinical_dataset.shared.constants import ROOT_DIR
from preduce.acu.pipeline import prepare, train_and_eval

load_dotenv()

DATE = '2025-03-29'
DATA_PATH = f'{ROOT_DIR}/data/final/data_{DATE}/processed/treatment_centered_data.parquet'
SAVE_PATH = os.getenv("SAVE_PATH")

def main():
    df = pd.read_parquet(DATA_PATH)
    out = prepare(df)
    target = 'target_ED_90d'
    res = train_and_eval(out, targets=[target], save_path=SAVE_PATH, load_model=False, train_kwargs=dict(time_limit=10e6))
    res['val'].to_csv(f'{SAVE_PATH}/{target}/val_score.csv', index=False)
    res['test'].to_csv(f'{SAVE_PATH}/{target}/test_score.csv', index=False)

if __name__ == '__main__':
    main()