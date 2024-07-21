import os

import datetime
import pandas as pd


###############################################################################
# I/O
###############################################################################
def initialize_folders():
    main_folders = ["logs", "models", "result/tables"]
    for folder in main_folders:
        if not os.path.exists(f"./{folder}"):
            os.makedirs(f"./{folder}")


def load_clinic_dates(data_dir: str):
    clinic = pd.read_parquet(f"{data_dir}/clinical_notes.parquet.gzip")

    # cleaning up columns
    clinic = clinic.rename(
        columns={
            "processed_date": "clinic_date",
            "EPRDate": "upload_date",
            "MRN": "mrn",
        }
    )
    assert all(clinic["clinic_date"].dt.time == datetime.time(0))  # date of visit
    assert all(
        clinic["upload_date"].dt.time == datetime.time(0)
    )  # date the note was uploaded to EPR (meaning it could have been revised)
    clinic["clinic_date"] = clinic["clinic_date"].dt.date
    clinic["upload_date"] = clinic["upload_date"].dt.date

    # removing erroneous entries
    mask = clinic["clinic_date"] <= clinic["clinic_date"].quantile(0.0001)
    print(
        f'Removing {sum(mask)} visits that "occured before" {clinic[mask].clinic_date.max()}'
    )
    clinic = clinic[~mask]

    return clinic
