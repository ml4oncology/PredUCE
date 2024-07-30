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
            # date of visit
            "processed_date": "clinic_date",
            # date the note was uploaded to EPR (meaning the note could have been revised)
            "EPRDate": "upload_date",
            "MRN": "mrn",
        }
    )
    clinic["clinic_date"] = clinic["clinic_date"].dt.tz_localize(None)
    clinic["upload_date"] = clinic["upload_date"].dt.tz_localize(None)

    # removing erroneous entries
    mask = clinic["clinic_date"] <= clinic["clinic_date"].quantile(0.0001)
    print(
        f'Removing {sum(mask)} visits that "occured before" {clinic[mask].clinic_date.max()}'
    )
    clinic = clinic[~mask]

    clinic = clinic.sort_values(by=["mrn", "clinic_date"])
    return clinic
