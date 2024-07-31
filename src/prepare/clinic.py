"""
Module to preprocess clinic visit data

TODO: move to make-clinical-dataset
"""

from typing import Optional

import pandas as pd


def get_clinic_visit_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = "./data/interim"

    treatment = pd.read_parquet(f"{data_dir}/treatment.parquet.gzip")
    clinic_visit = pd.read_parquet(f"{data_dir}/clinical_notes.parquet.gzip")

    clinic_visit = filter_clinic_visit_data(clinic_visit, treatment)
    clinic_visit = process_clinic_visit_data(clinic_visit, treatment)

    return clinic_visit


def process_clinic_visit_data(
    clinic: pd.DataFrame, treatment: pd.DataFrame
) -> pd.DataFrame:
    # Calculate the median height, weight, and body surface area prior to clinic date
    # TODO: put this in make_clinical_dataset.combine.add_engineered_features
    # TODO: use pd.merge instead of split_and_parallelize when it makes sense
    cols = ["height", "weight", "body_surface_area"]
    prior_median = pd.merge(
        clinic[["mrn", "clinic_date"]],
        treatment[["mrn", "treatment_date"] + cols],
        on="mrn",
        how="inner",
    )
    prior_median = prior_median.rename(
        columns={col: f"prior_median_{col}" for col in cols}
    )
    prior_median = prior_median.query("treatment_date < clinic_date").drop(
        columns=["treatment_date"]
    )
    prior_median = prior_median.groupby(["mrn", "clinic_date"]).median().reset_index()
    # sns.displot(data=prior_median, x='prior_median_height', y='prior_median_weight')
    clinic = pd.merge(clinic, prior_median, on=["mrn", "clinic_date"], how="left")

    clinic = clinic.sort_values(by=["mrn", "clinic_date"])
    return clinic


def filter_clinic_visit_data(
    clinic: pd.DataFrame, treatment: pd.DataFrame
) -> pd.DataFrame:
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

    # combine clinic and treatment
    cols = ["treatment_date", "regimen", "line_of_therapy", "intent", "cycle_number"]
    df = pd.merge(clinic, treatment[["mrn"] + cols], on="mrn", how="inner")
    df = df.rename(columns={col: f"next_{col}" for col in cols})

    # filter out clinic dates where the next treatment session does not occur within 5 days
    mask = df["next_treatment_date"].between(
        df["clinic_date"], df["clinic_date"] + pd.Timedelta(days=5)
    )
    df = df[mask]

    # filter out clinic dates where notes were uploaded after the next treatment session
    mask = df["upload_date"] < df["next_treatment_date"]
    df = df[mask]

    # remove duplicates from the merging
    df = df.sort_values(by=["mrn", "next_treatment_date"])
    df = df.drop_duplicates(subset=["mrn", "clinic_date"], keep="first")

    return df


def process_clinic_visits_prior_to_treatment(df: pd.DataFrame):
    """Process missing treatment information

    Patient visit flow
        -> first clinic visit (book treatment plan)
        -> second clinic visit (check up)
        -> start treatment
    For the clinic visits right before starting a new treatment, we are missing treatment information
    But the treatment is pre-booked at that point
    So we can pull treatment information back

    NOTE: this procedure occurs after anchoring treatment information to clinic visits
    """
    # backfill treatment information if no prior treatments within X days
    no_trts_prior = df["treatment_date"].isnull()
    for col in ["regimen", "line_of_therapy", "intent", "cycle_number"]:
        df.loc[no_trts_prior, col] = df.pop(f"next_{col}")
    df.loc[no_trts_prior, "days_since_starting_treatment"] = 0

    # for missing height, weight, body surface area, take the median value prior to clinic visit
    for col in ["height", "weight", "body_surface_area"]:
        df[col] = df[col].fillna(df.pop(f"prior_median_{col}"))

    return df
