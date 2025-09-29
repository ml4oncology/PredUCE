"""
Module to create summary tables
"""

from typing import Optional, Sequence

import pandas as pd
from make_clinical_dataset.shared.constants import (
    LAB_COLS,
    SYMP_COLS,
    UNIT_MAP,
)


def pre_and_post_treatment_missingness_summary(
    df: pd.DataFrame,
    pretreatment_cols: list[str],
    posttreatment_cols: list[str],
    event_cols: list[str],
) -> pd.DataFrame:
    """Creates a summary table of the following:

    any_missingness_trt:
        proportion (No, %) of treatments where the pre- and/or post-treatment score was not measured
    target_missingness_trt:
        proportion (No, %) of treatments where the post-treatment target is not measured, when the pre-treatment
        feature is measured
    event_rate_trt:
        proportion (No, %) of treatments followed by the event, when the pre-treatment feature and post-treatment
        target is measured
    any_missingness_mrn:
        proportion (No, %) of patients with at least one treatment with any missingness
    event_rate_mrn:
        proportion (No, %) of patients with at least one treatment followed by an event, among patients who have at
        least one target measurement
    """
    result = {}
    for pre_col, post_col, event_col in zip(
        pretreatment_cols, posttreatment_cols, event_cols
    ):
        pretreatment_exists = df[pre_col].notnull()
        posttreatment_exists = df[post_col].notnull()
        event_occured = df[event_col] == 1
        result[pre_col] = {
            # any missingness of treatments
            "num_trts_without_pre_or_post": (
                ~posttreatment_exists | ~pretreatment_exists
            ).sum(),
            # target missingness of treatments
            "num_trts_with_pre_and_without_post": (
                ~posttreatment_exists & pretreatment_exists
            ).sum(),
            "num_trts_with_pre": pretreatment_exists.sum(),
            # event rate of treatments
            "num_trts_with_event": event_occured.sum(),
            "num_trts_with_pre_and_post": (
                pretreatment_exists & posttreatment_exists
            ).sum(),
            # any missingness of patients
            "num_mrns_without_pre_or_post": df.loc[
                (~posttreatment_exists | ~pretreatment_exists), "mrn"
            ].nunique(),
            # event rate of patients
            "num_mrns_with_event": df.loc[event_occured, "mrn"].nunique(),
            "num_mrns_with_post": df.loc[posttreatment_exists, "mrn"].nunique(),
        }
    result = pd.DataFrame(result).T
    # NOTE: the reason I store the numerator and denominator separately is to compute the mean easier
    result.loc["Mean"] = result.mean().astype(int)

    # calculate the proportion
    def calc(numerator, denominator):
        return (
            numerator.astype(str)
            + " ("
            + (numerator / denominator * 100).round(1).astype(str)
            + ")"
        )

    result["any_missingness_trt"] = calc(
        result["num_trts_without_pre_or_post"], len(df)
    )
    result["target_missingness_trt"] = calc(
        result["num_trts_with_pre_and_without_post"], result["num_trts_with_pre"]
    )
    result["event_rate_trt"] = calc(
        result["num_trts_with_event"], result["num_trts_with_pre_and_post"]
    )
    result["any_missingness_mrn"] = calc(
        result["num_mrns_without_pre_or_post"], df["mrn"].nunique()
    )
    result["event_rate_mrn"] = calc(
        result["num_mrns_with_event"], result["num_mrns_with_post"]
    )

    return result


def feature_summary(feats: pd.DataFrame) -> pd.DataFrame:
    N = len(feats)

    # get number of missing values, mean, and standard deviation for each feature in the training set
    summary = feats.astype(float).describe()
    summary = summary.loc[["count", "mean", "std"]].T
    count = N - summary["count"]
    mean = summary["mean"].round(3).apply(lambda x: f"{x:.3f}")
    std = summary["std"].round(3).apply(lambda x: f"{x:.3f}")
    count[count.between(1, 5)] = 6  # mask small cells less than 6
    summary["Mean (SD)"] = mean + " (" + std + ")"
    summary["Missingness (%)"] = (count / N * 100).round(1)
    summary = summary.drop(columns=["count", "mean", "std"])

    # assign the groupings for each feature
    feature_groupings_by_keyword = {
        "Acute care use": "ED",
        "Cancer": "cancer_type",
        "Demographic": "height|weight|body_surface_area|female|age",
        "Laboratory": "|".join(LAB_COLS),
        "Treatment": "regimen|intent|treatment|dose|drug|therapy|cycle",
        "Symptoms": "|".join(SYMP_COLS),
    }
    features = summary.index
    for group, keyword in feature_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), "Group"] = group
    summary = summary[["Group", "Mean (SD)", "Missingness (%)"]]

    # clean name and insert units
    unit_map = {
        feat: f" ({unit})" for unit, feats in UNIT_MAP.items() for feat in feats
    }
    unit_map["female"] = " (yes/no)"
    summary.index = [f"{clean_feature_name(feat)}{unit_map.get(feat, '')}" for feat in summary.index]
    summary = summary.reset_index(names="Features")
    summary = summary.sort_values(by=["Group", "Features"])
    return summary


def clean_feature_name(name: str) -> str:
    if name == "ecog":
        return "Eastern Cooperative Oncology Group (ECOG) Performance Status"
    name = name.replace("_", " ").title()
    name = name.replace(" Ed ", " ED ").replace(" Of ", " of ").replace(" Egfr ", " eGFR ")
    return name


def cohort_summary(
    df: pd.DataFrame,
    top_regimens: Optional[Sequence] = None,
    top_cancers: Optional[Sequence] = None,
    targets: Optional[Sequence] = None,
):
    """Get summary statistics of the patients in the cohort"""
    if top_cancers is None:
        top_cancers = []
    if top_regimens is None:
        top_regimens = []
    if targets is None:
        targets = []

    N = len(df)
    pc = {}  # patient characteristics

    # number of treatment sessions
    num_sessions = df.groupby("mrn").apply(len, include_groups=False)
    median = int(num_sessions.median())
    q25, q75 = num_sessions.quantile([0.25, 0.75]).astype(int)
    pc["Number of Treatments, Median (IQR)"] = f"{median} ({q25}-{q75})"

    # age
    age = df["age"]
    median = int(age.median())
    q25, q75 = age.quantile([0.25, 0.75]).astype(int)
    pc["Age (years), Median (IQR)"] = f"{median} ({q25}-{q75})"

    # height
    height = df["height"]
    median = height.median()
    q25, q75 = height.quantile([0.25, 0.75]).round(1)
    pc["Height (cm), Median (IQR)"] = f"{median:.1f} ({q25}-{q75})"

    # weight
    weight = df["weight"]
    median = weight.median()
    q25, q75 = weight.quantile([0.25, 0.75]).round(1)
    pc["Weight (kg), Median (IQR)"] = f"{median:.1f} ({q25}-{q75})"

    # sex
    # num_females = df["female"].sum()
    # pc["Female, No. (%)"] = f"{num_females} ({num_females/N*100:.1f})"

    # regimens
    for regimen in top_regimens:
        num_regimens = sum(df["regimen"] == regimen)
        pc[f"Regimen {regimen}, No. (%)"] = f"{num_regimens} ({num_regimens/N*100:.1f})"

    # cancers
    for cancer in top_cancers:
        num_cancers = sum(df["cancer_type"] == cancer)
        pc[f"Cancer Site {cancer}, No. (%)"] = (f"{num_cancers} ({num_cancers/N*100:.1f})")

    # targets
    for target in targets:
        num_targets = df[target].sum()
        pc[f"{target.upper()}, No. (%)"] = f"{num_targets} ({num_targets/N*100:.1f})"

    return pc
