"""
Module to create summary tables
"""

from typing import Optional, Sequence

import pandas as pd

from common.src.constants import CANCER_CODE_MAP, DRUG_COLS, LAB_COLS, UNIT_MAP


def pre_and_post_treatment_missingness_summary(
    df: pd.DataFrame,
    pretreatment_cols: list[str],
    posttreatment_cols: list[str],
    event_cols: list[str],
) -> pd.DataFrame:
    """Creates a summary table of the following:

    any_missingness_trt:
        proportion (No, %) of treatments where the pre- and/or post-treatment symptom score was not measured
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


def feature_summary(
    X_train: pd.DataFrame,
    save_path: Optional[str] = None,
    keep_orig_names: bool = False,
    remove_missingness_feats: bool = True,
) -> pd.DataFrame:
    """
    Args:
        X_train (pd.DataFrame): table of the original data (not one-hot encoded, normalized, clipped, etc) for the
            training set
    """
    N = len(X_train)

    if remove_missingness_feats:
        # remove missingness features
        cols = X_train.columns
        drop_cols = cols[cols.str.contains("is_missing")]
        X_train = X_train.drop(columns=drop_cols)

    # get number of missing values, mean, and standard deviation for each feature in the training set
    summary = X_train.astype(float).describe()
    summary = summary.loc[["count", "mean", "std"]].T
    count = N - summary["count"]
    mean = summary["mean"].round(3).apply(lambda x: f"{x:.3f}")
    std = summary["std"].round(3).apply(lambda x: f"{x:.3f}")
    count[count.between(1, 5)] = 6  # mask small cells less than 6
    summary["Mean (SD)"] = mean + " (" + std + ")"
    summary["Missingness (%)"] = (count / N * 100).round(1)
    summary = summary.drop(columns=["count", "mean", "std"])
    # special case for drug features (percentage of dose given)
    for col in DRUG_COLS:
        if col not in X_train.columns:
            continue
        mask = (
            X_train[col] != 0
        )  # 0 indicates no drugs were given (not the percentage of the given dose)
        vals = X_train.loc[mask, col]
        summary.loc[col, "Mean (SD)"] = f"{vals.mean():.3f} ({vals.std():.3f})"

    # assign the groupings for each feature
    feature_groupings_by_keyword = {
        "Acute care use": "ED_visit",
        "Cancer": "cancer_site|morphology",
        "Demographic": "height|weight|body_surface_area|female|age",
        "Laboratory": "|".join(LAB_COLS),
        "Treatment": "visit_month|regimen|intent|treatment|dose|therapy|cycle",
        "Symptoms": "esas|ecog",
    }
    features = summary.index
    for group, keyword in feature_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), "Group"] = group
    summary = summary[["Group", "Mean (SD)", "Missingness (%)"]]

    if keep_orig_names:
        summary["Features (original)"] = summary.index

    # insert units
    rename_map = {
        feat: f"{feat} ({unit})" for unit, feats in UNIT_MAP.items() for feat in feats
    }
    rename_map["female"] = "female (yes/no)"
    summary = summary.rename(index=rename_map)

    summary.index = [clean_feature_name(feat) for feat in summary.index]
    summary = summary.reset_index(names="Features")
    summary = summary.sort_values(by=["Group", "Features"])
    if save_path is not None:
        summary.to_csv(f"{save_path}", index=False)
    return summary


def clean_feature_name(name: str) -> str:
    if name == "patient_ecog":
        return "Eastern Cooperative Oncology Group (ECOG) Performance Status"

    mapping = {
        "prev": "previous",
        "num_": "number_of_",
        "%_ideal_dose": "percentage_of_ideal_dose",
        "intent": "intent_of_systemic_treatment",
        "cancer_site": "topography_ICD-0-3",
        "morphology": "morphology_ICD-0-3",
        "shortness_of_breath": "dyspnea",
        "tiredness": "fatigue",
        "patient_ecog": "eastern_cooperative_oncology_group_(ECOG)_performance_status",
        "cycle_number": "chemotherapy_cycle",
    }
    for orig, new in mapping.items():
        name = name.replace(orig, new)

    # title the name and replace underscores with space, but don't modify anything inside brackets at the end
    if name.endswith(")") and not name.startswith("regimen"):
        name, extra_info = name.split("(")
        name = "(".join([name.replace("_", " ").title(), extra_info])
    else:
        name = name.replace("_", " ").title()

    # capitalize certain substrings
    for substr in ["Ed V", "Icd", "Other", "Esas", "Ecog"]:
        name = name.replace(substr, substr.upper())
    # lowercase certain substrings
    for substr in [" Of "]:
        name = name.replace(substr, substr.lower())

    if name.startswith("Topography ") or name.startswith("Morphology "):
        # get full cancer description
        code = name.split(" ")[-1]
        if code in CANCER_CODE_MAP:
            name = f"{name}, {CANCER_CODE_MAP[code]}"
    elif name.startswith("ESAS "):
        # add 'score'
        if "Change" in name:
            name = name.replace("Change", "Score Change")
        else:
            name += " Score"

    for prefix in ["Regimen ", "Percentage of Ideal Dose Given "]:
        if name.startswith(prefix):
            # capitalize all regimen / drug names
            name = f"{prefix}{name.split(prefix)[-1].upper()}"

    return name


def get_patient_characteristics(
    df: pd.DataFrame,
    top_regimens: Optional[Sequence] = None,
    top_cancers: Optional[Sequence] = None,
):
    """Get summary statistics of the patients in the cohort"""
    if top_cancers is None:
        top_cancers = []
    if top_regimens is None:
        top_regimens = []

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
    median = height.median().round(1)
    q25, q75 = height.quantile([0.25, 0.75]).round(1)
    pc["Height (cm), Median (IQR)"] = f"{median} ({q25}-{q75})"

    # weight
    weight = df["weight"]
    median = weight.median().round(1)
    q25, q75 = weight.quantile([0.25, 0.75]).round(1)
    pc["Weight (kg), Median (IQR)"] = f"{median} ({q25}-{q75})"

    # sex
    num_females = df["female"].sum()
    pc["Female, No. (%)"] = f"{num_females} ({num_females/N*100:.1f})"

    # regimens
    for regimen in top_regimens:
        num_regimens = sum(df["regimen"] == regimen)
        pc[f"Regimen {regimen}, No. (%)"] = f"{num_regimens} ({num_regimens/N*100:.1f})"

    # cancers
    for cancer in top_cancers:
        num_cancers = df[cancer].sum()
        pc[f"Cancer Site {CANCER_CODE_MAP[cancer[-3:]]}, No. (%)"] = (
            f"{num_cancers} ({num_cancers/N*100:.1f})"
        )

    return pc
