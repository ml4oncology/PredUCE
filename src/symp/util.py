# Standard library imports
import logging
import os

os.environ["OUTDATED_IGNORE"] = "1"
from functools import reduce

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from .train import EHRDataset, pad_collate

logger = logging.getLogger(__name__)


###############################################################################
# I/O
###############################################################################
def initialize_folders():
    main_folders = ["logs", "models", "result/tables"]
    for folder in main_folders:
        if not os.path.exists(f"./{folder}"):
            os.makedirs(f"./{folder}")


###############################################################################
#  Data manipulation
###############################################################################
def calculate_mean_cal_error(prob_true, prob_pred, n_bins=10):
    """Compute mean calibration error (MCE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    calibration_errors = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(prob_pred >= bin_lower, prob_pred < bin_upper)
        if np.sum(in_bin) > 0:
            bin_true = np.mean(prob_true[in_bin])
            bin_pred = np.mean(prob_pred[in_bin])
            calibration_errors.append(np.abs(bin_true - bin_pred))

    return np.mean(calibration_errors)


def calculate_max_cal_error(prob_true, prob_pred, n_bins=10):
    """Compute maximum calibration error (MaxCE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    max_ce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(prob_pred >= bin_lower, prob_pred < bin_upper)
        if np.sum(in_bin) > 0:
            bin_true = np.mean(prob_true[in_bin])
            bin_pred = np.mean(prob_pred[in_bin])
            max_ce = max(max_ce, np.abs(bin_true - bin_pred))

    return max_ce


def find_largest_threshold_interval(df):
    sorted_df = df.sort_values(by="Threshold")
    interval_list = []

    i = 0
    while i < len(sorted_df):
        if (
            sorted_df["System"].iloc[i] > sorted_df["All"].iloc[i]
            and sorted_df["System"].iloc[i] >= 0
        ):
            start = sorted_df["Threshold"].iloc[i]
            end = start

            j = i + 1
            while (
                j < len(sorted_df)
                and sorted_df["System"].iloc[j] > sorted_df["All"].iloc[j]
                and sorted_df["System"].iloc[j] > 0
            ):
                end = sorted_df["Threshold"].iloc[j]
                j += 1

            interval_list.append((start, end))
            i = j
        else:
            i += 1

    if not interval_list:
        return None

    # Find the largest interval by comparing their lengths
    largest_interval = max(interval_list, key=lambda x: x[1] - x[0])
    return largest_interval


def partial_predict(X, label, model_name, train_results, df, device):
    """
    Generates predictions for sequential models by processing modified input samples,
    facilitating the computation of SHAP values. This function duplicates a single
    treatment session with various feature permutations and integrates historical
    patient data to form a complete input sequence.
       Args:
           X (pd.DataFrame): DataFrame containing a single treatment session, duplicated for feature permutation.
           label (str): The target label for prediction.
           model_name (str): Identifier for the specific model used.
           train_results (dict): Contains trained models and their metadata.
           df (pd.DataFrame): Full dataset containing historical patient data.
           device (torch.device): Device on which computations will be performed (GPU or CPU).

       Returns:
           np.ndarray: Predictions for the last session in the sequence, adjusted for feature permutations.
    """

    torch.cuda.empty_cache()
    # X is a single treatment session, duplicated multiple times with different feature permutations
    # Reformat to sequential data
    X = X.copy()
    N = len(X)
    # get the patient id and visit date for this sample row
    res = {}
    for col in ["PatientID", "Trt_Date"]:
        mask = X[col] != -1
        val = X.loc[mask, col].unique()
        assert len(val) == 1
        res[col] = val[0]
    # get patient's historical data
    # NOTE: historical data is NOT permuted
    hist = df.query(
        f'PatientID == {res["PatientID"]} & Trt_Date < {res["Trt_Date"]}'
    ).copy()
    n = len(hist)
    hist = pd.concat(
        [hist] * N
    )  # repeat patient historical data for each duplicated sample row
    # set up new patient id for each sample row
    ikns = np.arange(0, N, 1) + res["PatientID"]
    hist["PatientID"] = np.repeat(ikns, n)
    X["PatientID"] = ikns
    # combine historical data and sample rows togethers
    X["Trt_Date"] = res["Trt_Date"]
    X = pd.concat([X, hist]).sort_values(
        by=["PatientID", "Trt_Date"], ignore_index=True
    )

    # Get predictions
    dataset = EHRDataset(X, [label])
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=pad_collate
    )
    model_input = next(iter(loader))[0]

    # change different model
    model = train_results[model_name, label]["best_model"]
    model.half()  # use fp16 instead of fp32
    pred = model(model_input.to(device).half())

    # pred = gru_pain_model(model_input)
    # print(pred.shape)
    last_session_pred = pred[:, -1, :]
    last_session_pred = torch.sigmoid(last_session_pred)
    return last_session_pred.cpu().detach().numpy()


def find_threshold_for_alarm_rate(
    y_labels, y_preds, target_alarm_rate=0.1, search_steps=3000
):
    best_threshold = 0
    closest_alarm_rate_diff = float("inf")
    closest_alarm_rate = 0

    for threshold in np.linspace(0, 1, search_steps):
        y_pred = (
            (y_preds >= threshold).astype(int)
            if y_preds.ndim == 1
            else (y_preds >= threshold).any(axis=1).astype(int)
        )
        tn, fp, fn, tp = confusion_matrix(y_labels, y_pred).ravel()
        alarm_rate = (tp + fp) / len(y_labels)
        current_diff = abs(target_alarm_rate - alarm_rate)

        # Ensure TP is not zero and the current difference is smaller than any previously found
        if current_diff < closest_alarm_rate_diff and tp > 0:
            closest_alarm_rate_diff = current_diff
            best_threshold = threshold
            closest_alarm_rate = alarm_rate

    # print(f"Best threshold: {best_threshold} with alarm rate: {closest_alarm_rate}, TP: {tp}")
    return best_threshold


def get_merge_df(results_df, test_df, labels_list, model_name, alarm_rate=0.1):
    """
    Find the threshold that achieves a 10% alarm rate across all events and return the merge df.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing the results.
    test_df (pd.DataFrame): DataFrame containing the test data.
    labels_list (list): List of labels to consider.
    model_name (str): The name of the model to filter results for.

    Returns:
    pd.DataFrame: A DataFrame with predictions merged and a new 'any_symptoms' column.
    """

    model_results = results_df[results_df["model"] == model_name]
    predict = []

    # Loop over all labels in labels_list
    for label in labels_list:
        # Filter the DataFrame to only include rows where the label is not -1
        filtered_df = test_df[test_df[label] != -1].copy()

        label_results = model_results[model_results["label"] == label]
        # prediction = label_results['all_preds'].iloc[0][0]
        prediction = label_results["all_preds"].iloc[0]

        pred_name = label + "_prediction"
        # Add the prediction column to the DataFrame
        filtered_df[pred_name] = prediction
        predict.append(filtered_df)

    # Get column names excluding the last one from the first dataframe
    merge_cols = predict[0].columns.tolist()[:-1]
    # Perform the merge operation
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=merge_cols, how="inner"), predict
    )

    # Create a new column 'any_symptoms' that is 1 if any of the labels is 1, and 0 otherwise
    merged_df["any_symptoms"] = merged_df[labels_list].apply(
        lambda row: int(any(row)), axis=1
    )
    y_labels = merged_df["any_symptoms"].values
    prediction_columns = [
        col for col in merged_df.columns if col.endswith("prediction")
    ]
    y_preds = merged_df[prediction_columns].values

    threshold = find_threshold_for_alarm_rate(
        y_labels, y_preds, target_alarm_rate=alarm_rate
    )
    print(f"The threshold that achieves a {alarm_rate*100}% alarm rate is {threshold}")

    return merged_df, threshold
