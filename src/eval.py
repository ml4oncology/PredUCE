# Standard library imports
import os

os.environ["OUTDATED_IGNORE"] = "1"
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.util import find_threshold_for_alarm_rate


def calculate_event_metrics(
    results_df,
    test_df,
    labels_list,
    model_name,
    label_mapping,
    legend_order,
    threshold=None,
    choice="per_event",
):
    """
    Calculate metrics for a 10% alarm rate either per event or for all events.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing the results.
    test_df (pd.DataFrame): DataFrame containing the test data.
    labels_list (list): List of labels to consider.
    model_name (str): The name of the model to filter results for.
    threshold (float): Optional threshold value for all events.
    choice (str): 'per_event' for individual event threshold, 'all_events' for a common threshold.

    Returns:
    pd.DataFrame: A DataFrame with precision, recall, and F1 scores for each event.
    """
    event_results = []

    for label in labels_list:
        if choice == "per_event":
            # For 10% alarm rate per event
            event_results.append(
                process_results(results_df, test_df, [label], model_name, label)
            )
        elif choice == "all_events":
            # For 10% alarm rate for all events
            event_results.append(
                process_results(
                    results_df, test_df, [label], model_name, label, threshold=threshold
                )
            )
        else:
            raise ValueError("Choice must be 'per_event' or 'all_events'")

    # Initialize an empty list to hold each row of the table
    table_rows = []

    for i, label in enumerate(labels_list):
        pred_name = label + "_Pred"

        # Extract the relevant data from the results
        results = event_results[i]
        threshold = results[(model_name, label, "threshold")]
        labels = results[(model_name, label, label)][label]
        predictions = results[(model_name, label, label)][pred_name]

        # Calculate the precision, sensitivity (recall), and F1 score
        precision = precision_score(labels, predictions)
        sensitivity = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Append the results for this label to the table_rows list
        table_rows.append(
            {
                "Event": label,
                "Threshold": threshold,
                "Precision": precision,
                "Sensitivity": sensitivity,
                "F1": f1,
            }
        )

    # Create a DataFrame from the rows
    results_table = pd.DataFrame(table_rows)

    # Set the 'Event' column as the index
    results_table.set_index("Event", inplace=True)

    # Rename the index and columns according to label_mapping
    results_table.index = results_table.index.map(label_mapping)
    results_table = results_table.reindex(legend_order)

    return results_table


def analyze_model_performance(results_df, labels_list, model_name):
    """
    Analyzes model performance for multiple labels and computes various metrics.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing the model results.
    labels_list (list of str): List of labels to analyze.
    model_name (str): The name of the model to filter results for.

    Returns:
    dict: A dictionary containing ROC curves, AUC scores, PR curves, calibration curves, and net benefit DataFrames.
    """
    model_results = results_df[results_df["model"] == model_name]
    roc_curves = []
    pr_curves = []
    calibration_curves = []
    net_benefit_df = []
    val_auc = []
    val_aup = []

    for label in labels_list:
        # Filter for the current label
        label_results = model_results[model_results["label"] == label]

        model_preds = label_results["all_preds"].iloc[0]
        model_labels = label_results["all_labels"].iloc[0]

        # ROC Curve and AUC
        fpr, tpr, roc_thresh = roc_curve(model_labels, model_preds)
        roc_curves.append((fpr, tpr))
        val_auc.append(roc_auc_score(model_labels, model_preds))

        # Precision-Recall Curve and AUPRC
        precision, recall, pr_thresh = precision_recall_curve(model_labels, model_preds)
        pr_curves.append((precision, recall))
        val_aup.append(average_precision_score(model_labels, model_preds))

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(
            model_labels, model_preds, n_bins=10, strategy="quantile"
        )
        calibration_curves.append((prob_true, prob_pred))

        # Net Benefit Calculation
        mask = roc_thresh > 0.999
        sensitivity, specificity = tpr[~mask], 1 - fpr[~mask]
        prevalence = model_labels.mean()
        odds = roc_thresh[~mask] / (1 - roc_thresh[~mask])
        net_benefit = (
            sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * odds
        )
        treat_all = prevalence - (1 - prevalence) * odds
        net_benefit_df.append(
            pd.DataFrame(
                {
                    "Threshold": roc_thresh[~mask][1:],
                    "System": net_benefit[1:],
                    "All": treat_all[1:],
                }
            )
        )

    # Return all the metrics in a dictionary
    return {
        "roc_curves": roc_curves,
        "val_auc": val_auc,
        "pr_curves": pr_curves,
        "val_aup": val_aup,
        "calibration_curves": calibration_curves,
        "net_benefit_df": net_benefit_df,
    }


def calculate_metrics(row):
    preds = row["all_preds"]
    labels = row["all_labels"]
    # print current row model_name and label
    # Calculate AUROC and AUPRC, and skip if label == 'All_Targets'
    if row["label"] == "All_Targets":
        return pd.Series([np.nan, np.nan], index=["auroc", "auprc"])
    else:
        auroc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)

    return pd.Series([auroc, auprc], index=["auroc", "auprc"])


def calculate_event_metrics(
    results_df,
    test_df,
    labels_list,
    model_name,
    label_mapping,
    legend_order,
    threshold=None,
    choice="per_event",
    target_alarm_rate=0.1,
):
    """
    Calculate metrics for a 10% alarm rate either per event or for all events.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing the results.
    test_df (pd.DataFrame): DataFrame containing the test data.
    labels_list (list): List of labels to consider.
    model_name (str): The name of the model to filter results for.
    threshold (float): Optional threshold value for all events.
    choice (str): 'per_event' for individual event threshold, 'all_events' for a common threshold.

    Returns:
    pd.DataFrame: A DataFrame with precision, recall, and F1 scores for each event.
    """
    event_results = []

    for label in labels_list:
        if choice == "per_event":
            # For 10% alarm rate per event
            event_results.append(
                process_results(
                    results_df,
                    test_df,
                    [label],
                    model_name,
                    label,
                    target_alarm_rate=target_alarm_rate,
                )
            )
        elif choice == "all_events":
            # For 10% alarm rate for all events
            event_results.append(
                process_results(
                    results_df, test_df, [label], model_name, label, threshold=threshold
                )
            )
        else:
            raise ValueError("Choice must be 'per_event' or 'all_events'")

    # Initialize an empty list to hold each row of the table
    table_rows = []

    for i, label in enumerate(labels_list):
        pred_name = label + "_Pred"

        # Extract the relevant data from the results
        results = event_results[i]
        threshold = results[(model_name, label, "threshold")]
        labels = results[(model_name, label, label)][label]
        predictions = results[(model_name, label, label)][pred_name]

        # Calculate the precision, sensitivity (recall), and F1 score
        precision = precision_score(labels, predictions)
        sensitivity = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Append the results for this label to the table_rows list
        table_rows.append(
            {
                "Event": label,
                "Threshold": threshold,
                "Precision": precision,
                "Sensitivity": sensitivity,
                "F1": f1,
            }
        )

    # Create a DataFrame from the rows
    results_table = pd.DataFrame(table_rows)

    # Set the 'Event' column as the index
    results_table.set_index("Event", inplace=True)

    # Rename the index and columns according to label_mapping
    results_table.index = results_table.index.map(label_mapping)
    results_table = results_table.reindex(legend_order)

    return results_table


def process_results(
    results_df,
    test_df,
    labels_list,
    model_name,
    target_label,
    threshold=None,
    palliative=False,
    target_alarm_rate=0.1,
):
    """
    Process the prediction results from a specified model for each label in the labels_list,
    binarize the predictions based on a threshold, and return a list of DataFrames containing
    the results for each label.
    """
    final_df_dict = {}
    # Filter once for the specified model
    model_results = results_df[results_df["model"] == model_name]

    for label in labels_list:
        # Filter for the current label
        label_results = model_results[model_results["label"] == label]

        if label_results.empty:
            print(f"No results found for model {model_name} and label {label}.")
            continue

        # Extract predictions and labels
        model_preds = label_results["all_preds"].iloc[0]
        model_labels = label_results["all_labels"].iloc[0]

        # Find the threshold for binary classification if not provided
        label_threshold = (
            threshold
            if threshold is not None
            else find_threshold_for_alarm_rate(
                model_labels,
                model_preds,
                target_alarm_rate=target_alarm_rate,
                search_steps=1000,
            )
        )
        # print(f"Threshold for {label}: {label_threshold}")

        # Binarize the predictions based on the threshold
        binary_preds = np.where(model_preds >= label_threshold, 1, 0)

        # Create a DataFrame from the predictions and labels
        preds_df = pd.DataFrame({"Label": model_labels, f"{label}_Pred": binary_preds})

        aligned_df = test_df[test_df[label].isin([0, 1])].reset_index(drop=True)
        # Align and join DataFrames
        aligned_df = aligned_df.join(preds_df)

        # Filter based on the target label
        aligned_df = aligned_df[aligned_df[target_label].isin([0, 1])]

        # Verify label consistency and keep necessary columns
        aligned_df[label] = aligned_df[label].astype(
            int
        )  # Convert columns to int for comparison
        aligned_df["Label"] = aligned_df["Label"].astype(int)
        if aligned_df[label].equals(aligned_df["Label"]):
            # iF PALIATIVE IS TRUE, FILTER FOR PALLIATIVE PATIENTS
            if palliative == True:
                aligned_df = aligned_df[aligned_df["PALLIATIVE"] == 1]
            aligned_df = aligned_df[
                ["PatientID", "Trt_Date", target_label, f"{label}_Pred"]
            ]

            # Add 'Time' column which enumerates the visits per patient
            aligned_df["Time"] = aligned_df.groupby("PatientID").cumcount() + 1

            # Add this DataFrame to the final_df_dict
            final_df_dict[(model_name, label, target_label)] = aligned_df
            # Store the threshold for the model
            final_df_dict[(model_name, label, "threshold")] = label_threshold
        else:
            print(f"Labels do NOT match for {label}.")

    return final_df_dict
