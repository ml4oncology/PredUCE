# Standard library imports
import os
from functools import reduce

os.environ['OUTDATED_IGNORE'] = '1'
# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score)

import torch
from torch.utils.data import DataLoader
import shap
import pingouin as pg

from src.train import EHRDataset, pad_collate


def get_neg_to_pos_ratios(train_df, labels_list):
    """
    Calculates the number of positive and negative examples for each label,
    their ratio, and the event rate.

    Parameters:
    - train_df (pd.DataFrame): The DataFrame containing the training data.
    - labels_list (list): List of labels to calculate the ratios for.

    Returns:
    - dict: A dictionary with the number of positive and negative examples,
            their ratio, and the event rate for each label.
    """
    neg_to_pos_ratios_dict = {}

    for label in labels_list:
        # Check for existence of both positive and negative labels to avoid KeyError
        pos_count = train_df[label].value_counts().get(1, 0)
        neg_count = train_df[label].value_counts().get(0, 0)

        # Calculate the ratio if positive count is not zero to avoid ZeroDivisionError
        ratio = neg_count / pos_count if pos_count != 0 else float('inf')

        # Event rate is the ratio of positive count to the total count
        event_rate = pos_count / (pos_count + neg_count)

        # Store calculations in the dictionary
        neg_to_pos_ratios_dict[label] = {
            'positive_count': pos_count,
            'negative_count': neg_count,
            'ratio': ratio,
            'event_rate': event_rate
        }

        # Optional: Print each label's stats for verification/debugging
        print(
            f'{label} has {pos_count} positive examples, {neg_count} negative examples, '
            f'ratio of {ratio:.2f} negative to positive examples. '
            f'The event rate is {event_rate:.4f}.'
        )

    return neg_to_pos_ratios_dict


def plot_repeated_corr_matrix(df, labels_list, label_mapping, legend_order, subsample_size=None):
    # Rename labels in the DataFrame according to label_mapping
    df = df.rename(columns=label_mapping)

    # Adjust labels_list according to label_mapping
    labels_list = [label_mapping[label] for label in labels_list]

    # Compute correlations
    correlations = {}
    for i, target1 in enumerate(labels_list):
        for target2 in labels_list[i+1:]:
            # Filter and clean data
            corr_df = df[(df[target1] != -1) & (df[target2] != -1)]
            # corr_df = corr_df[np.isfinite(corr_df[target1]) & np.isfinite(corr_df[target2])]
            corr_df = corr_df[[target1, target2, 'PatientID']].dropna()

            # Subsample unique patients if necessary
            if subsample_size and corr_df['PatientID'].nunique() > subsample_size:
                np.random.seed(42)
                unique_patients = corr_df['PatientID'].unique()
                sampled_patients = np.random.choice(unique_patients, size=subsample_size, replace=False)
                corr_df = corr_df[corr_df['PatientID'].isin(sampled_patients)]

            if len(corr_df) >= 3:
                try:
                    result = pg.rm_corr(data=corr_df, x=target1, y=target2, subject='PatientID')
                    correlations[(target1, target2)] = result['r'].values[0]
                    print(f"Correlation between {target1} and {target2}: {result['r'].values[0]:.4f}")
                except Exception as e:
                    correlations[(target1, target2)] = np.nan
                    # print fail to compute correlation
                    print(f"Failed to compute correlation between {target1} and {target2}: {e}")

    # Create and reorder correlation matrix
    corr_matrix = pd.DataFrame(np.nan, index=labels_list, columns=labels_list)
    for (target1, target2), corr_value in correlations.items():
        corr_matrix.loc[target1, target2] = corr_value
        corr_matrix.loc[target2, target1] = corr_value

    # Reorder matrix to match legend_order
    corr_matrix = corr_matrix.reindex(index=legend_order, columns=legend_order)

    # Assuming corr_matrix is already defined
    trimmed_corr_matrix = corr_matrix.iloc[1:, :-1]

    # Initialize a mask with all False values
    mask = np.zeros_like(trimmed_corr_matrix, dtype=bool)

    # Iterate through each row and set True for the elements to the right of NaN
    for i, row in enumerate(trimmed_corr_matrix.values):
        nan_indices = np.where(np.isnan(row))[0]
        if nan_indices.size > 0:
            first_nan_index = nan_indices[0]
            mask[i, first_nan_index:] = True

    # Create the plot with adjusted dimensions
    plt.figure(figsize=(10, 8), facecolor='white')
    sns.heatmap(trimmed_corr_matrix, annot=True, fmt=".2f", cmap='Oranges', vmin=0.13, vmax=0.52, mask=mask,
                square=True)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    return corr_matrix

def calculate_event_metrics(results_df, test_df, labels_list, model_name, label_mapping, legend_order, threshold=None, choice='per_event'):
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
        if choice == 'per_event':
            # For 10% alarm rate per event
            event_results.append(process_results(results_df, test_df, [label], model_name, label))
        elif choice == 'all_events':
            # For 10% alarm rate for all events
            event_results.append(process_results(results_df, test_df, [label], model_name, label, threshold=threshold))
        else:
            raise ValueError("Choice must be 'per_event' or 'all_events'")

    # Initialize an empty list to hold each row of the table
    table_rows = []

    for i, label in enumerate(labels_list):
        pred_name = label + '_Pred'

        # Extract the relevant data from the results
        results = event_results[i]
        threshold = results[(model_name, label, 'threshold')]
        labels = results[(model_name, label, label)][label]
        predictions = results[(model_name, label, label)][pred_name]

        # Calculate the precision, sensitivity (recall), and F1 score
        precision = precision_score(labels, predictions)
        sensitivity = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Append the results for this label to the table_rows list
        table_rows.append({
            'Event': label,
            'Threshold': threshold,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'F1': f1
        })

    # Create a DataFrame from the rows
    results_table = pd.DataFrame(table_rows)

    # Set the 'Event' column as the index
    results_table.set_index('Event', inplace=True)

    # Rename the index and columns according to label_mapping
    results_table.index = results_table.index.map(label_mapping)
    results_table = results_table.reindex(legend_order)

    return results_table


def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5 * height, width * 0.8, 0, length_includes_head=True, head_width=0.75 * height, color=orig_handle.get_facecolor())
    return p


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
    model_results = results_df[results_df['model'] == model_name]
    roc_curves = []
    pr_curves = []
    calibration_curves = []
    net_benefit_df = []
    val_auc = []
    val_aup = []

    for label in labels_list:
        # Filter for the current label
        label_results = model_results[model_results['label'] == label]

        model_preds = label_results['all_preds'].iloc[0]
        model_labels = label_results['all_labels'].iloc[0]

        # ROC Curve and AUC
        fpr, tpr, roc_thresh = roc_curve(model_labels, model_preds)
        roc_curves.append((fpr, tpr))
        val_auc.append(roc_auc_score(model_labels, model_preds))

        # Precision-Recall Curve and AUPRC
        precision, recall, pr_thresh = precision_recall_curve(model_labels, model_preds)
        pr_curves.append((precision, recall))
        val_aup.append(average_precision_score(model_labels, model_preds))

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(model_labels, model_preds, n_bins=10, strategy='quantile')
        calibration_curves.append((prob_true, prob_pred))

        # Net Benefit Calculation
        mask = roc_thresh > 0.999
        sensitivity, specificity = tpr[~mask], 1 - fpr[~mask]
        prevalence = model_labels.mean()
        odds = roc_thresh[~mask] / (1 - roc_thresh[~mask])
        net_benefit = sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * odds
        treat_all = prevalence - (1 - prevalence) * odds
        net_benefit_df.append(pd.DataFrame({
            'Threshold': roc_thresh[~mask][1:],
            'System': net_benefit[1:],
            'All': treat_all[1:]
        }))

    # Return all the metrics in a dictionary
    return {
        'roc_curves': roc_curves,
        'val_auc': val_auc,
        'pr_curves': pr_curves,
        'val_aup': val_aup,
        'calibration_curves': calibration_curves,
        'net_benefit_df': net_benefit_df
    }


def plot_performance_curves(metrics, labels_list, label_mapping, legend_order, colors):
    """
    Plots ROC and PR curves for the given metrics with converted label names and ordered legend.

    Parameters:
    metrics (dict): A dictionary containing the metrics including ROC and PR curves.
    labels_list (list of str): List of labels corresponding to the metrics.
    label_mapping (dict): A dictionary for converting label names.
    legend_order (list): An ordered list of labels for the legend.
    colors (list): A list of colors for each label in the legend order.
    """
    # Reorder labels_list according to legend_order
    ordered_labels = [label for name in legend_order for label in labels_list if label_mapping[label] == name]

    # Plot ROC curves
    plt.figure(figsize=(6, 6), facecolor='white')
    for i, label in enumerate(ordered_labels):
        fpr, tpr = metrics['roc_curves'][labels_list.index(label)]
        plt.plot(fpr, tpr, color=colors[i], label=f'{label_mapping[label]} (AUC = {metrics["val_auc"][labels_list.index(label)]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curves')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_facecolor('white')
    plt.grid(False)
    plt.show()

    # Plot PR curves
    plt.figure(figsize=(6, 6), facecolor='white')
    for i, label in enumerate(ordered_labels):
        precision, recall = metrics['pr_curves'][labels_list.index(label)]
        plt.plot(recall, precision, color=colors[i], label=f'{label_mapping[label]} (AUP = {metrics["val_aup"][labels_list.index(label)]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall (PR) Curves')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='both', length=0)
    plt.grid(False)
    plt.show()


def calculate_mean_cal_error(prob_true, prob_pred, n_bins=10):
    """ Compute mean calibration error (MCE) """
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
    """ Compute maximum calibration error (MaxCE) """
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


def plot_calibration_curves(calibration_curves, labels_list, label_mapping, legend_order, colors):
    plt.figure(figsize=(6, 6), facecolor='white')

    # Reorder labels_list according to legend_order
    ordered_labels = [label for name in legend_order for label in labels_list if label_mapping[label] == name]

    for i, label in enumerate(ordered_labels):
        prob_true, prob_pred = calibration_curves[labels_list.index(label)]
        mce = calculate_mean_cal_error(prob_true, prob_pred)
        max_ce = calculate_max_cal_error(prob_true, prob_pred)

        if i == 0:
            # Full description for the first entry
            legend_label = f'{label_mapping[label]} | MCE: {mce:.2f}, MaxCE: {max_ce:.2f}'
        else:
            # Just numbers for subsequent entries
            legend_label = f'{label_mapping[label]} | {mce:.2f}/{max_ce:.2f}'

        plt.plot(prob_pred, prob_true, color=colors[i], label=legend_label)

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")  # Adjusted range for calibration line

    plt.ylabel("Observed Rate")
    plt.ylim([0.0, 0.75])  # Adjusted y-axis limit
    plt.xlim([0.0, 0.75])  # Adjusted x-axis limit
    plt.xlabel("Predicted Risk")

    leg = plt.legend(loc="upper left", fontsize=9, frameon=False)
    leg.get_frame().set_facecolor('white')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='both', length=0)

    plt.grid(False)
    plt.show()


def plot_ecdf(merge_df, label_mapping, legend_order, colors):
    """
    Plots the Empirical Cumulative Distribution Function (ECDF) for prediction columns in merge_df,
    using provided label mapping, legend order, and colors.

    Parameters:
    merge_df (pd.DataFrame): DataFrame containing merged predictions.
    label_mapping (dict): A dictionary for converting label names with '_prediction' suffix.
    legend_order (list): An ordered list of labels for the legend.
    colors (list): A list of colors for each label in the legend order.
    """
    # Update label_mapping to include '_prediction' suffix
    label_mapping_with_suffix = {key + '_prediction': value for key, value in label_mapping.items()}

    # Identify columns that end with '_prediction'
    prediction_columns = [col for col in merge_df.columns if col.endswith('_prediction')]

    # Reorder labels_list according to legend_order
    ordered_predictions = [label for name in legend_order for label in prediction_columns if label_mapping_with_suffix.get(label, label) == name]

    plt.figure(figsize=(6, 6), facecolor='white')
    ax = plt.gca()
    # Create an empty figure
    # fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ECDF for each prediction column
    for i, column in enumerate(ordered_predictions):
        # Sort data
        x = np.sort(merge_df[column])
        # Calculate ECDF values
        y = np.arange(1, len(x) + 1) / len(x)
        # Get the label from mapping
        label = label_mapping_with_suffix.get(column, column)
        ax.plot(x, y, linestyle='-', color=colors[i], label=label)

    # Customize the plot
    ax.set_xlabel('Prediction Value')
    ax.set_ylabel('cumulative proportion')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)

    ax.set_facecolor('white')
    ax.grid(False)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # rescale the x-axis to 0.0 to 0.4
    ax.set_xlim([0.0, 1.0])

    # Show the plot
    # plt.tight_layout()
    plt.show()


def find_largest_threshold_interval(df):
    sorted_df = df.sort_values(by='Threshold')
    interval_list = []

    i = 0
    while i < len(sorted_df):
        if sorted_df['System'].iloc[i] > sorted_df['All'].iloc[i] and sorted_df['System'].iloc[i] >= 0:
            start = sorted_df['Threshold'].iloc[i]
            end = start

            j = i + 1
            while j < len(sorted_df) and sorted_df['System'].iloc[j] > sorted_df['All'].iloc[j] and sorted_df['System'].iloc[j] > 0:
                end = sorted_df['Threshold'].iloc[j]
                j += 1

            interval_list.append((start, end))
            i = j
        else:
            i += 1

    if not interval_list:
        return None

    # Find the largest interval by comparing their lengths
    largest_interval = max(interval_list, key=lambda x: x[1]-x[0])
    return largest_interval

def plot_net_benefit_curves(net_benefit_df_list, labels_list, label_mapping, legend_order, colors):
    ordered_labels = [label for name in legend_order for label in labels_list if label_mapping.get(label) == name]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()
    all_target_intervals = {}


    for i, label in enumerate(ordered_labels):
        df = net_benefit_df_list[labels_list.index(label)]
        largest_interval = find_largest_threshold_interval(df)

        mapped_label = label_mapping.get(label)
        ax = axes[i]

        y_max = df[['System', 'All']].max().max()
        thresh = df['Threshold'].values

        ax.plot(thresh, df['System'], label=mapped_label, color=colors[i])
        ax.plot(thresh, df['All'], label='All', color='black')
        ax.plot(thresh, np.zeros(thresh.shape), label='None', color='black', linestyle='--')

        # if largest_interval is not None:
        #     print(f"Optimal interval for {mapped_label}: {largest_interval}")
        all_target_intervals.update({mapped_label: largest_interval})

            # ax.axvspan(largest_interval[0], largest_interval[1], color=colors[i], alpha=0.3, label='Optimal Interval')

        ax.set(xlabel='Threshold Probability', ylabel='Net Benefit', xlim=(thresh.min(), thresh.max()), ylim=(-y_max/4, y_max*1.1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('white')
        ax.grid(False)

    # find the widest and narrowest interval in all_target_intervals and its corresponding labels
    widest_interval = max(all_target_intervals.items(), key=lambda x: x[1][1] - x[1][0])
    narrowest_interval = min(all_target_intervals.items(), key=lambda x: x[1][1] - x[1][0])
    print(f"Widest interval: {widest_interval[0]} - {widest_interval[1]}")
    print(f"Narrowest interval: {narrowest_interval[0]} - {narrowest_interval[1]}")

    plt.tight_layout()
    plt.show()

def partial_predict(X,label,model_name,train_results,df,device):
    torch.cuda.empty_cache()
    # X is a single treatment session, duplicated multiple times with different feature permutations
    # Reformat to sequential data
    X = X.copy()
    N = len(X)
    # get the patient id and visit date for this sample row
    res = {}
    for col in ['PatientID', 'Trt_Date']:
        mask = X[col] != -1
        val = X.loc[mask, col].unique()
        assert len(val) == 1
        res[col] = val[0]
    # get patient's historical data
    # NOTE: historical data is NOT permuted
    hist = df.query(f'PatientID == {res["PatientID"]} & Trt_Date < {res["Trt_Date"]}').copy()
    n = len(hist)
    hist = pd.concat([hist] * N) # repeat patient historical data for each duplicated sample row
    # set up new patient id for each sample row
    ikns = np.arange(0, N, 1) + res['PatientID']
    hist['PatientID'] = np.repeat(ikns, n)
    X['PatientID'] = ikns
    # combine historical data and sample rows togethers
    X['Trt_Date'] = res['Trt_Date']
    X = pd.concat([X, hist]).sort_values(by=['PatientID', 'Trt_Date'], ignore_index=True)

    # Get GRU predictions
    dataset = EHRDataset(X,[label])
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,collate_fn=pad_collate)
    model_input = next(iter(loader))[0]
    # print(model_input.shape)
    torch.cuda.empty_cache()
    # change different model
    model = train_results[model_name,label]['best_model']
    model.half() # use fp16 instead of fp32
    pred = model(model_input.to(device).half())
    torch.cuda.empty_cache()
    # pred = gru_pain_model(model_input)
    # print(pred.shape)
    last_session_pred = pred[:, -1, :]
    last_session_pred = torch.sigmoid(last_session_pred)
    return last_session_pred.cpu().detach().numpy()


def calculate_metrics(row):
    preds = row['all_preds']
    labels = row['all_labels']
    # print current row model_name and label
    # print(f"model_name: {row['model']}, label: {row['label']}")
    # Calculate AUROC and AUPRC, and skip if label == 'All_Targets'
    if row['label'] == 'All_Targets':
        return pd.Series([np.nan, np.nan], index=['auroc', 'auprc'])
    else:
        auroc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)

    return pd.Series([auroc, auprc], index=['auroc', 'auprc'])



def generate_shap_plots(shap_dict, labels_list, save_path):
    """
    Generates and saves SHAP layered violin and bar plots for each label.

    Parameters:
    - shap_dict (dict): Dictionary containing SHAP values and explainers for each label.
    - labels_list (list): List of labels for which to generate plots.
    - save_path (str): Path to the directory where plots will be saved.
    """
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    for label in labels_list:
        shap_explainer = shap_dict[label]['shap_values']
        shap_values = shap_explainer.values
        data = shap_dict[label]['data']
        feature_names = data.columns

        # Calculate mean absolute SHAP values and select top 10 features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(-mean_abs_shap)[:10]
        top_shap_values = shap_values[:, top_indices]
        top_features = feature_names[top_indices]

        # Determine the range for the x-axis
        min_shap_value = top_shap_values.min()
        max_shap_value = top_shap_values.max()

        # Layered Violin Plot
        shap.summary_plot(top_shap_values, data[top_features], plot_type="layered_violin", show=False)
        plt.xlim([min_shap_value, max_shap_value])  # Set dynamic x-axis range
        plt.savefig(os.path.join(save_path, f"{label}_layered_violin.png"))
        plt.close()

        # Bar Plot
        filtered_explainer = shap.Explanation(values=top_shap_values,
                                              base_values=shap_explainer.base_values,
                                              data=data[top_features].values,
                                              feature_names=top_features)
        shap.plots.bar(filtered_explainer, max_display=10, show=False)
        plt.gcf().subplots_adjust(left=0.7)
        plt.gca().set_xticklabels(['{:.3f}'.format(x) for x in plt.gca().get_xticks()], rotation=45)
        plt.savefig(os.path.join(save_path, f"{label}_bar.png"))
        plt.close()


def plot_mean_shap_bar(mean_shap_df, colors, top_n=10,sort_by='mean'):
    """
    Plots a horizontal bar chart for SHAP values, displaying only the top N features.

    Args:
    - mean_shap_df (pd.DataFrame): DataFrame with mean SHAP values for each feature and label.
    - colors (list): List of colors for each label.
    - top_n (int): Number of top features to display.
    """
    # Sort the DataFrame based on the mean SHAP values and select top N features
    sorted_df = mean_shap_df.sort_values(by=sort_by, axis=1, ascending=False).iloc[:, :top_n]

    # Set up the plot
    plt.figure(figsize=(10, top_n * 0.5))  # Adjust the figure size based on the number of features
    ax = plt.gca()

    # Plot the mean SHAP values as horizontal bars
    ax.barh(sorted_df.columns, sorted_df.loc[sort_by], color='grey', alpha=0.4)

    # Plot individual SHAP values for each label as dots
    for i, label in enumerate(sorted_df.index[:-1]):  # Exclude the last row ('mean')
        ax.scatter(sorted_df.loc[label], sorted_df.columns, color=colors[i % len(colors)], label=label)

    # Set labels and title
    ax.set_xlabel('mean(|SHAP Value|)')

    # Format x-axis tick labels to 5 decimal places
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))

    ax.invert_yaxis()  # Invert y-axis to show highest values on top
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # move lengend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,frameon=False)

    # plt.legend(frameon=False)

    # Show the plot
    plt.show()


def plot_mean_abs_shap_values(mean_shap_df, temporal_cohort_sheet):
    # Transpose mean_shap_df and merge with temporal_cohort_sheet
    transposed_df = pd.merge(mean_shap_df.T.abs(), temporal_cohort_sheet, left_index=True, right_on='Features (converted)')

    # Aggregate mean absolute SHAP values by group
    grouped_shap_means_abs = transposed_df.groupby('Group')['mean'].sum()


    # Sort the aggregated values for better visualization
    grouped_shap_means_abs_sorted = grouped_shap_means_abs.sort_values(ascending=True)

    # Plotting
    plt.figure(figsize=(10, len(grouped_shap_means_abs_sorted) * 0.5))  # Adjust the figure size
    grouped_shap_means_abs_sorted.plot(kind='barh',color='grey', alpha=0.4)
    plt.xlabel('mean(|SHAP Value|)')
    plt.ylabel('')  # Set ylabel to empty string

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def create_forest_plot(df, colors):
    # Plot settings
    fig, ax = plt.subplots(figsize=(14, 8))  # Adjusted for better fit

    # Removing frame borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot the Odds Ratios using the index for y-axis labels
    ax.errorbar(df['OddsRatio'], range(len(df)), xerr=[df['OddsRatio'] - df['CI.Lower'], df['CI.Upper'] - df['OddsRatio']],
                fmt='o', color='black', ecolor=colors, elinewidth=3, capsize=0)

    # Add a line for Odds Ratio of 1
    ax.axvline(x=1, linestyle='--', color='grey', linewidth=2.0)

    # Hide y-axis line and ticks
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])

    # Add X-axis title
    ax.set_xlabel('Odds Ratio for Emergency Department Visits', fontsize=16, labelpad=20)

    # Set the x-axis tick labels size
    ax.tick_params(axis='x', labelsize=14)  # Increase fontsize for x-axis tick labels

    # Hide the axes frame
    ax.frame_on = False

    # Calculate positions for equal spacing
    symptoms_position = -0.50
    odds_ratio_position = symptoms_position + 0.15  # Adjust spacing
    p_value_position = odds_ratio_position + 0.26  # Adjust spacing to ensure equal length

    # Add text for each symptom, odds ratio, CI, and p-value
    for i, (symptom, row) in enumerate(df.iterrows()):
        ax.text(symptoms_position, i, symptom, ha='left', va='center', transform=ax.get_yaxis_transform(), fontsize=16)
        ax.text(odds_ratio_position, i, f"{row['OddsRatio']:.2f} ({row['CI.Lower']:.2f}-{row['CI.Upper']:.2f})", ha='left', va='center', transform=ax.get_yaxis_transform(), fontsize=16)
        ax.text(p_value_position, i, f"{row['Pvalue_JAMA']}", ha='left', va='center', transform=ax.get_yaxis_transform(), fontsize=16)

    # Headers
    headers = ["Symptoms", "Odds Ratio(95% CI)", "P Value"]
    header_positions = [symptoms_position, odds_ratio_position, p_value_position]
    header_y_position = len(df) - 9.8  # Incremented by 0.5 to ensure it's above all text entries
    for header, pos in zip(headers, header_positions):
        ax.text(pos, header_y_position, header, ha='left', va='bottom', fontsize=16, transform=ax.get_yaxis_transform(), weight='bold')

    plt.show()


def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5 * height, width * 0.8, 0, length_includes_head=True, head_width=0.75 * height, color=orig_handle.get_facecolor())
    return p

def prepare_event_rate_dfs(test_df, results_df, labels_list, model_type, label_mapping, legend_order, alarm_rate=0.1):
    # Calculate the event rates for test_df
    event_rates_ordered = {}
    for label, readable_label in label_mapping.items():
        pos_count = test_df[label].value_counts().get(1, 0)
        neg_count = test_df[label].value_counts().get(0, 0)
        total_count = pos_count + neg_count
        event_rate = pos_count / total_count if total_count > 0 else 0
        event_rates_ordered[readable_label] = event_rate

    # Order the event rates according to legend_order
    ordered_event_rates = [event_rates_ordered[label] for label in legend_order]

    # Prepare metrics dataframes
    merge_df, threshold = get_merge_df(results_df, test_df, labels_list, model_type, alarm_rate=alarm_rate)
    table_per_event = calculate_event_metrics(results_df, test_df, labels_list, model_type, label_mapping, legend_order, choice='per_event',target_alarm_rate=alarm_rate)
    table_all_events = calculate_event_metrics(results_df, test_df, labels_list, model_type, label_mapping, legend_order, threshold=threshold, choice='all_events')

    # Prepare DataFrames for 10% alarm rate scenarios
    ten_percent_per_event_df = pd.DataFrame({
        'Event': legend_order,
        'Threshold': table_per_event['Threshold'].to_list(),
        'Precision': table_per_event['Precision'].to_list(),
        'Sensitivity': table_per_event['Sensitivity'].to_list(),
        'F1': table_per_event['F1'].to_list(),
        'Event Rate': ordered_event_rates
    })

    # calculate alarm rate for each target under 10% all event
    # Get all prediction columns, convert to binary, and compute alarm rates
    binary_columns = [(col, (merge_df[col] >= threshold).astype(int)) for col in merge_df.columns if
                      col.endswith('_prediction')]
    alarm_rates = {col: binary.sum() / len(merge_df) for col, binary in binary_columns}

    # Map the prediction column names to more readable labels and reorder according to legend_order
    label_mapping_reversed = {v: k for k, v in label_mapping.items()}  # Reverse the label_mapping for easier access
    ordered_alarm_rate_10_all_event = {label: alarm_rates[label_mapping_reversed[label] + '_prediction'] for label in legend_order
                           if label_mapping_reversed[label] + '_prediction' in alarm_rates}
    event_rate_10_all_event_list = list(ordered_alarm_rate_10_all_event.values())

    ten_percent_all_event_df = pd.DataFrame({
        'Event': legend_order,
        'Threshold': table_all_events['Threshold'].to_list(),
        'Alarm Rate': event_rate_10_all_event_list,
        'Precision': table_all_events['Precision'].to_list(),
        'Sensitivity': table_all_events['Sensitivity'].to_list(),
        'F1': table_all_events['F1'].to_list(),
    })

    # Calculate start values for drawing arrows
    start_values = [(ten_percent_per_event_df.loc[ten_percent_per_event_df['Event'] == row['Event'], 'Sensitivity'].values[0],
                     ten_percent_per_event_df.loc[ten_percent_per_event_df['Event'] == row['Event'], 'Precision'].values[0])
                    for _, row in ten_percent_all_event_df.iterrows()]

    return merge_df, ten_percent_per_event_df, ten_percent_all_event_df, start_values


def get_merge_df(results_df, test_df, labels_list, model_name,alarm_rate=0.1):
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

    model_results = results_df[results_df['model'] == model_name]
    predict = []

    # Loop over all labels in labels_list
    for label in labels_list:
        # Filter the DataFrame to only include rows where the label is not -1
        filtered_df = test_df[test_df[label] != -1].copy()

        label_results = model_results[model_results['label'] == label]
        # prediction = label_results['all_preds'].iloc[0][0]
        prediction = label_results['all_preds'].iloc[0]

        pred_name = label + '_prediction'
        # Add the prediction column to the DataFrame
        filtered_df[pred_name] = prediction
        predict.append(filtered_df)

    # Get column names excluding the last one from the first dataframe
    merge_cols = predict[0].columns.tolist()[:-1]
    # Perform the merge operation
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=merge_cols, how='inner'), predict)

    # Create a new column 'any_symptoms' that is 1 if any of the labels is 1, and 0 otherwise
    merged_df['any_symptoms'] = merged_df[labels_list].apply(lambda row: int(any(row)), axis=1)
    y_labels = merged_df['any_symptoms'].values
    prediction_columns = [col for col in merged_df.columns if col.endswith('prediction')]
    y_preds = merged_df[prediction_columns].values

    threshold = find_threshold_for_alarm_rate(y_labels, y_preds, target_alarm_rate=alarm_rate)
    print(f"The threshold that achieves a {alarm_rate*100}% alarm rate is {threshold}")

    return merged_df,threshold


def calculate_event_metrics(results_df, test_df, labels_list, model_name, label_mapping, legend_order, threshold=None, choice='per_event',target_alarm_rate = 0.1):
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
        if choice == 'per_event':
            # For 10% alarm rate per event
            event_results.append(process_results(results_df, test_df, [label], model_name, label,target_alarm_rate=target_alarm_rate))
        elif choice == 'all_events':
            # For 10% alarm rate for all events
            event_results.append(process_results(results_df, test_df, [label], model_name, label,threshold=threshold))
        else:
            raise ValueError("Choice must be 'per_event' or 'all_events'")

    # Initialize an empty list to hold each row of the table
    table_rows = []

    for i, label in enumerate(labels_list):
        pred_name = label + '_Pred'

        # Extract the relevant data from the results
        results = event_results[i]
        threshold = results[(model_name, label, 'threshold')]
        labels = results[(model_name, label, label)][label]
        predictions = results[(model_name, label, label)][pred_name]

        # Calculate the precision, sensitivity (recall), and F1 score
        precision = precision_score(labels, predictions)
        sensitivity = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        # Append the results for this label to the table_rows list
        table_rows.append({
            'Event': label,
            'Threshold': threshold,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'F1': f1
        })

    # Create a DataFrame from the rows
    results_table = pd.DataFrame(table_rows)

    # Set the 'Event' column as the index
    results_table.set_index('Event', inplace=True)

    # Rename the index and columns according to label_mapping
    results_table.index = results_table.index.map(label_mapping)
    results_table = results_table.reindex(legend_order)

    return results_table

def process_results(results_df, test_df, labels_list, model_name, target_label, threshold=None,palliative = False,target_alarm_rate = 0.1):
    """
    Process the prediction results from a specified model for each label in the labels_list,
    binarize the predictions based on a threshold, and return a list of DataFrames containing
    the results for each label.
    """
    final_df_dict = {}
    # Filter once for the specified model
    model_results = results_df[results_df['model'] == model_name]

    for label in labels_list:
        # Filter for the current label
        label_results = model_results[model_results['label'] == label]

        if label_results.empty:
            print(f"No results found for model {model_name} and label {label}.")
            continue

        # Extract predictions and labels
        model_preds = label_results['all_preds'].iloc[0]
        model_labels = label_results['all_labels'].iloc[0]

        # Find the threshold for binary classification if not provided
        label_threshold = threshold if threshold is not None else find_threshold_for_alarm_rate(model_labels, model_preds, target_alarm_rate=target_alarm_rate, search_steps=1000)
        # print(f"Threshold for {label}: {label_threshold}")

        # Binarize the predictions based on the threshold
        binary_preds = np.where(model_preds >= label_threshold, 1, 0)

        # Create a DataFrame from the predictions and labels
        preds_df = pd.DataFrame({'Label': model_labels, f'{label}_Pred': binary_preds})


        aligned_df = test_df[test_df[label].isin([0, 1])].reset_index(drop=True)
        # Align and join DataFrames
        aligned_df = aligned_df.join(preds_df)

        # Filter based on the target label
        aligned_df = aligned_df[aligned_df[target_label].isin([0, 1])]

        # Verify label consistency and keep necessary columns
        aligned_df[label] = aligned_df[label].astype(int)  # Convert columns to int for comparison
        aligned_df['Label'] = aligned_df['Label'].astype(int)
        if aligned_df[label].equals(aligned_df['Label']):

            # iF PALIATIVE IS TRUE, FILTER FOR PALLIATIVE PATIENTS
            if palliative == True:
                aligned_df = aligned_df[aligned_df['PALLIATIVE'] == 1]
            aligned_df = aligned_df[['PatientID','Trt_Date', target_label, f'{label}_Pred']]

            # Add 'Time' column which enumerates the visits per patient
            aligned_df['Time'] = aligned_df.groupby('PatientID').cumcount() + 1

            # Add this DataFrame to the final_df_dict
            final_df_dict[(model_name, label, target_label)] = aligned_df
            # Store the threshold for the model
            final_df_dict[(model_name, label, 'threshold')] = label_threshold
        else:
            print(f"Labels do NOT match for {label}.")

    return final_df_dict


def find_threshold_for_alarm_rate(y_labels, y_preds, target_alarm_rate=0.1, search_steps=3000):
    best_threshold = 0
    closest_alarm_rate_diff = float('inf')
    closest_alarm_rate = 0

    for threshold in np.linspace(0, 1, search_steps):
        y_pred = (y_preds >= threshold).astype(int) if y_preds.ndim == 1 else (y_preds >= threshold).any(axis=1).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_labels, y_pred).ravel()
        alarm_rate = (tp + fp) / len(y_labels)
        current_diff = abs(target_alarm_rate - alarm_rate)

        # Ensure TP is not zero and the current difference is smaller than any previously found
        if current_diff < closest_alarm_rate_diff and tp > 0:
            closest_alarm_rate_diff = current_diff
            best_threshold = threshold
            closest_alarm_rate = alarm_rate

    print(f"Best threshold: {best_threshold} with alarm rate: {closest_alarm_rate}, TP: {tp}")
    return best_threshold


def arrows_plot(dfs, colors, labels, start_values_list=None, invert_xaxis=False, show_text=True, ed_odds_ratio=None):
    if len(dfs) == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        axes = [ax]  # Make it iterable
    else:
        fig, axes = plt.subplots(1, len(dfs), figsize=(14, 6))

    for idx, df in enumerate(dfs):
        ax = axes[idx] if len(dfs) > 1 else axes[0]
        start_values = start_values_list[idx] if start_values_list else None

        for i, row in df.iterrows():
            start = start_values[i] if start_values else (1, row['Event Rate'])
            end = (row['Sensitivity'], row['Precision'])
            color = colors[i % len(colors)]

            if show_text:
                if row['Precision'] == 0:
                    ppv_text = "N/A"
                else:
                    ppv_fold_increase = row['Precision'] / start[1]
                    ppv_text = f"{ppv_fold_increase:.1f}"
                ax.text(end[0], end[1] + 0.00003, ppv_text, color='black', ha='left', va='bottom')

            arrow = mpatches.FancyArrowPatch(start, end, color=color, arrowstyle='-|>', mutation_scale=20, linewidth=2)
            ax.add_patch(arrow)

        ax.set_xlabel('Sensitivity')
        ax.set_ylabel('Precision')
        if invert_xaxis:
            ax.invert_xaxis()

        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.autoscale_view()
        ax.grid(False)

        legend_labels = []
        for label in labels:
            if ed_odds_ratio is not None and label in ed_odds_ratio.index:
                odds_ratio = ed_odds_ratio.loc[label, 'OddsRatio']
                legend_labels.append(f"{label} (OR: {odds_ratio:.2f})")
            else:
                legend_labels.append(label)

        legend_handles = [mpatches.FancyArrowPatch((0, 0), (0.1 * 0.8, 0), color=color, arrowstyle='-|>', mutation_scale=20) for color in colors]
        # set bbox_to_anchor=(0, 0.6) for per event and bbox_to_anchor=(0, 0.0) for all event
        ax.legend(legend_handles, legend_labels, handler_map={mpatches.FancyArrowPatch: HandlerPatch(patch_func=make_legend_arrow)}, loc='lower left', bbox_to_anchor=(0, 0.6), frameon=False, fontsize=9)

    #plt.tight_layout()
    plt.show()

def plot_radar_chart(df, models, start_from=0.5, end_at=1.0, scale=0.1, rotation_degrees=0, label_font_size=10, label_padding=0.01):
    categories = list(df.index)
    N = len(categories)

    # Calculate the angle for each axis and ensure the plot is circular
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Setup radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], [])

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(np.arange(start_from, end_at, scale), ["{:.1f}".format(x) for x in np.arange(start_from, end_at, scale)], color="grey", size=7)
    plt.ylim(start_from, end_at)

    # Plot data
    for key, value in models.items():
        values = df.loc[categories, key].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=value)
        ax.fill(angles, values, alpha=0.1)

    # Add model names as legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Adding category labels slightly outside the grid
    for i, label in enumerate(categories):
        angle_rad = angles[i]
        if angle_rad == 0:
            ha = 'center'
        elif 0 < angle_rad < np.pi:
            ha = 'left'
        elif angle_rad == np.pi:
            ha = 'center'
        else:
            ha = 'right'
        ax.text(angle_rad, end_at + label_padding, label, size=label_font_size, horizontalalignment=ha, verticalalignment="center", rotation_mode='anchor')

    plt.show()