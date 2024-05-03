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

import shap
import pingouin as pg
from src.util import calculate_mean_cal_error,calculate_max_cal_error, find_largest_threshold_interval


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
                    # print(f"Correlation between {target1} and {target2}: {result['r'].values[0]:.4f}")
                except Exception as e:
                    correlations[(target1, target2)] = np.nan
                    # print fail to compute correlation
                    # print(f"Failed to compute correlation between {target1} and {target2}: {e}")

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


def make_legend_arrow(
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize):
    p = mpatches.FancyArrow(0, 0.5 * height, width * 0.8, 0, length_includes_head=True, head_width=0.75 * height, color=orig_handle.get_facecolor())
    return p

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
    # print(f"Widest interval: {widest_interval[0]} - {widest_interval[1]}")
    # print(f"Narrowest interval: {narrowest_interval[0]} - {narrowest_interval[1]}")

    plt.tight_layout()
    plt.show()

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

def arrows_plot(
        dfs,
        colors,
        labels,
        start_values_list=None,
        invert_xaxis=False,
        show_text=True,
        ed_odds_ratio=None):

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

def plot_radar_chart(
        df,
        models,
        start_from=0.5,
        end_at=1.0,
        scale=0.1,
        rotation_degrees=0,
        label_font_size=10,
        label_padding=0.01):

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