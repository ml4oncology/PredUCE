import numpy as np
import pandas as pd
import shap
import dill
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    plt.style.use('seaborn-whitegrid')

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
        plt.grid(False)  # Turn off the grid
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
        plt.grid(False)  # Turn off the grid
        plt.savefig(os.path.join(save_path, f"{label}_bar.png"))
        plt.close()


def load_results(file_path):
    """
    Custom loading function to handle unpickling with missing modules gracefully.
    """
    class IgnoreMissingModuleUnpickler(dill.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                # print(f"Warning: Module {module}.{name} not found. Using placeholder.")
                return type(name, (), {})
    with open(file_path, 'rb') as file:
        return IgnoreMissingModuleUnpickler(file).load()

def calculate_shap_values(test_df, labels_list, train_results, excel_path, sheet_name, model_type='lgbm'):
    """
    Calculates SHAP values for each label in labels_list using the specified model from train_results.
    """
    shap_dict = {}
    for label in labels_list:
        # Filter and preprocess the test data for the current label
        current_X_test = test_df.drop([col for col in test_df.columns if col.startswith("Label")] + ['PatientID', 'Trt_Date'], axis=1)
        current_X_test_array = np.array(current_X_test)

        # Retrieve the model trained for the current label
        model = train_results[model_type, label]['best_model']

        # Create a SHAP explainer using the current data
        explainer = shap.Explainer(model, current_X_test_array, max_evals=50)

        # Calculate SHAP values
        shap_explainer = explainer(current_X_test_array, check_additivity=False)

        # Store both SHAP values and the corresponding data in the dictionary
        shap_dict[label] = {
            'shap_values': shap_explainer,
            'data': current_X_test
        }

        cohort_sheet = pd.read_excel(excel_path, sheet_name=sheet_name)
        rename_dict = dict(zip(cohort_sheet['Features(original)'], cohort_sheet['Features(lower)']))
        for label, data in shap_dict.items():
            data['data'].rename(columns=rename_dict, inplace=True)

    return shap_dict


def generate_shaps(test_df,labels_list, excel_path, sheet_name, train_results,save_path):
    """
    Main function to orchestrate SHAP value calculation and column renaming based on user inputs.
    """
    # labels_list = [s for s in test_df.columns.tolist() if s.startswith("Label") and s.endswith("3pt_change")]
    shap_dict = calculate_shap_values(test_df, labels_list, train_results,excel_path,sheet_name)

    generate_shap_plots(shap_dict, labels_list,save_path)
    print("Finished SHAP value calculation and column renaming.")
    return shap_dict


def shap_plots(plot_path, legend_order, label_mapping, plot_type="layered_violin"):
    """
    Display SHAP plots in a 3x3 grid.

    Parameters:
    - plot_path (str): Path to the directory where SHAP plots are stored.
    - legend_order (list): Order of labels for plotting.
    - label_mapping (dict): Mapping from technical label names to human-readable names.
    - plot_type (str): Type of plot to display ("layered_violin" or "bar").
    """
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))  # Initialize subplot grid with a larger size for each subplot
    axs = axs.flatten()  # Flatten the array for easy iteration

    for i, converted_label in enumerate(legend_order):
        original_label = [key for key, value in label_mapping.items() if value == converted_label][0]
        filename = f"{original_label}_{plot_type}.png"  # Construct the filename based on the plot type
        img_path = f"{plot_path}/{filename}"

        try:
            img = mpimg.imread(img_path)
            axs[i].imshow(img)
            axs[i].axis('off')  # Hide axes

            # Place the title using text to manually position it
            axs[i].text(0.6, 1.1, converted_label, transform=axs[i].transAxes, ha="center", va="top", fontsize=16)
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            continue

    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust layout to manage space between plots
    plt.tight_layout()
    plt.show()


def calculate_mean_shap_values(shap_dict, labels_list, test_df, label_mapping, legend_order, temporal_cohort_sheet):
    """
    Calculate mean SHAP values for each feature across all labels and transform column names to lower cases.

    Parameters:
    - shap_dict: Dictionary with label keys and values as {'shap_values': SHAP values array, 'data': feature data}.
    - labels_list: List of labels for which to calculate mean SHAP values.
    - test_df: Test DataFrame from which to extract feature columns.
    - label_mapping: Dictionary mapping original label names to readable names.
    - legend_order: List of labels in the desired order for the final DataFrame.
    - temporal_cohort_sheet: DataFrame with original and lower case feature names.

    Returns:
    - mean_shap_df: DataFrame with mean SHAP values for each feature, indexed by readable labels and including a 'mean' row.
    """


    rename_dict = dict(zip(temporal_cohort_sheet['Features(original)'], temporal_cohort_sheet['Features(lower)']))

    # Prepare data columns from test_df, excluding label and other non-feature columns
    data_columns = test_df.drop([col for col in test_df.columns if col.startswith("Label")] + ['PatientID', 'Trt_Date'], axis=1).columns.tolist()
    # Apply renaming to data_columns
    data_columns_lower = [rename_dict.get(col, col) for col in data_columns]

    # Initialize mean_shap_df with renamed lower case columns
    mean_shap_df = pd.DataFrame(columns=data_columns_lower)

    # Calculate mean absolute SHAP values for each label
    for label in labels_list:
        shap_values = np.abs(shap_dict[label]['shap_values'].values).mean(axis=0)
        # Rename columns in the shap_dict data according to lower cases
        data_lower_case = shap_dict[label]['data'].rename(columns=rename_dict)
        # Ensure the columns are in lower case according to Excel sheet
        mean_shap_df.loc[label, :] = shap_values[:len(data_columns_lower)]  # Adjust the slice as necessary

    # Map index using label_mapping and reorder according to legend_order
    mean_shap_df.index = mean_shap_df.index.map(label_mapping)
    mean_shap_df = mean_shap_df.reindex(legend_order)

    # Add a row for the mean of the mean_shap_df
    mean_shap_df.loc['mean'] = mean_shap_df.mean(axis=0)

    return mean_shap_df

def plot_mean_abs_shap_values(mean_shap_df, temporal_cohort_sheet, colors):

    plt.style.use('seaborn-whitegrid')
    transposed_df = pd.merge(mean_shap_df.T.abs(), temporal_cohort_sheet, left_index=True, right_on='Features(lower)')

    # Aggregate mean absolute SHAP values by group
    grouped_shap_means_abs = transposed_df.groupby('Group')['mean'].sum()

    # Sort the aggregated values for better visualization
    grouped_shap_means_abs_sorted = grouped_shap_means_abs.sort_values(ascending=True)

    # Plotting
    plt.figure(figsize=(10, len(grouped_shap_means_abs_sorted) * 0.5))  # Adjust the figure size
    bars = grouped_shap_means_abs_sorted.plot(kind='barh', color='grey', alpha=0.4)

    # Initialize an empty list for legend handles
    legend_handles = []

    # Plot individual model's values
    for i, label in enumerate(mean_shap_df.index[:-1]):  # Exclude the last row
        label_sums = transposed_df.groupby('Group')[label].sum()
        color = colors[i % len(colors)]
        # Plot dots for each group and add a legend entry
        for group, value in label_sums.iteritems():
            y_pos = grouped_shap_means_abs_sorted.index.get_loc(group)
            plt.scatter(value, y_pos, color=color, s=80)
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color,markersize=9))

    plt.xlabel('Sum of Mean(|SHAP Value|) Across Group',fontsize = 12,labelpad=10)
    plt.ylabel('')  # Set ylabel to empty string

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add the legend outside of the plot on the right-hand side, smaller legend font size
    plt.grid(False)
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.1, 0.5), frameon=False, fontsize=12)

    plt.show()

