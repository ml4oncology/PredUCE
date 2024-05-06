import argparse
import pickle
import pandas as pd

from src.eval import process_results
from src.shap import load_results

from src.config import data_path_dict


def main(train_results_path, test_df_path, output_file_path):
    train_results = load_results(train_results_path)

    results_df = pd.DataFrame.from_dict(train_results, orient="index")
    results_df.index.names = ["model", "label"]
    results_df.reset_index(inplace=True)

    test_df = pd.read_csv(test_df_path)
    labels_list = [
        s
        for s in test_df.columns.tolist()
        if s.startswith("Label") and s.endswith("3pt_change")
    ]

    # Generate dataframes for each label with ED visit based on 10% alarm rate per event.
    label_with_ed_per = process_results(
        results_df, test_df, labels_list, "lgbm", "Label_EDvisit"
    )

    # Generate dataframes for each labels with ED visit based on 10% alarm rate for all events.
    all_with_ed_all = process_results(
        results_df, test_df, labels_list, "lgbm", "Label_EDvisit", threshold=0.1
    )

    ed_death_association = {
        "label_with_ed_per": label_with_ed_per,
        "all_with_ed_all": all_with_ed_all,
    }

    with open(output_file_path, "wb") as file:
        pickle.dump(ed_death_association, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ED and Death Associations.")

    parser.add_argument(
        "--train_results_path",
        type=str,
        default=data_path_dict["train_results"],
        help="Path to train results pickle file",
    )
    parser.add_argument(
        "--test_df_path",
        type=str,
        default=data_path_dict["test"],
        help="Path to test dataset CSV file",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=data_path_dict["ed_association"],
        help="Path where to save the output pickle file",
    )

    args = parser.parse_args()

    main(args.train_results_path, args.test_df_path, args.output_file_path)

    # print success message
    print("ED Associations processed successfully!")
