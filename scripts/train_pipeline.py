# Standard library imports
from pathlib import Path
import warnings
import sys
import pickle
import argparse
import dill

# Related third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from bayes_opt import BayesianOptimization

# Local application/library specific imports
ROOT_DIR = Path(__file__).parent.parent.as_posix()
sys.path.append(ROOT_DIR)

from src.models.GRU import GRU
from src.models.LSTM import LSTM
from src.models.MLP import MLP
from src.models.TCN import TCN
from src.models.Transformer import Transformer

from src.train import (
    EHRDataset,
    pad_collate,
    train_and_evaluate_dl,
    masked_binary_cross_entropy_with_logits
)
from src.config import (
    bounds_dict,
    data_path_dict
)

from src.train import get_neg_to_pos_ratios


def perform_bayesian_optimization(objective_func, model_name, model_type, pbounds, train_df, tune_df,test_df, label,full_colmuns_list, n_iter=3, init_points=5, weights=None, return_optimizer=False,verbose = False,allow_duplicate_points = True,ci =False):
    #torch.cuda.empty_cache()
    # Initialize a variable to keep track of the best val_aucs for dl models
    best_val_aucs = None

    best_auroc = None  # Initialize a variable to keep track of the best auroc

    best_auroc_ci = None

    best_all_preds = None

    best_all_labels = None

    def objective_wrapper(**params):
        nonlocal best_val_aucs  # Declare best_val_aucs as nonlocal so that we can modify it
        nonlocal best_auroc  # Declare best_auroc as nonlocal so that we can modify it
        nonlocal best_auroc_ci
        nonlocal best_all_preds
        nonlocal best_all_labels


        objective_auroc, val_aucs ,auroc_ci, all_preds,all_labels,_,_ = objective_func(model_name, model_type, params, train_df, tune_df, test_df, label,full_colmuns_list, weights, verbose,ci)

        # If this is the first iteration or if a better auroc is found, update best_val_aucs and best_auroc
        if best_auroc is None or objective_auroc > best_auroc:
            best_val_aucs = val_aucs
            best_auroc = objective_auroc
            best_auroc_ci = auroc_ci
            best_all_preds = all_preds
            best_all_labels = all_labels

        return objective_auroc  # return objective_auroc to the optimizer

    optimizer = BayesianOptimization(
        f=objective_wrapper,  # use wrapper function here
        pbounds=pbounds,
        random_state=42,
        verbose=2,
        allow_duplicate_points=allow_duplicate_points
    )

    optimizer.maximize(n_iter=n_iter, init_points=init_points)

    if return_optimizer:
        # If return_optimizer is True, return a tuple containing both the optimizer and best_val_aucs
        return optimizer, best_val_aucs,best_auroc_ci,best_all_preds,best_all_labels
    else:
        # Otherwise, just return the best parameters as before
        return optimizer.max['params']

def train_model(model_name,model_type,params, train_df,tune_df,test_df,label,full_colmuns_list,weights=None,verbose = False,ci= False,testing = False):
    torch.cuda.empty_cache()
    # for dl models
    val_aucs = None
    best_model = None
    best_ir_model = None
    isotonic = None
    # store 95 ci
    auroc_ci = None

    if model_type == 'ml':
        # Initialize dataset for ml models

        if model_name == 'lr':
            model = LogisticRegression(C=params['C'], max_iter=1000)
        elif model_name == 'rf':
            model = RandomForestClassifier(n_estimators=int(params['n_estimators']), max_depth=int(params['max_depth']), max_features=params['max_features'])
        elif model_name == 'xgb':
            model = xgb.XGBClassifier(n_estimators=int(params['n_estimators']),
                                      max_depth=int(params['max_depth']), learning_rate=params['learning_rate'], gamma=params['gamma'],eval_metric='logloss')
        elif model_name == 'lgbm':
            model = lgb.LGBMClassifier(n_estimators=int(params['n_estimators']),
                                      max_depth=int(params['max_depth']), learning_rate=params['learning_rate'], num_leaves=int(params['num_leaves']),
                                      min_data_in_leaf=int(params['min_data_in_leaf']), feature_fraction=params['feature_fraction'],
                                      bagging_fraction=params['bagging_fraction'], bagging_freq=int(params['bagging_freq']),
                                      lambda_l1=params['lambda_l1'], lambda_l2=params['lambda_l2'],verbose= -1)

        auroc_scores = []
        # Train using train_df and tuned in tune_df, exclude labels from training
        labels_list = [s for s in full_colmuns_list if s.startswith("Label")]
        X_train = train_df[train_df[label] != -1].drop(labels_list + ['PatientID', 'Trt_Date'], axis=1)
        y_train = train_df[train_df[label] != -1][label]

        X_tune = tune_df[tune_df[label] != -1].drop(labels_list + ['PatientID', 'Trt_Date'], axis=1)
        y_tune = tune_df[tune_df[label] != -1][label]

        X_test = test_df[test_df[label] != -1].drop(labels_list + ['PatientID', 'Trt_Date'], axis=1)
        y_test = test_df[test_df[label] != -1][label]

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_tune)[:, 1]

        if testing == True:

            # Fit Isotonic Regression using the tunning cohort
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(y_pred_proba, y_tune)

            #Predict using the test cohort
            test_predictions = model.predict_proba(X_test)[:, 1]
            y_pred_proba = isotonic.transform(test_predictions)
            all_preds, all_labels = y_pred_proba,y_test.values
        else:
            all_preds, all_labels = y_pred_proba, y_tune.values
        # Testing
        auroc_scores.append(roc_auc_score(all_labels, all_preds))
        objective_auroc = np.mean(auroc_scores)

        best_model = model
        best_ir_model = isotonic


    if model_type == 'dl':
        torch.cuda.empty_cache()
        # Train and tune using full batch, get the number of unique patientID in train_df
        test_batch = test_df['PatientID'].unique().shape[0]

        batch_size_dict = {1: 64, 2: 128, 3: 256, 4: 512}

        # batch_size_dict = {1: 32,2:32}
        batch_size = batch_size_dict[int(params['batch_size'])]

        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']

         # Check whether label and weights are given as list
        if not isinstance(label, list):
            label = [label]
        if not isinstance(weights, list):
            weights = [weights]

        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Hyperparameters for DL models
        train_dataset = EHRDataset(train_df,label)
        tune_dataset = EHRDataset(tune_df,label)
        test_dataset = EHRDataset(test_df,label)

        train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        tune_loader =  DataLoader(tune_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate)
        test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, collate_fn=pad_collate)

        # input size = size - (features that starts with Label) - 2 (PatientID and Trt_Date column)
        input_size = train_df.shape[1] - (len([s for s in full_colmuns_list if s.startswith("Label")]) + 2)

        # hidden_size = 128

        output_size = len(label)
        criterion = masked_binary_cross_entropy_with_logits

        if model_name =='mlp':
            hidden_size_1 = int(params['hidden_size_1'])
            hidden_size_2 = int(params['hidden_size_2'])
            model = MLP(input_size, hidden_size_1,hidden_size_2,output_size,dropout=params['dropout']).to(device)

        elif model_name =='gru':
            hidden_size = int(params['hidden_size'])
            num_layers = int(params['num_layers'])
            model = GRU(input_size, hidden_size, output_size, num_layers=num_layers,dropout=params['dropout']).to(device)

        elif model_name =='lstm':
            hidden_size = int(params['hidden_size'])
            num_layers = int(params['num_layers'])
            model = LSTM(input_size,hidden_size,output_size,num_layers=num_layers,dropout=params['dropout']).to(device)

        elif model_name == 'tcn':
            #model = TCNModel(input_size, num_channels, output_size).to(device)
            num_channels = [int(params[f'num_channels_{i}']) for i in range(1, 4)]
            kernel_size = int(params['kernel_size'])
            # dropout = params['dropout']
            model = TCN(input_size, num_channels, output_size, kernel_size=kernel_size,dropout=params['dropout']).to(device)

        elif model_name == 'vanilla_transformer':
            num_heads = {1: 8, 2: 16, 3: 32, 4: 64}
            dim_feedforward = {1: 128, 2: 256, 3: 512}
            hidden_size =  {1: 128, 2: 256, 3: 512}
            model = Transformer(input_size, output_size,hidden_size =hidden_size[int(params['hidden_size'])],
                                     num_heads=num_heads[int(params['num_heads'])],
                                     num_layers=int(params['num_layers']),
                                     dim_feedforward= dim_feedforward[int(params['dim_feedforward'])],dropout=params['dropout']).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if testing == True:
            train_losses, train_aucs, train_aups, val_losses, val_aucs, val_aups, mean_val_losses, train_auc_cis, train_aup_cis, val_auc_cis, val_aup_cis,ir_models,models,all_preds,all_labels = train_and_evaluate_dl(
                30, model, train_loader, tune_loader,test_loader, optimizer, criterion, device,output_size,label, weights,scheduler =None,early_stopping_patience=5,verbose = verbose,ci=ci,my_copy=True,testing=True,calibration=True)
        else:
            train_losses, train_aucs, train_aups, val_losses, val_aucs, val_aups, mean_val_losses, train_auc_cis, train_aup_cis, val_auc_cis, val_aup_cis, ir_models, models, all_preds, all_labels = train_and_evaluate_dl(
                30, model, train_loader, tune_loader, None, optimizer, criterion, device, output_size, label,weights,scheduler=None, early_stopping_patience=3, verbose=verbose, ci=ci, my_copy=False)

        if output_size == 1:
            best_epoch = np.argmax(val_aucs[0])
            objective_auroc = val_aucs[0][best_epoch]  # find the epoch with the highest AUROC

        else:
            # Case when there are multiple outputs
            mean_aucs = [np.mean([val_aucs[i][epoch] for i in range(output_size)]) for epoch in
                         range(len(val_aucs[0]))]
            best_epoch = np.argmax(mean_aucs)  # find the epoch with the highest mean AUROC
            objective_auroc = mean_aucs[best_epoch]

        if ci == True:
            auroc_ci = val_auc_cis

        all_preds = all_preds[best_epoch]
        all_labels = all_labels[best_epoch]

        if len(models) > 0:
            best_model = models[best_epoch]
            best_ir_model = ir_models[best_epoch]

    return objective_auroc, val_aucs,auroc_ci,all_preds,all_labels,best_model,best_ir_model

def main(train_path, tune_path, test_path, full_path, search_save_path, test_save_path,target_change):

    train_df = pd.read_csv(train_path)
    tune_df = pd.read_csv(tune_path)
    test_df = pd.read_csv(test_path)
    df = pd.read_csv(full_path)

    bayopt_search_results = search_save_path
    bayopt_test_results = test_save_path

    # set labels_list as startswith "Label" and end with 3pt_change
    labels_list = [s for s in train_df.columns.tolist() if s.startswith("Label") and s.endswith(target_change)]
    full_colmuns_list = df.columns.tolist()
    neg_to_pos_ratios_dict = get_neg_to_pos_ratios(train_df, labels_list)
    weights_list = [neg_to_pos_ratios_dict[label]['ratio'] for label in labels_list]

    # List of models to consider for both ML and DL
    ml_models = ['lr', 'rf', 'xgb', 'lgbm']
    dl_models = ['mlp', 'gru', 'lstm', 'tcn', 'vanilla_transformer']

    warnings.filterwarnings("ignore")

    # # ========== Bayesian optimization ================
    search_results = {}
    all_evaluations = {}

    # Single-Task learning for ML and DL models
    for label in labels_list:
        # ML models
        for model in ml_models:
            print(f"Optimizing hyperparameters for Model: {model}, Label: {label} using Bayesian Optimization...")
            optimizer, _, _, all_preds, all_labels = perform_bayesian_optimization(
                train_model,
                model, 'ml', bounds_dict[model],
                train_df, tune_df, test_df, label, full_colmuns_list,
                weights=None, n_iter=30, init_points=15,
                return_optimizer=True,
                verbose=False,
                ci=False
            )
            search_results[(model, label)] = {
                'params': optimizer.max['params'],
                'target': optimizer.max['target'],  # Include the best objective value
                # 'auroc_ci': auroc_ci,
                'all_preds': all_preds,
                'all_labels': all_labels
            }
            all_evaluations[(model, label)] = optimizer.res

        for model in dl_models:
            print(
                f"Optimizing hyperparameters for Deep Learning Model: {model}, Label: {label} using Bayesian Optimization...")
            optimizer, _, _, all_preds, all_labels = perform_bayesian_optimization(
                train_model,
                model, 'dl', bounds_dict[model],
                train_df, tune_df, test_df, label, full_colmuns_list,
                weights=neg_to_pos_ratios_dict[label]['ratio'],
                n_iter=30, init_points=15,
                return_optimizer=True,
                verbose=False,
                ci=False
            )
            search_results[(model, label)] = {
                'params': optimizer.max['params'],
                'target': optimizer.max['target'],  # Include the best objective value
                'all_preds': all_preds[-1],
                'all_labels': all_labels[-1]
            }
            all_evaluations[(model, label)] = optimizer.res

        # Multi-task (all labels) for DL models
        for model in dl_models:
            print(
                f"Optimizing hyperparameters for Deep Learning Model: {model} for All Targets using Bayesian Optimization...")
            optimizer, val_aucs, auroc_ci, all_preds, all_labels = perform_bayesian_optimization(
                train_model,
                model, 'dl', bounds_dict[model],
                train_df, tune_df, test_df, labels_list, full_colmuns_list,
                weights=weights_list,
                n_iter=30, init_points=15,
                return_optimizer=True,
                verbose=False,
                ci=False
            )

            mean_aucs = [np.mean([val_aucs[i][epoch] for i in range(len(val_aucs))]) for epoch in
                         range(len(val_aucs[0]))]
            best_epoch = np.argmax(mean_aucs)  # find the epoch with the highest mean AUROC

            for i in range(len(labels_list)):
                search_results[('mtl_' + model, labels_list[i])] = {
                    'params': optimizer.max['params'],
                    'target': val_aucs[i][best_epoch],
                    'all_preds': all_preds[i],
                    'all_labels': all_labels[i]
                }

            search_results[('mtl_' + model, "All_Targets")] = {
                'params': optimizer.max['params'],
                'target': optimizer.max['target']  # Include the best objective value
            }

            print("=============================================")
            all_evaluations[(model, "All_Targets")] = optimizer.res

        # save the search results
        with open(bayopt_search_results, "wb") as f:
            pickle.dump(search_results, f)

    # =================== Model Training and Testing ===================
    search_df = pd.DataFrame.from_dict(search_results, orient='index')
    search_df.index.names = ['model', 'label']
    search_df.reset_index(inplace=True)

    train_results = {}

    # Single task training and testing
    for label in labels_list:
        # ML models
        for model in ml_models:
            objective_auroc, _, _, all_preds, all_labels, best_model, best_ir_model = \
                train_model(model, 'ml', search_results[(model, label)]['params'],
                            train_df, tune_df, test_df, label, full_colmuns_list,
                            weights=None, verbose=False, ci=False, testing=True
                            )

            train_results[(model, label)] = {
                'train_auroc': objective_auroc,
                'params': search_results[(model, label)]['params'],
                'all_preds': all_preds,
                'all_labels': all_labels,
                'best_model': best_model,
                'best_ir_model': best_ir_model
            }
            print(
                f"Running Model : {model}, Label: {label} with tuned hyperparameters. Best AUROC: {objective_auroc:.4f}")

        for model in dl_models:
            objective_auroc, _, _, all_preds, all_labels, best_model, best_ir_model = \
                train_model(model, 'dl', search_results[(model, label)]['params'],
                            train_df, tune_df, test_df, label, full_colmuns_list,
                            weights=neg_to_pos_ratios_dict[label]['ratio'], verbose=False, ci=False, testing=True
                            )
            train_results[(model, label)] = {
                'train_auroc': objective_auroc,
                'params': search_results[(model, label)]['params'],
                'all_preds': all_preds[-1],
                'all_labels': all_labels[-1],
                'best_model': best_model,
                'best_ir_model': best_ir_model
            }
            print(
                f"Running Model : {model}, Label: {label} with tuned hyperparameters. Best AUROC: {objective_auroc:.4f}")

    # Multi-task learning and testing
    for model in dl_models:
        # add mtl to model name
        params = search_results[('mtl_' + model, 'All_Targets')]['params']
        objective_auroc, val_aucs, auroc_ci, all_preds, all_labels, best_model, best_ir_model = \
            train_model(model, 'dl', params,
                        train_df, tune_df, test_df, labels_list, full_colmuns_list,
                        weights=weights_list, verbose=True, ci=False, testing=True
                        )

        mean_aucs = [np.mean([val_aucs[i][epoch] for i in range(len(val_aucs))]) for epoch in
                     range(len(val_aucs[0]))]
        best_epoch = np.argmax(mean_aucs)  # find the epoch with the highest mean AUROC
        # best_epoch_aucs = mean_aucs[best_epoch]

        for i in range(len(labels_list)):
            train_results[('mtl_' + model, labels_list[i])] = {
                'train_auroc': val_aucs[i][best_epoch],
                'params': params,
                'all_preds': all_preds[i],
                'all_labels': all_labels[i]
            }
            print(
                f"Running multi-task Model : {model}, Label: {labels_list[i]} with tuned hyperparameters. Best AUROC: {val_aucs[i][best_epoch]:.4f}")

        train_results[('mtl_' + model, "All_Targets")] = {
            'train_auroc': objective_auroc,  # Include the best objective value
            'params': params,
            'best_model': best_model,
            'best_ir_model': best_ir_model
        }

    # save the testing results
    with open(bayopt_test_results, "wb") as f:
        dill.dump(train_results, f)

# write main function to run
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Bayesian optimization and model training/testing.')
    parser.add_argument('--train_path', type=str, default=data_path_dict['train'], help='Path to the pre-processed training set CSV')
    parser.add_argument('--tune_path', type=str, default=data_path_dict['tune'], help='Path to the pre-processed validation set CSV')
    parser.add_argument('--test_path', type=str, default=data_path_dict['test'], help='Path to the pre-processed test set CSV')
    parser.add_argument('--full_path', type=str, default=data_path_dict['full'], help='Path to the pre-processed full dataset CSV')
    parser.add_argument('--search_save_path', type=str, default=data_path_dict['save_bayopt_search'], help='Save path for bayopt search results')
    parser.add_argument('--test_save_path', type=str, default=data_path_dict['save_bayopt_test'], help='Save path for bayopt test results')
    parser.add_argument('--target_change', type=str, default='3pt_change', help='Target change')

    args = parser.parse_args()
    main(args.train_path, args.tune_path, args.test_path, args.full_path, args.search_save_path, args.test_save_path,args.target_change)
