lr_bounds = {'C': (0.001, 10)}
rf_bounds = {'n_estimators': (50, 200), 'max_depth': (3, 7), 'max_features': (0.01, 1)}
xgb_bounds = {'n_estimators': (50, 200), 'max_depth': (3, 7), 'learning_rate': (0.001, 0.01), 'gamma': (0, 1)}

nn_models = {'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01)}
mlp_bounds = {'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01),'hidden_size_1':(64,256),'hidden_size_2':(64,256),'dropout':(0,0.5)}
gru_bounds = {'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01), 'hidden_size': (64, 256), 'num_layers': (1, 3),'dropout':(0,0.5)}

lstm_bounds = {'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01), 'hidden_size': (64, 256), 'num_layers': (1, 3),'dropout':(0,0.5)}

tcn_bounds = {
    'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01),
    'num_channels_1': (16, 128), 'num_channels_2': (16, 128), 'num_channels_3': (16, 128),
    'kernel_size': (2, 5),'dropout':(0,0.5)
}

transformer_bounds = {
    'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01),
    'num_layers': (1, 4), 'num_heads': (1, 4) , 'dim_feedforward':(1,3)
}


vanilla_transformer_bounds = {
    'batch_size': (1, 3), 'learning_rate': (0.001, 0.01), 'weight_decay': (0, 0.01),
    'num_layers': (1, 4), 'num_heads': (1, 4), 'dim_feedforward':(1,3), 'hidden_size':(1,3),'dropout':(0,0.5)
}


lgbm_bounds = {
    'n_estimators': (50, 200),
    'max_depth': (3, 7),
    'learning_rate': (0.001, 0.01),
    'num_leaves': (20, 40),
    'min_data_in_leaf': (20, 100),
    'feature_fraction': (0.5, 1.0),
    'bagging_fraction': (0.5, 1.0),
    'bagging_freq': (0, 1),
    'lambda_l1': (0, 1),
    'lambda_l2': (0, 1)
}

# Define bounds for each model
bounds_dict = {
    'lr': lr_bounds,
    'rf': rf_bounds,
    'xgb': xgb_bounds,
    'mlp': mlp_bounds,
    'gru': gru_bounds,
    'lstm': lstm_bounds,
    'tcn': tcn_bounds,
    'transformer':transformer_bounds,
    'vanilla_transformer':vanilla_transformer_bounds,
    'lgbm': lgbm_bounds
}

# Setup for labels and plotting
label_mapping = {
    'Label_Pain_3pt_change': 'Pain',
    'Label_Tired_3pt_change': 'Fatigue',
    'Label_Nausea_3pt_change': 'Nausea',
    'Label_Depress_3pt_change': 'Depression',
    'Label_Anxious_3pt_change': 'Anxiety',
    'Label_Drowsy_3pt_change': 'Drowsiness',
    'Label_Appetite_3pt_change': 'Appetite',
    'Label_WellBeing_3pt_change': 'Well-being',
    'Label_SOB_3pt_change': 'Dyspnea'
}

legend_order = ['Nausea', 'Appetite', 'Pain', 'Dyspnea', 'Fatigue', 'Drowsiness', 'Depression', 'Anxiety', 'Well-being']

data_path_dict = {
    # Load the pre-processed data
    'train': "./data/pre_processed_train_set.csv",
    'tune': "./data/pre_processed_valid_set.csv",
    'test': "./data/pre_processed_test_set.csv",
    'full': "./data/pre_processed_df.csv",

    # Save the bayesian optimization search and test results
    'save_bayopt_search': "./bayopt_results/search_results.pkl",
    'save_bayopt_test': "./bayopt_results/test_results.pkl",

    # Train results, features file, and odds ratio file
    "train_results": "./model_results/train_results.pkl",
    "features_file": "./data/features_list.xlsx",
    "odd_ratio_file": "./data/odds_ratio_analysis.csv",
    'ed_association': "./data/ed_death_association.pkl",

    # SHAP value plot save path
    "plots_save_path": "./plots",
    "sheet_name": "FeatureList"
}
