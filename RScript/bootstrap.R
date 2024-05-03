library(reticulate)
library(pROC)
library(PRROC) 
library(readr)

# Specify the path to your Python executable if necessary
use_python("./python.exe", required = TRUE)

# Define a function to read a pickle file using Python
read_pickle <- function(file_path) {
  py_run_string("import pickle")
  py_run_string(paste0("with open('", file_path, "', 'rb') as f:
                        data = pickle.load(f)"))
  return(py$data)
}


process_results <- function(search_results, label_list, models_list, save_path, n_bootstraps = 1000, seed = 123) {
  set.seed(seed) # For reproducibility
  
  # Prepare lists to store results
  avg_results <- setNames(vector("list", length(models_list)), models_list)
  roc_objects <- list()
  overall_avg_aucs_list <- setNames(vector("list", length(models_list)), models_list)  # Store overall averages for each model
  
  # Initialize progress bar
  total_operations <- length(models_list) * length(label_list) * n_bootstraps
  pb <- txtProgressBar(min = 0, max = total_operations, style = 3)
  progress <- 0
  
  for (model in models_list) {
    model_bootstrap_aucs <- list()
    
    for (label in label_list) {
      key <- paste("('", model, "', '", label, "')", sep = "")
      if (!is.null(search_results[[key]])) {
        preds <- as.vector(search_results[[key]]$all_preds)
        labels <- as.vector(search_results[[key]]$all_labels)
        
        # Perform bootstrapping
        bootstrap_aucs <- numeric(n_bootstraps)
        for (i in 1:n_bootstraps) {
          boot_indices <- sample(length(labels), replace = TRUE)
          boot_labels <- labels[boot_indices]
          boot_preds <- preds[boot_indices]
          
          suppressMessages({
            roc_obj <- roc(boot_labels, boot_preds)
          })
          bootstrap_aucs[i] <- auc(roc_obj)
          
          # Update progress bar
          progress <- progress + 1
          setTxtProgressBar(pb, progress)
        }
        
        suppressMessages({
          roc_objects[[key]] <- roc(labels, preds)  
        })
        
        # Store bootstrap AUCs for each label
        model_bootstrap_aucs[[label]] <- bootstrap_aucs
        
        # Calculate and store average AUC across all bootstraps for this label
        avg_auc <- mean(bootstrap_aucs)
        ci_auc <- quantile(bootstrap_aucs, probs = c(0.025, 0.975))
        avg_results[[model]][[label]] <- sprintf("%.3f (%.3f - %.3f)", avg_auc, ci_auc[1], ci_auc[2])
      }
    }
    
    # Calculate the overall average AUC across all labels for each bootstrap iteration
    if (length(model_bootstrap_aucs) > 0) {
      all_bootstrap_aucs <- do.call(cbind, model_bootstrap_aucs)
      overall_avg_aucs <- rowMeans(all_bootstrap_aucs)
      
      # Store the overall average AUCs for each model
      overall_avg_aucs_list[[model]] <- overall_avg_aucs
      
      # Compute the point estimate and CI for the overall average
      overall_point_estimate <- mean(overall_avg_aucs)
      overall_ci <- quantile(overall_avg_aucs, probs = c(0.025, 0.975))
      
      avg_results[[model]][["average"]] <- sprintf("%.3f (%.3f - %.3f)", overall_point_estimate, overall_ci[1], overall_ci[2])
    }
  }
  
  # Close the progress bar
  close(pb)
  
  # Convert the avg_results list to a dataframe for CSV output
  results_df <- do.call(rbind, lapply(avg_results, function(x) {
    data.frame(t(x), stringsAsFactors = FALSE)
  }))
  
  # Transpose the results dataframe to get labels as rows and models as columns
  transposed_results_df <- as.data.frame(t(results_df), stringsAsFactors = FALSE)
  
  # Set column names to the models
  colnames(transposed_results_df) <- models_list
  
  # Set row names to the labels plus "Average"
  rownames(transposed_results_df) <- c(label_list, "average")
  
  # Write the transposed results to a CSV file
  transposed_results_df <-data.frame(lapply(transposed_results_df, unlist), stringsAsFactors = FALSE)
  
  # Assuming 'results_with_roc' is a list containing the 'results' dataframe
  write.csv(transposed_results_df, 
            save_path, 
            row.names = TRUE,  # Include row names (Label_Pain, Label_Tired, etc.)
            quote = TRUE)      # Ensure that strings are quoted
  # Return the results dataframe, ROC objects, and overall average AUCs list
  return(list(results = transposed_results_df, roc_objects = roc_objects, overall_avg_aucs = overall_avg_aucs_list))
}


compare_model_individual_auc <- function(roc_objects, labels_list, model1, model2, save_path,method) {
  comparisons_df <- data.frame(
    PickModel = character(),
    ComparedModel = character(),
    AUROC_BestModel = numeric(),
    AUROC_ComparedModel = numeric(),
    PValue_DeLong = numeric(),
    TargetLabel = character(),
    stringsAsFactors = FALSE
  )
  
  for (label in c(labels_list)) {
    key1 <- paste("('", model1, "', '", label, "')", sep = "")
    key2 <- paste("('", model2, "', '", label, "')", sep = "")
    
    # Ensure both models have ROC data for the label
    if (!is.null(roc_objects[[key1]]) && !is.null(roc_objects[[key2]])) {
      # Perform DeLong test
      delong_test <- roc.test(roc_objects[[key1]], roc_objects[[key2]], method)
      
      # Compile results
      comparison <- data.frame(
        PickModel = model1,
        ComparedModel = model2,
        AUROC_BestModel = auc(roc_objects[[key1]]),
        AUROC_ComparedModel = auc(roc_objects[[key2]]),
        PValue_DeLong = delong_test$p.value,
        TargetLabel = label,
        stringsAsFactors = FALSE
      )
      
      # Append to the results dataframe
      comparisons_df <- rbind(comparisons_df, comparison)
    }
  }
  
  # Write the comparisons to a CSV file
  write.csv(comparisons_df, 
            save_path, 
            row.names = FALSE, 
            quote = TRUE)
  
  return(comparisons_df)
}


# 2*pnorm(-abs((AUROC_{average, xgb} - AUROC_{average, lr}) / sd_{across s bootstraps}(AUROC_{average, xgb} - AUROC_{average, lr}))
calculate_p_value_for_mean <- function(search_results_with_roc, model1_name, model2_name) {
  # Extracting the list of mean AUROC values for 1000 bootstrap results for both models
  mean_auroc_model1_list <- search_results_with_roc$overall_avg_aucs[[model1_name]]
  mean_auroc_model2_list <- search_results_with_roc$overall_avg_aucs[[model2_name]]
  
  # Calculating the mean difference in AUROC between the two models
  mean_diff_auroc <- mean(mean_auroc_model1_list - mean_auroc_model2_list)
  
  # Calculating the standard deviation of the difference across bootstrap samples
  sd_diff <- sd(mean_auroc_model1_list - mean_auroc_model2_list)
  
  # Calculating the z-score and then the p-value
  z_score <- abs(mean_diff_auroc / sd_diff)
  p_value <- 2 * pnorm(-z_score)
  
  # Returning the p-value
  return(p_value)
}


