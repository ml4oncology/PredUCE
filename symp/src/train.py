# Standard library imports
from copy import copy

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import *
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


class EHRDataset(Dataset):
    def __init__(self, df, labels_list):
        self.df = df
        self.patients = df["PatientID"].unique()
        self.labels_list = labels_list
        self.full_label_list = [s for s in df.columns.tolist() if s.startswith("Label")]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_data = self.df[self.df["PatientID"] == self.patients[idx]]
        patient_features = patient_data.drop(
            columns=["PatientID", "Trt_Date"] + self.full_label_list
        ).values

        # Training only using symptoms labels
        patient_labels = patient_data[self.labels_list].values
        return torch.tensor(patient_features, dtype=torch.float32), torch.tensor(
            patient_labels, dtype=torch.float32
        )


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
        ratio = neg_count / pos_count if pos_count != 0 else float("inf")

        # Event rate is the ratio of positive count to the total count
        event_rate = pos_count / (pos_count + neg_count)

        # Store calculations in the dictionary
        neg_to_pos_ratios_dict[label] = {
            "positive_count": pos_count,
            "negative_count": neg_count,
            "ratio": ratio,
            "event_rate": event_rate,
        }

        # Optional: Print each label's stats for verification/debugging
        print(
            f"{label} has {pos_count} positive examples, {neg_count} negative examples, "
            f"ratio of {ratio:.2f} negative to positive examples. "
            f"The event rate is {event_rate:.4f}."
        )

    return neg_to_pos_ratios_dict


def pad_collate(batch):
    """
    Custom collate function to pad the sequences in the batch for DataLoader.

    Args:
        batch (list): A list of tuples containing the features and labels for each sample.

    Returns:
        A tuple containing the padded features and labels tensors.
    """
    max_seq_len = max([len(item[0]) for item in batch])

    padded_features_batch = []
    padded_labels_batch = []

    for features, labels in batch:
        seq_len = len(features)
        pad_len = max_seq_len - seq_len
        padded_features = torch.cat(
            [
                features,
                torch.tensor([[-1e6] * features.shape[1]] * pad_len, dtype=torch.float),
            ]
        )
        padded_labels = torch.cat(
            [
                labels,
                torch.tensor([[-1] * labels.shape[1]] * pad_len, dtype=torch.float),
            ]
        )
        padded_features_batch.append(padded_features)
        padded_labels_batch.append(padded_labels)

    padded_features_batch = torch.stack(padded_features_batch)
    padded_labels_batch = torch.stack(padded_labels_batch)

    return padded_features_batch, padded_labels_batch


def masked_binary_cross_entropy_with_logits(logits, targets, mask, neg_to_pos_ratios):
    """
    Custom loss function that calculates the masked binary cross-entropy loss for the DL model.

    Args:
        logits (torch.Tensor): The logits output from the model.
        targets (torch.Tensor): The ground truth labels.
        mask (torch.Tensor): The mask for valid target values.
        neg_to_pos_ratios (list): The list of negative to positive ratios for each task.

    Returns:
        A scalar value representing the masked binary cross-entropy loss.
    """
    losses = 0
    for i in range(len(neg_to_pos_ratios)):
        # Create a mask for valid target values (not equal to -1)
        valid_targets_mask = targets[:, :, i] != -1
        # Filter out invalid elements from logits and targets using the mask
        filtered_logits = logits[:, :, i][valid_targets_mask]
        filtered_targets = targets[:, :, i][valid_targets_mask]
        # Calculate the loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            filtered_logits,
            filtered_targets,
            reduction="none",
            pos_weight=torch.tensor(neg_to_pos_ratios[i]),
        )
        masked_loss = loss * mask[:, :, i][valid_targets_mask]
        losses += masked_loss.sum() / mask[:, :, i][valid_targets_mask].sum()

    losses = losses / len(neg_to_pos_ratios)
    return losses


def calibrated_model(model, dataloader, neg_to_pos_ratios, output_size, device):
    model.eval()
    running_loss = [0.0] * output_size
    all_labels = [[] for _ in range(output_size)]
    all_preds = [[] for _ in range(output_size)]

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = (
                inputs.to(device, dtype=torch.float),
                labels.to(device, dtype=torch.float),
            )
            outputs = model(inputs)

            mask = (labels != -1).float()

            # put outputs into sigmoid
            outputs = torch.sigmoid(outputs)

            for j in range(output_size):
                current_mask = mask[:, :, j].float()
                current_labels = labels[:, :, j]
                current_outputs = outputs[:, :, j]

                if current_mask.sum().item() > 0:
                    task_loss = nn.functional.binary_cross_entropy_with_logits(
                        current_outputs,
                        current_labels,
                        reduction="none",
                        pos_weight=torch.tensor(neg_to_pos_ratios[j]),
                    )
                    task_loss = task_loss * current_mask
                    masked_loss = task_loss.sum() / current_mask.sum()
                    running_loss[j] += masked_loss

                current_mask = current_mask.bool()
                all_labels[j].append(
                    labels[:, :, j][current_mask].detach().cpu().numpy()
                )
                all_preds[j].append(
                    outputs[:, :, j][current_mask].detach().cpu().numpy()
                )

    ir_models = []
    for j in range(output_size):
        if len(all_labels[j]) > 0 and len(all_preds[j]) > 0:
            all_labels[j] = np.concatenate(all_labels[j])
            all_preds[j] = np.concatenate(all_preds[j])
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(all_preds[j], all_labels[j])
            ir_models.append(ir)
        else:
            ir_models.append(None)
    return ir_models


def train_dl(model, dataloader, optimizer, criterion, neg_to_pos_ratios, device):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = batch
        inputs, labels = (
            inputs.to(device, dtype=torch.float),
            labels.to(device, dtype=torch.float),
        )
        outputs = model(inputs)

        mask = (labels != -1).float()

        # Dont't put it to sigmoid here, because we need to use BCEWithLogitsLoss
        loss = criterion(outputs, labels, mask, neg_to_pos_ratios)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / (i + 1)


def validate_dl(
    model,
    dataloader,
    neg_to_pos_ratios,
    output_size,
    device,
    ir_models=None,
    ci=False,
    evaluate_net_benefit=False,
    calibration=False,
):
    model.eval()
    running_loss = [0.0] * output_size
    all_labels = [[] for _ in range(output_size)]
    all_preds = [[] for _ in range(output_size)]

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = (
                inputs.to(device, dtype=torch.float),
                labels.to(device, dtype=torch.float),
            )
            outputs = model(inputs)

            mask = (labels != -1).float()

            # put outputs into sigmoid
            outputs = torch.sigmoid(outputs)

            for j in range(output_size):
                current_mask = mask[:, :, j].float()
                current_labels = labels[:, :, j]
                current_outputs = outputs[:, :, j]

                if current_mask.sum().item() > 0:
                    task_loss = nn.functional.binary_cross_entropy_with_logits(
                        current_outputs,
                        current_labels,
                        reduction="none",
                        pos_weight=torch.tensor(neg_to_pos_ratios[j]),
                    )
                    task_loss = task_loss * current_mask
                    masked_loss = task_loss.sum() / current_mask.sum()
                    running_loss[j] += masked_loss

                current_mask = current_mask.bool()
                all_labels[j].append(
                    labels[:, :, j][current_mask].detach().cpu().numpy()
                )
                all_preds[j].append(
                    outputs[:, :, j][current_mask].detach().cpu().numpy()
                )

    individual_auc = []
    individual_aup = []

    # Add 95% confidence intervals for AUC and AUPRC
    individual_auc_ci = []
    individual_aup_ci = []

    # For plotting
    roc_curves = []
    pr_curves = []

    # Net Benefit
    net_benefit_df = []

    # Calibration
    calibration_curves = []

    for j in range(output_size):
        if len(all_labels[j]) > 0 and len(all_preds[j]) > 0:
            all_labels[j] = np.concatenate(all_labels[j])
            all_preds[j] = np.concatenate(all_preds[j])
            if calibration == True and ir_models is not None:
                calibrated_preds = ir_models[j].transform(all_preds[j])
                prob_true, prob_pred = calibration_curve(
                    all_labels[j], calibrated_preds, n_bins=10, strategy="quantile"
                )
                calibration_curves.append((prob_true, prob_pred))
                # set predictions to calibrated predictions
                all_preds[j] = calibrated_preds
            else:
                calibrated_preds = all_preds[j]

            # Compute ROC curve and ROC AUC for each class
            fpr, tpr, _ = roc_curve(all_labels[j], calibrated_preds)
            roc_curves.append((fpr, tpr))

            # Compute Precision-Recall curve and AUPRC for each class
            precision, recall, _ = precision_recall_curve(
                all_labels[j], calibrated_preds
            )
            pr_curves.append((precision, recall))

            if ci == True:
                # Compute bootstrapped 95% CI for AUC and AUP
                auc, auc_ci_lower, auc_ci_upper, aup, aup_ci_lower, aup_ci_upper = (
                    bootstrap_ci(all_labels[j], calibrated_preds)
                )

                individual_auc_ci.append((auc_ci_lower, auc_ci_upper))
                individual_aup_ci.append((aup_ci_lower, aup_ci_upper))

                individual_auc.append(auc)
                individual_aup.append(aup)

            else:
                individual_auc_ci.append((0, 0))
                individual_aup_ci.append((0, 0))

                individual_auc.append(roc_auc_score(all_labels[j], calibrated_preds))
                individual_aup.append(
                    average_precision_score(all_labels[j], calibrated_preds)
                )

            if evaluate_net_benefit == True:
                # Compute the net benefit and treat all
                fpr, tpr, thresh = roc_curve(all_labels[j], calibrated_preds)

                # the odds approaches infinity at these thresholds, let's remove them
                mask = thresh > 0.999
                fpr, tpr, thresh = fpr[~mask], tpr[~mask], thresh[~mask]

                # compute net benefit for model and treat all
                sensitivity, specificity, prevalence = (
                    tpr,
                    1 - fpr,
                    all_labels[j].mean(),
                )
                odds = thresh / (1 - thresh)
                net_benefit = (
                    sensitivity * prevalence
                    - (1 - specificity) * (1 - prevalence) * odds
                )
                treat_all = prevalence - (1 - prevalence) * odds
                thresh, net_benefit, treat_all = (
                    thresh[1:],
                    net_benefit[1:],
                    treat_all[1:],
                )

                df = pd.DataFrame(
                    data=np.array([thresh, net_benefit, treat_all]).T,
                    columns=["Threshold", "System", "All"],
                )
                net_benefit_df.append(df)
        else:
            individual_auc.append(np.nan)
            individual_auc_ci.append((np.nan, np.nan))
            individual_aup_ci.append((np.nan, np.nan))

        running_loss[j] /= i + 1

    return (
        running_loss,
        individual_auc,
        individual_aup,
        individual_auc_ci,
        individual_aup_ci,
        roc_curves,
        pr_curves,
        net_benefit_df,
        calibration_curves,
        all_preds,
        all_labels,
    )


def train_and_evaluate_dl(
    epochs,
    model,
    train_dataloader,
    tune_dataloader,
    test_dataloader,
    optimizer,
    criterion,
    device,
    output_size,
    labels_list,
    neg_to_pos_ratios,
    scheduler=None,
    early_stopping_patience=5,
    ci=False,
    my_copy=True,
    verbose=False,
    testing=False,
    calibration=False,
):
    train_losses = []
    train_aucs = [[] for _ in range(output_size)]
    train_aups = [[] for _ in range(output_size)]
    val_losses = [[] for _ in range(output_size)]
    val_aucs = [[] for _ in range(output_size)]
    val_aups = [[] for _ in range(output_size)]
    mean_val_losses = []
    all_preds_list = []
    all_labels_list = []

    best_val_loss = float("inf")
    best_epochs = 0
    epochs_since_best = 0
    ir_models = []
    models = []

    # Add 95% CI
    train_auc_cis = [[] for _ in range(output_size)]
    train_aup_cis = [[] for _ in range(output_size)]
    val_auc_cis = [[] for _ in range(output_size)]
    val_aup_cis = [[] for _ in range(output_size)]

    for epoch in range(epochs):
        train_loss = train_dl(
            model, train_dataloader, optimizer, criterion, neg_to_pos_ratios, device
        )

        # Ccalibrated model
        if calibration == True:
            ir_model_epoch = calibrated_model(
                model, tune_dataloader, neg_to_pos_ratios, output_size, device
            )
        else:
            ir_model_epoch = None

        if my_copy == True:
            model_copy = copy.deepcopy(model)
            models.append(model_copy)
            ir_models.append(ir_model_epoch)

        # Set calibrate to True when real training
        (
            train_loss_eval,
            train_auc_eval,
            train_aup_eval,
            train_auc_ci_eval,
            train_aup_ci_eval,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = validate_dl(
            model,
            train_dataloader,
            neg_to_pos_ratios,
            output_size,
            device,
            ir_models=ir_model_epoch,
            ci=ci,
            calibration=False,
        )

        if testing == True:
            (
                val_loss,
                val_auc,
                val_aup,
                val_auc_ci,
                val_aup_ci,
                _,
                _,
                _,
                _,
                all_preds,
                all_labels,
            ) = validate_dl(
                model,
                test_dataloader,
                neg_to_pos_ratios,
                output_size,
                device,
                ir_models=ir_model_epoch,
                ci=ci,
                calibration=True,
            )
        else:
            (
                val_loss,
                val_auc,
                val_aup,
                val_auc_ci,
                val_aup_ci,
                _,
                _,
                _,
                _,
                all_preds,
                all_labels,
            ) = validate_dl(
                model,
                tune_dataloader,
                neg_to_pos_ratios,
                output_size,
                device,
                ir_models=ir_model_epoch,
                ci=ci,
                calibration=False,
            )
        all_preds_list.append(all_preds)
        all_labels_list.append(all_labels)

        train_losses.append(train_loss)

        for i, (auc, aup, auc_ci, aup_ci) in enumerate(
            zip(train_auc_eval, train_aup_eval, train_auc_ci_eval, train_aup_ci_eval)
        ):
            train_aucs[i].append(auc)
            train_aups[i].append(aup)
            train_auc_cis[i].append(auc_ci)
            train_aup_cis[i].append(aup_ci)

        for i, (loss, auc, aup, auc_ci, aup_ci) in enumerate(
            zip(val_loss, val_auc, val_aup, val_auc_ci, val_aup_ci)
        ):
            val_losses[i].append(loss)
            val_aucs[i].append(auc)
            val_aups[i].append(aup)
            val_auc_cis[i].append(auc_ci)
            val_aup_cis[i].append(aup_ci)

        # Calculate mean validation loss across all tasks
        mean_val_loss = sum(val_loss[-1] for val_loss in val_losses) / output_size
        mean_val_losses.append(mean_val_loss)

        if verbose == True:
            if ci == False:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Mean Val Loss: {mean_val_loss:.4f}"
                )
                for i, (train_auc, train_aup, val_loss, val_auc, val_aup) in enumerate(
                    zip(train_aucs, train_aups, val_losses, val_aucs, val_aups)
                ):
                    print(
                        f"Task {i + 1} {labels_list[i]}: Train AUC: {train_auc[-1]:.4f}, Train AUP: {train_aup[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val AUC: {val_auc[-1]:.4f}, Val AUP: {val_aup[-1]:.4f}"
                    )
                print()
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Mean Val Loss: {mean_val_loss:.4f}"
                )
                for i, (
                    train_auc,
                    train_aup,
                    train_auc_ci,
                    train_aup_ci,
                    val_loss,
                    val_auc,
                    val_aup,
                    val_auc_ci,
                    val_aup_ci,
                ) in enumerate(
                    zip(
                        train_aucs,
                        train_aups,
                        train_auc_cis,
                        train_aup_cis,
                        val_losses,
                        val_aucs,
                        val_aups,
                        val_auc_cis,
                        val_aup_cis,
                    )
                ):
                    print(
                        f"Task {i + 1} {labels_list[i]}: Train AUC: {train_auc[-1]:.4f} (95% CI: {train_auc_ci[-1][0]:.4f}, {train_auc_ci[-1][1]:.4f}), Train AUP: {train_aup[-1]:.4f} (95% CI: {train_aup_ci[-1][0]:.4f}, {train_aup_ci[-1][1]:.4f}), Val Loss: {val_loss[-1]:.4f}, Val AUC: {val_auc[-1]:.4f} (95% CI: {val_auc_ci[-1][0]:.4f}, {val_auc_ci[-1][1]:.4f}), Val AUP: {val_aup[-1]:.4f} (95% CI: {val_aup_ci[-1][0]:.4f}, {val_aup_ci[-1][1]:.4f})"
                    )
                print()

        if scheduler is not None:
            # Update the learning rate using the scheduler
            # scheduler.step(mean_val_loss)
            scheduler.step()

        if mean_val_loss < best_val_loss:
            print(
                f"best val loss is {mean_val_loss} and previous loss is {best_val_loss}"
            ) if verbose == True else None
            best_val_loss = mean_val_loss
            # best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # If early stopping patience has been reached, stop training
        if epochs_since_best >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.") if verbose == True else None
            break

    if output_size == 1:
        best_epoch = np.argmax(val_aucs[0])  # find the epoch with the highest AUROC
    else:
        # Case when there are multiple outputs
        mean_aucs = [
            np.mean([val_aucs[i][epoch] for i in range(output_size)])
            for epoch in range(len(val_aucs[0]))
        ]
        best_epoch = np.argmax(mean_aucs)  # find the epoch with the highest mean AUROC

    # print(f"best epoch is {best_epoch+1}")
    return (
        train_losses,
        train_aucs,
        train_aups,
        val_losses,
        val_aucs,
        val_aups,
        mean_val_losses,
        train_auc_cis,
        train_aup_cis,
        val_auc_cis,
        val_aup_cis,
        ir_models,
        models,
        all_preds_list,
        all_labels_list,
    )


def bootstrap_ci(labels, predictions, n_iterations=100, alpha=0.95):
    """
    Compute bootstrap confidence intervals for AUROC and AUPRC.

    Args:
    - labels (array-like): True labels.
    - predictions (array-like): Model predictions or probabilities.
    - n_iterations (int): Number of bootstrap iterations.
    - alpha (float): Confidence level, e.g., 0.95 for 95% CI.

    Returns:
    - Tuple: AUROC, AUROC CI lower, AUROC CI upper, AUPRC, AUPRC CI lower, AUPRC CI upper
    """
    n = len(labels)
    auroc_scores = []
    auprc_scores = []

    # Bootstrapping
    for _ in range(n_iterations):
        sample_indices = np.random.choice(n, n, replace=True)
        sample_labels = labels[sample_indices]
        sample_preds = predictions[sample_indices]

        auroc_scores.append(roc_auc_score(sample_labels, sample_preds))
        auprc_scores.append(average_precision_score(sample_labels, sample_preds))

    # Calculating percentiles
    p_lower = ((1.0 - alpha) / 2.0) * 100
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100

    auroc_ci_lower = np.percentile(auroc_scores, p_lower)
    auroc_ci_upper = np.percentile(auroc_scores, p_upper)
    auprc_ci_lower = np.percentile(auprc_scores, p_lower)
    auprc_ci_upper = np.percentile(auprc_scores, p_upper)

    auroc = np.mean(auroc_scores)
    auprc = np.mean(auprc_scores)

    return auroc, auroc_ci_lower, auroc_ci_upper, auprc, auprc_ci_lower, auprc_ci_upper
