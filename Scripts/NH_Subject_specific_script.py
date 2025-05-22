import os
import git
import numpy as np
import torch
from scipy.stats import pearsonr
import json
from src.models.dnn import CNN
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from torch.utils.data import DataLoader
from pathlib import Path
from src.evaluation.training_functions import train_dnn

# Determine whether to use GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

srate = 125  # Sampling rate of the data

# Model identifier for this hyperparameter setting
model_id = '109'

# Base directory to locate data
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
data_dir = os.path.join(base_dir, "..", "..", "..", "data_nfs", "shared", "CJ_Semeco_Datasets", "control_group_301_328.hdf5")

# Subject IDs and trial indices
subjects = list(range(301, 329))
test_indices_list = [np.array([i]) for i in range(8, 20)]  # Trials for testing

# Validation indices (rotate test indices for validation)
val_indices = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
val_indices_list = [np.array([i]) for i in val_indices]

# Training indices (excluding test and validation indices)
train_indices_list = [np.delete(np.arange(0, 20), np.hstack((test_indices_list[i], val_indices[i]))) for i in range(len(test_indices_list))]

# Hyperparameters for CNN training
cnn_train_params = {
    "data_dir": data_dir,
    "lr": 0.00005,
    "batch_size": 256,
    "weight_decay": 1e-08,
    "epochs": 10,
    "early_stopping_patience": 7
}

cnn_hyperparameters = {
    "dropout_rate": 0.35,
    "F1": 8,
    "D": 8,
    "F2": 64,
    "input_length": 100,
    "num_input_channels": 31
}

# Define feature extraction type
feature = 'env'

def get_best_model(checkpoint_base_path, cnn_hyperparameters):
    """
    Loads the best model from a checkpoint file.
    """
    checkpoint_file = os.path.join(checkpoint_base_path, 'best_model.ckpt')

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    # Load CNN model and checkpoint
    model = CNN(**cnn_hyperparameters)
    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict)
    return model

def get_dnn_predictions(dnn, loader, device='cuda'):
    """
    Evaluates the DNN on the provided DataLoader and calculates prediction accuracy and Pearson correlations.
    """
    predictions = []
    corr_class, false_class = 0, 0
    pearson_attended_list, pearson_distractor_list = [], []
    
    dnn.eval()  # Set model to evaluation mode
    dnn = dnn.to(device)
    
    with torch.no_grad():
        for x, y_attended, y_distractor in loader:
            x = x.to(device, dtype=torch.float)
            y_hat = dnn(x)
            predictions.append(y_hat.cpu().numpy())

            # Calculate Pearson correlation for attended and distractor stimuli
            pearson_attended, _ = pearsonr(y_hat, y_attended)
            pearson_distractor, _ = pearsonr(y_hat, y_distractor)

            pearson_attended_list.append(pearson_attended)
            pearson_distractor_list.append(pearson_distractor)

            if pearson_attended > pearson_distractor:
                corr_class += 1
            elif pearson_attended < pearson_distractor:
                false_class += 1
            else:
                corr_class += 1
                false_class += 1

        acc = corr_class / (corr_class + false_class)
    
    return predictions, acc, np.mean(pearson_attended_list), pearson_attended_list, np.mean(pearson_distractor_list), pearson_distractor_list

# for subject in subjects:
#     for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):

#        subject = str(subject)  # Convert subject from int to str
#        checkpoint_path = os.path.join(base_dir,'models','cnn', 'NH_Subject_specific_model', model_id, subject, 'test_ind_' + str(test_indices[0].item()))
#        if not os.path.exists(checkpoint_path):
#             os.makedirs(checkpoint_path)
          
#         #_, _ = train_dnn(subject_string_train = subject, checkpoint_path = checkpoint_path, train_indices = train_indices, 
#         #                         subject_string_val = [subject], val_indices = val_indices, ica=False, feature=feature, workers=0, mdl_checkpointing=True, 
#         #                         use_pretrained=False, **cnn_train_params, **cnn_hyperparameters)


# Loop over different decision windows
decision_windows = [2, 5, 10, 20, 30, 60]
for decision_window in decision_windows:
    print(f"Processing decision window: {decision_window}")

    all_subject_results = []
    subject_specific_accuracy = []
        
    # Iterate over each subject
    for subject in subjects:
        subject_results = {
            "subject": subject,
            "subject_specific_accuracy": subject_specific_accuracy,
            "results": [] 
        }

        # Iterate over training, validation, and test indices for each subject
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
            trial = str(test_indices[0].item())  # Get the trial number
        
            # Load the best model for this subject and trial
            model = get_best_model(os.path.join(base_dir, 'models', 'cnn', 'NH_Subject_specific_model', model_id, subject, 'test_ind_' + trial), cnn_hyperparameters)
            
            # Create dataset and dataloader for testing
            test_dataset = EegAttentionDataset(dir_h5=data_dir, subj_string=[subject], trials_indices=test_indices, window_size_training=100, sampling_rate=srate)
            test_dataloader = DataLoader(test_dataset, batch_size=decision_window * srate, shuffle=False, drop_last=True)

            # Get DNN predictions and accuracy
            prediction, acc, pearson_attended_mean, pearson_attended_list, pearson_distractor_mean, pearson_distractor_list = get_dnn_predictions(model, test_dataloader, device='cpu')
            
            # Print results for this subject
            print(f"Subject: {subject}")
            print(f"Accuracy: {acc}")
            print(f"Mean Pearson Correlation (Attended): {pearson_attended_mean}")
            print(f"Mean Pearson Correlation (Distractor): {pearson_distractor_mean}")
    
            # Save results for this trial
            subject_results["results"].append({
                "accuracy": acc,
                "mean_pearson_attended": pearson_attended_mean,
                "pearson_attended_list": pearson_attended_list,
                "mean_pearson_distractor": pearson_distractor_mean,
                "pearson_distractor_list": pearson_distractor_list,
                "predictions": [pred.tolist() for pred in prediction]
            })    

        # Calculate subject-specific accuracy
        accuracies = [trial["accuracy"] for trial in subject_results["results"]]
        subject_results["subject_specific_accuracy"] = [np.mean(accuracies[i:i + 12]) for i in range(0, len(accuracies), 12)]
        print(f"Subject {subject} Specific Accuracy: {subject_results['subject_specific_accuracy']}")
        
        all_subject_results.append(subject_results)

    # Calculate final accuracy across all subjects
    final_acc = np.mean([trial["accuracy"] for subject in all_subject_results for trial in subject["results"]])
    print(f"Final Accuracy across all subjects: {final_acc}")
        
    # Save results to JSON file
    final_results_folder = os.path.join(base_dir, "results", "NH_Subject_specific_model")
    final_results_file = os.path.join(final_results_folder, f"NH_final_score_{decision_window}s.json")

    # Ensure the directory exists
    if not os.path.exists(final_results_folder):
        os.makedirs(final_results_folder)

    # Write the results to a JSON file
    with open(final_results_file, "w") as f:
        json.dump({"final_accuracy": final_acc, "subjects": all_subject_results}, f, indent=4)

    print(f"Final accuracy and all subject results have been saved to '{final_results_file}'.")
    
print(f"Results for decision window {decision_window} saved in {final_results_file}")