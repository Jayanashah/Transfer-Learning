import pandas as pd
import numpy as np
import os
import git
import torch
from src.models.dnn import CNN
from src.evaluation.training_population import train_dnn
from src.data.eeg_attention_pytorch_dataset import EegAttentionDataset
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import json

# Set up paths for checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List of all subjects
subjects = list(range(102,115))
subjects = subjects + list(range(116,117))
subjects = subjects + list(range(118,126))
subjects = subjects + [127, 128, 130]

# Sampling rate of the data
srate = 125

# Base directory and parameters
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
data_dir = os.path.join(base_dir, "data", "processed", "ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5")

model_id = "109"

# Define the trials directly as the indices from 8 to 19
trials = np.arange(8, 20)  # Trials from 8 to 19 (inclusive)

# Function to create random splits ensuring no indices are left
def create_random_splits():
    np.random.shuffle(trials)  # Shuffle the trials

    # Select one index for validation and one index for testing
    val_index = trials[0]
    test_index = trials[1]

    # Remove validation and test indices to get the training indices
    train_indices = np.delete(trials, [0, 1])

    return train_indices, [val_index], [test_index]

def get_best_model(checkpoint_base_path, cnn_hyperparameters):
    checkpoint_path = os.path.join(checkpoint_base_path)
    checkpoint_file = os.path.join(checkpoint_path, 'best_model.ckpt')

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    model = CNN(**cnn_hyperparameters)
    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict)
    return model  

def get_dnn_predictions(dnn, loader, device='cuda'): 
    predictions = []
    corr_class, false_class = 0, 0
    pearson_attended_list, pearson_distractor_list = [], []

    dnn.eval()
    dnn = dnn.to(device)

    with torch.no_grad():
        for x, y_attended, y_distractor in loader:
            x = x.to(device, dtype=torch.float)
            y_hat = dnn(x)
            predictions.append(y_hat.cpu().numpy())

            pearson_attended, _ = pearsonr(y_hat, y_attended)
            pearson_distractor, _ = pearsonr(y_hat, y_distractor)

            pearson_attended_list.append(pearson_attended)
            pearson_distractor_list.append(pearson_distractor)

            if pearson_attended > pearson_distractor:
                corr_class+=1
            elif pearson_attended < pearson_distractor:
                false_class+=1
            else:
                corr_class+=1
                false_class+=1
        acc = corr_class / (corr_class + false_class)
    return predictions, acc, np.mean(np.array(pearson_attended_list)), pearson_attended_list, np.mean(np.array(pearson_distractor_list)), pearson_distractor_list

# CNN training parameters
cnn_train_params = {"data_dir": data_dir, "lr": 0.00005, "batch_size": 256, "weight_decay": 1e-08, "epochs": 7, "early_stopping_patience": 7}
cnn_hyperparameters = {"dropout_rate": 0.35, "F1": 8, "D": 8, "F2": 64, "input_length": 100, "num_input_channels": 31}

feature = "env"

# Generate random train, validation, and test splits for all subjects
train_indices_all, val_indices_all, test_indices_all = [], [], []
for subject in subjects:
    train_indices, val_indices, test_indices = create_random_splits()

    # Append indices for each subject
    train_indices_all.extend([idx for idx in train_indices])
    val_indices_all.extend([idx for idx in val_indices])
    test_indices_all.extend([idx for idx in test_indices])

    # Print the splits for better understanding
    print("-" * 40)
    print(f"Subject {subject}:")
    print(f"  Train Indices: {sorted(train_indices)}")
    print(f"  Validation Indices: {sorted(val_indices)}")
    print(f"  Test Indices: {sorted(test_indices)}")
    print("-" * 40)

# Uncomment to train the model with the given splits and save checkpoints
# checkpoint_path = os.path.join(base_dir, "models", "cnn", "CI_Subject_independent_model")
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)

# Train the model for all subjects together using the random splits
# final_model, _ = train_dnn(
#     subject_string_train=[str(subject) for subject in subjects],
#     train_indices=train_indices_all, 
#     subject_string_val=[str(subject) for subject in subjects],
#     val_indices=val_indices_all,  
#     ica=False,
#     feature=feature,
#     mdl_checkpointing=True,
#     checkpoint_path=checkpoint_path,
#     **cnn_train_params,
#     **cnn_hyperparameters
# )


# Loop over decision windows for analysis

decision_windows = [2, 5, 10, 20, 30, 60]

for decision_window in decision_windows:
    print(f"Processing decision window: {decision_window}")

    # List to store all results across subjects for this decision window
    all_subject_results = []
    subject_specific_accuracy = []

    # Loop over each subject
    for subject in subjects:
        subject_results = {
            "subject": subject,
            "subject_specific_accuracy": subject_specific_accuracy,
            "results": []
        }

        # Loop over training, validation, and test indices
        for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
            trial = str(test_indices[0].item())  # Get the trial number
            
            # Load the best model for the current subject and trial
            model = get_best_model(os.path.join(base_dir, 'models', 'cnn', 'CI_Subject_independent_model'), cnn_hyperparameters)

            # Get the test dataset for this subject and trial
            test_dataset = EegAttentionDataset(dir_h5=data_dir, subj_string=[subject], trials_indices=test_indices, window_size_training=100, sampling_rate=125)
            test_dataloader = DataLoader(test_dataset, batch_size=decision_window * srate, shuffle=False, drop_last=True)

            # Get predictions and accuracy
            prediction, acc, pearson_attended_mean, pearson_attended_list, pearson_distractor_mean, pearson_distractor_list = get_dnn_predictions(model, test_dataloader, device='cpu')
            
            # Print results for each subject
            print(f"Subject: {subject}")
            print(f"Accuracy: {acc}")
            print(f"Mean Pearson Correlation (Attended): {pearson_attended_mean}")
            print(f"Mean Pearson Correlation (Distractor): {pearson_distractor_mean}")

            # Store results for the current subject
            subject_results["results"].append({
                "accuracy": acc,
                "mean_pearson_attended": pearson_attended_mean,
                "pearson_attended_list": pearson_attended_list,
                "mean_pearson_distractor": pearson_distractor_mean,
                "pearson_distractor_list": pearson_distractor_list,
                "predictions": [pred.tolist() for pred in prediction]
            })

        # Calculate subject-specific accuracy after all trials are completed
        accuracies = [trial["accuracy"] for trial in subject_results["results"]]
        subject_results["subject_specific_accuracy"] = [np.mean(accuracies[i:i+12]) for i in range(0, len(accuracies), 12)]
        print(f"Subject {subject} Specific Accuracy: {subject_results['subject_specific_accuracy']}")

        # Append subject results
        all_subject_results.append(subject_results)

    # Calculate the final accuracy across all subjects
    final_acc = np.mean([trial["accuracy"] for subject in all_subject_results for trial in subject["results"]])
    print(f"Final Accuracy across all subjects: {final_acc}")

    # Define the output folder for saving the results
    final_results_folder = os.path.join(base_dir, "results", "CI_Subject_independent_model")
    final_results_file = os.path.join(final_results_folder, f"CI_final_score{decision_window}s.json")

    # Create output folder if it doesn't exist
    if not os.path.exists(final_results_folder):
        os.makedirs(final_results_folder)

    # Save results to JSON file
    with open(final_results_file, "w") as f:
        json.dump({"final_accuracy": final_acc, "subjects": all_subject_results}, f, indent=4)

    print(f"Final accuracy and all subject results have been saved to '{final_results_file}'.")

print(f"Results for decision window {decision_window} saved in {final_results_file}")
