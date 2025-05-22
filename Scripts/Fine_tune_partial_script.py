#Libraries
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

# List of all subjects in CI dataset
subjects = list(range(102,115))
subjects = subjects + list(range(116,117))
subjects = subjects + list(range(118,126))
subjects = subjects + [127, 128, 130]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
   
# sampling rate of the data
srate = 125

# Base directory and dataset directory
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
data_dir = os.path.join(base_dir, "data", "processed", "ci_attention_final_l_1,1_h_32,50_out_125_130_incl_ica.hdf5")

# The 12 trials of competing-speaker scenerio as the indices from 8 to 19 (inclusive)
trials = np.arange(8, 20)  

# CNN training hyperparameters
cnn_train_params = {"data_dir": data_dir, "batch_size": 256, "weight_decay": 1e-08, "epochs": 7, "early_stopping_patience": 7}
cnn_hyperparameters = {"dropout_rate": 0.35, "F1": 8, "D": 8, "F2": 64, "input_length": 100, "num_input_channels": 31}

def create_random_splits(): # Function to create random splits for population model ensuring no indices are left
    np.random.shuffle(trials)  # Shuffle the trials
    val_index = trials[0] # Select one index for validation 
    test_index = trials[1] # Select one index for testing
    train_index = np.delete(trials, [0, 1]) # Remaining indices as training indices
    return train_index, [val_index], [test_index]

def get_best_model(checkpoint_base_path, cnn_hyperparameters): #Loads the best saved CNN model from a checkpoint file.
    """
    Args:checkpoint_base_path (str): Path to the directory containing the checkpoint file.
         cnn_hyperparameters (dict): Dictionary of hyperparameters to initialize the CNN model.

    Returns:model (CNN): The CNN model with loaded weights, with conv1 and conv2 layers frozen.

    Raises:FileNotFoundError: If the checkpoint file is not found at the specified path.
    """

    # Check if the checkpoint file exists and the location of model saved
    checkpoint_path = os.path.join(checkpoint_base_path)
    checkpoint_file = os.path.join(checkpoint_path, 'best_model.ckpt')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    # Initialize the model with the given hyperparameters
    model = CNN(**cnn_hyperparameters)

    # Load the saved state dictionary into the model
    state_dict = torch.load(checkpoint_file)
    model.load_state_dict(state_dict)

    # Freeze the convolutional layers to prevent further training
    for param in model.conv1.parameters(): #Freeze conv1 layers
        param.requires_grad = False
    for param in model.conv2.parameters(): #Freeze conv2 layers
        param.requires_grad = False
    return model 

def get_dnn_predictions(dnn, loader, device='cuda'): #Computes predictions on a given data loader and evaluates performance using Pearson correlation.
    """
    Args:dnn (torch.nn.Module): The deep neural network model.
         loader (torch.utils.data.DataLoader): DataLoader providing batches of input data.
         device (str, optional): The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.

    Returns: tuple: 
            - predictions (list): List of model predictions.
            - acc (float): Accuracy based on Pearson correlation comparison.
            - mean_pearson_attended (float): Mean Pearson correlation with attended speech.
            - pearson_attended_list (list): List of Pearson correlations for attended speech.
            - mean_pearson_distractor (float): Mean Pearson correlation with distractor speech.
            - pearson_distractor_list (list): List of Pearson correlations for distractor speech.
    """
     predictions = []
     corr_class, false_class = 0, 0

     pearson_attended_list, pearson_distractor_list = [], []
    
     # Set model to evaluation mode
     dnn.eval()

     # Move model to specified device
     dnn = dnn.to(device)
     
     with torch.no_grad():
        for x, y_attended, y_distractor in loader:
            # Move input data to the specified device
            x = x.to(device, dtype=torch.float)

            # Get model predictions
            y_hat = dnn(x)
            predictions.append(y_hat.cpu().numpy())

            # Compute Pearson correlation
            pearson_attended, _ = pearsonr(y_hat, y_attended)
            pearson_distractor, _ = pearsonr(y_hat, y_distractor)

            pearson_attended_list.append(pearson_attended)
            pearson_distractor_list.append(pearson_distractor)
            
            # Classify as correct or false based on Pearson correlation
            if pearson_attended > pearson_distractor:
                corr_class+=1
            elif pearson_attended < pearson_distractor:
                false_class+=1
            else:
                corr_class+=1
                false_class+=1
        # Compute accuracy        
        acc = corr_class / (corr_class + false_class)
     return predictions, acc, np.mean(np.array(pearson_attended_list)), pearson_attended_list, np.mean(np.array(pearson_distractor_list)), pearson_distractor_list


# Different learning rates for fine-tuning
learning_rates = [1e-05,1e-06,3e-05,5e-06,6e-06,7e-06]

feature = "env"

# train_indices_all, val_indices_all, test_indices_all = [], [], [] # Generate random train, validation, and test splits for all subjects for Population model
# for subject in subjects:
#     train_indices, val_indices, test_indices = create_random_splits()
    
#     # Append indices for each subject (only the indices, not (subject, idx) tuple)
#     train_indices_all.extend([idx for idx in train_indices])
#     val_indices_all.extend([idx for idx in val_indices])
#     test_indices_all.extend([idx for idx in test_indices])

#     # Print the splits for better understanding
#     print("-" * 40)
#     print(f" Subject {subject}:")
#     print(f"  Train Indices: {sorted(train_indices)}")
#     print(f"  Validation Indices: {sorted(val_indices)}")
#     print(f"  Test Indices: {sorted(test_indices)}")
#     print("-" * 40)


# # Fine-tune NH model on CI data with different learning rates and save the model
# for lr in learning_rates:
#     print(f"Fine-tuning with learning rate: {lr}")
#     checkpoint_path_ci = os.path.join(base_dir, "models", "cnn", "fine_tune_partial", str(lr))
#     os.makedirs(checkpoint_path_ci, exist_ok=True)
    
#     fine_tuned_ci_model, _ = train_dnn(
#         subject_string_train=[str(subject) for subject in subjects],
#         train_indices=train_indices_all,
#         subject_string_val=[str(subject) for subject in subjects],
#         val_indices=val_indices_all,
#         feature="env",
#         mdl_checkpointing=True,
#         checkpoint_path=checkpoint_path_ci,
#         lr=lr,  # Apply varying learning rate
#         **cnn_train_params,
#         **cnn_hyperparameters
#     )

# The indices relate to the trials in the dataset for Subject Specific Model
test_indices_list = [np.array([i]) for i in range(8,20)]
# validation indices
val_indices_ = np.roll(np.array(test_indices_list), 1).reshape(-1).tolist()
val_indices_list = [np.array([i]) for i in val_indices_]
train_indices_list = [np.delete(np.arange(0,20), np.hstack((test_indices_list[i],val_indices_[i]))) for i in range(0,len(test_indices_list))]

# The CI-population model (NH population with CI data and freezed 2 layers) as base-model and fine-tune it for individual CI-subjects
# Loop through different learning rates for fine-tuning
for lr in learning_rates:
    print(f"Fine-tuning with learning rate: {lr}")
    decision_windows = [2, 5, 10, 20, 30, 60]  # Different decision window sizes
    
    for decision_window in decision_windows:
        print(f"Processing decision window: {decision_window}")

        all_subject_results = []  # Store results for all subjects
        subject_specific_accuracy = []  # Store subject-specific accuracy values

        for subject in subjects:  # Iterate over individual subjects
            subject_results = {
                "subject": subject,
                "subject_specific_accuracy": subject_specific_accuracy,
                "results": []
            }

            # Iterate through different train, validation, and test splits
            for train_indices, val_indices, test_indices in zip(train_indices_list, val_indices_list, test_indices_list):
                # Load the best pre-trained model for fine-tuning
                model = get_best_model(os.path.join(base_dir, 'models', 'cnn', 'fine_tune_partial', str(lr)), cnn_hyperparameters)
                
                # Create test dataset and dataloader
                test_dataset = EegAttentionDataset(
                    dir_h5=data_dir, 
                    subj_string=[subject], 
                    trials_indices=test_indices, 
                    window_size_training=100, 
                    sampling_rate=125
                )
                test_dataloader = DataLoader(test_dataset, batch_size=decision_window * srate, shuffle=False, drop_last=True)
                
                # Get predictions and evaluation metrics from the model
                prediction, acc, pearson_attended_mean, pearson_attended_list, pearson_distractor_mean, pearson_distractor_list = get_dnn_predictions(model, test_dataloader, device='cpu')
                
                # Print evaluation metrics
                print("Subject: ", subject)
                print("Accuracy:", acc)
                print("Mean Pearson Correlation (Attended):", pearson_attended_mean)
                print("Mean Pearson Correlation (Distractor):", pearson_distractor_mean)
                
                # Store results for the current trial
                subject_results["results"].append({
                    "accuracy": acc,
                    "mean_pearson_attended": pearson_attended_mean,
                    "pearson_attended_list": pearson_attended_list,
                    "mean_pearson_distractor": pearson_distractor_mean,
                    "pearson_distractor_list": pearson_distractor_list,
                    "predictions": [pred.tolist() for pred in prediction]  # Convert predictions to list format
                })

            # Compute subject-specific accuracy across trials
            accuracies = [trial["accuracy"] for trial in subject_results["results"]]
            subject_results["subject_specific_accuracy"] = [np.mean(accuracies[i:i+12]) for i in range(0, len(accuracies), 12)]
            print(f"Subject {subject} Specific Accuracy: {subject_results['subject_specific_accuracy']}")
            
            all_subject_results.append(subject_results)  # Append subject results to the final list

        # Compute final accuracy across all subjects
        final_acc = np.mean([trial["accuracy"] for subject in all_subject_results for trial in subject["results"]])
        print(f"Final Accuracy across all subjects: {final_acc}")

        # Define folder and filename to store results
        final_results_folder = os.path.join(base_dir, "results", "Fine_tune_partial", str(lr))
        final_results_file = os.path.join(final_results_folder, "results" + str(decision_window) + "s.json")

        # Create results folder if it does not exist
        if not os.path.exists(final_results_folder):
            os.makedirs(final_results_folder)

        # Save results to a JSON file
        with open(final_results_file, "w") as f:
            json.dump({"final_accuracy": final_acc, "subjects": all_subject_results}, f, indent=4)

        print(f"Final accuracy and all subject results have been saved to '{final_results_file}'.")
        print(f"Results for decision window {decision_window} saved in {final_results_file}")
    
    print(f"Fine-tuning done for learning rate: {lr}")
