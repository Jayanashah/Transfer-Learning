import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import git
import numpy as np
import scipy.stats
import scipy.optimize

# Get the base directory using the git repository root
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

# ---------------------------------------- Subject-Specific Data ----------------------------------------

# List of JSON files containing accuracy results for subject-specific models
files1 = [
    ("CI_Subject_specific_model", "109", "CI_final_score2s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score5s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score10s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score20s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score30s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score60s.json"),
]

# Function to extract the final accuracy from a JSON file
def extract_final_accuracy(file_path):
    """
    Reads a JSON file and extracts the "final_accuracy" value.
    :param file_path: Path to the JSON file
    :return: Final accuracy value or None if not found
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("final_accuracy", None)

# List to store accuracy values for subject-specific models
ci_accuracies1 = []

# Iterate over the subject-specific model files and extract accuracy
for i in range(len(files1)):
    ci_path = os.path.join(base_dir, "results", *files1[i])  # Construct file path
    if os.path.exists(ci_path):  # Check if the file exists
        ci_acc = extract_final_accuracy(ci_path)  # Extract accuracy value
        if ci_acc is not None:
            ci_accuracies1.append(ci_acc * 100)  # Convert accuracy to percentage
    
# Print extracted accuracies for subject-specific models
print('ci_accuracies1:', ci_accuracies1)

# Compute mean and standard deviation for subject-specific models
ci_mean1 = np.mean(ci_accuracies1)
ci_std1 = np.std(ci_accuracies1)

# ---------------------------------------- Subject-Independent Data ----------------------------------------

# List of JSON files containing accuracy results for subject-independent models
ci_files = [
    ("CI_Subject_independent_model", "CI_final_score2s.json"),
    ("CI_Subject_independent_model", "CI_final_score5s.json"),
    ("CI_Subject_independent_model", "CI_final_score10s.json"),
    ("CI_Subject_independent_model", "CI_final_score20s.json"),
    ("CI_Subject_independent_model", "CI_final_score30s.json"),
    ("CI_Subject_independent_model", "CI_final_score60s.json")
]

# List to store accuracy values for subject-independent models
ci_accuracies2 = []

# Iterate over the subject-independent model files and extract accuracy
for i in range(len(ci_files)):
    ci_path = os.path.join(base_dir, "results", *ci_files[i])  # Construct file path
    if os.path.exists(ci_path):  # Check if the file exists
        ci_acc = extract_final_accuracy(ci_path)  # Extract accuracy value
        if ci_acc is not None:
            ci_accuracies2.append(ci_acc * 100)  # Convert accuracy to percentage

# Print extracted accuracies for subject-independent models
print('ci_accuracies2:', ci_accuracies2)

# Compute mean and standard deviation for subject-independent models
ci_mean2 = np.mean(ci_accuracies2)
ci_std2 = np.std(ci_accuracies2)

# ---------------------------------------- Fine Tune Partial ----------------------------------------

# Define the list of files for fine-tuned models with different decision window durations
files2 = [
    ("fine_tuned", "6e-06", "results2s.json"),
    ("fine_tuned", "6e-06", "results5s.json"),
    ("fine_tuned", "6e-06", "results10s.json"),
    ("fine_tuned", "6e-06", "results20s.json"),
    ("fine_tuned", "6e-06", "results30s.json"),
    ("fine_tuned", "6e-06", "results60s.json"),
]

# Function to extract final accuracy from JSON files
def extract_final_accuracy(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)  # Load JSON content
        return data.get("final_accuracy")  # Extract "final_accuracy" key, return None if missing
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return None

# List to store accuracy values for fine-tuned models
ci_accuracies3 = []

# Iterate through each fine-tuned model result file
for i in range(0, len(files2)):
    ci_path = os.path.join(base_dir, "results", *files2[i])  # Construct file path
    if os.path.exists(ci_path):  # Check if file exists
        ci_acc = extract_final_accuracy(ci_path)  # Extract accuracy value
        if ci_acc is not None:
            ci_accuracies3.append(ci_acc * 100)  # Convert accuracy to percentage

# Print extracted accuracy values
print('ci_accuracies3', ci_accuracies3)

# Compute mean and standard deviation of fine-tuned model accuracies
ci_mean3 = np.mean(ci_accuracies3)
ci_std3 = np.std(ci_accuracies3)
 
# -------------------------------------------------------------------------------------------------------------------------------
# Function to compute 95% chance interval threshold for accuracy
def ci_95_acc(n_window):
    p = 1.0  # Initial probability value
    k = int(np.ceil(n_window / 2))  # Define success threshold
    while p > 0.05:  # Find the minimum k where p-value drops below 0.05
        res = scipy.stats.binomtest(k, n_window, p=0.5, alternative='greater')
        p = res.pvalue
        k += 1
    return k / n_window  # Return the threshold as a proportion

# Define a square-root-based function for curve fitting
def parametrized_sqrt(x, a, b, c, d):
    return a * np.sqrt(b * x - c) + d

# Define number of trials, subjects, and decision windows (time intervals)
n_trial = 12  # Number of trials per subject
n_subjects = 28  # Number of subjects
decision_window = np.array([2, 5, 10, 20, 30, 60])  # Different decision time windows in seconds

# Calculate the number of decision windows for each trial configuration
n_windows = np.floor(120 / decision_window) * n_trial * n_subjects
n_windows = n_windows.astype(int)  # Convert to integer values

# Compute 95% confidence intervals for different decision windows
ci_95 = np.array([ci_95_acc(n) for n in n_windows])

# Fit a curve to the 95% confidence interval data
params_ci, pcov_ci = scipy.optimize.curve_fit(
    parametrized_sqrt, decision_window, ci_95, p0=[1, 1, 0.1, 0.5], method='trf'
)
sqrt_fit_ci = parametrized_sqrt(decision_window, *params_ci)

# Define color palette for the plots
palette = sns.color_palette("Set2")
ci_color1 = palette[0]  # Subject-specific model color
ci_color2 = palette[1]  # Subject-independent model color
ci_color3 = palette[2]  # Fine-tuned model color

# Labels for decision windows
decision_windows = ["2", "5", "10", "20", "30", "60"]

# -------------------------------------------------------------------------------------------------------------------------------
# Set Seaborn style for visualization
sns.set(style="whitegrid", font_scale=1.5)

# Create an x-axis for plotting
x = np.arange(len(decision_windows))

# Create a figure for the accuracy comparison plot
plt.figure(figsize=(17, 7))

# Plot accuracy results for subject-specific models
plt.plot(x, ci_accuracies1, label='Subject Specific', color=ci_color1)
plt.fill_between(x, np.array(ci_accuracies1) - ci_std1, np.array(ci_accuracies1) + ci_std1, color=ci_color1, alpha=0.2)

# Plot accuracy results for subject-independent models
plt.plot(x, ci_accuracies2, label='Subject Independent', color=ci_color2)
plt.fill_between(x, np.array(ci_accuracies2) - ci_std2, np.array(ci_accuracies2) + ci_std2, color=ci_color2, alpha=0.2)

# Plot accuracy results for fine-tuned models
plt.plot(x, ci_accuracies3, label='Fine Tuned', color=ci_color3)
plt.fill_between(x, np.array(ci_accuracies3) - ci_std3, np.array(ci_accuracies3) + ci_std3, color=ci_color3, alpha=0.2)

# Plot 95% confidence interval threshold as a dashed line
plt.plot(x, sqrt_fit_ci * 100, color="gray", label="95% Chance Level", linestyle="--")

# Configure x-axis labels and title
plt.xticks(x, decision_windows)
plt.xlabel("Decision Windows (seconds)")
plt.ylabel("Accuracy (%)")
plt.title("CI Variations")

# Add a legend to distinguish different models
plt.legend(loc="upper left")

# Set y-axis limits and enable grid lines
plt.ylim(40, 100)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Define output directory for saving the plot
output_dir = os.path.join(base_dir, "Plots", "Fine_tune_partial", '6e-06')
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the plot as an image file
plt.savefig(os.path.join(output_dir, "line_plot_CI_variations.png"))

# Display the plot
plt.show()
