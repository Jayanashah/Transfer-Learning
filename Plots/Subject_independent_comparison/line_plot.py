import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import git
import numpy as np
import scipy.stats
import scipy.optimize

# Get the base directory using the git repository
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

# Define a list of tuples containing file paths for both CI (Cochlear Implant) and NH (Normal Hearing) data
files = [ 
    ("CI_Subject_independent_model", "CI_final_score2s.json"),
    ("NH_Subject_independent_model", "NH_final_score2s.json"),
    ("CI_Subject_independent_model", "CI_final_score5s.json"),
    ("NH_Subject_independent_model", "NH_final_score5s.json"),
    ("CI_Subject_independent_model", "CI_final_score10s.json"),
    ("NH_Subject_independent_model", "NH_final_score10s.json"),
    ("CI_Subject_independent_model", "CI_final_score20s.json"),
    ("NH_Subject_independent_model", "NH_final_score20s.json"),
    ("CI_Subject_independent_model", "CI_final_score30s.json"),
    ("NH_Subject_independent_model", "NH_final_score30s.json"),
    ("CI_Subject_independent_model", "CI_final_score60s.json"),
    ("NH_Subject_independent_model", "NH_final_score60s.json")
]

# Function to load accuracy data from JSON files
def load_accuracies(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Extract and return the subject-specific accuracies from the data
    return [acc for subject in data["subjects"] for acc in subject["subject_specific_accuracy"]]

# Load all accuracies from the JSON files into a list
accuracies = [load_accuracies(os.path.join(base_dir, "results", *f)) for f in files]

# Define decision windows for plotting and analysis
decision_windows = ["2", "5", "10", "20", "30", "60"]

# Function to calculate the 95% chance level accuracy based on a binomial distribution
def ci_95_acc(n_window):
    p = 1.0
    k = int(np.ceil(n_window / 2))  # Success threshold for binomial test
    while p > 0.05:  # Continue until the p-value is less than 0.05
        res = scipy.stats.binomtest(k, n_window, p=0.5, alternative='greater')
        p = res.pvalue
        k += 1
    return k / n_window  # Return the calculated 95% chance level

# Function for square root fitting
def parametrized_sqrt(x, a, b, c, d):
    return a * np.sqrt(b * x - c) + d  # A square root curve model

# Extract final accuracy from the JSON files for CI and NH groups
def extract_final_accuracy(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("final_accuracy", None)  # Return the final accuracy value

# Define the number of trials and subjects, calculate decision window accuracies
n_trial = 12  # Number of trials per decision window
n_subjects = 28  # Number of subjects in the experiment
decision_windows = np.array([2, 5, 10, 20, 30, 60])  # Time windows in seconds
n_windows = np.floor(120 / decision_windows) * n_trial * n_subjects  # Number of windows per trial
n_windows = n_windows.astype(int)
ci_95 = np.array([ci_95_acc(n) for n in n_windows])  # Calculate the 95% chance level for each window

# Fit a square root curve to the calculated 95% CI data
params_ci, pcov_ci = scipy.optimize.curve_fit(parametrized_sqrt, decision_windows, ci_95, p0=[1, 1, 0.1, 0.5], method='trf')
sqrt_fit_ci = parametrized_sqrt(decision_windows, *params_ci)  # Compute the fitted curve

# Extract mean accuracies and standard deviations for both CI and NH groups
ci_accuracies = []
nh_accuracies = []

# Loop through files to extract accuracy data for CI and NH groups
for i in range(0, len(files), 2):
    ci_path = os.path.join(base_dir, "results", *files[i])
    if os.path.exists(ci_path):
        ci_acc = extract_final_accuracy(ci_path)
        if ci_acc is not None:
            ci_accuracies.append(ci_acc * 100)  # Convert to percentage
    
    nh_path = os.path.join(base_dir, "results", *files[i + 1])
    if os.path.exists(nh_path):
        nh_acc = extract_final_accuracy(nh_path)
        if nh_acc is not None:
            nh_accuracies.append(nh_acc * 100)  # Convert to percentage

# Calculate mean and standard deviation for both CI and NH groups
ci_mean = np.mean(ci_accuracies)
ci_std = np.std(ci_accuracies)
nh_mean = np.mean(nh_accuracies)
nh_std = np.std(nh_accuracies)

# Define color palette for plotting
palette = sns.color_palette("Set2")
ci_color = palette[0]
nh_color = palette[1]

# Print out CI and NH accuracies for verification
print(nh_accuracies, ci_accuracies)

# Set up Seaborn plot styling and initialize the plot
sns.set(style="whitegrid", font_scale=1.5)
x = np.arange(len(decision_windows))  # x-axis positions for the plot

plt.figure(figsize=(12, 7))  # Set figure size
# Plot the CI and NH accuracies with filled regions for standard deviations
plt.plot(x, ci_accuracies, label='Cochlear Implant (CI)', color=ci_color)
plt.fill_between(x, np.array(ci_accuracies) - ci_std, np.array(ci_accuracies) + ci_std, color=ci_color, alpha=0.2)
plt.plot(x, nh_accuracies, label='Normal Hearing (NH)', color=nh_color)
plt.fill_between(x, np.array(nh_accuracies) - nh_std, np.array(nh_accuracies) + nh_std, color=nh_color, alpha=0.2)

# Plot the fitted 95% chance level curve
plt.plot(x, sqrt_fit_ci * 100, color="gray", label="95% Chance Level", linestyle="--")
plt.xticks(x, decision_windows)  # Set x-axis tick labels
plt.xlabel("Decision Windows (seconds)")  # x-axis label
plt.ylabel("Accuracy (%)")  # y-axis label
plt.legend(loc="upper left")  # Legend for the plot
plt.ylim(40, 100)  # Set the y-axis limits
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid lines for y-axis

# Save the plot to the output directory and display it
output_dir = os.path.join(base_dir, "Plots", "Subject_independent_comparison")
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
plt.savefig(os.path.join(output_dir, "line_plot.png"))  # Save the plot as a PNG file
plt.show()  # Display the plot
