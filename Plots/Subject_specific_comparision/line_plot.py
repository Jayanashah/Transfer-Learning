import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import git
import numpy as np
import scipy.stats
import scipy.optimize

# Get the base directory of the git repository
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

# Define paths to the JSON files for Cochlear Implant (CI) and Normal Hearing (NH) data
files = [
    ("CI_Subject_specific_model", "109", "CI_final_score2s.json"),
    ("NH_Subject_specific_model", "109", "NH_final_score2s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score5s.json"),
    ("NH_Subject_specific_model", "109", "NH_final_score5s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score10s.json"),
    ("NH_Subject_specific_model", "109", "NH_final_score10s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score20s.json"),
    ("NH_Subject_specific_model", "109", "NH_final_score20s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score30s.json"),
    ("NH_Subject_specific_model", "109", "NH_final_score30s.json"),
    ("CI_Subject_specific_model", "109", "CI_final_score60s.json"),
    ("NH_Subject_specific_model", "109", "NH_final_score60s.json")
]

# Function to load accuracies from JSON files
def load_accuracies(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Extract accuracy for each subject
    return [acc for subject in data["subjects"] for acc in subject["subject_specific_accuracy"]]

# Load accuracies from all files
accuracies = [load_accuracies(os.path.join(base_dir, "results", *f)) for f in files]

# Define decision windows (time intervals in seconds)
decision_windows = ["2", "5", "10", "20", "30", "60"]

# Function to calculate the 95% chance level accuracy based on binomial test
def ci_95_acc(n_window):
    p = 1.0
    # Success threshold
    k = int(np.ceil(n_window / 2))
    while p > 0.05:
        res = scipy.stats.binomtest(k, n_window, p=0.5, alternative='greater')
        p = res.pvalue
        k += 1
    return k / n_window

# Function to fit a square root curve for the CI data
def parametrized_sqrt(x, a, b, c, d):
    return a * np.sqrt(b * x - c) + d

# Function to extract the final accuracy from a given JSON file
def extract_final_accuracy(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Return the final accuracy value
    return data.get("final_accuracy", None)

# Define the number of trials and subjects
n_trial = 12
n_subjects = 28

# Define decision windows (time intervals in seconds)
decision_windows = np.array([2, 5, 10, 20, 30, 60])

# Calculate the number of windows
n_windows = np.floor(120 / decision_windows) * n_trial * n_subjects
n_windows = n_windows.astype(int)

# Calculate the 95% confidence interval (CI)
ci_95 = np.array([ci_95_acc(n) for n in n_windows])

# Fit a curve to the CI data using a square root function
params_ci, pcov_ci = scipy.optimize.curve_fit(parametrized_sqrt, decision_windows, ci_95, p0=[1, 1, 0.1, 0.5], method='trf')
sqrt_fit_ci = parametrized_sqrt(decision_windows, *params_ci)

# Extract accuracies for CI and NH subjects
ci_accuracies = []
nh_accuracies = []

# Iterate through the files and extract accuracies for both CI and NH
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

# Calculate the mean and standard deviation for CI and NH accuracies
ci_mean = np.mean(ci_accuracies)
ci_std = np.std(ci_accuracies)

nh_mean = np.mean(nh_accuracies)
nh_std = np.std(nh_accuracies)

# Set color palette for the plot
palette = sns.color_palette("Set2")
ci_color = palette[0]
nh_color = palette[1]

# Set the seaborn style for the plot
sns.set(style="whitegrid", font_scale=1.5)

# Create an x-axis array corresponding to the decision windows
x = np.arange(len(decision_windows))

# Create the plot
plt.figure(figsize=(12, 7))

# Plot the accuracies for CI and NH
plt.plot(x, ci_accuracies, label='Cochlear Implant (CI)', color=ci_color)
plt.fill_between(x, np.array(ci_accuracies) - ci_std, np.array(ci_accuracies) + ci_std , color=ci_color, alpha=0.2)
plt.plot(x, nh_accuracies, label='Normal Hearing (NH)', color=nh_color)
plt.fill_between(x, np.array(nh_accuracies) - nh_std, np.array(nh_accuracies) + nh_std , color=nh_color, alpha=0.2)

# Plot the 95% chance level line
plt.plot(x, sqrt_fit_ci * 100, color="gray", label="95% Chance Level", linestyle="--")

# Customize x-axis ticks and labels
plt.xticks(x, decision_windows)
plt.xlabel("Decision Windows (seconds)")
plt.ylabel("Accuracy (%)")

# Add legend to the plot
plt.legend(loc="upper left")

# Set the y-axis limits and add gridlines
plt.ylim(40, 100) 
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Define the output directory for the plot
output_dir = os.path.join(base_dir, "Plots", "Subject_specific_comparision")
os.makedirs(output_dir, exist_ok=True)

# Save and display the plot
plt.savefig(os.path.join(output_dir, "line_plot.png"))
plt.show()
