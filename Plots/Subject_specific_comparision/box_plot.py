import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import git
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

# Get the base directory using git repo to work with files in the repository
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

# Define paths to the JSON files containing accuracy data for CI (Cochlear Implant) and NH (Normal Hearing) groups
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

# Function to load accuracy data from JSON files
def load_accuracies(file_path):
    """Load the accuracies from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return [acc for subject in data["subjects"] for acc in subject["subject_specific_accuracy"]]

# Load all accuracies from the defined JSON files
accuracies = [load_accuracies(os.path.join(base_dir, "results", *f)) for f in files]
# Define decision windows
decision_windows = ["2", "5", "10", "20", "30", "60"]

# Create empty lists for storing dataframes and t-test results
df_list = []
t_test_results = []
p_values = []

# Loop through each decision window (2s, 5s, 10s, etc.)
for i, window in enumerate(decision_windows):
    # Extract accuracies for CI and NH groups for the current window size
    ci_accuracies = [acc * 100 for acc in accuracies[i*2]]  # Cochlear Implant group
    nh_accuracies = [acc * 100 for acc in accuracies[i*2 + 1]]  # Normal Hearing group
    
    # Create a dataframe for boxplot with accuracy values and labels for CI and NH
    df = pd.DataFrame({
        "Accuracy": ci_accuracies + nh_accuracies,
        "Source": ["CI"] * len(ci_accuracies) + ["NH"] * len(nh_accuracies),
        "Label": ["CI"] * len(ci_accuracies) + ["NH"] * len(nh_accuracies)
    })
    df_list.append((df, window))  # Store the dataframe for the current window
    
    # Perform unpaired t-test to compare CI and NH accuracies
    t_stat, p_val = scipy.stats.ttest_ind(ci_accuracies, nh_accuracies, equal_var=False)
    degrees_of_freedom = len(ci_accuracies) + len(nh_accuracies) - 2  # Degrees of freedom for t-test
    t_test_results.append((window, t_stat, p_val, degrees_of_freedom))
    p_values.append(p_val)

# Apply FDR correction using statsmodels to adjust p-values for multiple comparisons
rejected, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# Define output directory for saving plots
output_dir = os.path.join(base_dir, "Plots", "Subject_specific_comparision")
os.makedirs(output_dir, exist_ok=True)

# Print t-test results with FDR correction applied
print("T-Test Results (Unpaired) with FDR Correction:")
for (window, t_stat, _, dof), p_val_corr, reject in zip(t_test_results, p_values_corrected, rejected):
    print(f"Decision Window: {window}s - T-Statistic: {t_stat:.4f}, Corrected P-Value: {p_val_corr:.4f}, Reject H0: {reject}, Degrees of Freedom: {dof}")

# Create subplots for each decision window to show boxplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows and 3 columns for subplots
axes = axes.flatten()  # Flatten the axes array to loop over each subplot

sns.set(style="whitegrid", font_scale=1.5)  # Set style for seaborn

# Loop through each decision window and plot the boxplot
for i, ((data, window), p_val_corr, reject) in enumerate(zip(df_list, p_values_corrected, rejected)):
    sns.boxplot(
        x="Label", y="Accuracy", data=data, palette="Set2", hue="Label", dodge=False,
        medianprops={"color": "black", "linewidth": 1.4}, ax=axes[i]  # Set boxplot properties
    )

    # Add significance marker if the p-value is less than 0.05 after FDR correction
    if reject:
        x1, x2 = 0, 1  # Positions of CI and NH boxes
        y_max = max(data["Accuracy"])
        h = 2  # Height of the significance line
        y = y_max + h  # Y position for the significance line
        axes[i].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black")  # Significance line
        axes[i].text((x1 + x2) * 0.5, y + h, "***", ha="center", va="bottom", color="black", fontsize=18)  # Add text for significance

    # Customize plot appearance
    axes[i].set_ylim(45, 110)  # Y-axis limits for accuracy percentages
    axes[i].set_yticks(range(50, 110, 10))  # Set Y-axis ticks
    axes[i].grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for y-axis
   
    axes[i].set_title(f"Decision Window Size - {window}s", fontsize=18)  # Title for each subplot
   
    axes[i].set_xlabel("Cohort", fontsize=18)  # X-axis label
    axes[i].set_ylabel("Accuracy (%)", fontsize=18)  # Y-axis label
   
    axes[i].tick_params(axis="x", labelsize=17)  # Set font size for X ticks
    axes[i].tick_params(axis="y", labelsize=17)  # Set font size for Y ticks

# Adjust the layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "box_plot.png"))
plt.show()  # Display the plot
