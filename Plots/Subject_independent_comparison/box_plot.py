import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import git
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

# Get the base directory using git repo
base_dir = git.Repo('.', search_parent_directories=True).working_tree_dir

# Define the list of files to load, where each file corresponds to a specific model and decision window
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

# Function to load data from JSON files
def load_accuracies(file_path):
    """Load the accuracies from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return [acc for subject in data["subjects"] for acc in subject["subject_specific_accuracy"]]

# Load all accuracies from the list of files
accuracies = [load_accuracies(os.path.join(base_dir, "results", *f)) for f in files]
# Define decision windows
decision_windows = ["2", "5", "10", "20", "30", "60"]

# Create DataFrames for boxplots and list to store t-test results
df_list = []
t_test_results = []
p_values = []

# Loop through each decision window, calculate accuracies, and store the results
for i, window in enumerate(decision_windows):
    # Separate accuracies for Cochlear Implant (CI) and Normal Hearing (NH) groups
    ci_accuracies = [acc * 100 for acc in accuracies[i*2]]  # Cochlear Implant group
    nh_accuracies = [acc * 100 for acc in accuracies[i*2 + 1]]  # Normal Hearing group
    print(ci_accuracies, nh_accuracies)
    
    # Create a DataFrame for each window, combining CI and NH accuracies
    df = pd.DataFrame({
        "Accuracy": ci_accuracies + nh_accuracies,
        "Source": ["CI"] * len(ci_accuracies) + ["NH"] * len(nh_accuracies),
        "Label": ["CI"] * len(ci_accuracies) + ["NH"] * len(nh_accuracies)
    })
    
    # Append DataFrame for boxplot and window for t-test results
    df_list.append((df, window))
    
    # Perform unpaired t-test to compare CI and NH accuracies
    t_stat, p_val = scipy.stats.ttest_ind(ci_accuracies, nh_accuracies, equal_var=False)
    degrees_of_freedom = len(ci_accuracies) + len(nh_accuracies) - 2  # Degrees of freedom for t-test
    t_test_results.append((window, t_stat, p_val, degrees_of_freedom))
    p_values.append(p_val)

# Apply FDR correction using statsmodels to adjust p-values for multiple comparisons
rejected, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

# Define output directory for saving plots
output_dir = os.path.join(base_dir, "Plots", "Subject_independent_comparison")
os.makedirs(output_dir, exist_ok=True)

# Print t-test results with FDR correction applied
print("T-Test Results (Unpaired) with FDR Correction:")
for (window, t_stat, _, dof), p_val_corr, reject in zip(t_test_results, p_values_corrected, rejected):
    print(f"Decision Window: {window}s - T-Statistic: {t_stat:.4f}, Corrected P-Value: {p_val_corr:.4f}, Reject H0: {reject}, Degrees of Freedom: {dof}")

# Create subplots for all decision windows
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 grid for subplots
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Set Seaborn style and font scale
sns.set(style="whitegrid", font_scale=1.5)  # Base font scale for Seaborn

# Loop through each window and corresponding DataFrame to plot
for i, ((data, window), (win, t_stat, p_val, _)) in enumerate(zip(df_list, t_test_results)):
    # Create boxplot for CI and NH accuracies for the current window
    sns.boxplot(
        x="Label", y="Accuracy", data=data, palette="Set2", hue="Label", dodge=False,
        medianprops={"color": "black", "linewidth": 1.4}, ax=axes[i]
    )

    # Add significance bar and asterisk if p-value < 0.05 (indicating significant difference)
    if reject:
        x1, x2 = 0, 1  # Positions of CI and NH boxes
        y_max = max(data["Accuracy"])
        h = 2  # Height of the significance line
        y = y_max + h  # Y position for the significance line
        axes[i].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black")  # Significance line
        axes[i].text((x1 + x2) * 0.5, y + h, "***", ha="center", va="bottom", color="black", fontsize=18)  # Add text for significance

    # Customize axes and labels
    axes[i].set_ylim(45, 110)  # Full plot range for y-axis
    axes[i].set_yticks(range(50, 110, 10))  # Y-axis labels limited to 50-100 for clarity
    axes[i].grid(axis="y", linestyle="--", alpha=0.7)  # Y-axis grid lines for readability

    # Titles and labels
    title = f"Decision Window Size - {window}s"
    axes[i].set_title(title, fontsize=18)  # Set title font size explicitly

    # Adjust axis label font sizes
    axes[i].set_xlabel("Cohort", fontsize=18)
    axes[i].set_ylabel("Accuracy (%)", fontsize=18)

    # Adjust tick label font sizes (both x and y ticks)
    axes[i].tick_params(axis="x", labelsize=17)
    axes[i].tick_params(axis="y", labelsize=17)

# Adjust layout and save the figure
plt.tight_layout()  # Adjust layout for better spacing
plt.savefig(os.path.join(output_dir, "box_plot.png"))  # Save the plot as a PNG file
plt.show()  # Show the plot
