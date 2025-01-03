import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
file1 = "results/validation_results/weight-loss-per-sample-atom-dist/Protein_RMSD_Results_39.csv"
file2 = "results/validation_results/delete-TM-align-save-in-csv/Protein_RMSD_Results_0.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Take the intersection based on the specified columns, renaming only 'Pocket RMSD'
columns_to_match = ['ligand', 'src protein', 'tar protein']
intersection = pd.merge(
    df1, 
    df2, 
    on=columns_to_match, 
    suffixes=('', '_file2')
)

# Ensure 'Pocket RMSD' from df1 and df2 are distinguishable
intersection.rename(columns={'Pocket RMSD': 'Pocket RMSD_softbbs', 'Pocket RMSD_file2': 'Pocket RMSD_TMalign'}, inplace=True)

# Bin the 'bbr' column into intervals of 0.05
bins = np.arange(0, intersection['bbr'].max() + 0.02, 0.02)
intersection['bbr_bin'] = pd.cut(intersection['bbr'], bins=bins, include_lowest=True)

# Group by the bins and calculate mean Pocket RMSD for each file
binned_means = intersection.groupby('bbr_bin').agg({
    'Pocket RMSD_softbbs': 'mean',
    'Pocket RMSD_TMalign': 'mean'
}).reset_index()

# Convert bins to the center of the intervals for plotting
binned_means['bbr_center'] = binned_means['bbr_bin'].apply(lambda x: x.mid)

# Plot the mean Pocket RMSD for each bin
plt.figure(figsize=(10, 6))
plt.scatter(
    binned_means['bbr_center'], 
    binned_means['Pocket RMSD_softbbs'], 
    label='SoftBBS', 
    color='blue', 
    marker='o'
)
plt.scatter(
    binned_means['bbr_center'], 
    binned_means['Pocket RMSD_TMalign'], 
    label='TM-align', 
    color='red', 
    marker='x'
)

# Add labels, title, and legend
plt.xlabel('BBR (Binned)')
plt.ylabel('Mean Pocket RMSD')
plt.title('Mean Pocket RMSD per Binned BBR')
plt.legend()
plt.grid()

# Save and show the plot
plt.savefig("Binned_Pocket_RMSD_vs_BBR.png")
plt.show()
