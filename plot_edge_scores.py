import matplotlib.pyplot as plt
import numpy as np
plt.close()
# Get edge scores and detach from graph
edge_scores = self.edge_learner(edge_features).detach().cpu().numpy()

# Optional: print min/max to verify
print(f"Edge score range: {edge_scores.min():.4f} - {edge_scores.max():.4f}")

# Plot histogram with proper range
plt.hist(edge_scores, bins=100, range=(edge_scores.min(), edge_scores.max()), alpha=0.75, color='steelblue')
plt.xlabel("Edge Score")
plt.ylabel("Frequency")
plt.title("Histogram of Edge Scores")
plt.grid(True)
plt.savefig("edge_score_histogram.png")


import matplotlib.pyplot as plt

# Get colors from edge_learner
colors = self.edge_learner(edge_features).detach().cpu().numpy()  # ensure it's numpy and detached from torch

# Flatten inputs
x = dist_A[:, i_idx, j_idx].flatten().cpu().numpy()
y = dist_B[:, i_idx, j_idx].flatten().cpu().numpy()

# Create scatter plot
sc = plt.scatter(x, y, s=1.0, alpha=0.2, c=colors, cmap='viridis')  # You can change the colormap if needed

# Add colorbar
plt.colorbar(sc, label='Edge Score')  # Optional: add label

plt.xlabel("dist_A")
plt.ylabel("dist_B")
plt.title("Scatter plot with edge scores as color")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

b, k= top_k_values.shape
# Flatten tensors
before = top_k_values[0].squeeze().detach().cpu().numpy()    # shape: (300,)
after = graph_data.x[:k].squeeze().detach().cpu().numpy()      # shape: (300,)
indices = np.arange(len(before))  # [0, 1, ..., 299]

plt.figure(figsize=(10, 5))
plt.scatter(indices, before, label='Before (Top-K)', s=10, alpha=0.7)
plt.scatter(indices, after, label='After (graph_data.x)', s=10, alpha=0.7)
plt.xlabel("Node Index")
plt.ylabel("Feature Value (log scale)")
plt.yscale("log")
plt.title("Feature Values Before and After GNN Update (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('nodes_before_and_after2.png')


import torch
import matplotlib.pyplot as plt
import numpy as np

# Use only new_mask (where new soft correspondences > 0)
mask = soft_correspondences > 0

# Extract values
orig_vals = orig_soft_correspondences[mask]
new_vals = soft_correspondences[mask]
gt_vals = gt_distance[mask]

# Convert to numpy
orig_vals_np = orig_vals.detach().cpu().numpy()
new_vals_np = new_vals.detach().cpu().numpy()
gt_vals_np = gt_vals.detach().cpu().numpy()

#####
orig_vals_np /= orig_vals_np.sum()
new_vals_np /= new_vals_np.sum()

growth_vals_np = np.log10(new_vals_np/orig_vals_np)
plt.scatter(new_vals_np , gt_vals_np,  c=growth_vals_np)
plt.xscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("soft_corr_trendlines_manualfit.png", dpi=300)
plt.close()


# Log x values
log_orig_vals = np.log(orig_vals_np)
log_new_vals = np.log(new_vals_np)

# Fit linear trend lines (on log-x)
coeffs_orig = np.polyfit(log_orig_vals, gt_vals_np, deg=1)
coeffs_new = np.polyfit(log_new_vals, gt_vals_np, deg=1)

# Generate smooth x range
x_range = np.logspace(np.log10(min(new_vals_np)), np.log10(max(new_vals_np)), 500)
log_x_range = np.log(x_range)

# Evaluate trend lines
trend_orig = np.polyval(coeffs_orig, log_x_range)
trend_new = np.polyval(coeffs_new, log_x_range)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(orig_vals_np, gt_vals_np, color='blue', label='Original', alpha=0.4, s=10)
plt.scatter(new_vals_np, gt_vals_np, color='red', label='After Consistency', alpha=0.4, s=10)
plt.plot(x_range, trend_orig, color='blue', linestyle='--', label='Original Trend')
plt.plot(x_range, trend_new, color='red', linestyle='--', label='Refined Trend')
plt.xscale('log')
plt.xlabel("Soft Correspondence Value (log scale)")
plt.ylabel("Ground Truth Distance")
plt.title("Trend Comparison (Log X-axis, New Mask Only)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("soft_corr_trendlines_manualfit.png", dpi=300)
plt.close()
