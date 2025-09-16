import io
import tempfile
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from scipy.cluster import hierarchy
from PIL import Image

# from models.utils.collate import move_batch_to_device

def plot_mse_matrix(matrix: np.ndarray, save_dir: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='MSE')
    plt.title('MSE Matrix')
    plt.xlabel('Transformation Index')
    plt.ylabel('Transformation Index')
    plt.grid(False)
    plt.savefig(os.path.join(save_dir, 'mse_matrix.png')) 
    
    
def plot_hierarchical_clustering(linkage_matrix: np.ndarray, save_dir: str) -> None:
     # Plot dendrogram
    plt.figure(figsize=(8, 6))
    dendrogram = hierarchy.dendrogram(linkage_matrix)
    plt.title('Dendrogram of Hierarchical Clustering')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.axhline(y=1, color='r', linestyle='--')

    plt.savefig(os.path.join(save_dir, 'hierarchical_clustring.png'))  


def plot_clustered_rmse_fitness(rmse: list[float], fitness: list[float],
                                 cluster_assignments: np.ndarray,  save_dir: str) -> None:
    
    plt.figure(figsize=(8, 6))
    for i in range(len(rmse)):
        color = plt.cm.nipy_spectral(cluster_assignments[i] / float(np.max(cluster_assignments)))
        plt.scatter(rmse[i], fitness[i], color=color)
    
    unique_clusters = np.unique(cluster_assignments)
    centroids_feature1 = np.zeros(len(unique_clusters))
    centroids_feature2 = np.zeros(len(unique_clusters))
    for i, cluster in enumerate(unique_clusters):
        centroids_feature1[i] = np.mean(rmse[cluster_assignments == cluster])
        centroids_feature2[i] = np.mean(fitness[cluster_assignments == cluster])
        color = plt.cm.nipy_spectral(cluster / float(np.max(cluster_assignments)))
        plt.scatter(centroids_feature1[i], centroids_feature2[i], color=color, marker='x', s=100)
        
    plt.title('Clustered transformations')
    plt.xlabel('RMSE')
    plt.ylabel('Coverage')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'rasnac_results_clusters.png'))  



def plot_gamma(combined_mask, gamma, postfix=""):
    plt.clf()

    path = f'plots/{postfix}_gamma.png'
    if os.path.exists(path):
        return

    non_zero_rows = combined_mask[0].sum(dim=1) != 0
    non_zero_cols = combined_mask[0].sum(dim=0) != 0

    # Mask the gamma matrix
    masked_l2 = gamma[0][non_zero_rows][:, non_zero_cols]

    # Plot the gamma matrix
    plt.title(f"Gamma matrix for soft BB")
    plt.xlabel(f"source embedding")
    plt.ylabel(f"target embedding")

    plt.imshow(masked_l2, cmap='viridis', interpolation='none')
    plt.colorbar()

    # Identify the top 5 (x, y) values
    top_k = 5
    flattened_indices = torch.topk(masked_l2.flatten(), top_k).indices
    top_values = [(int(idx // masked_l2.shape[1]), int(idx % masked_l2.shape[1])) for idx in flattened_indices]

    # Display the top 5 (x, y) values as a list above the plot
    top_list_str = f"Top {top_k} (x, y) values: {top_values}"
    plt.figtext(0.5, 0.95, top_list_str, ha='center', fontsize=10, wrap=True)

    # Save the plot
    plt.savefig(path, dpi=500)
    plt.show()

def generate_and_log_scatter_plot(metrics):
    """
    Generate a scatter plot of CATH degree vs Pocket RMSD and return as a PIL Image.

    Args:
        metrics (dict): The dictionary containing the metrics (e.g., 'cath_degree_per_sample' and 'pocket_rmsd_per_sample').
        
    Returns:
        Image: A PIL Image object of the scatter plot.
    """
    cath_degrees = metrics['cath_degree_per_sample']
    pocket_rmsds = metrics['pocket_rmsd_per_sample']

    # Calculate mean and std for each CATH degree
    unique_degrees = sorted(set(cath_degrees))
    means, stds = [], []
    for degree in unique_degrees:
        rmsd_values = [rmsd for d, rmsd in zip(cath_degrees, pocket_rmsds) if d == degree]
        means.append(np.mean(rmsd_values))
        stds.append(np.std(rmsd_values))

    plt.figure(figsize=(10, 6))
    plt.scatter(cath_degrees, pocket_rmsds, c=cath_degrees, cmap='viridis', alpha=0.5, label="Data Points")
    plt.errorbar(unique_degrees, means, yerr=stds, fmt='o', color='red', label="Mean ± STD", capsize=5)

    # Add titles and labels
    plt.title('Pocket RMSD')
    plt.xlabel('CATH Degree')
    plt.ylabel('Pocket RMSD')
    plt.legend()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        plt.savefig(temp_file.name, format='png', bbox_inches='tight')
        plt.close()
    return temp_file.name