import io
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from scipy.cluster import hierarchy
from PIL import Image

from models.utils.collate import move_batch_to_device

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

def plot_transformed_point_clouds(batch, transformations, loss_value: float | None = None, batch_idx=0, compose: bool=True):
    plt.ioff()
    batch = move_batch_to_device(batch, 'cpu')
    transformations = move_batch_to_device(transformations, 'cpu')
    plt.clf()
    
    path = f"plots/{batch['metadata'][0]['mov_protein']}_to_{batch['metadata'][0]['ref_protein']}_transformations.png"
    if os.path.exists(path):
        return
    src = batch['src_coordinates'][batch_idx]
    tar = batch['tar_coordinates'][batch_idx]
    
    perspectives = [(30, 45), (60, 90), (90, 0)]
    
    src_transformed = src.clone()
    tar_transformed = tar.clone()
    
    fig = plt.figure(figsize=(15, 20))  # Increase figure height for more space

    plt.subplots_adjust(hspace=0.6)  # Increase vertical space between rows
    n_transformation = len(list(transformations.keys()))
    for idx, (aligner_name, (R_gamma, t_gamma)) in enumerate(transformations.items()):
        if compose:
            src_transformed = (torch.matmul(src_transformed, R_gamma[batch_idx]) + t_gamma[batch_idx].unsqueeze(0))
        else:
            src_transformed = (torch.matmul(src.clone(), R_gamma[batch_idx]) + t_gamma[batch_idx].unsqueeze(0))
        for i, (elev, azim) in enumerate(perspectives):
            ax = fig.add_subplot(n_transformation, len(perspectives), idx * len(perspectives) + i + 1, projection='3d')

            ax.scatter(src_transformed[:, 0], src_transformed[:, 1], src_transformed[:, 2], c='r', marker='o', label='Source', s=1)
            ax.scatter(tar_transformed[:, 0], tar_transformed[:, 1], tar_transformed[:, 2], c='b', marker='^', label='Target', s=1)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{aligner_name} - View {i+1}')
            ax.view_init(elev=elev, azim=azim)
    
    plt.suptitle(f"{batch['metadata'][0]['mov_protein']} to {batch['metadata'][0]['ref_protein']}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=500)
    buf.seek(0)
    
    image = Image.open(buf)
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1)
    plt.close()
    buf.close()
    return image


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

