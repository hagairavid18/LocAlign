import matplotlib.pyplot as plt
import os
import numpy as np

from scipy.cluster import hierarchy


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