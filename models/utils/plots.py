import io
import tempfile
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image

from models.utils.collate import move_batch_to_device


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



def log_histograms(logger, cath_degrees, pocket_rmsds, epoch, bins=20):
    """
    Log histograms for Pocket RMSD per CATH degree to the Comet logger.

    Args:
        logger: The Comet logger instance.
        cath_degrees (list): List of CATH degree values for each sample.
        pocket_rmsds (list): List of Pocket RMSD values corresponding to each sample.
        epoch (int): The current training epoch, used as the step in the logger.
        bins (int): Number of bins for the histogram.
    """
    unique_cath_degrees = sorted(set(cath_degrees))
    histogram_data = {degree: [] for degree in unique_cath_degrees}

    # Group RMSD values by CATH degree
    for degree, rmsd in zip(cath_degrees, pocket_rmsds):
        histogram_data[degree].append(rmsd)

    # Compute and log histograms
    for degree, rmsds in histogram_data.items():
        if len(rmsds) > 1:  # At least two points needed to create a histogram
            logger.experiment.log_histogram_3d(
                name=f'Histogram_CATH_Degree_{degree}',
                values=rmsds,
                step=epoch
            )

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