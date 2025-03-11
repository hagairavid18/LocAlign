import tempfile
from typing import Any
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import plotly.graph_objects as go

from models.utils.collate import move_batch_to_device
from models.utils.math import compute_rmsd_torch


def plot_correspondences(soft_correspondences_list: list[torch.Tensor], mask: torch.Tensor, metadata: dict[str, Any], losses: dict[str, torch.Tensor], plot_dir: str) -> None:
        """
        Plots the correspondences as a 2D image, showing only the non-masked parts, using subplots.
        Adds small text for the top 10 values.

        Args:
            soft_correspondences_list (list[torch.Tensor]): List of correspondence tensors.
            mask (torch.Tensor): Mask to indicate valid correspondences.
            metadata (dict[str, Any]): Metadata containing information about the batch.
            losses (dict[str, torch.Tensor]): Loss information.
        """
        # Ensure tensors are on CPU and convert to numpy
        mask = mask.cpu().numpy()

        # Keep only the first two correspondence maps
        soft_correspondences_list = soft_correspondences_list[:2]
        num_correspondences = len(soft_correspondences_list)

        # Compute global vmin and vmax across both images
        all_values = [
            np.where(mask, corr.cpu().numpy(), np.nan)  # Apply mask to each correspondence
            for corr in soft_correspondences_list
        ]
        
        vmin = np.nanmin(all_values)  # Global minimum ignoring NaN
        vmax = np.nanmax(all_values)  # Global maximum ignoring NaN

        # Create subplots (adjust dynamically based on count)
        fig, axes = plt.subplots(1, num_correspondences, figsize=(5 * num_correspondences, 5))

        if num_correspondences == 1:
            axes = [axes]  # Ensure iterable when there's only one subplot

        # Process each correspondence map
        for i, (soft_correspondences, ax) in enumerate(zip(soft_correspondences_list, axes)):
            soft_correspondences = soft_correspondences.cpu().numpy()
            masked_correspondences = np.where(mask, soft_correspondences, np.nan)  # Apply mask

            # Plot on the corresponding subplot with consistent vmin/vmax
            im = ax.imshow(masked_correspondences[0], cmap='jet', interpolation='none', vmin=vmin, vmax=vmax)
            ax.set_title(f"Correspondence {i+1} - Loss: {losses[i]:.4f}")
            ax.axis("off")

            valid_mask = ~np.isnan(masked_correspondences[0])  # Mask to exclude NaN
            valid_values = masked_correspondences[0][valid_mask]  # Extract valid values

            if valid_values.size > 0:  # Ensure there are valid values
                top_indices = np.argsort(valid_values)[-10:]  # Get top 10 values
                valid_coords = np.argwhere(valid_mask)  # Get corresponding coordinates
                top_coords = valid_coords[top_indices]  # Get top 10 coordinate pairs

                # Annotate top 10 values
                for r, c in top_coords:
                    value = masked_correspondences[0, r, c]
                    ax.text(c, r + 1.0, f"{value:.2f}", color="white", fontsize=4, ha='center', va='center')

            # Add colorbar
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Generate filename
        name = f"{metadata[0]['Ligand_ID']}_{metadata[0]['mov_protein']}_{metadata[0]['ref_protein']}_{metadata[0]['cath_degree']}"
        plt.suptitle(f"Soft Correspondences Cath: {metadata[0]['cath_degree']}")

        # Save figure
        plt.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=500)
        plt.show()


def plot_offsets(offsets, frames, metadata, plot_dir) -> None:
        """
        Plots histograms for each XYZ offset and a scatter plot of the offsets and frames in 3D.

        Args:
            offsets (torch.Tensor or np.ndarray): Offsets in XYZ dimensions. Shape should be (N, 3), where N is the number of offsets.
            frames (torch.Tensor or np.ndarray): Frames in XYZ dimensions. Shape should be (N, 3), where N is the number of frames.
        """
        # Ensure offsets and frames are on CPU and convert to numpy if they are torch tensors
        offsets = offsets.cpu().numpy() if isinstance(offsets, torch.Tensor) else offsets
        frames = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames

        # Separate the offsets into XYZ components
        x_offsets = offsets[:, 0]
        y_offsets = offsets[:, 1]
        z_offsets = offsets[:, 2]

        # Plot Histograms for X, Y, Z components of offsets
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot histogram for X component
        axes[0].hist(x_offsets, bins=30, color='r', alpha=0.7)
        axes[0].set_title("Histogram of X Offsets")
        axes[0].set_xlabel("X Offset")
        axes[0].set_ylabel("Frequency")

        # Plot histogram for Y component
        axes[1].hist(y_offsets, bins=30, color='g', alpha=0.7)
        axes[1].set_title("Histogram of Y Offsets")
        axes[1].set_xlabel("Y Offset")
        axes[1].set_ylabel("Frequency")

        # Plot histogram for Z component
        axes[2].hist(z_offsets, bins=30, color='b', alpha=0.7)
        axes[2].set_title("Histogram of Z Offsets")
        axes[2].set_xlabel("Z Offset")
        axes[2].set_ylabel("Frequency")

        plt.tight_layout()
        name = f"{metadata[0]['Ligand_ID']}_{metadata[0]['mov_protein']}_{metadata[0]['ref_protein']}_{metadata[0]['cath_degree']}_offsets"
        plt.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=500)
        plt.show()

        # Plot the scatter plot of offsets and frames in 3D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for the offsets in black
        ax.scatter(x_offsets, y_offsets, z_offsets, c='k', marker='o', label='Offsets')

        # Scatter plot for the frames in a different color (e.g., red)
        x_frames = frames[:, 0]
        y_frames = frames[:, 1]
        z_frames = frames[:, 2]
        ax.scatter(x_frames, y_frames, z_frames, c='r', marker='^', label='Frames')

        ax.set_title("3D Scatter Plot of Offsets and Frames")
        ax.set_xlabel("X Offset")
        ax.set_ylabel("Y Offset")
        ax.set_zlabel("Z Offset")

        # Add legend to differentiate between offsets and frames
        ax.legend()

        # Save the 3D scatter plot
        name = f"{metadata[0]['Ligand_ID']}_{metadata[0]['mov_protein']}_{metadata[0]['ref_protein']}_{metadata[0]['cath_degree']}_histograms"
        plt.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=500)
        plt.show()


def plot_transformed_point_clouds_interactive(logger, batch, transformations, batch_idx=0, step: int = 0, epoch: int = 0):
    batch = move_batch_to_device(batch, 'cpu')
    transformations = move_batch_to_device(transformations, 'cpu')
    src = batch['src_frames'][:, :, 0, :][batch_idx][batch['src_mask'][batch_idx]]
    tar = batch['tar_frames'][:,:,0,:][batch_idx][batch['tar_mask'][batch_idx]]

    src_ligand = batch['src_ligand_coordinates'][batch_idx][batch['src_ligand_mask'][batch_idx]]
    tar_ligand = batch['tar_ligand_coordinates'][batch_idx][batch['tar_ligand_mask'][batch_idx]]
    
    # Extract the predicted rotation and translation from the transformations
    R_pred = transformations['pred_R']
    t_pred = transformations['pred_t']
    
    # Prepare figure
    fig = go.Figure()

    # Plot the target points (non-pocket)
    fig.add_trace(go.Scatter3d(
        x=tar[:, 0], y=tar[:, 1], z=tar[:, 2],
        mode='markers', marker=dict(size=2, color='blue'),
        name='Target (non-pocket)', visible=True  # Non-pocket points hidden by default
    ))

    # Plot the ligand points
    fig.add_trace(go.Scatter3d(
        x=tar_ligand[:, 0], y=tar_ligand[:, 1], z=tar_ligand[:, 2],
        mode='markers', marker=dict(size=5, color='purple'),
        name='Target (ligand)', visible=True  # Ligand points hidden by default
    ))

    # Apply the transformation: src_transformed = R_pred * src + t_pred
    src_transformed = (torch.matmul(src.clone(), R_pred[batch_idx]) + t_pred[batch_idx].unsqueeze(0)).detach().numpy()
    src_ligand_transformed = (torch.matmul(src_ligand.clone(), R_pred[batch_idx]) + t_pred[batch_idx].unsqueeze(0)).detach().numpy()
    # Add the transformed source points to the plot
    fig.add_trace(go.Scatter3d(
        x=src_transformed[:, 0], y=src_transformed[:, 1], z=src_transformed[:, 2],
        mode='markers', 
        marker=dict(size=2, color='gray'),
        name=f'Source (transformed)', visible=True
    ))

    # Add the transformed ligand points to the plot
    fig.add_trace(go.Scatter3d(
        x=src_ligand_transformed[:, 0], y=src_ligand_transformed[:, 1], z=src_ligand_transformed[:, 2],
        mode='markers',
        marker=dict(size=5, color='green'),
        name=f'Source (transformed ligand)', visible=True
    ))

    # Compute RMSD (if needed)
    rmsd = compute_rmsd_torch(
        batch['src_frames'][:, :, 0, :], batch['gt_R'], batch['gt_t'], R_pred, t_pred, batch['src_mask']
    )[0]

    ligand_rmsd = compute_rmsd_torch(
        batch['src_ligand_coordinates'], batch['gt_R'], batch['gt_t'], R_pred, t_pred, batch['src_ligand_mask']
    )[0]

    # Update layout with RMSD
    aligner_rmsd_str = f"pocket RMSD: {rmsd:.4f}"  # Update with computed RMSD value
    ligand_rmsd_str = f"Ligand RMSD: {ligand_rmsd:.4f}"  # Update with computed RMSD value
    fig.update_layout(
        title=f"3D Scatter: {aligner_rmsd_str}, {ligand_rmsd_str} - Step {step}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save the HTML plot
    html_folder = os.path.join("plots", logger._experiment_name, str(epoch))
    os.makedirs(html_folder, exist_ok=True)
    html_file_path = os.path.join(html_folder, f"plot_{step}.html")
    fig.write_html(html_file_path)


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