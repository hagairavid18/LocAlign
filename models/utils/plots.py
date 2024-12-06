import tempfile
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import plotly.graph_objects as go

from models.utils.collate import move_batch_to_device
from models.utils.math import compute_rmsd_torch



def plot_transformed_point_clouds_interactive(logger, batch, transformations, loss_value: float | None = None, batch_idx=0, step: int = 0, compose: bool=True):
    batch = move_batch_to_device(batch, 'cpu')
    transformations = move_batch_to_device(transformations, 'cpu')
    src = batch['src_coordinates'][batch_idx]
    tar = batch['tar_coordinates'][batch_idx]
    
    src_pocket_indices = batch['src_residue_indices'][batch_idx]
    src_pocket_mask = batch['src_pocket_mask'][batch_idx]
    
    tar_pocket_indices = batch['tar_residue_indices'][batch_idx]
    tar_pocket_mask = batch['tar_pocket_mask'][batch_idx]
    
    src_pocket_coords = src[src_pocket_indices[src_pocket_mask]]
    tar_pocket_coords = tar[tar_pocket_indices[tar_pocket_mask]]

    # Choose 5 points to highlight
    highlight_indices = torch.arange(min(5, len(src_pocket_coords)))

    # Prepare figure
    fig = go.Figure()

    # Colors for each aligner
    aligner_colors = {
        'GT': 'rgba(255, 0, 0, 0.6)',   # Red
        'TMalign': 'rgba(0, 255, 0, 0.6)',   # Green
        'SoftBBS': 'rgba(0, 0, 255, 0.6)',   # Blue
        # Add more colors if needed
    }

    # RMSD storage
    rmsd_values = {}

    # Plot the target points (non-pocket and pocket), with default visibility for the pocket points only
    fig.add_trace(go.Scatter3d(
        x=tar[:, 0], y=tar[:, 1], z=tar[:, 2],
        mode='markers', marker=dict(size=5, color='blue'),
        name='Target (non-pocket)', visible=False  # Non-pocket points hidden by default
    ))

    fig.add_trace(go.Scatter3d(
        x=tar_pocket_coords[:, 0], y=tar_pocket_coords[:, 1], z=tar_pocket_coords[:, 2],
        mode='markers', marker=dict(size=5, color='purple'),
        name='Target (pocket)', visible=True  # Pocket points visible by default
    ))

    # Loop through aligners and plot transformed points
    for aligner_name, (R_gamma, t_gamma) in transformations.items():
        src_transformed = (torch.matmul(src.clone(), R_gamma[batch_idx]) + t_gamma[batch_idx].unsqueeze(0))
        src_pocket_transformed = (torch.matmul(src_pocket_coords, R_gamma[batch_idx]) + t_gamma[batch_idx].unsqueeze(0))

        fig.add_trace(go.Scatter3d(
            x=src_transformed[:, 0], y=src_transformed[:, 1], z=src_transformed[:, 2],
            mode='markers', 
            marker=dict(size=5, color='gray'),
            name=f'Source (transformed protein) - {aligner_name}', visible=True  # Hidden by default
        ))

        fig.add_trace(go.Scatter3d(
            x=src_pocket_transformed[:, 0], y=src_pocket_transformed[:, 1], z=src_pocket_transformed[:, 2],
            mode='markers', 
            marker=dict(size=5, color=aligner_colors.get(aligner_name, 'gray')),
            name=f'Source (transformed pocket) - {aligner_name}', visible=True  # Pocket points visible by default
        ))

        # Highlight 5 specific points with indices
        fig.add_trace(go.Scatter3d(
            x=src_pocket_transformed[highlight_indices, 0],
            y=src_pocket_transformed[highlight_indices, 1],
            z=src_pocket_transformed[highlight_indices, 2],
            mode='markers+text',
            marker=dict(size=10, color=aligner_colors.get(aligner_name, 'gray'), symbol="diamond"),
            text=[str(i.item()) for i in highlight_indices],
            textposition="top center",
            name=f'Highlights ({aligner_name})', visible=True  # Highlighted points visible by default
        ))

        # Compute RMSD
        rmsd = compute_rmsd_torch(
            batch['src_pocket'], batch['gt_R'], batch['gt_t'], R_gamma, t_gamma, batch['src_pocket_mask']
        )[0]
        rmsd_values[aligner_name] = rmsd.item()

    # Update layout with RMSD and matrices below the plot
    aligner_rmsd_str = ", ".join([f"{aligner}: {rmsd:.4f}" for aligner, rmsd in rmsd_values.items()])
    matrix_text = "<br>".join([
        f"<b>{aligner}:</b><br>" + "<br>".join([" &nbsp; ".join([f"{v:.2f}" for v in row]) for row in R_gamma[batch_idx].numpy()])
        for aligner, (R_gamma, _) in transformations.items()
    ])

    fig.update_layout(
        title=f"3D Scatter and RMSD: {aligner_rmsd_str} - Step {step}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=[
            dict(
                text=matrix_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.3,
                align="left",
                font=dict(size=10),
            )
        ]
    )

    # Save the HTML plot
    html_folder = os.path.join("plots", logger._experiment_name)
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