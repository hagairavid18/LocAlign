import os
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import torch

from models.soft_bb_base import SoftBBBase
from models.utils import move_batch_to_device, build_object, compute_transformation_from_corr_and_coord, mask_and_normalize_matrix
from models.utils.plots import plot_transformed_point_clouds_interactive2
class VirtualSoftBB(SoftBBBase):
    def __init__(self, loss: dict[str, Any], optimizer: dict[str, Any], input_layer, virtual_layer: dict, scalar_layer: dict, max_iter: int = 5, n_iter_train: int = 2, plot_dir: str | None = None) -> None:
       
        super().__init__(loss=loss, optimizer=optimizer, max_iter=max_iter, n_iter_train=n_iter_train, plot_dir=plot_dir)
        self._input_block = build_object(input_layer, 'models.layers')
        self._linear = build_object(scalar_layer, 'models.layers')  # Projects tar_embedding to a scalar
        self._virtual_point_block = build_object(virtual_layer, 'models.layers')
        self._compute_pocket_importance = compute_pocket_importance
    
    def _get_distance_matrix(self, src_embedding: torch.Tensor, tar_embedding: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor) -> dict[str, torch.Tensor]:        

        distance_matrix = torch.sqrt(torch.sum((src_embedding.unsqueeze(1) - tar_embedding.unsqueeze(2)) ** 2, dim=-1) + 1e-4) / torch.sqrt(torch.tensor(src_embedding.shape[-1], dtype=torch.float32))
        # src_embedding = F.normalize(src_embedding, p=2, dim=-1)
        # tar_embedding = F.normalize(tar_embedding, p=2, dim=-1)

        # distance_matrix  =- torch.matmul(src_embedding, tar_embedding.transpose(1, 2)) / torch.sqrt(torch.tensor(src_embedding.shape[-1], dtype=torch.float32))
        # distance_matrix  =- torch.matmul(src_embedding, tar_embedding.transpose(1, 2)) 

        
        tar_scalar = self._linear(tar_embedding, mask=tar_mask).squeeze(-1) 
        src_scalar = self._linear(src_embedding, mask=src_mask).squeeze(-1)
            
        tar_matrix = tar_scalar.unsqueeze(-1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]
        src_matrix = src_scalar.unsqueeze(1).expand_as(distance_matrix)  # Shape [B, N_tar, N_src]

        distance_matrix = distance_matrix + tar_matrix + src_matrix
            
        return distance_matrix
    
    def _transform_offsets_with_frames(self, offsets: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """
        Transforms the offsets with the frames of each residue.
        Each predicted offset is transformed with the corresponding frame of the residue, by applying rotation and translation.
        

        Args:
            offsets (torch.Tensor): relative 3D coordinates of points.
            frames (torch.Tensor): corresponding frames of the residues. 

        Returns:
            torch.Tensor: _description_
        """
        transformations = torch.zeros(frames.shape[0], frames.shape[1], 4, 4, device=frames.device)
        transformations[:, :, :3, :3] = frames[:,:, 1:4, :].transpose(2,3)
        transformations[:, :, :3, 3] = frames[:, :, 0, :] # TODO: wait for bugfix

        transformations[:, :, 3, 3] = 1
        offsets = torch.cat([offsets, torch.ones_like(offsets[:, :, :1])], dim=-1)
        return torch.matmul(transformations, offsets.unsqueeze(-1)).squeeze(-1)[..., :3]

    def _plot_offsets(self, offsets, frames, metadata) -> None:
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
        plt.savefig(os.path.join(self._plot_dir, f"{name}.png"), dpi=500)
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
        plt.savefig(os.path.join(self._plot_dir, f"{name}.png"), dpi=500)
        plt.show()

    def _create_virtual_coordinates(self, embedding: torch.Tensor, frames: torch.Tensor, mask: torch.Tensor, metadata: str) -> torch.Tensor:
        """

        Args:
            embedding (torch.Tensor): embedding.
            frames (torch.Tensor): Frames of the residues.
            mask (torch.Tensor): Mask of the residues.

        Returns:
            torch.Tensor: virtual points per input, transformed with the frames.
        """        
        offsets = self._virtual_point_block(embedding, mask)
        transformed_offsets = self._transform_offsets_with_frames(offsets, frames)
        # if self._plot:
        #     self._plot_offsets(offsets[mask], frames[:, :, 0, :][mask], metadata)
        transformed_offsets  = transformed_offsets * mask.unsqueeze(-1)
        return transformed_offsets, offsets * mask.unsqueeze(-1)
    
    def _plot_correspondences(self, soft_correspondences_list: list[torch.Tensor], mask: torch.Tensor, metadata: dict[str, Any], losses: dict[str, torch.Tensor]) -> None:
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
        plt.savefig(os.path.join(self._plot_dir, f"{name}.png"), dpi=500)
        plt.show()
        
    def _compute_soft_bb_algorithm(self, batch: dict[torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The main algorithm of the model. Computes the soft correspondence matrix and the optimal transformation between two sets of embeddings.

        Args:
            batch (dict[torch.Tensor]): Dictionary containing the input tensors.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing the transformation dictionary and the embeddings dictionary.
        """
        tar_embedding, src_embedding  = self._input_block(batch['tar_embedding'], mask=batch['tar_mask']), self._input_block(batch['src_embedding'], mask =batch['src_mask'])
        # tar_embedding, src_embedding  = batch['tar_embedding'], batch['src_embedding']
        distance_matrix: torch.Tensor = self._get_distance_matrix(src_embedding=src_embedding, tar_embedding=tar_embedding, src_mask=batch['src_mask'], tar_mask=batch['tar_mask'])
        soft_correspondences, mask_2d = mask_and_normalize_matrix(distance_matrix, batch['src_mask'], batch['tar_mask'], src_embedding, tar_embedding)
        src_coord, tar_coord = batch['src_frames'][:, :, 0, :], batch['tar_frames'][:, :, 0, :]
        virtual_src_coord, src_offsets= self._create_virtual_coordinates(src_embedding, batch['src_frames'], batch['src_mask'], metadata=batch['metadata'])
        virtual_tar_coord, tar_offsets, =  self._create_virtual_coordinates(tar_embedding, batch['tar_frames'], batch['tar_mask'], metadata=batch['metadata'])
        optimal_transformation: dict[str, torch.Tensor] = compute_transformation_from_corr_and_coord(batch['max_length'], soft_correspondences, virtual_src_coord, virtual_tar_coord, src_coord, tar_coord, batch['src_mask'], mask_2d, iter_limit=self._max_iter if not self.training else self._n_iter_train)
        return optimal_transformation, mask_2d, src_offsets, tar_offsets

    def group_lasso_regularization(self, tensor, lambda_gl=1e-2):
        """
        Computes Group Lasso regularization for a B x N x 3 tensor.
        - Sums over B first, then N, then computes sqrt of the sum of squares across 3.
        """
        reg_loss = torch.sqrt(torch.sum(tensor ** 2, dim=(0, 1)) + 1e-4)  # Sum over B, then N
        return lambda_gl * reg_loss.mean()  # Sum over 3 (last dimension)

    def training_step(self, batch: dict[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = move_batch_to_device(batch, self.device)
        # print((batch['metadata'][0]['Ligand_ID'], batch['metadata'][0]['mov_protein'], batch['metadata'][0]['ref_protein'], batch['metadata'][0]['idx']))
        transformation_dict, _, virtual_src_coord, virtual_tar_coord = self._compute_soft_bb_algorithm(batch)
        
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        loss = loss + self.group_lasso_regularization(virtual_src_coord) + self.group_lasso_regularization(virtual_tar_coord)
        # if self._plot:
        #     plot_transformed_point_clouds_interactive2(self.logger, batch, transformation_dict, step=self.global_step)
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict' :transformation_dict}    
        return outputs

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validation step for the model. Computes the loss and the metrics for the model.

        Args:
            batch (dict[str, torch.Tensor]): 

        Returns:
            dict[str, torch.Tensor]: 
        """        
        batch = move_batch_to_device(batch, self.device)
        transformation_dict, mask, virtual_src_coord, virtual_tar_coord = self._compute_soft_bb_algorithm(batch)
        loss, loss_dict = self._compute_loss(batch, transformation_dict['pred_R'], transformation_dict['pred_t'])
        loss = loss + self.group_lasso_regularization(virtual_src_coord) + self.group_lasso_regularization(virtual_tar_coord)
        if self._plot:
            # loss_iter1, loss_dict2 = self._compute_loss(batch, transformation_dict['all_R'][0].detach(), transformation_dict['all_t'][0].detach())
            # self._plot_correspondences(transformation_dict['all_gamma'], mask, batch['metadata'], [loss_iter1, loss])
            print(f"Loss: {loss.item()}")
        
        outputs = {'loss': loss , 'loss_dict': loss_dict, 'transformation_dict': transformation_dict}
        self._metrics.update(batch, outputs)
        return outputs
