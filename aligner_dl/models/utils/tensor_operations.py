import torch


def create_2d_mask(src_mask: torch.Tensor, tar_mask: torch.Tensor) -> torch.Tensor:
        """
        Create a 2D mask for the combined mask. Uses broadcasting to create a mask that is the product of the two input masks.

        Args:
            src_mask (torch.Tensor): Mask for the source embedding.
            tar_mask (torch.Tensor): Mask for the target embedding.

        Returns:
            torch.Tensor: a 2D mask that is the product of the two input masks.
        """        
        batch_size = src_mask.shape[0]
        N = src_mask.shape[1]
        mask_dim1_expanded = tar_mask.unsqueeze(2).expand(batch_size, -1, N) 
        mask_dim2_expanded = src_mask.unsqueeze(1).expand(batch_size, N, -1)  
        combined_mask = mask_dim1_expanded * mask_dim2_expanded
        return combined_mask