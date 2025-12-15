import torch
from models.utils.kabsch import weighted_kabsch_torch


def compute_transformation_from_corr_and_coord(soft_corr: torch.Tensor, src_coordinates: torch.Tensor, tar_coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Compute the transformation matrices from the soft correspondences and the source and target coordinates.
    First iteration uses only soft correspondences, while the rest use also the transformed source coordinates to tarine the transformation. 

    Args:
        max_protein_length (int): maximum length of the protein, for d0 factor calculation.
        soft_corr (torch.Tensor): soft correspondences matrix/vector.
        src_coordinates (torch.Tensor): source coordinates.
        tar_coordinates (torch.Tensor): target coordinates.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing the predicted rotation and translation matrices, as well as all the intermediate rotations, translations and soft correspond
    """        
        
    R, t, weighted_corr_rmsd = weighted_kabsch_torch(src_coordinates, tar_coordinates, soft_corr)

    return {'pred_R': R, 'pred_t': t, 'corr_rmsd': weighted_corr_rmsd}

