import torch
from models.utils.math import compose_transformations
from models.utils.kabsch import weighted_kabsch_torch


def compute_transformation_from_corr_and_coord(soft_corr: torch.Tensor, src_coordinates: torch.Tensor, tar_coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Compute the transformation matrices from the soft correspondences and the source and target coordinates.
    First iteration uses only soft correspondences, while the rest use also the transformed source coordinates to refine the transformation. 

    Args:
        max_protein_length (int): maximum length of the protein, for d0 factor calculation.
        soft_corr (torch.Tensor): soft correspondences matrix.
        src_coordinates (torch.Tensor): source coordinates.
        tar_coordinates (torch.Tensor): target coordinates.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing the predicted rotation and translation matrices, as well as all the intermediate rotations, translations and soft correspond
    """        
        
    R, t, weighted_corr_rmsd = weighted_kabsch_torch(src_coordinates.float(), tar_coordinates.float(), soft_corr.float())

    rotation, translation = compose_transformations(rotations=[R], translations=[t])
    return {'pred_R': rotation, 'pred_t': translation, 'all_R': [R], 'all_t': [t], 'all_gamma': [soft_corr], 'corr_rmsd': weighted_corr_rmsd}

