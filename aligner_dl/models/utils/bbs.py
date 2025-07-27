import torch
from models.utils.math import compose_transformations
from models.utils.tensor_operations import create_2d_mask
from models.utils.deepbbs_utils import guess_best_alpha_torch, softargmin_rows_torch
from models.utils.kabsch import weighted_kabsch_torch


def get_d0(max_length: float) -> torch.Tensor:
    """
    Compute the d0 value based on the maximum length of the protein, used as a factor in the combination of the euclidean and 
    embedding distances. Based on TMAlign paper.

    Args:
        max_length (float): maximum length of the protein

    Returns:
        torch.Tensor: d0 value
    """    
    return 1.24*(torch.tensor(max_length) - 15) **(1/3) -1.8


def compute_transformation_from_corr_and_coord(max_protein_length: int, soft_corr: torch.Tensor, src_coordinates: torch.Tensor,
                                                tar_coordinates: torch.Tensor,  iter_limit: int = 2) -> dict[str, torch.Tensor]:
    """
    Compute the transformation matrices from the soft correspondences and the source and target coordinates.
    First iteration uses only soft correspondences, while the rest use also the transformed source coordinates to refine the transformation. 

    Args:
        max_protein_length (int): maximum length of the protein, for d0 factor calculation.
        soft_corr (torch.Tensor): soft correspondences matrix.
        src_coordinates (torch.Tensor): source coordinates.
        tar_coordinates (torch.Tensor): target coordinates.
        iter_limit (int, optional): Maximum number of refinement iteration. Defaults to 2.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing the predicted rotation and translation matrices, as well as all the intermediate rotations, translations and soft correspond
    """        
        
    gamma = soft_corr.clone()
    dtype = gamma.dtype
    all_R, all_t, all_gamma = [], [], []

    for iter_num in range(iter_limit):
        all_gamma.append(gamma)
        R_gamma, t_gamma, rmsd= weighted_kabsch_torch(src_coordinates.float(), tar_coordinates.float(), gamma.float())
        all_R.append(R_gamma)
        all_t.append(t_gamma)
        src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) 
        src_tgt_euc_dist = torch.sqrt(torch.sum((tar_coordinates - src_coordinates) ** 2, dim=2))
        gamma = ((soft_corr / ((1 + (src_tgt_euc_dist / get_d0(max_protein_length).to(soft_corr.device)[:, None])**2)**2)).to(soft_corr.device)).to(dtype)

    rotation, translation = compose_transformations(rotations=all_R, translations=all_t)
    return {'pred_R': rotation, 'pred_t': translation, 'all_R': all_R, 'all_t': all_t, 'all_gamma': all_gamma, 'corr_rmsd': rmsd}

