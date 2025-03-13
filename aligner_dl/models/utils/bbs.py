import torch
from models.utils.math import compose_transformations
from models.utils.tensor_operations import create_2d_mask
from models.utils.deepbbs_utils import cdist_torch, guess_best_alpha_torch, softargmin_rows_torch
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
                                                tar_coordinates: torch.Tensor, src_mask: torch.Tensor, combined_mask: torch.Tensor, iter_limit: int = 2) -> dict[str, torch.Tensor]:
    """
    Compute the transformation matrices from the soft correspondences and the source and target coordinates.
    First iteration uses only soft correspondences, while the rest use also the transformed source coordinates to refine the transformation. 

    Args:
        max_protein_length (int): maximum length of the protein, for d0 factor calculation.
        soft_corr (torch.Tensor): soft correspondences matrix.
        src_coordinates (torch.Tensor): source coordinates.
        tar_coordinates (torch.Tensor): target coordinates.
        src_mask (torch.Tensor): source mask.
        combined_mask (torch.Tensor): combined mask.
        iter_limit (int, optional): Maximum number of refinement iteration. Defaults to 2.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing the predicted rotation and translation matrices, as well as all the intermediate rotations, translations and soft correspond
    """        
        
    gamma = soft_corr.clone()
    all_R, all_t, all_gamma = [], [], []

    for iter_num in range(iter_limit):
        all_gamma.append(gamma)
        R_gamma, t_gamma, _, _ = weighted_kabsch_torch(src_coordinates, tar_coordinates, gamma.float())
        all_R.append(R_gamma)
        all_t.append(t_gamma)
        src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * src_mask.unsqueeze(-1).expand_as(tar_coordinates)
        src_tgt_euc_dist = cdist_torch(tar_coordinates, src_coordinates, 3)
        gamma = (soft_corr / ((1 + (src_tgt_euc_dist / get_d0(max_protein_length).to(soft_corr.device)[:, None, None])**2)**2)).to(soft_corr.device) * combined_mask

    rotation, translation = compose_transformations(rotations=all_R, translations=all_t)
    return {'pred_R': rotation, 'pred_t': translation, 'all_R': all_R, 'all_t': all_t, 'all_gamma': all_gamma}


def mask_and_normalize_matrix(distance_matrix: torch.Tensor, src_mask: torch.Tensor, tar_mask: torch.Tensor, src_embedding: torch.Tensor, tar_embedding: torch.Tensor) -> list[tuple]:
    """
    Apply softmin on rows and columns of the distance matrix and normalize the matrix using the source and target masks.

    Args:
        distance_matrix (torch.Tensor): _description_
        src_mask (torch.Tensor): _description_
        tar_mask (torch.Tensor): _description_
        src_embedding (torch.Tensor): _description_
        tar_embedding (_type_): _description_

    Returns:
        list[tuple]: _description_
    """    
    batch_size = distance_matrix.shape[0]
    device = distance_matrix.device
    combined_mask = create_2d_mask(src_mask, tar_mask)
    distance_matrix = distance_matrix * combined_mask
    distance_matrix = distance_matrix.masked_fill(~combined_mask, float('inf'))
    t = torch.tensor([guess_best_alpha_torch(src_embedding[i,:][src_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)], device=device)
    # print(f't1 {t}')

    R = torch.stack([softargmin_rows_torch(distance_matrix[i], t[i]) for i in range(batch_size)], dim=0)
    t = torch.tensor([guess_best_alpha_torch(tar_embedding[i,:][tar_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)] , device=device)
    # print(f't2 {t}')
    C = torch.stack([softargmin_rows_torch(torch.transpose(distance_matrix, dim0=1, dim1=2)[i], t[i]) for i in range(batch_size)], dim=0)
    C = torch.transpose(C, dim0=1, dim1=2)
    B = torch.mul(R, C)

    B = B * combined_mask
    return B, combined_mask