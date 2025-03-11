import torch
from models.utils.math import compose_transformations
from models.utils.tensor_operations import create_2d_mask
from utils.kabsch import weighted_kabsch_torch
from utils.deepbbs_utils import *



def expand_embeddings_to_atoms(
    atom_coordinates: torch.Tensor,
    residue_indices: torch.Tensor,
    residue_embeddings: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Expand residue embeddings to atom-level resolution based on residue index mapping (with batch dimension), considering masks.

    Args:
        atom_coordinates (torch.Tensor): Atom coordinates with the last column containing residue indices (B, N_atoms, 4).
        residue_indices (torch.Tensor): Residue indices mapped to the residue embeddings (B, N_residues).
        residue_embeddings (torch.Tensor): Residue embeddings in the same order as residue_indices (B, N_residues, embedding_dim).
        atom_mask (torch.Tensor): Mask for valid atoms (B, N_atoms). Masked atoms will be excluded.
        residue_mask (torch.Tensor): Mask for valid residues (B, N_residues). Masked residues will be excluded.

    Returns:
        torch.Tensor: Atom-level embeddings (B, N_atoms, embedding_dim).
    """
    batch_size, n_atoms, _ = atom_coordinates.shape
    embedding_dim = residue_embeddings.shape[-1]

    # Extract residue indices from the last column of atom_coordinates
    atom_residue_indices = atom_coordinates[:, :, -1].long()  # Shape [B, N_atoms]

    # Create atom-level embeddings tensor
    atom_embeddings = torch.zeros(batch_size, n_atoms, embedding_dim, device=residue_embeddings.device)

    for b in range(batch_size):
        # Apply residue mask to filter out invalid residues
        valid_residues = residue_mask[b]
        valid_residue_indices = residue_indices[b][valid_residues]  # Only valid residues

        # Create a mapping from residue index to embedding, only for valid residues
        residue_index_to_embedding = torch.zeros(residue_indices[b].max() + 1, embedding_dim, device=residue_embeddings.device)
        residue_index_to_embedding[valid_residue_indices] = residue_embeddings[b][valid_residues]

        # Apply atom mask to filter out invalid atoms
        valid_atoms = atom_mask[b]
        valid_atom_residue_indices = atom_residue_indices[b][valid_atoms]  # Only valid atoms

        # Assign atom-level embeddings using the residue index, considering the mask
        atom_embeddings[b][valid_atoms] = residue_index_to_embedding[valid_atom_residue_indices]

    return atom_embeddings


def compute_mean_bbs_pocket_values(matrix_values: torch.Tensor, pocket_indices: torch.Tensor, pocket_mask: torch.Tensor, num_residues: torch.Tensor):
    """
    Compute the mean of means for pocket and non-pocket vectors in a BxNxN matrix.

    Args:
        matrix_values (torch.Tensor): BxNxN matrix containing values.
        pocket_indices (torch.Tensor): BxN tensor with residue indices.
        pocket_mask (torch.Tensor): BxN tensor indicating pocket residues.
        num_residues (torch.Tensor): B tensor indicating the number of residues per batch.

    Returns:
        tuple: Mean of means for pocket and non-pocket vectors.
    """
    # Handle indices out of bounds by masking
    above_thresh_mask = pocket_indices >= matrix_values.size(1)
    pocket_indices[above_thresh_mask] = 0
    pocket_mask[above_thresh_mask] = 0

    # Gather the pocket vectors
    pocket_vectors = matrix_values.gather(1, pocket_indices.unsqueeze(-1).expand(-1, -1, matrix_values.size(-1)))
    pocket_vectors = pocket_vectors[pocket_mask]  # Keep only pocket locations

    # Compute per-batch means for pocket vectors
    pocket_means = []
    for b in range(matrix_values.size(0)):  # Iterate over batches
        pocket_values = pocket_vectors[b]
        if pocket_values.numel() > 0:
            pocket_means.append(pocket_values.mean(dim=0))
        else:
            pocket_means.append(torch.tensor(0.0, device=matrix_values.device))  # Handle no pocket case

    # Compute all indices for each batch
    all_indices = [torch.arange(n, device=pocket_indices.device) for n in num_residues]
    
    # Find non-pocket indices
    non_pocket_indices = [
        all_idx[~torch.isin(all_idx, residue_idx)]  # Exclude indices in pocket_indices
        for all_idx, residue_idx in zip(all_indices, pocket_indices)
    ]
    non_pocket_indices = torch.stack(non_pocket_indices, dim=0)

    # Gather non-pocket vectors
    non_pocket_vectors = matrix_values.gather(1, non_pocket_indices.unsqueeze(-1).expand(-1, -1, matrix_values.size(-1))).squeeze(0)

    # Compute per-batch means for non-pocket vectors
    non_pocket_means = []
    for b in range(matrix_values.size(0)):  # Iterate over batches
        non_pocket_values = non_pocket_vectors[b]
        if non_pocket_values.numel() > 0:
            non_pocket_means.append(non_pocket_values.mean(dim=0))
        else:
            non_pocket_means.append(torch.tensor(0.0, device=matrix_values.device))  # Handle no non-pocket case

    # # Compute the mean of means
    # pocket_mean = torch.stack(pocket_means, dim=0).mean(dim=0)
    # non_pocket_mean = torch.stack(non_pocket_means, dim=0).mean(dim=0)

    return pocket_means[0], non_pocket_means[0]

def get_d0(max_length: float) -> torch.Tensor:
    """
    Compute the d0 value based on the maximum length of the protein, used as a factor in the combination of the euclidean and 
    embedding distances

    Args:
        max_length (float): maximum length of the protein

    Returns:
        torch.Tensor: d0 value
    """    
    return 1.24*(torch.tensor(max_length) - 15) **(1/3) -1.8


def compute_transformation_from_corr_and_coord(max_protein_length: int, soft_corr: torch.Tensor, src_coordinates: torch.Tensor,
                                                tar_coordinates: torch.Tensor, src_orig_coord, tar_orig_coord, src_mask: torch.Tensor, combined_mask: torch.Tensor, iter_limit: int = 2) -> dict[str, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]: All residual transformations and the final rotation and translation matrices.
    """        
        
    gamma = soft_corr.clone()
    all_R, all_t, all_gamma = [], [], []
    iter_num = 0
    while iter_num < iter_limit:
        all_gamma.append(gamma)
        R_gamma, t_gamma, _, _ = weighted_kabsch_torch(src_coordinates, tar_coordinates, gamma.float())
        all_R.append(R_gamma)
        all_t.append(t_gamma)
        src_coordinates = (torch.matmul(src_coordinates, R_gamma) + t_gamma.unsqueeze(1)) * src_mask.unsqueeze(-1).expand_as(tar_coordinates)
        src_tgt_euc_dist = cdist_torch(tar_coordinates, src_coordinates, 3)
        gamma = (soft_corr / ((1 + (src_tgt_euc_dist / get_d0(max_protein_length).to(soft_corr.device)[:, None, None])**2)**2)).to(soft_corr.device) * combined_mask
        iter_num += 1

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
    R = torch.stack([softargmin_rows_torch(distance_matrix[i], t[i]) for i in range(batch_size)], dim=0)
    t = torch.tensor([guess_best_alpha_torch(tar_embedding[i,:][tar_mask[i]], dim_num=tar_embedding.shape[-1], transpose=False) for i in range(batch_size)] , device=device)
    C = torch.stack([softargmin_rows_torch(torch.transpose(distance_matrix, dim0=1, dim1=2)[i], t[i]) for i in range(batch_size)], dim=0)
    C = torch.transpose(C, dim0=1, dim1=2)
    B = torch.mul(R, C)

    B = B * combined_mask
    return B, combined_mask


def _log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport_mask(scores: torch.Tensor, alpha: torch.Tensor, iters: int,
        mask_rows: torch.Tensor, mask_cols: torch.Tensor) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    mask_rows = mask_rows.float()
    mask_cols = mask_cols.float()

    mask = (1-(mask_cols.unsqueeze(1) * mask_rows.unsqueeze(2))) * -1e9  # log-values for mask: 0 -> -1e9 ~ -inf
    scores = mask + scores
    one = torch.tensor([1], dtype=scores.dtype, device=scores.device)
    ms, ns = (mask_rows.sum(1) * one).to(scores), (mask_cols.sum(1) * one).to(scores)

    bins0 = torch.cat([torch.cat(
        [alpha.expand(1, ms[i].int(), 1), torch.tensor(-float('Inf')).type_as(alpha).expand(1, (m - ms[i]).int(), 1)], 1) for i in
                       range(ms.shape[0])], 0)
    bins1 = torch.cat([torch.cat(
        [alpha.expand(1, 1, ns[i].int()), torch.tensor(-float('Inf')).type_as(alpha).expand(1, 1, (n - ns[i]).int())], 2) for i in
                       range(ns.shape[0])], 0)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], dim=1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.repeat(m, 1), (ns.log() + norm)[None]]).T
    log_mu[:, :-1][mask_rows == 0.0] = torch.tensor(-float('Inf')).type_as(scores)
    log_nu = torch.cat([norm.repeat(n, 1), (ms.log() + norm)[None]]).T
    log_nu[:, :-1][mask_cols == 0.0] = torch.tensor(-float('Inf')).type_as(scores)
    Z = _log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm.unsqueeze(-1).unsqueeze(-1)  # multiply probabilities by M+N
    return Z


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """
    Perform Differentiable Optimal Transport in Log-space for stability
    for further reading:
    https://arxiv.org/pdf/1905.11885.pdf
    https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    good luck with that
    Args:
        scores: (torch.Tensor) [b,M,N] of scoring matrix between M features to N features of image pair
        alpha: (torch.Tensor) nonlearnable / learnable matrix transport matrix element parameter (defaults to 1)
        iters: (int) number of sinkhorn iterations

    Returns:
        torch.Tensor "soft max" of the scoring matrix
    """

    b, m, n = scores.shape
    ms = torch.tensor(m, device=scores.device)
    ns = torch.tensor(n, device=scores.device)
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = _log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z