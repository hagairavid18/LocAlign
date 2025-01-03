#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import numpy as np
from scipy.spatial.transform import Rotation


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
# Part of the code is referred from: https://github.com/WangYueFt/dcp

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def square_dist_torch(A, B):
    AA = (A**2).sum(dim=1, keepdim=True)
    BB = (B**2).sum(dim=1, keepdim=True)
    inner = torch.matmul(A.float(), B.float().T)

    R = AA + (-2)*inner + BB.T

    return R

def new_cdist(x1, x2):
        x1 = x1.float()
        x2 = x2.float()
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True).float()
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True).float()
        res = -2*torch.matmul(x1, x2.transpose(-2, -1)) + x2_norm.transpose(-2, -1) + x1_norm
        res = res.clamp_min_(1e-30).sqrt_()
        return res

def dist_torch(A,B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """
    s = square_dist_torch(A,B)
    s[s<0]=0
    return torch.sqrt(s)

def cdist_torch(A,B,points_dim=None):
    num_features = 512
    if points_dim is not None:
        num_features = points_dim
    if (A.shape[-1] != num_features):
        A = torch.transpose(A, dim0=-2, dim1=-1)
    if (B.shape[-1] != num_features):
        B = torch.transpose(B, dim0=-2, dim1=-1)
    assert A.shape[-1] == num_features
    assert B.shape[-1] == num_features
    A = A.double().contiguous()
    B = B.double().contiguous()
    C = new_cdist(A,B)
    return C

def compute_residue_min_distances(
    src_all_coordinates: torch.Tensor,
    tar_all_coordinates: torch.Tensor,
    src_residue_indices: torch.Tensor,
    tar_residue_indices: torch.Tensor,
    src_all_mask: torch.Tensor,
    tar_all_mask: torch.Tensor,
    src_mask: torch.Tensor,
    tar_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the minimum Euclidean distance between residues based on atom-level coordinates,
    ensuring distances align with the original residue order.

    Args:
        src_all_coordinates (torch.Tensor): Source atom coordinates with residue indices in the last dimension (B, N_src_atoms, 4).
        tar_all_coordinates (torch.Tensor): Target atom coordinates with residue indices in the last dimension (B, N_tar_atoms, 4).
        src_residue_indices (torch.Tensor): Residue indices for source residues (B, N_src_residues).
        tar_residue_indices (torch.Tensor): Residue indices for target residues (B, N_tar_residues).
        src_all_mask (torch.Tensor): Mask for valid source atoms (B, N_src_atoms).
        tar_all_mask (torch.Tensor): Mask for valid target atoms (B, N_tar_atoms).
        src_mask (torch.Tensor): Mask for valid source residues (B, N_src_residues).
        tar_mask (torch.Tensor): Mask for valid target residues (B, N_tar_residues).

    Returns:
        torch.Tensor: Minimum residue-level distances (B, N_src_residues, N_tar_residues).
    """
    # Extract coordinates and residue indices
    src_coordinates = src_all_coordinates[..., :3]  # First 3 columns are coordinates
    tar_coordinates = tar_all_coordinates[..., :3]
    src_atom_residue_indices = src_all_coordinates[..., 3].long()  # 4th column is residue index
    tar_atom_residue_indices = tar_all_coordinates[..., 3].long()

    batch_size, n_src_residues = src_residue_indices.shape
    n_tar_residues = tar_residue_indices.shape[1]


    # Compute all pairwise distances at atom level
    all_atom_distances = torch.cdist(src_coordinates, tar_coordinates)  # (B, N_src_atoms, N_tar_atoms)

    # Initialize distance matrix
    residue_distances = torch.full((batch_size, n_src_residues, n_tar_residues), float("inf"), device=src_coordinates.device)

    for b in range(batch_size):
        # Map residue indices to atom distances
        src_map = src_residue_indices[b][src_mask[b].bool()]  # Valid source residues
        tar_map = tar_residue_indices[b][tar_mask[b].bool()]  # Valid target residues

        for i, src_res in enumerate(src_map):
            src_atom_mask = (src_atom_residue_indices[b] == src_res)
            if not src_atom_mask.any():
                continue

            for j, tar_res in enumerate(tar_map):
                tar_atom_mask = (tar_atom_residue_indices[b] == tar_res) 
                if not tar_atom_mask.any():
                    continue

                # Compute minimum pairwise distance for the residue pair
                residue_distances[b, i, j] = all_atom_distances[b][src_atom_mask][:, tar_atom_mask].min()

    return residue_distances

import torch

import torch

def compute_residue_min_distances2(
    src_all_coordinates: torch.Tensor,
    tar_all_coordinates: torch.Tensor,
    src_residue_indices: torch.Tensor,
    tar_residue_indices: torch.Tensor,
    src_all_mask: torch.Tensor,
    tar_all_mask: torch.Tensor,
    src_mask: torch.Tensor,
    tar_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the minimum Euclidean distance between residues based on atom-level coordinates,
    ensuring distances align with the original residue order, using min-pooling on atom-level distances.

    Args:
        src_all_coordinates (torch.Tensor): Source atom coordinates with residue indices in the last dimension (B, N_src_atoms, 4).
        tar_all_coordinates (torch.Tensor): Target atom coordinates with residue indices in the last dimension (B, N_tar_atoms, 4).
        src_residue_indices (torch.Tensor): Residue indices for source residues (B, N_src_residues).
        tar_residue_indices (torch.Tensor): Residue indices for target residues (B, N_tar_residues).
        src_all_mask (torch.Tensor): Mask for valid source atoms (B, N_src_atoms).
        tar_all_mask (torch.Tensor): Mask for valid target atoms (B, N_tar_atoms).
        src_mask (torch.Tensor): Mask for valid source residues (B, N_src_residues).
        tar_mask (torch.Tensor): Mask for valid target residues (B, N_tar_residues).

    Returns:
        torch.Tensor: Minimum residue-level distances (B, N_src_residues, N_tar_residues).
    """
    # Extract coordinates and residue indices
    src_coordinates = src_all_coordinates[..., :3]  # (B, N_src_atoms, 3)
    tar_coordinates = tar_all_coordinates[..., :3]  # (B, N_tar_atoms, 3)

    # Compute all pairwise distances at the atom level
    all_atom_distances = torch.cdist(src_coordinates, tar_coordinates)  # (B, N_src_atoms, N_tar_atoms)
    B, src_tar_size, _ = all_atom_distances.shape
     # Ensure the size is divisible by 3 (i.e., 3N is a multiple of 3)
    assert src_tar_size % 3 == 0, "Input size must be divisible by 3"
    
    N = src_tar_size // 3  # Number of residues (since each residue has 3 atoms)

    # Use unfold to create sliding windows of size 3x3 (across both source and target residues)
    unfolded_tensor = all_atom_distances.unfold(1, 3, 3).unfold(2, 3, 3)  # (B, N, 3, N, 3)

    # unfolded_tensor now has shape (B, N, 3, N, 3)
    # Apply min pooling across the 3x3 windows along the last two dimensions
    min_pooled_tensor = unfolded_tensor.min(dim=3)[0].min(dim=3)[0]  # (B, N, N)

    return min_pooled_tensor



def min_without_self_per_row_torch(D):
    """
    Accepts a distance matrix between all points in a set. For each point,
    returns its distance from the closest point that is not itself.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: vector of distances to nearest neighbor for each point.
    """
    E = D.clone()
    diag_ind = range(E.shape[0])
    E[diag_ind,diag_ind] = np.inf
    m = E.min(dim=1).values
    return m

def representative_neighbor_dist_torch(D):
    """
    Accepts a distance matrix between all points in a set,
    returns a number that is representative of the distances in this set.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: The representative distance in this set
    """

    assert D.shape[0] == D.shape[1], "Input to representative_neighbor_dist should be a matrix of distances from a point cloud to itself"
    m = min_without_self_per_row_torch(D)
    neighbor_dist = m.median()
    return neighbor_dist.cpu().detach().numpy()

def guess_best_alpha_torch(A, dim_num=3, transpose=None):
    """
        A good guess for the temperature of the soft argmin (alpha) can
        be calculated as a linear function of the representative (e.g. median)
        distance of points to their nearest neighbor in a point cloud.

        :param A: Point Cloud of size Nx3
        :return: Estimated value of alpha
        """

    COEFF = 0.1
    EPS = 1e-8
    if transpose is None:
        assert A.shape[0] != A.shape[1], 'Number of points and number of dimensions can''t be same'
    if (A.shape[1] != dim_num and transpose is None) or transpose:
        A = A.T
    assert A.shape[1]==dim_num
    rep = representative_neighbor_dist_torch(dist_torch(A, A))
    return COEFF * rep + EPS

def soft_BBS_loss_torch(T, S, t, points_dim=None, return_mat=False, transpose=None):
    num_features = 512
    if transpose is None:
        assert S.shape[0] != S.shape[1] and T.shape[0] != T.shape[1], 'Number of points and number of dimensions can''t be same'
    if points_dim is not None:
        num_features = points_dim
    if (T.shape[1] is not num_features and transpose is None) or transpose:
        T = torch.transpose(T, dim0=0, dim1=1)
    if (S.shape[1] is not num_features and transpose is None) or transpose:
        S = torch.transpose(S, dim0=0, dim1=1)
    assert S.shape[1] == num_features and T.shape[1] == num_features, 'Points dimension dismatch'

    T_num_samples = T.shape[0]
    S_num_samples = S.shape[0]
    mean_num_samples = np.mean([T_num_samples, S_num_samples])
    D = cdist_torch(T, S, points_dim)
    R = torch.squeeze(softargmin_rows_torch(D, t))
    C = torch.squeeze(softargmin_rows_torch(torch.transpose(D, dim0=0, dim1=1), t))
    C = torch.transpose(C, dim0=0, dim1=1)
    B = torch.mul(R, C)
    loss = torch.div(-torch.sum(B), mean_num_samples).view(1)
    if return_mat:
        return B
    else:
        return loss

def my_softmax(x, eps=1e-12, dim=0):
    x_exp = torch.exp(x - x.max())
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp/(x_exp_sum + eps)

def softargmin_rows_torch(X, t, eps=1e-12):
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=eps, dim=1)
    return weights