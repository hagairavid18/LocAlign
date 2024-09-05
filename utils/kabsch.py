import torch
from utils.deepbbs_utils import *

def kabsch_torch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))
    t = centroid_Q - torch.matmul(R, centroid_P)

    diff = (torch.mm(P, R.T) + t) - Q
    # diff = torch.matmul(p, R.transpose(0, 1)) - q
    rmsd_per_bb = torch.sqrt(torch.sum(torch.square(diff), axis=1))
    rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=1)))

    return R.T, t, rmsd, rmsd_per_bb

    
def weighted_kabsch_torch(P: torch.Tensor, Q: torch.Tensor, weights: torch.Tensor, P_mask, Q_mask):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # rows_normalized_weights = normalize_rows(weights)
    P_weights = torch.sum(weights, axis=1)
    Q_weights = torch.sum(weights, axis=2)
    weighted_centroids_P = torch.sum(P * P_weights.unsqueeze(2), axis = 1) / torch.sum(P_weights, axis=1).unsqueeze(dim=1)
    weighted_centroids_Q = torch.sum(Q * Q_weights.unsqueeze(2), axis = 1) / torch.sum(Q_weights, axis=1).unsqueeze(dim=1)

    # Center the points
    p = P - weighted_centroids_P[:, None, :]
    q = Q - weighted_centroids_Q[:, None, :]
    # diagonal_w = torch.diag(weights.squeeze(1))

    U, S, V = [], [], []
    R = []

    # Compute the covariance matrix
    H = torch.bmm(torch.bmm(q.transpose(1, 2), weights), p)
    # for i in range(1):
    #     u, s, v = torch.svd(H[i])
    #     r = torch.matmul(v, u.transpose(1, 0).contiguous())
    #     r_det = torch.det(r)
    #     if r_det < 0:
    #         u, s, v = torch.svd(H[i])
    #         v = torch.matmul(v, torch.eye(3))
    #         r = torch.matmul(v, u.transpose(1, 0).contiguous())
    #     R.append(r)

    #     U.append(u)
    #     S.append(s)
    #     V.append(v)

    # R = torch.stack(R, dim=0)


    # SVD
    U, S, Vt = torch.linalg.svd(H)
    # S = S.detach()

    # Validate right-handed coordinate system
    if torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))) < 0.0:
        Vt[0, :, -1] = -Vt[0, :, -1] # change 0 to batch idx

    # Optimal rotation
    # R = torch.linalg.inv(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    t =  weighted_centroids_Q - torch.bmm(R.transpose(1,2), weighted_centroids_P[:, :, None]).squeeze(2)
    diff = (torch.bmm(P, R.transpose(1,2)) + t[:, None, :]) - Q
    # diff = torch.matmul(p, R.transpose(0, 1)) - q
    rmsd_per_bb = torch.sqrt(torch.sum(torch.square(diff), axis=1))
    rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=1)))

    return R, t, rmsd, rmsd_per_bb



# def weighted_kabsch_torch_deepbbs(src_embedding, tgt_embedding, src, tgt, src_mask, tgt_mask, all_mask ):
#     eps = 1e-12

#     batch_size = src.size(0)
#     device = src.device

#     t = 4.0*torch.tensor([guess_best_alpha_torch(src_embedding[i,:][src_mask[i]], dim_num=128, transpose=False) for i in range(batch_size)], device=device)
#     scores = torch.cat(
#         [soft_BBS_loss_torch(src_embedding[i,:][src_mask[i]], tgt_embedding[i,:][tgt_mask[i]], t[i], points_dim=128, return_mat=True, transpose=True).float().unsqueeze(0)
#         for i in range(batch_size)], dim=0)
#     scores_norm = scores / (scores.sum(dim=2, keepdim=True)+ eps)
#     src_corr = torch.matmul(tgt, scores_norm.float().transpose(2, 1).contiguous())
#     src_tgt_euc_dist = cdist_torch(src, tgt, 3)
#     T = torch.clamp(torch.abs(src_embedding.mean(dim=2) - tgt_embedding.mean(dim=2)), 0.01, 100).view(-1,1,1)
#     # T = T/2**(iter-1)
#     gamma = (scores * torch.exp(-src_tgt_euc_dist / T)).sum(dim=2, keepdim=True).float().transpose(2,1)
#     src_weighted_mean = (src * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True)+eps)
#     src_centered = src - src_weighted_mean

#     src_corr_weighted_mean = (src_corr * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True) + eps)
#     src_corr_centered = src_corr - src_corr_weighted_mean

#     H = torch.matmul(src_centered * gamma, src_corr_centered.transpose(2, 1).contiguous()) + eps*torch.diag(torch.tensor([1,2,3], device=device)).unsqueeze(0).repeat(batch_size,1,1)

#     U, S, V = [], [], []
#     R = []

#     for i in range(src.size(0)):
#         u, s, v = torch.svd(H[i])
#         r = torch.matmul(v, u.transpose(1, 0).contiguous())
#         r_det = torch.det(r)
#         if r_det < 0:
#             u, s, v = torch.svd(H[i])
#             v = torch.matmul(v, torch.eye(3))
#             r = torch.matmul(v, u.transpose(1, 0).contiguous())
#         R.append(r)

#         U.append(u)
#         S.append(s)
#         V.append(v)

#     R = torch.stack(R, dim=0)

#     t = torch.matmul(-R, src_weighted_mean) + src_corr_weighted_mean
#     return R, t.view(batch_size, 3), src_corr
