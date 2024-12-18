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
    rmsd_per_bb = torch.sqrt(torch.sum(torch.square(diff), axis=1))
    rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=1)))

    return R.T, t, rmsd, rmsd_per_bb

    
def weighted_kabsch_torch(P: torch.Tensor, Q: torch.Tensor, weights: torch.Tensor):
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

    # SVD
    U, S, raw_Vt = torch.linalg.svd(H)
    
    # Validate right-handed coordinate system
    Vt = raw_Vt.clone()
    for i, value in  enumerate(torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))):
        if value < 0.0:
        # Vt[0, :, -1] = -Vt[0, :, -1] # change 0 to batch idx
            Vt[i, -1, :] = -raw_Vt[i, -1, :] # change 0 to batch idx

    # Optimal rotation
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    t =  weighted_centroids_Q - torch.bmm(R.transpose(1,2), weighted_centroids_P[:, :, None]).squeeze(2)
    diff = (torch.bmm(P, R.transpose(1,2)) + t[:, None, :]) - Q
    rmsd_per_bb = torch.sqrt(torch.sum(torch.square(diff), axis=1))
    rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=1)))

    return R, t, rmsd, rmsd_per_bb
