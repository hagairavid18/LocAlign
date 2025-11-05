import torch


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
    assert P.shape == Q.shape, "P and Q must be the same shape"
    assert weights.shape == P.shape[:2], "Weights must be [B, K]"

    B, K, _ = P.shape

    zero_mask = (weights.sum(dim=(1), keepdim=True) == 0)  # Shape: (B, 1, 1)
    weights = weights + zero_mask * 1e-6  # Avoid zero weights

    # Compute weighted centroids
    centroid_P = torch.sum(P * weights.unsqueeze(-1), dim=1)  # [B, 3]
    centroid_Q = torch.sum(Q * weights.unsqueeze(-1), dim=1)  # [B, 3]

    # Center the point clouds
    P_centered = P - centroid_P.unsqueeze(1)  # [B, K, 3]
    Q_centered = Q - centroid_Q.unsqueeze(1)  # [B, K, 3]

    # Compute covariance matrix: H = Q^T * W * P
    H = torch.bmm(Q_centered.transpose(1, 2), P_centered * weights.unsqueeze(-1))  # [B, 3, 3]

    # SVD
    U, S, raw_Vt = torch.linalg.svd(H, full_matrices=False)

    # Compute determinant signs: (B,)
    det_sign = torch.sign(torch.det(torch.bmm(raw_Vt.transpose(1, 2), U.transpose(1, 2))))

    # Fix reflection by adjusting the last row of Vt
    eye = torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0).repeat(H.shape[0], 1, 1)
    eye[:, -1, -1] = det_sign  # last singular value sign correction

    # Corrected Vt
    Vt = torch.bmm(eye, raw_Vt)

    # Optimal rotation
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    t =  centroid_Q - torch.bmm(R.transpose(1,2), centroid_P[:, :, None]).squeeze(2)
    
    diff = (torch.bmm(P, R) + t[:, None, :]) - Q
    sq_dist = torch.sum(diff ** 2, dim=2)
    weighted_sq = weights * sq_dist
    weighted_corr_rmsd = torch.sqrt(weighted_sq.sum(dim=1))

    return R, t, weighted_corr_rmsd
