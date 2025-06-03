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
    # Normalize weights per batch (optional but stable)
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
    # print(f"weights_sum: {weights_sum}")
    norm_weights = weights / weights_sum  # [B, K]
    # print(f"norm_weights max: {norm_weights.max()}")

    # Compute weighted centroids
    centroid_P = torch.sum(P * norm_weights.unsqueeze(-1), dim=1)  # [B, 3]
    centroid_Q = torch.sum(Q * norm_weights.unsqueeze(-1), dim=1)  # [B, 3]
    # print(f"centroid_P: {centroid_P}, centroid_Q: {centroid_Q}")

    # Center the point clouds
    P_centered = P - centroid_P.unsqueeze(1)  # [B, K, 3]
    Q_centered = Q - centroid_Q.unsqueeze(1)  # [B, K, 3]

    # Compute covariance matrix: H = Q^T * W * P
    H = torch.bmm(Q_centered.transpose(1, 2), P_centered * norm_weights.unsqueeze(-1))  # [B, 3, 3]

    # SVD
    dtype = H.dtype
    with torch.autocast(device_type="cuda", enabled=False):
        U, S, raw_Vt = torch.linalg.svd(H.float(), full_matrices=False)
        
        # Validate right-handed coordinate system
        Vt = raw_Vt.clone()
        for i, value in  enumerate(torch.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))):
            if value < 0.0:
                Vt[i, -1, :] = -raw_Vt[i, -1, :] # change 0 to batch idx

        # Optimal rotation
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
        t =  centroid_Q - torch.bmm(R.transpose(1,2), centroid_P[:, :, None]).squeeze(2)
        diff = (torch.bmm(P, R.transpose(1,2)) + t[:, None, :]) - Q
        rmsd_per_bb = torch.sqrt(torch.sum(torch.square(diff), axis=1))
        rmsd = torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=1)))

    return R.to(dtype), t.to(dtype), rmsd, rmsd_per_bb
