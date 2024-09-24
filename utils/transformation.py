
import numpy as np
import torch

from utils.metrics import rigid_transformation_mse


def create_transformation_mse_matrix(transformations: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    
    mse_matrix = np.zeros((len(transformations), len(transformations)))
    
    for i in range(len(transformations)):
        for j in range(i, len(transformations)):
            R1, t1 = transformations[i][:3,:3], transformations[i][:3,3]
            R2, t2 = transformations[j][:3,:3], transformations[j][:3,3]
            mse = rigid_transformation_mse(R1, t1, R2, t2, points) 
            mse_matrix[i, j] = mse
            mse_matrix[j, i] = mse
             
    return mse_matrix


def rotation_matrix_to_euler_angles(R: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    x = torch.where(singular, torch.atan2(-R[:, 1, 2], R[:, 1, 1]), x)
    z = torch.where(singular, torch.zeros_like(z), z)

    return torch.stack((x, y, z), dim=-1)

