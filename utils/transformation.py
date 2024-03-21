
import numpy as np

from utils.metrics import rigid_transformation_mse


def create_transformation_mse_matrix(transformations: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    
    mse_matrix = np.zeros((len(transformations), len(transformations)))
    
    for i in range(len(transformations)):
        for j in range(i, len(transformations)):
            R1, t1 = transformations[i][:3,:3], transformations[i][:,3]
            R2, t2 = transformations[j][:3,:3], transformations[j][:,3]
            mse = rigid_transformation_mse(R1, t1, R2, t2, points) 
            mse_matrix[i, j] = mse
            mse_matrix[j, i] = mse
             
    return mse_matrix

