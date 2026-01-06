
import numpy as np

def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    transformed_points = np.dot(R, points.T).T + t
    return transformed_points

def rigid_transformation_mse(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray,
                             t2: np.ndarray, P: np.ndarray) -> float:

     # Calculate centroids
    
    transformed_points_1 = transform_points(P, R1, t1[:3])
    transformed_points_2 = transform_points(P, R2, t2[:3])
    
    squared_diff = np.mean(np.linalg.norm(transformed_points_1 - transformed_points_2, axis=1)**2)

    return squared_diff