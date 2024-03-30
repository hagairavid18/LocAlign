
import numpy as np

def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    transformed_points = np.dot(R, points.T).T + t
    return transformed_points

def rigid_transformation_mse(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray,
                             t2: np.ndarray, P: np.ndarray) -> float:
    #TODO: solve it analiticly instead of using the norm formula.
    
    ## ||(R_1 * P + t_1) - (R_2 * P + t_2)||^2 = ((R_1 * P + t_1) - (R_2 * P + t_2))^T ((R_1 * P + t_1) - (R_2 * P + t_2)) 
    ## = trance(((R_1 * P + t_1) - (R_2 * P + t_2))^T ((R_1 * P + t_1) - (R_2 * P + t_2)))
    ## = trance((P^TR_1^T + t_1^T -P^T R_2^T - t_2^T) (R_1 * P + t_1 -R_2 * P - t_2))
    ## = trance((P^TR_1^T + t_1^T -P^T R_2^T - t_2^T) (R_1 * P + t_1 -R_2 * P - t_2))
    
    # Calculate the squared differences between rotation matrices.T
    # rotation_diff_1 = np.dot((R1 - R2), np.dot(P.T, P))
    # rotation_diff_2 = np.dot((R1 - R2), np.dot(P.T, P))
    
    # # Calculate the trace of the squared differences
    # trace_rotation_diff_1 = np.trace(rotation_diff_1)
    # trace_rotation_diff_2 = np.trace(rotation_diff_2)
    
    # points_inner_p = np.dot(P, P.T)
    # second_term = 2 * np.dot(P, np.dot((R1 - R2).T, (t1 - t2)))
    # last_term = np.dot((t1 - t2).T, (t1 - t2))
    

     # Calculate centroids
    
    transformed_points_1 = transform_points(P, R1, t1[:3])
    transformed_points_2 = transform_points(P, R2, t2[:3])
    
    # centroid1 = np.mean(transformed_points_1, axis=0)
    # centroid2 = np.mean(transformed_points_2, axis=0)
    
    # # Calculate translation component of MSE
    # translation_component = np.linalg.norm(centroid1 - centroid2) ** 2
    
    # # Calculate rotation component of MSE
    # rotation_component = np.mean(np.linalg.norm(transformed_points_1 - transformed_points_2 - centroid1 + centroid2, axis=1) ** 2)
    
    # # Average translation and rotation components
    # mse = (translation_component + rotation_component) / 2
    # Calculate the squared difference between translation vectors
    # squared_diff_translation = np.linalg.norm(t1 - t2)**2
    # Calculate squared differences between corresponding transformed points
    squared_diff = np.mean(np.linalg.norm(transformed_points_1 - transformed_points_2, axis=1)**2)

    
    # Calculate the MSE
    # mse = 2 * (trace_rotation_diff_1 - trace_rotation_diff_2) + squared_diff_translation
    return squared_diff