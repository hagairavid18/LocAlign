import logging
from matplotlib import pyplot as plt
from Bio.PDB.Atom import Atom
import open3d as o3d
import os
from open3d.pipelines.registration import registration_ransac_based_on_correspondence, RegistrationResult
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster

from alligners import BaseStructureAlligner
from utils.transformation import create_transformation_mse_matrix
from utils.plots import plot_mse_matrix, plot_hierarchical_clustring, plot_clustered_rmse_fintness

logger = logging.getLogger(__name__)


class RANSACAlligner(BaseStructureAlligner):
    def __init__(self, n_ransac: int = 300, iter_per_ransac: int = 5, criterion_threshold: float = 0.3) -> None:
  
        super().__init__()
        self.name = "RANSACAlligner"
        self._n_ransac = n_ransac
        self._iter_per_ransac = iter_per_ransac
        self._criterion_threshold = criterion_threshold

    @staticmethod
    def cluster_and_select_transformations(ransac_results: list[RegistrationResult], mov_points: list[np.ndarray],
                                plot_save_dir: str|None = None) -> list[np.ndarray]:
        
        transformations = [np.asarray(res.transformation) for res in ransac_results]
        mse_matrix = create_transformation_mse_matrix(transformations, np.stack(mov_points))
            
        linkage_matrix = hierarchy.linkage(mse_matrix, method='average')

        cluster_assignments = fcluster(linkage_matrix, 200, criterion='distance')
        
        rmse = np.array([res.inlier_rmse for res in ransac_results])
        fitness = np.array([res.fitness for res in ransac_results])
        unique_clusters = np.unique(cluster_assignments)
        cluster_representive = np.zeros(len(unique_clusters))
        for i, cluster in enumerate(unique_clusters):
            represntive_idx = np.argmax(np.add(1 - rmse[cluster_assignments == cluster], fitness[cluster_assignments == cluster]))
            cluster_representive[i] = np.where(cluster_assignments == cluster)[0][represntive_idx]
        
        if plot_save_dir:
            plot_mse_matrix(mse_matrix, plot_save_dir)
            plot_hierarchical_clustring(linkage_matrix, plot_save_dir)
            plot_clustered_rmse_fintness(rmse, fitness, cluster_assignments, plot_save_dir)
        
        representive_results: list[RegistrationResult] = [ransac_results[index] for index in list(cluster_representive.astype('int'))]
        return [np.asarray(ransac_result.transformation) for ransac_result in representive_results]
        
    def impose_structure(self, fix_points: list[Atom], mov_points: list[Atom],
                         save_dir: str | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
                
        fixed_coord, moving_coord, = [], []
        for i in range(len(fix_points)):
            fixed_coord.append(fix_points[i].get_coord())
            moving_coord.append(mov_points[i].get_coord())
            
        fixed_coord_o3d, moving_coord_o3d = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        moving_coord_o3d.points = o3d.utility.Vector3dVector(moving_coord)
        fixed_coord_o3d.points = o3d.utility.Vector3dVector(fixed_coord)
        corr = o3d.cpu.pybind.utility.Vector2iVector([[i, i] for i in range(len(fix_points))])

        ransac_results = []
        for _ in range(self._n_ransac):
            result: RegistrationResult = registration_ransac_based_on_correspondence(
                source=moving_coord_o3d,
                target=fixed_coord_o3d,
                corres=corr,
                max_correspondence_distance=self._criterion_threshold,
                # o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=6,
                # [],
                criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(10, 1.0))
            
            if result.fitness > 0.3:
                ransac_results.append(result)        
                
        if len(ransac_results) == 0:
            return [], []
        
        selected_transformations: list[np.ndarray] =  RANSACAlligner.cluster_and_select_transformations(ransac_results, moving_coord, save_dir)
                
        rotations = [np.linalg.inv(trans[:3,:3].astype("f")) for trans in  selected_transformations] 
        translations = [trans[:,3].astype("f") for trans in  selected_transformations]
        
        return rotations, translations