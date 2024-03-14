import logging
from matplotlib import pyplot as plt
from Bio.PDB.Atom import Atom
import open3d as o3d
import os
from open3d.pipelines.registration import registration_ransac_based_on_correspondence, RegistrationResult
import numpy as np
from sklearn.cluster import KMeans

from alligners import BaseStructureAlligner

logger = logging.getLogger(__name__)


class RANSACAlligner(BaseStructureAlligner):
    def __init__(self, n_ransac: int = 1000, iter_per_ransac: int = 5, criterion_threshold: float = 0.5) -> None:
  
        super().__init__()
        self.name = "RANSACAlligner"
        self._n_ransac = n_ransac
        self._iter_per_ransac = iter_per_ransac
        self._criterion_threshold = criterion_threshold
  
    @staticmethod
    def _plot_rmse_vs_coverage(all_ransac_resilts: list[RegistrationResult], save_dir: str) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rmse = [res.inlier_rmse for res in all_ransac_resilts]
        fitness = [res.fitness for res in all_ransac_resilts]
        labels, centroids = RANSACAlligner.cluster_transformations(np.column_stack((rmse, fitness)))
        ax.scatter(rmse, fitness, c=labels, marker='o', s=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=20, label='Centroids')

        ax.set_xlabel('inlier rmse')
        ax.set_ylabel('fitness (coverage)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir,"rasnac_results.png"))
    
    @staticmethod
    def cluster_transformations(X: list[RegistrationResult], num_clusters: int = 10) -> tuple[np.ndarray, np.ndarray]:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        return labels, centroids

    def impose_structure(self, fix_points: list[Atom], mov_points: list[Atom], save_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
                
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
            result: RegistrationResult = registration_ransac_based_on_correspondence(moving_coord_o3d, fixed_coord_o3d, corr, self._criterion_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 6, [],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                10, 1.0))

            ransac_results.append(result)        
                
        #TODO: find a proper criterion for choosing the right solution
        best_run_id = np.argmax([(np.asarray(res.correspondence_set).shape[0]/ len(fixed_coord))  + 8 * (1-res.inlier_rmse) for res in ransac_results])
        best_transforamtion = np.asarray(ransac_results[best_run_id].transformation)
        
        self._plot_rmse_vs_coverage(ransac_results, save_dir)
        rot = np.linalg.inv(best_transforamtion[:3,:3].astype("f")) # open3d returns the inverse matrix
        tran = best_transforamtion[:,3].astype("f")
        
        return rot, tran