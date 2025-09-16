import logging
from Bio.PDB.Atom import Atom
import open3d as o3d
from open3d.pipelines.registration import registration_ransac_based_on_correspondence, RegistrationResult
import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster, ClusterWarning

from aligners import BaseStructureAligner
from miners.utils.transformation import create_transformation_mse_matrix
from miners.utils.plots import plot_mse_matrix, plot_hierarchical_clustering, plot_clustered_rmse_fitness

logging.getLogger('matplotlib').setLevel(logging.ERROR)



logger = logging.getLogger(__name__)


class RANSACAligner(BaseStructureAligner):
    def __init__(self, n_ransac: int = 300, iter_per_ransac: int = 5, criterion_threshold: float = 0.5, save_plots: bool = False, cluster_thresh: int = 200) -> None:
  
        super().__init__()
        self.name = "RANSACAligner"
        self._n_ransac = n_ransac
        self._iter_per_ransac = iter_per_ransac
        self._criterion_threshold = criterion_threshold
        self._save_plots = save_plots
        self._cluster_thresh = cluster_thresh

    def _cluster_and_select_transformations(self, ransac_results: list[RegistrationResult], mov_points: list[np.ndarray],
                                plot_save_dir: str|None = None) -> list[np.ndarray]:
        """
        First, calculates and MSE matrix between all transformations. The MSE is between the locations of the transformed point.
        Later it use Hirechical Clustering and fcluster to cluster the points into different clusters, which refers to different
        groups of transformations. The purpose is to get n_clusters that corresponds to the possible alignments in the 3d space
        of the two proteins. Finally, for each cluster we select a representative transformation by a criterion.

        Args:
            ransac_results (list[RegistrationResult]): Each results contains the RT that will be later clustered.
            mov_points (list[np.ndarray]): NX3 array contains all xyz coordinates of the ligand we try to align.
            plot_save_dir (str | None, optional): If give, plots related the clustering will be saved. Defaults to None.

        Returns:
            list[np.ndarray]: List of 4X4 rigid transformation matrices, each one is a representative of one cluster. 
        """        
        
        transformations = [np.asarray(res.transformation) for res in ransac_results]
        mse_matrix = create_transformation_mse_matrix(transformations, np.stack(mov_points))
            
        linkage_matrix = hierarchy.linkage(mse_matrix, method='average')
        
        #TODO: find a proper threshold for unite groups 
        cluster_assignments = fcluster(linkage_matrix, self._cluster_thresh, criterion='distance')
        
        rmse = np.array([res.inlier_rmse for res in ransac_results])
        fitness = np.array([res.fitness for res in ransac_results])
        unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
        unique_clusters = unique_clusters[cluster_counts >= np.sum(cluster_counts) * 0.1]

        cluster_representative = np.zeros(len(unique_clusters))
        # if len(unique_clusters) > 1:
        #     print(plot_save_dir)
        for i, cluster in enumerate(unique_clusters):
            representative_idx = np.argmax(np.add(1 - rmse[cluster_assignments == cluster], fitness[cluster_assignments == cluster]))
            cluster_representative[i] = np.where(cluster_assignments == cluster)[0][representative_idx]
        
        if plot_save_dir and self._save_plots:
            plot_mse_matrix(mse_matrix, plot_save_dir)
            plot_hierarchical_clustering(linkage_matrix, plot_save_dir)
            plot_clustered_rmse_fitness(rmse, fitness, cluster_assignments, plot_save_dir)
        
        representative_results: list[RegistrationResult] = [ransac_results[index] for index in list(cluster_representative.astype('int'))]
        return representative_results
        
    def impose_structure(self, fix_points: list[Atom], mov_points: list[Atom],
                         save_dir: str | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
                
        fixed_coord = [points.get_coord() for points in fix_points]
        moving_coord = [points.get_coord() for points in mov_points]
        
        fixed_coord_o3d, moving_coord_o3d = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        moving_coord_o3d.points = o3d.utility.Vector3dVector(moving_coord)
        fixed_coord_o3d.points = o3d.utility.Vector3dVector(fixed_coord)

        id_to_index_map = {atom.id: index for index, atom in enumerate(mov_points)}
        correspondence_list = [[index1, id_to_index_map[atom.id]] for index1, atom in enumerate(fix_points) if atom.id in id_to_index_map]
        corr = o3d.cpu.pybind.utility.Vector2iVector(correspondence_list)

        ransac_results = []
        for _ in range(self._n_ransac):
            result: RegistrationResult = registration_ransac_based_on_correspondence(
                source=moving_coord_o3d,
                target=fixed_coord_o3d,
                corres=corr,
                max_correspondence_distance=self._criterion_threshold,
                ransac_n=min(6, len(fixed_coord)),
                criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(self._iter_per_ransac))
            
            if result.fitness > 0.3:
                ransac_results.append(result)        
                
        logger.debug(f"Found {len(ransac_results)} valid alignments")
        if len(ransac_results)  == 0:
            return [], [], [], []
        
        if len(ransac_results)  > 1: 
            representative_results: list[np.ndarray] = self._cluster_and_select_transformations(ransac_results, moving_coord, save_dir)
        else:
            representative_results = ransac_results
        selected_transformations = [np.asarray(ransac_result.transformation) for ransac_result in representative_results]
                
        rotations = [np.linalg.inv(trans[:3,:3].astype("f")) for trans in  selected_transformations] 
        translations = [trans[:,3].astype("f") for trans in  selected_transformations]
        rmse = [trans.inlier_rmse for trans in  representative_results]
        coverage = [trans.fitness for trans in  representative_results]
        
        return np.array(rotations), np.array(translations), np.array(rmse), np.array(coverage)