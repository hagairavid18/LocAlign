from Bio.PDB.Model import Model
from Bio.PDB.Atom import Atom
from pytorch3d.ops.points_alignment import iterative_closest_point, ICPSolution
import torch
import numpy as np
import logging

from aligners import BaseStructurealigner

logger = logging.getLogger(__name__)


class Pytorch3dAligner(BaseStructurealigner):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Pytorch3daligner"
  
    def impose_structure(self, points1: list[Atom], points2: list[Atom], save_dir: str | None = None) -> Model:
        # atom_to_int ={key: idx for idx, key in enumerate(set([point.name for point in points1] + [point.name for point in points2]))}
        
        fixed_coord, moving_coord, = [], []
        for i in range(len(points1)):
            fixed_coord.append(points1[i].get_coord())
            moving_coord.append(points2[i].get_coord())
        
        fixed_coord = torch.Tensor(np.array(fixed_coord)).unsqueeze(0)
        moving_coord = torch.Tensor(np.array(moving_coord)).unsqueeze(0)
        solution: ICPSolution = iterative_closest_point(moving_coord, fixed_coord)
        
        rot = np.array(solution.RTs[0][0]).astype("f")
        tran = np.array(solution.RTs[1][0]).astype("f")
       
        return rot, tran