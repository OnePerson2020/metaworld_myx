# success_checker.py
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict
from params import ControlParams
from coordinate_systems import TaskCoordinateSystem

class InsertionSuccessChecker:
    """插入成功检测器"""
    
    def __init__(self, params: ControlParams, task_coord_system: TaskCoordinateSystem):
        self.params = params
        self.task_coord_system = task_coord_system
        
    def check_insertion_success(self, peg_head_pos: np.ndarray, peg_pos: np.ndarray, 
                              peg_rotation: Rotation) -> Dict:
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        peg_center_task = self.task_coord_system.world_to_task(peg_pos)
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        radial_distance = np.sqrt(peg_head_task[0]**2 + peg_head_task[2]**2)
        insertion_depth = max(0, -peg_head_task[1])
        
        is_inside_hole = radial_distance <= self.params.insertion_tolerance
        sufficient_depth = insertion_depth >= self.params.min_insertion_depth
        
        center_radial_distance = np.sqrt(peg_center_task[0]**2 + peg_center_task[2]**2)
        is_aligned = center_radial_distance <= self.params.insertion_tolerance * 2
        
        orientation_magnitude = np.linalg.norm(orientation_error)
        is_orientation_aligned = orientation_magnitude <= self.params.max_orientation_error
        
        success = is_inside_hole and sufficient_depth and is_aligned and is_orientation_aligned
        
        return {
            'success': success,
            'insertion_depth': insertion_depth,
            'radial_distance': radial_distance,
        }