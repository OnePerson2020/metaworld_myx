# coordinate_systems.py
import numpy as np
from scipy.spatial.transform import Rotation

class TaskCoordinateSystem:
    """任务坐标系：以hole为基准建立坐标系"""
    
    def __init__(self, hole_pos: np.ndarray, hole_orientation: np.ndarray):
        self.hole_pos = hole_pos.copy()
        
        # Y轴为插入方向（指向hole内部，即Y负方向）
        self.y_axis = -hole_orientation / np.linalg.norm(hole_orientation)
        
        # 构建正交的X、Z轴
        if abs(self.y_axis[2]) < 0.9:
            self.z_axis = np.cross(self.y_axis, [0, 0, 1])
        else:
            self.z_axis = np.cross(self.y_axis, [1, 0, 0])
        self.z_axis /= np.linalg.norm(self.z_axis)
        self.x_axis = np.cross(self.y_axis, self.z_axis)
        
        # 旋转矩阵：世界坐标系 -> 任务坐标系
        self.rotation_matrix = np.column_stack([self.x_axis, self.y_axis, self.z_axis])
        self.target_rotation = Rotation.from_matrix(self.rotation_matrix.T)
        
    def world_to_task(self, world_pos: np.ndarray) -> np.ndarray:
        relative_pos = world_pos - self.hole_pos
        return self.rotation_matrix.T @ relative_pos
    
    def task_to_world(self, task_pos: np.ndarray) -> np.ndarray:
        world_relative = self.rotation_matrix @ task_pos
        return world_relative + self.hole_pos
    
    def world_force_to_task(self, world_force: np.ndarray) -> np.ndarray:
        return self.rotation_matrix.T @ world_force
    
    def task_force_to_world(self, task_force: np.ndarray) -> np.ndarray:
        return self.rotation_matrix @ task_force
    
    def get_orientation_error(self, current_rotation: Rotation) -> np.ndarray:
        relative_rotation = self.target_rotation * current_rotation.inv()
        axis_angle = relative_rotation.as_rotvec()
        return self.rotation_matrix.T @ axis_angle