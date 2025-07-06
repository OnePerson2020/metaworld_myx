# controllers.py
import numpy as np
from typing import Tuple
from params import ControlParams

class Enhanced6DOFController:
    """增强的6DOF力位混合控制器"""
    
    def __init__(self, params: ControlParams):
        self.params = params
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        
    def reset(self):
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        
    def compute_control(self, 
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       current_orientation_error: np.ndarray,
                       target_orientation_error: np.ndarray,
                       current_force: np.ndarray,
                       target_force: np.ndarray,
                       current_torque: np.ndarray,
                       target_torque: np.ndarray,
                       selection_matrix_pos: np.ndarray,
                       selection_matrix_rot: np.ndarray,
                       dt: float) -> Tuple[np.ndarray, np.ndarray]:

        pos_error = target_pos - current_pos
        rot_error = target_orientation_error - current_orientation_error
        
        force_error = target_force - current_force
        torque_error = target_torque - current_torque
        
        force_mask = 1 - np.diag(selection_matrix_pos)
        torque_mask = 1 - np.diag(selection_matrix_rot)
        
        self.force_integral += force_error * force_mask * dt
        self.torque_integral += torque_error * torque_mask * dt
        
        self.force_integral = np.clip(self.force_integral, -1.0, 1.0)
        self.torque_integral = np.clip(self.torque_integral, -0.5, 0.5)
        
        pos_control = self.params.kp_pos * pos_error
        rot_control = self.params.kp_rot * rot_error
        
        force_control_delta = (self.params.kp_force * force_error + 
                               self.params.ki_force * self.force_integral)
        torque_control_delta = (self.params.kp_torque * torque_error + 
                                self.params.ki_torque * self.torque_integral)
        
        pos_output = (selection_matrix_pos @ pos_control + 
                     (np.eye(3) - selection_matrix_pos) @ force_control_delta)
        rot_output = (selection_matrix_rot @ rot_control + 
                     (np.eye(3) - selection_matrix_rot) @ torque_control_delta)
        
        return pos_output, rot_output