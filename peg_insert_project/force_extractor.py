# force_extractor.py
import numpy as np
import mujoco
from typing import Tuple

class ForceExtractor:
    """从仿真中提取接触力和力矩"""
    
    def __init__(self, env):
        self.env = env
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        self.peg_geom_ids = []
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'peg' in geom_name.lower():
                self.peg_geom_ids.append(i)
    
    def get_contact_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            if contact.geom1 in self.peg_geom_ids or contact.geom2 in self.peg_geom_ids:
                c_array = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force = c_array[:3]
                
                contact_frame = contact.frame.reshape(3, 3)
                world_force = contact_frame @ contact_force
                
                peg_pos = self.env.unwrapped._get_pos_objects()
                r = contact.pos - peg_pos
                contact_torque = np.cross(r, world_force)
                
                total_force += world_force
                total_torque += contact_torque
        
        return total_force, total_torque