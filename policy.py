# policy.py
import numpy as np
import mujoco

class SimplePolicy:
    """提供高级运动指令的简单策略"""
    def __init__(self):
        self.phase = "reach"
        self.grasp_threshold = 0.05
        self.transport_threshold = 0.1  # 新增：运输阶段的距离阈值
        
    def reset(self):
        self.phase = "reach"

    def get_action(self, obs, env_unwrapped):
        hand_pos = obs[:3]
        gripper_open = obs[3] > 0.5 # 简化为布尔值
        obj_pos = obs[4:7]
        goal_pos = env_unwrapped._target_pos

        action = np.zeros(4)
        hand_to_obj_dist = np.linalg.norm(hand_pos - obj_pos)
        obj_to_goal_dist = np.linalg.norm(obj_pos - goal_pos)

        if self.phase == "reach":
            # 更温和的接近动作
            direction = obj_pos - hand_pos
            distance = np.linalg.norm(direction)
            if distance > 0.001:  # 避免除零
                action[:3] = direction / distance * min(distance * 5.0, 1.0)  # 限制速度
            action[3] = -1  # 打开夹爪
            
            if hand_to_obj_dist < self.grasp_threshold:
                self.phase = "grasp"
        
        elif self.phase == "grasp":
            # 保持在物体附近
            direction = obj_pos - hand_pos
            distance = np.linalg.norm(direction)
            if distance > 0.001:
                action[:3] = direction / distance * min(distance * 2.0, 0.5)  # 更温和的移动
            action[3] = 1 # 关闭夹爪
            
            # 检测是否抓取成功
            if hand_to_obj_dist < 0.04 and not gripper_open:
                self.phase = "transport"

        elif self.phase == "transport":
            # 更智能的运输策略 - 分为接近和插入两个子阶段
            goal_pos = env_unwrapped._target_pos
            
            # 计算peg head到goal的距离（更准确的距离计算）
            peg_head_site_id = mujoco.mj_name2id(env_unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'pegHead')
            peg_head_pos = env_unwrapped.data.site_xpos[peg_head_site_id].copy()
            head_to_goal_dist = np.linalg.norm(peg_head_pos - goal_pos)

            
            if head_to_goal_dist > 0.10:  # 距离较远时，快速接近
                direction = goal_pos - obj_pos
                distance = np.linalg.norm(direction)
                if distance > 0.001:
                    # 根据距离调整速度
                    speed_factor = min(distance * 1.5, 0.6)
                    action[:3] = direction / distance * speed_factor
                    
            elif head_to_goal_dist > 0.05:  # 中等距离，准备插入
                # 这个阶段需要更精确的对齐
                direction = goal_pos - obj_pos
                distance = np.linalg.norm(direction)
                if distance > 0.001:
                    # 缓慢接近，准备插入
                    action[:3] = direction / distance * min(distance * 5.0, 0.3)
                    
            else:  # 非常接近时，进行插入动作
                # 获取hole的方向进行插入
                hole_site_id = mujoco.mj_name2id(env_unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'hole')
                hole_pos = env_unwrapped.data.site_xpos[hole_site_id].copy()
                
                # 计算插入方向（从peg位置指向hole内部）
                insertion_direction = hole_pos - obj_pos
                distance = np.linalg.norm(insertion_direction)
                if distance > 0.001:
                    # 沿着插入方向缓慢移动
                    action[:3] = insertion_direction / distance * 0.2
                    
            action[3] = 1 # 保持抓取

        # 额外的安全限制
        action[:3] = np.clip(action[:3], -0.8, 0.8)  # 限制最大动作幅度
        
        return action