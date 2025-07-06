# policy.py
import numpy as np

class SimplePolicy:
    """提供高级运动指令的简单策略"""
    def __init__(self):
        self.phase = "reach"
        self.grasp_threshold = 0.05
        
    def reset(self):
        self.phase = "reach"

    def get_action(self, obs, env_unwrapped):
        hand_pos = obs[:3]
        gripper_open = obs[3] > 0.5 # 简化为布尔值
        obj_pos = obs[4:7]
        goal_pos = env_unwrapped._target_pos

        action = np.zeros(4)
        hand_to_obj_dist = np.linalg.norm(hand_pos - obj_pos)

        if self.phase == "reach":
            action[:3] = (obj_pos - hand_pos) * 2.0
            action[3] = -1  # 打开夹爪
            if hand_to_obj_dist < self.grasp_threshold:
                self.phase = "grasp"
        
        elif self.phase == "grasp":
            action[:3] = (obj_pos - hand_pos) * 2.0
            action[3] = 1 # 关闭夹爪
            # 检测是否抓取成功 (peg和gripper很近且gripper闭合)
            if hand_to_obj_dist < 0.04 and not gripper_open:
                self.phase = "transport"

        elif self.phase == "transport":
            # 直接以目标位置作为方向，让底层的混合控制器处理细节
            direction = goal_pos - obj_pos
            action[:3] = direction * 5.0 # 给予一个朝向目标的驱动力
            action[3] = 1 # 保持抓取

        return np.clip(action, -1, 1)