# params.py
from dataclasses import dataclass

@dataclass
class ControlParams:
    """力位混合控制参数"""
    # 位置控制参数 - 大幅降低增益
    kp_pos: float = 10.0   # 从1000.0降到10.0
    kd_pos: float = 1.0    # 从100.0降到1.0
    
    # 姿态控制参数 - 降低增益
    kp_rot: float = 5.0    # 从500.0降到5.0
    kd_rot: float = 0.5    # 从50.0降到0.5
    
    # 力控制参数 - 保持较小
    kp_force: float = 0.001
    ki_force: float = 0.0001
    kp_torque: float = 0.0005
    ki_torque: float = 0.00005
    force_deadzone: float = 0.5
    torque_deadzone: float = 0.1
    
    # 几何参数
    peg_radius: float = 0.015
    hole_radius: float = 0.025
    insertion_tolerance: float = 0.008
    min_insertion_depth: float = 0.06
    
    # 插入策略参数
    approach_distance: float = 0.0    # 接近阶段的距离阈值
    alignment_distance: float = 0.9   # 对齐阶段目标位置距离hole的距离
    max_orientation_error: float = 0.15  # 允许的最大姿态误差(弧度)
    
    # 控制频率
    position_control_freq: float = 500.0
    force_control_freq: float = 1000.0
    
    # 切换阈值
    switch_distance: float = 0.05