# demo_position_control.py
import metaworld
import numpy as np
import time

# 从本地模块导入
from wrapper import HybridControlWrapper
from policy import SimplePolicy
from params import ControlParams

# 1. 创建环境
ml1 = metaworld.ML1('peg-insert-side-v3', seed=42) 
env = ml1.train_classes['peg-insert-side-v3'](render_mode='human')
task = ml1.train_tasks[0]
env.set_task(task)

# 2. 使用修改后的控制参数
control_params = ControlParams()
hybrid_env = HybridControlWrapper(env, control_params)

# 3. 定义并设置选择矩阵函数（纯位置控制）
def pure_position_control_matrices():
    """返回位置和姿态都为纯位置控制的选择矩阵"""
    pos_selection = np.eye(3)  # [1, 1, 1] on diag -> All axes position controlled
    rot_selection = np.eye(3)  # [1, 1, 1] on diag -> All axes orientation controlled
    return pos_selection, rot_selection
    
hybrid_env.set_selection_matrices_func(pure_position_control_matrices)
print("控制模式: 纯位置控制 (选择矩阵为单位阵)")

# 4. 初始化策略和环境
policy = SimplePolicy()
obs, info = hybrid_env.reset()
policy.reset()

# 5. 添加初始状态检查
print(f"初始状态:")
print(f"  Hand position: {obs[:3]}")
print(f"  Object position: {obs[4:7]}")
print(f"  Goal position: {env.unwrapped._target_pos}")
print(f"  Distance to object: {np.linalg.norm(obs[:3] - obs[4:7]):.3f}")

# 6. 运行一个episode
max_steps = 800
prev_obj_pos = obs[4:7].copy()
stuck_counter = 0

for step in range(max_steps):
    # 渲染环境
    hybrid_env.render()
    
    # 从简单策略获取高级动作
    action = policy.get_action(obs, hybrid_env.unwrapped)
    
    # 执行一步
    obs, reward, terminated, truncated, info = hybrid_env.step(action)
    
    # 检查是否卡住
    current_obj_pos = obs[4:7]
    if np.linalg.norm(current_obj_pos - prev_obj_pos) < 0.001:
        stuck_counter += 1
    else:
        stuck_counter = 0
    prev_obj_pos = current_obj_pos.copy()
    
    # 详细的调试信息
    if (step + 1) % 50 == 0:
        hand_pos = obs[:3]
        obj_pos = obs[4:7]
        goal_pos = env.unwrapped._target_pos
        
        print(f"\nStep: {step+1}")
        print(f"  Policy phase: {policy.phase}")
        print(f"  Insertion phase: {info.get('insertion_phase', 'unknown')}")
        print(f"  Hand pos: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        print(f"  Object pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        print(f"  Goal pos: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
        print(f"  Hand-Object dist: {np.linalg.norm(hand_pos - obj_pos):.3f}")
        print(f"  Object-Goal dist: {np.linalg.norm(obj_pos - goal_pos):.3f}")
        print(f"  Distance to target: {info.get('distance_to_target', 0):.3f}")
        print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")
        print(f"  Success: {info.get('success', False)}")
        print(f"  Insertion depth: {info.get('insertion_depth', 0):.3f}")
        print(f"  Stuck counter: {stuck_counter}")

    # 安全检查：如果长时间卡住，重置环境
    if stuck_counter > 100:
        print("\n警告：检测到长时间卡住，重置环境...")
        obs, info = hybrid_env.reset()
        policy.reset()
        stuck_counter = 0
        continue

    # 如果任务成功，提前结束
    if info.get('success', False):
        print("\n*** 任务成功! ***")
        time.sleep(3) # 暂停3秒查看结果
        break

    if terminated or truncated:
        print(f"\n任务结束: terminated={terminated}, truncated={truncated}")
        break
        
    time.sleep(0.02) # 减慢渲染速度，便于观察
    
print("\n演示结束。")
hybrid_env.close()