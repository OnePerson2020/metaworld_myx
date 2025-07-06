# demo_position_control.py
import metaworld
import numpy as np
import time

# 从本地模块导入
from wrapper import HybridControlWrapper
from policy import SimplePolicy


ml1 = metaworld.ML1('peg-insert-side-v3', seed=42) 
env = ml1.train_classes['peg-insert-side-v3'](render_mode='human')
task = ml1.train_tasks[0]
env.set_task(task)

# 2. 使用混合控制包装器
hybrid_env = HybridControlWrapper(env)

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

# 5. 运行一个 episode
max_steps = 800
for step in range(max_steps):
    # 渲染环境
    hybrid_env.render()
    
    # 从简单策略获取高级动作
    action = policy.get_action(obs, hybrid_env.unwrapped)
    
    # 执行一步
    obs, reward, terminated, truncated, info = hybrid_env.step(action)
    
    if (step + 1) % 50 == 0:
        print(f"Step: {step+1}, Phase: {policy.phase}, Success: {info.get('success', False)}, Depth: {info.get('insertion_depth', 0):.3f}")

    # 如果任务成功，提前结束
    if info.get('success', False):
        print("\n*** 任务成功! ***")
        time.sleep(3) # 暂停3秒查看结果
        break

    if terminated or truncated:
        break
        
    time.sleep(0.02) # 减慢渲染速度，便于观察
    
print("\n演示结束。")
hybrid_env.close()