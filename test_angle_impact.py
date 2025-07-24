# test_angle_impact.py
import metaworld
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mujoco

# 从本地模块导入
from wrapper import HybridControlWrapper
from policy import SimplePolicy
from params import ControlParams

def run_test_with_angle(angle_degrees):
    """运行指定角度偏差的测试"""
    print(f"\n=== 测试角度偏差: {angle_degrees}度 ===")
    
    # 1. 创建环境
    ml1 = metaworld.ML1('peg-insert-side-v3', seed=42) 
    env = ml1.train_classes['peg-insert-side-v3'](render_mode='human')
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    # 2. 修改peg的初始角度（引入偏差）
    if angle_degrees != 0:
        # 重置环境以获取初始状态
        obs, info = env.reset()
        
        # 获取peg的body id
        peg_body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, 'peg')
        
        # 获取原始位置和姿态
        original_pos = env.unwrapped.data.xpos[peg_body_id].copy()
        original_quat = env.unwrapped.data.xquat[peg_body_id].copy()
        
        # 应用角度偏差（绕Z轴旋转）
        angle_rad = np.radians(angle_degrees)
        rotation = Rotation.from_euler('z', angle_rad)
        original_rotation = Rotation.from_quat(original_quat)
        new_rotation = rotation * original_rotation
        new_quat = new_rotation.as_quat()
        
        # 关闭环境并重新创建（避免渲染问题）
        env.close()
        env = ml1.train_classes['peg-insert-side-v3'](render_mode=None)
        env.set_task(task)
        obs, info = env.reset()
        
        # 设置新的姿态
        env.unwrapped.data.xquat[peg_body_id] = new_quat
        env.unwrapped.data.xpos[peg_body_id] = original_pos
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        
        # 重新开启渲染
        env.close()
        env = ml1.train_classes['peg-insert-side-v3'](render_mode='human')
        env.set_task(task)
        obs, info = env.reset()
        env.unwrapped.data.xquat[peg_body_id] = new_quat
        env.unwrapped.data.xpos[peg_body_id] = original_pos
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    
    # 3. 使用控制参数
    control_params = ControlParams()
    hybrid_env = HybridControlWrapper(env, control_params)

    # 4. 设置纯位置控制
    def pure_position_control_matrices():
        """返回位置和姿态都为纯位置控制的选择矩阵"""
        pos_selection = np.eye(3)  # [1, 1, 1] on diag -> All axes position controlled
        rot_selection = np.eye(3)  # [1, 1, 1] on diag -> All axes orientation controlled
        return pos_selection, rot_selection
        
    hybrid_env.set_selection_matrices_func(pure_position_control_matrices)
    print("控制模式: 纯位置控制 (选择矩阵为单位阵)")

    # 5. 初始化策略和环境
    policy = SimplePolicy()
    obs, info = hybrid_env.reset()
    policy.reset()

    # 6. 运行测试
    max_steps = 500
    max_contact_force = 0
    force_history = []
    step_history = []
    
    for step in range(max_steps):
        # 渲染环境
        hybrid_env.render()
        
        # 从简单策略获取高级动作
        action = policy.get_action(obs, hybrid_env.unwrapped)
        
        # 执行一步
        obs, reward, terminated, truncated, info = hybrid_env.step(action)
        
        # 记录接触力
        contact_force, contact_torque = hybrid_env.force_extractor.get_contact_forces_and_torques()
        contact_force_magnitude = np.linalg.norm(contact_force)
        force_history.append(contact_force_magnitude)
        step_history.append(step)
        
        if contact_force_magnitude > max_contact_force:
            max_contact_force = contact_force_magnitude
            
        # 每50步打印一次信息
        if (step + 1) % 50 == 0:
            hand_pos = obs[:3]
            obj_pos = obs[4:7]
            goal_pos = env.unwrapped._target_pos
            
            print(f"\nStep: {step+1}")
            print(f"  Hand pos: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
            print(f"  Object pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
            print(f"  Goal pos: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
            print(f"  Contact force: {contact_force_magnitude:.3f}")
            print(f"  Max contact force so far: {max_contact_force:.3f}")
            print(f"  Success: {info.get('success', False)}")

        # 如果任务成功或失败，提前结束
        if info.get('success', False) or terminated or truncated:
            print(f"\n任务结束: success={info.get('success', False)}, terminated={terminated}, truncated={truncated}")
            break
            
        time.sleep(0.01) # 减慢渲染速度，便于观察
    
    hybrid_env.close()
    
    return max_contact_force, force_history, step_history

def main():
    """主测试函数"""
    print("开始测试不同角度偏差对接触力的影响...")
    
    angles = [0, 5, 10, 15, 20]  # 测试的角度偏差（度）
    max_forces = []
    
    for angle in angles:
        max_force, force_history, step_history = run_test_with_angle(angle)
        max_forces.append(max_force)
        print(f"角度 {angle}° 的最大接触力: {max_force:.3f}")
        
        # 绘制接触力历史
        plt.figure()
        plt.plot(step_history, force_history)
        plt.xlabel('Step')
        plt.ylabel('Contact Force Magnitude')
        plt.title(f'Contact Force History (Angle Error: {angle}°)')
        plt.grid(True)
        plt.savefig(f'contact_force_angle_{angle}.png')
        plt.close()
    
    # 绘制角度与最大接触力的关系
    plt.figure()
    plt.plot(angles, max_forces, 'o-')
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Max Contact Force')
    plt.title('Max Contact Force vs Angle Error')
    plt.grid(True)
    plt.savefig('angle_vs_force.png')
    plt.close()
    
    print("\n测试完成！结果已保存到图片文件中。")
    print("角度与最大接触力的关系:")
    for angle, force in zip(angles, max_forces):
        print(f"  {angle}°: {force:.3f}")

if __name__ == "__main__":
    main()
