import numpy as np
import gymnasium as gym
import ppo_test # 导入ppo_test包

def keyboard_control_peg_insertion():
    # 使用ppo_test.make_mt_envs创建环境，并指定渲染模式为 'human'
    # 'peg-insert-side-v3' 是ppo_test中提供的任务名称
    env_name = 'peg-insert-side-v3'
    env = ppo_test.make_mt_envs(env_name, render_mode='human')

    print(f"环境 '{env_name}' 已加载。")
    print("控制说明：")
    print("每次输入需要提供7个浮点数，用空格分隔，表示机械臂的动作：")
    print("[delta_pos_x, delta_pos_y, delta_pos_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper_effort]")
    print("  - delta_pos_x/y/z: 末端在X/Y/Z轴上的位置增量 (-1.0 到 1.0)")
    print("  - delta_rot_x/y/z: 末端绕X/Y/Z轴的欧拉角旋转增量 (-1.0 到 1.0)")
    print("  - gripper_effort: 夹爪开合力度 (-1.0 为张开, 1.0 为闭合)")
    print("输入示例: 0.1 0.0 0.0 0.0 0.0 0.0 -1.0 (X轴正向移动，夹爪张开)")
    print("关闭窗口或按 Ctrl+C 退出程序。")

    obs, info = env.reset()
    terminated = False
    truncated = False

    try:
        while not terminated and not truncated:
            env.render() # 渲染环境

            # 从观测 (obs) 中获取当前末端执行器位置和夹爪距离
            # 根据 ppo_test/policies/sawyer_peg_insertion_side_v3_policy.py 中的 _parse_obs 方法
            # obs[:3] 是 hand_pos (3维)
            # obs[7] 是 gripper_distance_apart (1维)
            hand_pos = obs[:3]
            gripper_distance = obs[7]
            print(f"\n当前末端位置: {hand_pos.round(3)}， 夹爪距离: {gripper_distance:.3f}")

            # 提示用户输入动作
            action_input = input("请输入7个动作值 (dx dy dz drx dry drz gripper): ")
            try:
                action_values = [float(val) for val in action_input.split()]
                if len(action_values) != 7:
                    print("错误: 请输入7个浮点数。")
                    continue
                action = np.array(action_values, dtype=np.float32)
                
                # 裁剪动作值到有效范围 [-1, 1]
                action = np.clip(action, -1.0, 1.0)

            except ValueError:
                print("错误: 输入无效，请确保输入的是数字。")
                continue

            # 执行一步动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 打印一些关键信息
            print(f"奖励: {reward:.3f}, 成功: {info['success'] == 1.0}")
            # 检查info字典中是否存在这些键，以防止KeyError
            if 'obj_to_target' in info:
                print(f"对象到目标距离: {info['obj_to_target']:.3f}")
            if 'grasp_success' in info:
                print(f"抓取成功: {info['grasp_success'] == 1.0}")
            if 'pegHead_force_magnitude' in info:
                print(f"pegHead 受力大小: {info['pegHead_force_magnitude']:.3f}")
            if 'insertion_phase' in info:
                print(f"插入阶段: {info['insertion_phase']}")

    except KeyboardInterrupt:
        print("\n程序终止。")
    finally:
        env.close() # 关闭环境

if __name__ == '__main__':
    keyboard_control_peg_insertion()