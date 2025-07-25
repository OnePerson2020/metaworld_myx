# keyboard_control_demo_pygame.py

import gymnasium as gym
import numpy as np
import pygame
import time

# 导入 ppo_test 以注册自定义环境
import ppo_test

def print_instructions():
    """打印控制指令"""
    print("机械臂键盘控制 Demo (Pygame 版本)")
    print("-" * 40)
    print("窗口必须处于激活状态才能接收按键！")
    print("\n位置控制:")
    print("  W/S: 向前/向后 (X轴)")
    print("  A/D: 向左/向右 (Y轴)")
    print("  Q/E: 向上/向下 (Z轴)")
    print("\n旋转控制:")
    print("  J/L: 翻滚 (Roll, 绕X轴)")
    print("  I/K: 俯仰 (Pitch, 绕Y轴)")
    print("  U/O: 偏航 (Yaw, 绕Z轴)")
    print("\n夹爪控制:")
    print("  [空格]: 按住以闭合夹爪")
    print("\n其他:")
    print("  [Esc] 或 关闭窗口: 退出程序")
    print("-" * 40)
    print("正在启动环境...")

def main():
    """主执行函数"""
    print_instructions()

    # 初始化 pygame 用于事件处理
    pygame.init()
    # 我们只需要 pygame 的事件系统，但创建一个小窗口是标准做法
    pygame.display.set_mode((200, 200))
    pygame.display.set_caption("控制器")

    # 创建并配置 Meta-World 环境
    env = gym.make(
        'Meta-World/MT1',
        env_name='peg-insert-side-v3',
        render_mode='human',
        camera_name='corner2'
    )
    
    # 重置环境
    obs, info = env.reset()

    # 7维动作: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
    current_action = np.zeros(7, dtype=np.float32)
    current_action[6] = -1.0  # 默认: 张开夹爪

    # 键盘映射 (pygame.key 常量)
    KEY_MAP = {
        pygame.K_w: (0, 1.0), pygame.K_s: (0, -1.0),
        pygame.K_d: (1, 1.0), pygame.K_a: (1, -1.0),
        pygame.K_q: (2, 1.0), pygame.K_e: (2, -1.0),
        pygame.K_j: (3, 1.0), pygame.K_l: (3, -1.0),
        pygame.K_i: (4, 1.0), pygame.K_k: (4, -1.0),
        pygame.K_u: (5, 1.0), pygame.K_o: (5, -1.0),
        pygame.K_SPACE: (6, 1.0),
    }

    running = True
    while running:
        # --- 事件处理 (都在主线程) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_MAP:
                    index, value = KEY_MAP[event.key]
                    current_action[index] = value
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    index, _ = KEY_MAP[event.key]
                    # 对于夹爪，松开按键时恢复张开状态
                    if index == 6:
                        current_action[index] = -1.0
                    else:
                        current_action[index] = 0.0

        # --- 仿真步进 ---
        obs, reward, terminated, truncated, info = env.step(current_action)
        
        if terminated or truncated:
            print("回合结束, 环境已重置。")
            obs, info = env.reset()

        # 控制循环速率
        time.sleep(1 / env.metadata['render_fps'])

    # 清理
    env.close()
    pygame.quit()
    print("程序已退出。")

if __name__ == "__main__":
    main()