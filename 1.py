# keyboard_control_demo_pygame_optimized.py

import gymnasium as gym
import numpy as np
import pygame
import time

# 导入 ppo_test 以注册自定义环境
import ppo_test

def print_instructions():
    """打印控制指令"""
    print("机械臂键盘控制 Demo (优化版)")
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

    pygame.init()
    pygame.display.set_mode((200, 200))
    pygame.display.set_caption("控制器")

    # 创建环境，并传入优化参数
    env = gym.make(
        'Meta-World/MT1',
        env_name='peg-insert-side-v3',
        render_mode='human',
        camera_name='corner2',
        # 优化参数: 提升移动/旋转速度
        action_scale=5.0 / 100.0,  # 提高移动灵敏度
        action_rot_scale=0.5,      # 提高旋转灵敏度
        # 优化参数: 降低分辨率以提高帧率
        width=320,
        height=240
    )
    
    obs, info = env.reset()

    current_action = np.zeros(7, dtype=np.float32)
    current_action[6] = -1.0

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
    clock = pygame.time.Clock()

    while running:
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
                    current_action[index] = -1.0 if index == 6 else 0.0

        obs, reward, terminated, truncated, info = env.step(current_action)
        
        if terminated or truncated:
            print("回合结束, 环境已重置。")
            obs, info = env.reset()

        # 使用pygame的时钟来稳定帧率，而不是time.sleep()
        clock.tick(env.metadata['render_fps'])

    env.close()
    pygame.quit()
    print("程序已退出。")

if __name__ == "__main__":
    main()