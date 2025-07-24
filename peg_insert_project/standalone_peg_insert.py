#!/usr/bin/env python3
"""
完全独立的Peg Insert Side环境测试脚本
直接使用MuJoCo环境而不依赖metaworld模块
"""

import numpy as np
import time
import mujoco

class SimplePegInsertEnv:
    """简化版的Peg Insert Side环境"""
    
    def __init__(self, render_mode='human'):
        # 加载模型
        import os
        xml_path = os.path.join(os.path.dirname(__file__), 'xml/sawyer_peg_insertion_side.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化参数
        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = 500
        self.current_step = 0
        
        # 动作空间: [dx, dy, dz, grip]
        self.action_space = np.array([[-1, -1, -1, -1], [1, 1, 1, 1]])
        
        # 观测空间维度
        self.obs_dim = 18  # 简化观测空间
        
        # 目标位置
        self.target_pos = np.array([-0.3, 0.6, 0.13])
        
        # 重置环境
        self.reset()
        
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # 设置初始状态
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        
        # 设置手部初始位置
        self.data.qpos[0:3] = [0, 0.6, 0.2]  # 手部位置
        self.data.qpos[3] = 0.05  # 夹爪初始张开
        
        # 设置peg初始位置
        self.data.qpos[9:12] = [0, 0.6, 0.02]  # peg位置
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        """获取观测值"""
        # 手部位置
        hand_pos = self.data.body('hand').xpos.copy()
        
        # 夹爪开合度
        gripper_state = self.data.qpos[3]
        
        # peg位置
        peg_pos = self.data.body('peg').xpos.copy()
        
        # peg四元数
        peg_quat = self.data.body('peg').xquat.copy()
        
        # 组合观测值
        obs = np.concatenate([
            hand_pos,           # 3: 手部位置
            [gripper_state],    # 1: 夹爪状态
            peg_pos,            # 3: peg位置
            peg_quat,           # 4: peg四元数
            hand_pos,           # 3: 重复手部位置（用于帧堆叠）
            [gripper_state],    # 1: 重复夹爪状态
            peg_pos,            # 3: 重复peg位置
        ])
        
        return obs
        
    def step(self, action):
        """执行一步动作"""
        self.current_step += 1
        
        # 确保动作在范围内
        action = np.clip(action, self.action_space[0], self.action_space[1])
        
        # 应用动作到机械臂
        # 控制手部位置
        self.data.ctrl[0] = action[0] * 0.1  # dx
        self.data.ctrl[1] = action[1] * 0.1  # dy
        self.data.ctrl[2] = action[2] * 0.1  # dz
        
        # 控制夹爪
        self.data.ctrl[3] = action[3]  # grip
        
        # 执行仿真
        mujoco.mj_step(self.model, self.data)
        
        # 获取新状态
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._compute_reward(obs)
        
        # 检查是否完成
        terminated = self._check_success(obs)
        truncated = self.current_step >= self.max_steps
        
        # 信息
        info = {
            'success': terminated,
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
        
    def _compute_reward(self, obs):
        """计算奖励"""
        hand_pos = obs[0:3]
        peg_pos = obs[4:7]
        target_pos = self.target_pos
        
        # 计算peg到目标的距离
        peg_to_target = np.linalg.norm(peg_pos - target_pos)
        
        # 计算手到peg的距离
        hand_to_peg = np.linalg.norm(hand_pos - peg_pos)
        
        # 基础奖励：peg接近目标
        reward = 1.0 - min(float(peg_to_target / 0.5), 1.0)
        
        # 额外奖励：手接近peg（抓取阶段）
        if hand_to_peg < 0.1:
            reward += 0.5
            
        # 成功奖励
        if peg_to_target < 0.07:
            reward += 5.0
            
        return reward
        
    def _check_success(self, obs):
        """检查是否成功"""
        peg_pos = obs[4:7]
        target_pos = self.target_pos
        peg_to_target = np.linalg.norm(peg_pos - target_pos)
        return peg_to_target < 0.07
        
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
                
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()

class SimpleController:
    """简单的控制器"""
    
    def __init__(self):
        self.kp = 5.0
        
    def compute_control(self, current_pos, target_pos):
        """计算位置控制输出"""
        pos_error = target_pos - current_pos
        control_output = self.kp * pos_error
        return np.clip(control_output, -1.0, 1.0)

def demo_simple_control():
    """简单控制演示"""
    print("=== 简单位置控制演示 ===")
    
    # 创建环境实例
    env = SimplePegInsertEnv(render_mode='human')
    
    obs, info = env.reset()
    episode_reward = 0
    max_steps = 500
    
    print("环境初始化完成")
    
    # 初始化控制器
    controller = SimpleController()
    
    for step in range(max_steps):
        env.render()
        
        # 获取状态信息
        hand_pos = obs[0:3]
        peg_pos = obs[4:7]
        target_pos = env.target_pos
        
        # 简单的三阶段控制策略
        hand_to_peg_dist = np.linalg.norm(hand_pos - peg_pos)
        peg_to_target_dist = np.linalg.norm(peg_pos - target_pos)
        
        if hand_to_peg_dist > 0.05:
            # 第一阶段：接近peg
            target_pos_ctrl = peg_pos + np.array([0, 0, 0.1])
            grip_action = -1.0  # 张开夹爪
        elif peg_to_target_dist > 0.1:
            # 第二阶段：运输到目标上方
            target_pos_ctrl = target_pos + np.array([0, 0, 0.1])
            grip_action = 1.0   # 闭合夹爪
        else:
            # 第三阶段：向下插入
            target_pos_ctrl = target_pos - np.array([0, 0, 0.1])
            grip_action = 1.0   # 保持闭合
            
        # 计算控制输出
        control_output = controller.compute_control(hand_pos, target_pos_ctrl)
        
        # 组合动作为最终控制输入
        action = np.concatenate([control_output, [grip_action]])
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.3f}, "
                  f"PegDist={peg_to_target_dist:.3f}, "
                  f"Success={info.get('success', False)}")
            
        if info.get('success', False):
            print(f"任务成功! 总奖励: {episode_reward:.3f}")
            time.sleep(2)
            break
            
        if terminated or truncated:
            break
            
        time.sleep(0.02)
    
    env.close()
    return episode_reward, info.get('success', False)

def main():
    """主函数"""
    print("基于MuJoCo的简化Peg Insert环境演示")
    print("==================================")
    
    try:
        print("\n运行简单位置控制演示...")
        reward, success = demo_simple_control()
        print(f"\n结果: 奖励={reward:.3f}, 成功={success}")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
