import mujoco
from mujoco import viewer
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

class MetaWorldMuJoCoAdapter:
    """
    将MuJoCo环境适配为MetaWorld策略可以使用的形式
    """
    
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 创建渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 初始化环境状态
        self._initialize_env_state()
        
        # 观测相关
        self._obs_obj_max_len = 14
        self._prev_obs = None
        
        # 任务相关
        self._target_pos = np.array([-0.3, 0.6, 0.0])  # 目标位置
        self.obj_init_pos = np.array([0, 0.6, 0.02])   # 物体初始位置
        self.hand_init_pos = np.array([0, 0.6, 0.2])   # 手的初始位置
        
        # 环境参数
        self.max_path_length = 500
        self.curr_path_length = 0
        
    def _initialize_env_state(self):
        """初始化环境状态"""
        # 设置初始关节位置
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置机器人初始姿态
        if self.model.nq > 0:
            # Sawyer机器人的初始关节角度
            init_qpos = np.array([0.0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
            if len(init_qpos) <= self.model.nq:
                self.data.qpos[:len(init_qpos)] = init_qpos
        
        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.curr_path_length = 0
        
        # 重置mujoco数据
        mujoco.mj_resetData(self.model, self.data)
        self._initialize_env_state()
        
        # 设置物体位置（可以添加随机化）
        self._set_object_position(self.obj_init_pos)
        
        # 设置目标位置
        self._set_target_position(self._target_pos)
        
        # 计算初始观测
        obs = self._get_obs()
        self._prev_obs = obs[:18].copy()
        
        return obs
    
    def _set_object_position(self, pos: np.ndarray):
        """设置物体位置"""
        # 查找物体的关节ID
        try:
            # 尝试找到物体相关的关节
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and 'obj' in joint_name.lower():
                    # 假设物体是自由关节
                    if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                        qpos_addr = self.model.jnt_qposadr[i]
                        self.data.qpos[qpos_addr:qpos_addr+3] = pos
                        break
            else:
                # 如果没有找到关节，尝试直接设置body位置
                for i in range(self.model.nbody):
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and 'peg' in body_name.lower():
                        self.data.xpos[i] = pos
                        break
        except Exception as e:
            print(f"设置物体位置时出错: {e}")
    
    def _set_target_position(self, pos: np.ndarray):
        """设置目标位置"""
        try:
            # 查找目标site
            for i in range(self.model.nsite):
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name and 'goal' in site_name.lower():
                    self.data.site_xpos[i] = pos
                    break
        except Exception as e:
            print(f"设置目标位置时出错: {e}")
    
    def _get_endeff_pos(self) -> np.ndarray:
        """获取末端执行器位置"""
        try:
            # 查找手部body
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and 'hand' in body_name.lower():
                    return self.data.xpos[i].copy()
            
            # 如果没找到，返回默认位置
            return np.array([0.0, 0.6, 0.2])
        except:
            return np.array([0.0, 0.6, 0.2])
    
    def _get_gripper_distance(self) -> float:
        """获取夹爪张开程度"""
        try:
            # 查找左右夹爪
            left_pos = None
            right_pos = None
            
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    if 'left' in body_name.lower() and 'claw' in body_name.lower():
                        left_pos = self.data.xpos[i]
                    elif 'right' in body_name.lower() and 'claw' in body_name.lower():
                        right_pos = self.data.xpos[i]
            
            if left_pos is not None and right_pos is not None:
                distance = np.linalg.norm(right_pos - left_pos)
                return np.clip(distance / 0.1, 0.0, 1.0)
            else:
                return 0.5  # 默认值
        except:
            return 0.5
    
    def _get_object_pos(self) -> np.ndarray:
        """获取物体位置"""
        try:
            # 查找物体site
            for i in range(self.model.nsite):
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name and 'peg' in site_name.lower() and 'grasp' in site_name.lower():
                    return self.data.site_xpos[i].copy()
            
            # 如果没找到site，查找body
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and 'peg' in body_name.lower():
                    return self.data.xpos[i].copy()
            
            return self.obj_init_pos.copy()
        except:
            return self.obj_init_pos.copy()
    
    def _get_object_quat(self) -> np.ndarray:
        """获取物体四元数"""
        try:
            # 查找物体body的旋转
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and 'peg' in body_name.lower():
                    # 从旋转矩阵获取四元数
                    xmat = self.data.xmat[i].reshape(3, 3)
                    return Rotation.from_matrix(xmat).as_quat()
            
            return np.array([1, 0, 0, 0])  # 默认四元数
        except:
            return np.array([1, 0, 0, 0])
    
    def _get_obs(self) -> np.ndarray:
        """获取观测值，格式与MetaWorld兼容"""
        # 获取末端执行器位置
        pos_hand = self._get_endeff_pos()
        
        # 获取夹爪状态
        gripper_distance = self._get_gripper_distance()
        
        # 获取物体位置和方向
        obj_pos = self._get_object_pos()
        obj_quat = self._get_object_quat()
        
        # 构建物体观测（填充到固定长度）
        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_info = np.hstack([obj_pos, obj_quat])
        obs_obj_padded[:len(obj_info)] = obj_info
        
        # 当前观测
        curr_obs = np.hstack([pos_hand, gripper_distance, obs_obj_padded])
        
        # 帧堆叠
        if self._prev_obs is None:
            self._prev_obs = curr_obs.copy()
        
        # 目标位置（MetaWorld策略需要）
        goal_pos = self._target_pos.copy()
        
        # 完整观测：当前 + 前一帧 + 目标
        obs = np.hstack([curr_obs, self._prev_obs, goal_pos])
        
        # 更新前一帧观测
        self._prev_obs = curr_obs.copy()
        
        return obs.astype(np.float64)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真"""
        # 应用动作到控制器
        self._apply_action(action)
        
        # 执行仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新路径长度
        self.curr_path_length += 1
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励和完成状态
        reward, info = self._compute_reward_and_info(obs, action)
        
        # 检查终止条件
        terminated = info.get('success', False)
        truncated = self.curr_path_length >= self.max_path_length
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """将MetaWorld动作应用到MuJoCo控制器"""
        if len(action) >= 4:
            # 前3个是位置增量，第4个是夹爪控制
            pos_delta = action[:3] * 0.01  # 缩放因子
            gripper_action = action[3]
            
            # 应用位置控制（通过mocap或者直接控制关节）
            # 这里需要根据你的具体模型来调整
            try:
                # 如果有mocap body
                if self.model.nmocap > 0:
                    current_pos = self.data.mocap_pos[0].copy()
                    new_pos = current_pos + pos_delta
                    # 限制在工作空间内
                    new_pos = np.clip(new_pos, 
                                    [-0.5, 0.4, 0.05], 
                                    [0.5, 1.0, 0.5])
                    self.data.mocap_pos[0] = new_pos
                    self.data.mocap_quat[0] = [1, 0, 1, 0]
                
                # 应用夹爪控制
                if self.model.nu > 0:
                    # 找到夹爪控制器
                    self.data.ctrl[-2:] = [gripper_action, -gripper_action]
                    
            except Exception as e:
                print(f"应用动作时出错: {e}")
    
    def _compute_reward_and_info(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, dict]:
        """计算奖励和信息"""
        # 简化的奖励函数
        obj_pos = self._get_object_pos()
        target_pos = self._target_pos
        
        # 距离奖励
        obj_to_target = np.linalg.norm(obj_pos - target_pos)
        success = obj_to_target < 0.07
        
        # 基本奖励
        reward = -obj_to_target
        if success:
            reward += 10.0
        
        info = {
            'success': float(success),
            'obj_to_target': obj_to_target,
            'reward': reward
        }
        
        return reward, info


def create_custom_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """创建自定义相机"""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    
    # 设置为自由相机
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.5
    camera.azimuth = 135
    camera.elevation = -25
    camera.lookat = np.array([0.0, 0.6, 0.15])
    
    return camera


def main(xml_path):
    """主程序"""
    # 创建适配器
    env_adapter = MetaWorldMuJoCoAdapter(xml_path)
    
    # 创建策略
    from metaworld.policies import SawyerPegInsertionSideV3Policy
    policy = SawyerPegInsertionSideV3Policy()
    
    # 创建自定义相机
    camera = create_custom_camera(env_adapter.model)
    
    # 重置环境
    obs = env_adapter.reset()
    
    print("开始仿真...")
    print(f"观测维度: {len(obs)}")
    
    # 使用被动viewer进行可视化
    with viewer.launch_passive(env_adapter.model, env_adapter.data) as v:
        # 设置相机
        v.cam.distance = 1.5
        v.cam.azimuth = 135
        v.cam.elevation = -25
        v.cam.lookat = [0.0, 0.6, 0.15]
        
        step_count = 0
        while step_count < 1000:
            # 获取策略动作
            try:
                action = policy.get_action(obs)
                
                # 执行步骤
                obs, reward, terminated, truncated, info = env_adapter.step(action)
                
                # 同步viewer
                v.sync()
                
                # 打印信息
                if step_count % 50 == 0:
                    print(f"Step {step_count}: Reward={reward:.3f}, "
                          f"Success={info['success']}, "
                          f"Distance={info['obj_to_target']:.3f}")
                
                # 检查完成
                if terminated or truncated:
                    print(f"Episode finished at step {step_count}")
                    if info['success']:
                        print("任务成功完成！")
                    # 重置环境
                    obs = env_adapter.reset()
                    step_count = 0
                    continue
                
                step_count += 1
                time.sleep(0.02)  # 控制仿真速度
                
            except KeyboardInterrupt:
                print("用户中断")
                break
            except Exception as e:
                print(f"仿真出错: {e}")
                break


def test_environment(xml_path):
    
    try:
        env_adapter = MetaWorldMuJoCoAdapter(xml_path)
        print("✓ 环境创建成功")
        
        obs = env_adapter.reset()
        print(f"✓ 环境重置成功，观测维度: {len(obs)}")
        
        # 测试动作
        action = np.array([0.01, 0.0, 0.0, 0.5])
        obs, reward, terminated, truncated, info = env_adapter.step(action)
        print(f"✓ 步骤执行成功，奖励: {reward}")
        
        # 测试策略
        from metaworld.policies import SawyerPegInsertionSideV3Policy
        policy = SawyerPegInsertionSideV3Policy()
        action = policy.get_action(obs)
        print(f"✓ 策略调用成功，动作: {action}")
        
        print("所有测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    xml_path = "./metaworld/xml/sawyer_peg_insertion_side.xml"
    # 首先测试环境
    if test_environment(xml_path):
        # 如果测试通过，运行主程序
        main(xml_path)
    else:
        print("请检查XML路径和依赖项")