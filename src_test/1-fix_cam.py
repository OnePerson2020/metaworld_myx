import gymnasium as gym
import ppo_test
import time

def inspect_env_structure(env):
    """检查环境结构以找到viewer"""
    print("环境检查:")
    print(f"环境类型: {type(env)}")
    print(f"Unwrapped类型: {type(env.unwrapped)}")
    
    # 检查环境属性
    attrs = [attr for attr in dir(env.unwrapped) if not attr.startswith('__')]
    viewer_attrs = [attr for attr in attrs if 'view' in attr.lower() or 'render' in attr.lower()]
    print(f"与viewer/render相关的属性: {viewer_attrs}")
    
    # 检查是否有特定属性
    check_attrs = ['viewer', '_viewer', '_viewers', 'mujoco_renderer', 'renderer']
    for attr in check_attrs:
        if hasattr(env.unwrapped, attr):
            value = getattr(env.unwrapped, attr)
            print(f"{attr}: {type(value)} = {value}")
    
    return env.unwrapped

def set_camera_view_v2(env, distance=1.5, azimuth=90.0, elevation=-30.0, lookat=None):
    """
    改进的相机设置函数，适用于新版本的gymnasium/mujoco
    """
    if lookat is None:
        lookat = [0.0, 0.6, 0.2]
    
    mujoco_env = env.unwrapped
    
    # 确保先渲染一次以初始化viewer
    if not hasattr(mujoco_env, '_initialized_viewer'):
        env.render()
        mujoco_env._initialized_viewer = True
    
    # 方法1: 通过mujoco_renderer访问
    if hasattr(mujoco_env, 'mujoco_renderer'):
        renderer = mujoco_env.mujoco_renderer
        print(f"Found mujoco_renderer: {type(renderer)}")
        if hasattr(renderer, 'viewer'):
            viewer = renderer.viewer
            if viewer and hasattr(viewer, 'cam'):
                print("Setting camera via mujoco_renderer.viewer")
                viewer.cam.distance = distance
                viewer.cam.azimuth = azimuth
                viewer.cam.elevation = elevation
                viewer.cam.lookat[:] = lookat
                return True
    
    # 方法2: 通过_viewers字典访问
    if hasattr(mujoco_env, '_viewers'):
        viewers = mujoco_env._viewers
        print(f"Found _viewers: {viewers}")
        if viewers:
            for mode, viewer in viewers.items():
                if viewer and hasattr(viewer, 'cam'):
                    print(f"Setting camera via _viewers[{mode}]")
                    viewer.cam.distance = distance
                    viewer.cam.azimuth = azimuth
                    viewer.cam.elevation = elevation
                    viewer.cam.lookat[:] = lookat
                    return True
    
    # 方法3: 查找所有可能的viewer属性
    for attr_name in dir(mujoco_env):
        if 'view' in attr_name.lower():
            try:
                attr_value = getattr(mujoco_env, attr_name)
                if attr_value and hasattr(attr_value, 'cam'):
                    print(f"Setting camera via {attr_name}")
                    attr_value.cam.distance = distance
                    attr_value.cam.azimuth = azimuth
                    attr_value.cam.elevation = elevation
                    attr_value.cam.lookat[:] = lookat
                    return True
            except:
                continue
    
    # 方法4: 尝试通过render_mode直接访问
    try:
        # 获取当前的viewer
        current_viewer = None
        if hasattr(mujoco_env, '_get_viewer'):
            current_viewer = mujoco_env._get_viewer('human')
        elif hasattr(mujoco_env, 'viewer'):
            current_viewer = mujoco_env.viewer
        
        if current_viewer and hasattr(current_viewer, 'cam'):
            print("Setting camera via direct viewer access")
            current_viewer.cam.distance = distance
            current_viewer.cam.azimuth = azimuth
            current_viewer.cam.elevation = elevation
            current_viewer.cam.lookat[:] = lookat
            return True
    except Exception as e:
        print(f"Direct viewer access failed: {e}")
    
    print("无法找到或设置viewer相机")
    return False

def create_camera_wrapper(env_name, camera_config=None):
    """
    创建一个能够正确设置相机的环境包装器
    """
    if camera_config is None:
        camera_config = {
            'distance': 1.2,
            'azimuth': 90.0,
            'elevation': -20.0,
            'lookat': [0.0, 0.6, 0.2]
        }
    
    class SmartCameraWrapper(gym.Wrapper):
        def __init__(self, env, camera_config):
            super().__init__(env)
            self.camera_config = camera_config
            self._camera_attempts = 0
            self._camera_success = False
            
        def render(self):
            # 先执行原始渲染
            result = self.env.render()
            
            # 如果还没成功设置相机，继续尝试
            if not self._camera_success and self._camera_attempts < 5:
                self._camera_attempts += 1
                print(f"尝试设置相机 (第{self._camera_attempts}次)")
                success = set_camera_view_v2(self.env, **self.camera_config)
                if success:
                    self._camera_success = True
                    print("相机设置成功!")
                elif self._camera_attempts == 1:
                    # 第一次失败时，打印环境结构信息
                    inspect_env_structure(self.env)
            
            return result
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            # 重置时不重置相机成功状态，因为viewer可能仍然存在
            return obs, info
        
        def step(self, action):
            # 每隔一段时间重新尝试设置相机
            result = self.env.step(action)
            if hasattr(self, '_step_count'):
                self._step_count += 1
            else:
                self._step_count = 1
                
            # 每50步尝试一次相机设置（以防viewer被重新创建）
            if self._step_count % 50 == 0 and not self._camera_success:
                set_camera_view_v2(self.env, **self.camera_config)
                
            return result
    
    # 创建基础环境
    base_env = ppo_test.make_mt_envs(
        env_name,
        render_mode='human',
        width=1080,
        height=1920
    )
    
    return SmartCameraWrapper(base_env, camera_config)


if __name__ == "__main__":
    print("方法1: 使用智能相机包装器")
    camera_config = {
        'distance': 1.5,
        'azimuth': 135.0,
        'elevation': -25.0,
        'lookat': [0.0, 0.6, 0.15]
    }
    
    try:
        env = create_camera_wrapper('peg-insert-side-v4', camera_config)
        
        from ppo_test.policies import SawyerPegInsertionSideV3Policy
        policy = SawyerPegInsertionSideV3Policy()
        
        obs, info = env.reset()
        
        for i in range(200):
            env.render()
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info['success'] > 0.5:
                print("任务成功!")
                break
                
            time.sleep(0.02)
        
        env.close()
        
    except Exception as e:
        print(f"包装器方法失败: {e}")