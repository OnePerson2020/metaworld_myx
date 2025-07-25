import metaworld
import time

env_name = 'peg-insert-side-v3'
env_class = metaworld._env_dict.ALL_V3_ENVIRONMENTS[env_name]
env = env_class(render_mode='human', width=1080, height=1920)

benchmark = metaworld.MT1(env_name)
task = benchmark.train_tasks[0]  # 使用第一个训练任务
env.set_task(task)

from metaworld.policies import SawyerPegInsertionSideV3Policy
policy = SawyerPegInsertionSideV3Policy()

obs, info = env.reset()
env.mujoco_renderer.viewer.cam.azimuth = 145

done = False
count = 0

print("Initial observation shape:", obs.shape)
print("Initial info:", info)

while count < 500 and not done:
    # 渲染环境
    env.render()
    # 根据当前观测值获取动作
    action = policy.get_action(obs)
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  PegHead Force Magnitude: {info.get('pegHead_force_magnitude', 'N/A'):.4f}")
    print(f"  PegHead Force Direction: {info.get('pegHead_force_direction', 'N/A')}")
    
    # 检查任务是否成功
    if info['success'] > 0.5:
        print("任务成功！")
        done = True
        
    time.sleep(0.02)
    count += 1

print(f"最终信息: {info}")
# env.close()

