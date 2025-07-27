import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# mpl_toolkits.mplot3d is necessary for '3d' projection
from mpl_toolkits.mplot3d import Axes3D

def animate_force_vector(filepath="force_analysis.csv", episode_to_animate=1):
    """
    为特定轮次的数据创建一个力的3D矢量动画。

    Args:
        filepath (str): CSV数据文件的路径。
        episode_to_animate (int): 您希望生成动画的轮次（episode）编号。
    """
    # --- 1. 加载并准备数据 ---
    print("正在加载数据...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{filepath}'。")
        print("请确保CSV文件与此脚本位于同一目录下。")
        return

    # 筛选出要制作动画的特定轮次的数据
    episode_df = df[df['episode'] == episode_to_animate].reset_index()
    if episode_df.empty:
        print(f"错误: 在CSV文件中找不到轮次 {episode_to_animate} 的数据。")
        return

    # --- 2. 设置3D绘图区 ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 计算一个缩放因子，使最长的矢量长度为0.8，以获得最佳观感
    max_magnitude = df['magnitude'].max()
    scale_factor = 1.5

    # --- 3. 定义动画更新函数 ---
    # 这个函数会为动画的每一帧被调用
    def update(frame):
        ax.cla()  # 清除上一帧的图像

        # 获取当前帧的数据
        row = episode_df.iloc[frame]
        magnitude = row['magnitude']
        direction = np.array([row['direction_x'], row['direction_y'], row['direction_z']])

        # 根据力的大小计算矢量的显示长度
        length = (magnitude / max_magnitude) * scale_factor if max_magnitude > 0 else 0
        vector = direction * length

        # 绘制代表力的矢量（从原点出发）
        ax.quiver(0, 0, 0,  # 矢量起点
                  vector[0], vector[1], vector[2],  # 矢量终点
                  color='r',
                  arrow_length_ratio=0.15, # 箭头相对于杆的长度比例
                  label='Force Vector'
                  )

        # --- 设置图表样式 ---
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X Direction')
        ax.set_ylabel('Y Direction')
        ax.set_zlabel('Z Direction')

        # 设置一个固定的、更合适的观察角度
        ax.view_init(elev=30, azim=45)

        # 添加标题和信息文本
        ax.set_title(f"Force Vector Animation (Episode {episode_to_animate})")
        ax.text2D(0.05, 0.95, f"Step: {int(row['step'])}", transform=ax.transAxes, fontsize=12)
        ax.text2D(0.05, 0.90, f"Magnitude: {magnitude:.3f}", transform=ax.transAxes, fontsize=12)

    # --- 4. 创建并保存动画 ---
    num_frames = len(episode_df)
    # interval参数控制帧之间的延迟（毫秒），影响播放速度
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1 )

    output_filename = f'force_animation_episode_{episode_to_animate}.gif'
    print(f"正在保存动画到 {output_filename} ...")
    print("这可能需要一些时间，具体取决于您的数据量大小。")
    anim.save(output_filename, writer='pillow')
    print(f"动画保存成功！请查看 {output_filename}")
    plt.close()


if __name__ == '__main__':
    # 您可以在这里更改想制作动画的轮次编号
    EPISODE_NUMBER = 1
    animate_force_vector(episode_to_animate=EPISODE_NUMBER)