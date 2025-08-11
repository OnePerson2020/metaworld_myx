import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_force_data(filepath="force_analysis.csv"):
    """
    读取力分析的CSV文件并生成可视化图表。
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{filepath}'。")
        print("请确保您已经运行了仿真脚本来生成CSV文件。")
        return

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    print(f"成功读取 {filepath}，开始生成图表...")

    # --- 图表1: 所有轮次中，力的大小随时间的变化 ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x="step", y="magnitude", hue="episode", palette="viridis", legend="full")
    plt.title("Force Magnitude Over Time (All Episodes)", fontsize=16)
    plt.xlabel("Simulation Step")
    plt.ylabel("Force Magnitude")
    plt.legend(title="Episode")
    plt.tight_layout()
    plt.savefig("force_magnitude_vs_time.png")
    print("图表已保存: force_magnitude_vs_time.png")
    plt.close()

    # --- 图表2: 每个轮次中，力的分量随时间的变化 ---
    # 使用 melt 函数重塑数据，以便用 seaborn 进行分面绘图
    df_melted = df.melt(id_vars=['episode', 'step'], 
                        value_vars=['direction_x', 'direction_y', 'direction_z'],
                        var_name='component', 
                        value_name='value')

    g = sns.relplot(
        data=df_melted,
        x="step", y="value",
        hue="component", col="episode",
        kind="line", col_wrap=3,  # 每行显示的图表数量
        height=4, aspect=1.2,
        palette="bright"
    )
    g.fig.suptitle("Force Components Over Time by Episode", y=1.03, fontsize=16)
    g.set_axis_labels("Simulation Step", "Direction Component Value")
    g.set_titles("Episode {col_name}")
    g.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("force_components_vs_time.png")
    print("图表已保存: force_components_vs_time.png")
    plt.close()

    # --- 图表3: 每个轮次的最大作用力对比 ---
    plt.figure(figsize=(10, 6))
    max_forces = df.groupby('episode')['magnitude'].max().reset_index()
    sns.barplot(data=max_forces, x="episode", y="magnitude", palette="plasma")
    plt.title("Maximum Force Magnitude per Episode", fontsize=16)
    plt.xlabel("Episode")
    plt.ylabel("Maximum Force Magnitude")
    plt.tight_layout()
    plt.savefig("max_force_per_episode.png")
    print("图表已保存: max_force_per_episode.png")
    plt.close()

    print("\n所有图表生成完毕！")

if __name__ == "__main__":
    visualize_force_data()