import math

import matplotlib.pyplot as plt
import numpy as np

def plot_staircase(data: list[float], title: str = "Staircase Plot", xlabel: str = "Steps", ylabel: str = "Cumulative Sum"):
    """
    绘制一个阶梯状的数据示意图。
    阶梯之间的间隔由输入数组的元素决定。

    Args:
        data (list[float]): 输入数据数组，其元素代表阶梯的间隔。
        title (str): 图表的标题。
        xlabel (str): X轴的标签。
        ylabel (str): Y轴的标签。
    """
    if not data:
        print("输入数组为空，无法绘制阶梯图。")
        return

    x_values = [0]  # 起始点
    y_values = [0]  # 起始高度

    current_x = 0
    for i, interval in enumerate(data):
        # 绘制当前阶梯的水平段
        x_values.append(current_x)
        y_values.append(i + 1) # 提升到下一个高度

        current_x += interval # 移动到下一个水平位置
        x_values.append(current_x)
        y_values.append(i + 1) # 保持当前高度

    x_coords = [0.0] + list(np.cumsum(data))

    plot_x = [0.0]
    plot_y = [0.0] # 初始高度为0

    cumulative_sum = 0.0
    for i, val in enumerate(data):
        cumulative_sum += val
        plot_x.append(cumulative_sum) # 阶梯的终点X坐标
        plot_y.append(i + 1)        # 阶梯的终点Y坐标（高度）

    # 构造 x 和 y 序列
    final_x = [0.0]
    final_y = [0.0]

    current_height = 0.0
    current_x_pos = 0.0

    for i, interval in enumerate(data):
        # 水平段：从当前x位置到下一个x位置，高度保持不变
        current_x_pos += interval
        final_x.append(current_x_pos)
        final_y.append(current_height)

        # 垂直段：x位置不变，高度增加
        current_height += 1.0 # 每次增加1，或者可以设置为data[i]来表示阶梯高度由数据决定
        final_x.append(current_x_pos)
        final_y.append(current_height)

    # 绘制图像
    

    # plt.step 的直接使用方式
    y_step_coords = [0.0] + list(np.cumsum(data))
    x_step_coords = [0.0] + list(range(1, len(data) + 1))
    
    

    plt.figure(figsize=(10, 6))
    plt.plot([0,len(data)], [0,sum(data)], linestyle='-', linewidth=2, color='blue', label='Linear')
    plt.step(x_step_coords, y_step_coords, where='post', linestyle='-', marker='o', markersize=5, color='orange', label='Staircase Plot')

    # 添加每个阶梯的文本标签 (可选)
    # for i in range(len(data)):
    #     mid_x = (x_step_coords[i] + x_step_coords[i+1]) / 2
    #     mid_y = y_step_coords[i] # 文本放在阶梯的中间高度
    #     plt.text(mid_x, mid_y - 0.2, f'{data[i]}', ha='center', va='top', color='gray', fontsize=9)


    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    #plt.yticks(range(max(1, len(data) + 1))) # 确保Y轴刻度清晰
    plt.savefig(f"figures/{title}.png") 

def plot_histogram(data: list[float], title: str = "Histogram", xlabel: str = "Value", ylabel: str = "Frequency", yscale: str = "linear", bins: int = 20):
    """
    绘制数据的直方图分布。

    Args:
        data (list[float]): 输入数据数组。
        title (str): 图表的标题。
        xlabel (str): X轴的标签。
        ylabel (str): Y轴的标签。
        bins (int): 直方图的柱子数量。
    """
    if not data:
        print("输入数组为空，无法绘制直方图。")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', color='skyblue', alpha=0.7)
    plt.yscale(yscale) # 设置Y轴的刻度类型，可以是'linear'或'log'
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"figures/{title}.png") # 保存图像为PNG文件

def plot_comparison(x,y1,y2,title,x_title,y_title,y1_label,y2_label):
    """
    绘制两组数据的对比图。

    Args:
        x (list[float]): X轴数据。
        y1 (list[float]): 第一组Y轴数据。
        y2 (list[float]): 第二组Y轴数据。
        title (str): 图表的标题。
        x_title (str): X轴的标签。
        y_title (str): Y轴的标签。
        y1_label (str): 第一组数据的标签。
        y2_label (str): 第二组数据的标签。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=y1_label, marker='o')
    plt.plot(x, y2, label=y2_label, marker='o')
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/{title}.png") # 保存图像为PNG文件

def plot_dual_axis(x, y1, y2, x_label, y1_label, y2_label, title):
    """Plot TTFF and TBF on dual y-axes"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot TTFF on left y-axis
    ax1.plot(x, y1, marker='o', color='blue', linewidth=2, label='TTFF')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create right y-axis and plot TBF
    ax2 = ax1.twinx()
    ax2.plot(x, y2, marker='s', color='orange', linewidth=2, label='TBF')
    ax2.set_ylabel(y2_label, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"figures/{title}.png")

def plot_frame_times(start_time, data, title, clipp_range_max=300):
    # 1. 显式创建 figure 和 axes 对象
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for key, request_data in data.items():
        for conv in request_data:
            TTNF_list = [t for t, model_idx in conv["TTNF"]]
            sublist = [t - start_time for t in TTNF_list] 

            input_time = conv["req_input_time"] - start_time
            # Clip values to [0, clipp_range_max]
            clipped_sublist = [min(max(x, 0), clipp_range_max) for x in sublist]
        
            # Plot every n-th element
            n = max(1, len(clipped_sublist) // 50)  # Adjust n based on list length
            x_positions = clipped_sublist
            y_positions = [conv["req_id"]] * len(clipped_sublist)
        
            # 2. 使用 ax 对象进行绘图，而不是 plt
            ax.scatter(input_time, conv["req_id"], color='red', marker='x')
            ax.scatter(x_positions[::n], y_positions[::n], alpha=0.6, s=20)
    
    # 3. 使用 ax 对象设置属性
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('List Index')
    ax.set_xlim(0, clipp_range_max)  # Lock x-axis range to 0-clipp_range_max
    ax.grid(True, alpha=0.3)
    
    # 4. 调整布局并保存
    fig.tight_layout()
    fig.savefig(f"figures/critical/{title}.png")
    
    # 5. 返回 figure 对象
    return fig

def plot_time_distribution(data, interval, title, xlabel, ylabel, clipp_range_max=300):
    # 显式创建 figure 和 axes 对象
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = [i * interval for i in range(len(data))]
    
    # 在指定的 ax 上绘图
    ax.plot(indices, data, marker='o', linewidth=2, color='blue')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, clipp_range_max)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 注意：这里不要 savefig，因为我们要最后合并了再保存
    # 如果需要单独保存，可以保留，但返回 fig 是关键
    
    return fig  # 【关键修改】返回图表对象，而不是 plt 模块

def combine_figures_vertical(fig_list, title):
    num_figs = len(fig_list)
    if num_figs == 0:
        print("列表为空，无法合并。")
        return

    # 1. 计算行数（向上取整）
    n_rows = math.ceil(num_figs / 2)
    
    # 2. 创建画布：n_rows行，2列
    # 宽度设为单图的两倍左右(例如20)，高度根据行数调整
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))
    
    # 3. 将 axes 展平为一维列表，以便按顺序索引
    # 这样 axes[0] 是第一行第一列，axes[1] 是第一行第二列，axes[2] 是第二行第一列...
    axes = axes.flatten()
    
    for i, old_fig in enumerate(fig_list):
        old_ax = old_fig.axes[0]
        
        # --- 复制线条 ---
        for line in old_ax.get_lines():
            axes[i].plot(line.get_xdata(), line.get_ydata(), 
                         label=line.get_label(), 
                         marker=line.get_marker(), 
                         color=line.get_color(),
                         linewidth=line.get_linewidth())
        
        # --- 复制散点图 ---
        # 因为你的 plot_frame_times 用到了 scatter，所以必须处理 collections
        for coll in old_ax.collections:
            offsets = coll.get_offsets()
            # 有些散点图可能为空
            if len(offsets) > 0:
                axes[i].scatter(offsets[:, 0], offsets[:, 1],
                                s=coll.get_sizes(),
                                c=coll.get_facecolors(),
                                alpha=coll.get_alpha(),
                                edgecolors=coll.get_edgecolors(),
                                marker=coll.get_paths()[0].vertices[0] if coll.get_paths() else 'o'
                               )

        # --- 复制属性 ---
        axes[i].set_title(old_ax.get_title())
        axes[i].set_xlabel(old_ax.get_xlabel())
        axes[i].set_ylabel(old_ax.get_ylabel())
        axes[i].set_xlim(old_ax.get_xlim())
        axes[i].set_ylim(old_ax.get_ylim())
        axes[i].grid(True, alpha=0.3)
        
        if old_ax.get_legend():
            axes[i].legend()
            
        plt.close(old_fig)
    
    # 4. 隐藏多余的空白子图
    # 如果图的数量是奇数，最后一个 axes 是多余的
    for j in range(num_figs, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    save_path = f"figures/critical/{title}.png"
    plt.savefig(save_path)
    print(f"合并图表已保存至: {save_path}")

# 示例用法
if __name__ == "__main__":
    lbd_rates = [5,10,12,15,17,20]
    TTFF = [0.03468,0.08522,0.1235,0.4894,0.5152,0.5917]
    TBF = [0.0279,0.0618,0.0662,0.1731,0.1851,0.2130]
    plot_dual_axis(lbd_rates, TTFF, TBF, x_label="Lambda Rate", y1_label="TTFF violation rate", y2_label="TBF violation rate", title="Violation vs Lambda Rate")