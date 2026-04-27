import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_rewards(json_path, mode='reward'):
    """
    从 json 文件中读取数据 - 支持多种格式（适配你的数据结构）
    
    参数:
        json_path: str, JSON文件路径
        mode: str, 'reward' 或 'diff'
    
    返回:
        np.ndarray: 提取的数值数组
    
    支持的格式:
        1. 字典格式: {"episode_rewards": [1,2,3], ...}
        2. 记录列表: [{"episode_reward": 1, ...}, {"episode_reward": 2, ...}]
        3. 简单列表: [1, 2, 3] (仅支持 reward 模式)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"❌ 无法读取JSON文件 '{json_path}': {e}")
    
    # 处理字典格式
    
    if isinstance(data, dict):
        return _extract_from_dict(data, json_path, mode)
    
    # 处理列表格式（你的文件很可能是这种格式）
    elif isinstance(data, list):
        return _extract_from_list(data, json_path, mode)
    
    else:
        raise TypeError(f"文件 '{json_path}' 格式错误: 期望 dict 或 list，得到 {type(data)}")

def _extract_from_dict(data, json_path, mode):
    """从字典格式提取数据"""
    if mode == 'reward':
        key = "episode_rewards"
        if key not in data:
            raise KeyError(f"文件 '{json_path}' 中未找到 '{key}' 键")
        values = data[key]
    elif mode == 'diff':
        key_b, key_e = "episode_xi_B", "episode_xi_E"
        if key_b not in data or key_e not in data:
            missing = [k for k in (key_b, key_e) if k not in data]
            raise KeyError(f"文件 '{json_path}' 中未找到以下键: {missing}")
        values = np.array(data[key_b]) - np.array(data[key_e])
    else:
        raise ValueError(f"未知的 mode: '{mode}'")
    
    return np.array(values, dtype=float)

def _extract_from_list(data, json_path, mode):
    """从列表格式提取数据"""
    if len(data) == 0:
        raise ValueError(f"文件 '{json_path}' 包含空列表")
    
    # 如果是数字列表，直接返回
    if isinstance(data[0], (int, float)):
        if mode != 'reward':
            raise ValueError(f"文件 '{json_path}' 是简单的数字列表，不支持 '{mode}' 模式")
        return np.array(data, dtype=float)
    
    # 如果是字典列表，提取字段
    elif isinstance(data[0], dict):
        return _extract_from_record_list(data, json_path, mode)
    
    else:
        raise TypeError(f"文件 '{json_path}' 包含未知类型的列表元素: {type(data[0])}")

def _extract_from_record_list(data, json_path, mode):
    """从记录列表提取数据"""
    if mode == 'reward':
        # 尝试可能的字段名
        for key in ["episode_reward", "episode_rewards", "reward"]:
            if key in data[0]:
                values = [record.get(key) for record in data]
                return np.array(values, dtype=float)
        
        raise KeyError(f"文件 '{json_path}' 的字典元素中未找到 reward 相关字段")
    
    elif mode == 'diff':
        key_b, key_e = "episode_xi_B", "episode_xi_E"
        if key_b not in data[0] or key_e not in data[0]:
            missing = [k for k in (key_b, key_e) if k not in data[0]]
            raise KeyError(f"文件 '{json_path}' 的字典元素中未找到以下键: {missing}")
        
        data_b = np.array([record[key_b] for record in data], dtype=float)
        data_e = np.array([record[key_e] for record in data], dtype=float)
        return data_b - data_e

def moving_average(x, window_size=200):
    """滑动平均平滑"""
    if window_size <= 1:
        return x
    kernel = np.ones(window_size) / window_size
    ma = np.convolve(x, kernel, mode="valid")
    pad = np.full(window_size - 1, np.nan)
    return np.concatenate([pad, ma])

def get_color_palette(n):
    """
    生成 n 组高对比度的深浅颜色组合
    raw: 浅色 (_light)
    smooth: 深色 (_dark)
    """
    # 预定义8组高对比度颜色（可根据需要继续扩展）
    color_pairs = [
        # 蓝色系
        {'light': (0.6, 0.7, 1.0), 'dark': (0.0, 0.2, 0.8)},
        # 橙色系
        {'light': (1.0, 0.7, 0.4), 'dark': (0.8, 0.4, 0.0)},
        # 绿色系
        {'light': (0.5, 0.9, 0.5), 'dark': (0.0, 0.6, 0.0)},
        # 红色系
        {'light': (1.0, 0.5, 0.5), 'dark': (0.8, 0.0, 0.0)},
        # 紫色系
        {'light': (0.8, 0.6, 1.0), 'dark': (0.5, 0.0, 0.7)},
        # 青色系
        {'light': (0.4, 0.9, 0.9), 'dark': (0.0, 0.6, 0.6)},
        # 粉色系
        {'light': (1.0, 0.6, 0.8), 'dark': (0.8, 0.2, 0.5)},
        # 黄色系
        {'light': (1.0, 0.9, 0.4), 'dark': (0.8, 0.7, 0.0)},
    ]
    
    colors = []
    for i in range(n):
        # 循环使用预定义颜色对，但通过亮度微调确保区分度
        base_pair = color_pairs[i % len(color_pairs)]
        
        # 如果 n 超过预定义颜色数量，通过调整亮度生成变种
        variation_factor = (i // len(color_pairs)) * 0.15
        light_color = tuple(min(1.0, c + variation_factor) for c in base_pair['light'])
        dark_color = tuple(max(0.0, c - variation_factor) for c in base_pair['dark'])
        
        colors.append({
            'raw': light_color + (1.0,),
            'smooth': dark_color + (1.0,)
        })
    
    return colors

def get_color_palette1(n):
    """生成 n 组协调的颜色"""
    cmap = plt.colormaps['tab10'] if n <= 10 else plt.colormaps['tab20']
    colors = []
    for i in range(n):
        base_color = cmap(i % cmap.N)[:3]
        colors.append({
            'raw': base_color + (0.8,),
            'smooth': base_color + (1.0,)
        })
    return colors

def plot_rewards_multi(
    json_paths,
    labels,
    modes=None,
    window_size=200,
    episode_limit=None,
    save_dir="./figures",
    save_name="episode_rewards_comparison_multi.png",
    title="Episode Rewards Comparison",
    figsize=(16, 8)
):
    """绘制多条训练曲线对比图"""
    if len(json_paths) != len(labels):
        raise ValueError(f"json_paths 长度 ({len(json_paths)}) 与 labels 长度 ({len(labels)}) 不匹配！")
    
    n_curves = len(json_paths)
    if modes is None:
        modes = ['reward'] * n_curves
    elif len(modes) != n_curves:
        raise ValueError(f"modes 长度 ({len(modes)}) 与 json_paths 长度 ({n_curves}) 不匹配！")
    
    colors = get_color_palette(n_curves)
    plt.figure(figsize=figsize)
    
    successful_indices = []
    
    # 循环绘制每条曲线，添加错误处理
    for i, (json_path, label, mode, color_dict) in enumerate(zip(json_paths, labels, modes, colors)):
        try:
            print(f"\n{'='*50}")
            print(f"正在处理第 {i+1}/{n_curves} 条曲线: {label}")
            print(f"文件路径: {json_path}")
            
            rewards = load_rewards(json_path, mode=mode)
            
            episodes = np.arange(1, len(rewards) + 1)
            smoothed = moving_average(rewards, window_size=window_size)
            
            if episode_limit is not None:
                mask = episodes <= episode_limit
                episodes = episodes[mask]
                rewards = rewards[mask]
                smoothed = smoothed[:len(episodes)]
            
            # 绘制曲线
            plt.plot(episodes, rewards, color=color_dict['raw'], 
                    linewidth=0.5, label=f"{label} (raw)", alpha=0.7)
            
            
        except Exception as e:
            print(f"❌ 处理文件失败: {json_path}")
            print(f"错误信息: {e}")
            print(f"跳过此文件...")
            continue

    for i, (json_path, label, mode, color_dict) in enumerate(zip(json_paths, labels, modes, colors)):
        try:
            print(f"\n{'='*50}")
            print(f"正在处理第 {i+1}/{n_curves} 条曲线: {label}")
            print(f"文件路径: {json_path}")
            
            rewards = load_rewards(json_path, mode=mode)
            
            episodes = np.arange(1, len(rewards) + 1)
            smoothed = moving_average(rewards, window_size=window_size)
            
            if episode_limit is not None:
                mask = episodes <= episode_limit
                episodes = episodes[mask]
                rewards = rewards[mask]
                smoothed = smoothed[:len(episodes)]
            
            # 绘制曲线
            plt.plot(episodes, smoothed, color=color_dict['smooth'], 
                    linewidth=2.2, label=f"{label} (smoothed)")
            
            
        except Exception as e:
            print(f"❌ 处理文件失败: {json_path}")
            print(f"错误信息: {e}")
            print(f"跳过此文件...")
            continue
    
    # 检查是否有成功的曲线
    
    
    # 绘制图表元素
    # plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # 只显示平滑曲线的图例
    # handles, labels_legend = plt.gca().get_legend_handles_labels()
    # n_success = len(successful_indices)
    # smooth_handles = handles[1::2][:n_success]
    # smooth_labels = labels_legend[1::2][:n_success]
    # plt.legend(smooth_handles, smooth_labels, 
    #            loc="lower right", fontsize=10, framealpha=0.9)
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)
    
    # 保存图表
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    print(f"\n{'='*50}")
    print(f"✅ 图表保存成功: {save_path}")
    print(f"📊 成功绘制 {len(successful_indices)}/{n_curves} 条曲线")
    
    plt.show()

def plot_with_config(n, json_paths, labels, modes=None, **kwargs):
    """简化调用接口"""
    if len(json_paths) != n or len(labels) != n:
        raise ValueError(f"n={n} 但 json_paths 有 {len(json_paths)} 个，labels 有 {len(labels)} 个")
    
    if modes is not None and len(modes) != n:
        raise ValueError(f"modes 数量 ({len(modes)}) 必须与 n ({n}) 匹配！")
    
    return plot_rewards_multi(json_paths, labels, modes, **kwargs)

if __name__ == "__main__":
    # 配置
    n = 5
    json_paths = [
        "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_gen_log.json",
        "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_refresh_every_5000_log.json",
        "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen_teacher/Qwen_teacher_refresh_every_5000_log.json",
        "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/baselines/baseline_heuristic_threshold_avg5_log.json",
        "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/baselines/baseline_random_log.json",
    ]
    
    labels = [
        "Random Refresh",
        "Qwen2.5 Refresh",
        "Qwen2.5 Teacher",
        "baseline_greedy", 
        "baseline_random"
    ]

    modes_example = ['reward', 'reward', 'diff', 'reward', 'reward']
    
    plot_with_config(
        n=n,
        json_paths=json_paths,
        labels=labels,
        modes=modes_example,
        window_size=100,
        episode_limit=5000,
        save_dir="/models/Qwen/lyq_data/pythoncode/DeepSCmaster/figure",
        save_name="rewards_comparison_5_curves.pdf",
        title="Training Rewards Comparison (5 Methods)",
        figsize=(18, 8)
    )