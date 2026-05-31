import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 模块 1：伪造数据生成器 (Mock Data Generator)
# ==========================================
def generate_mock_coeff_maps(t1_grid, t2_grid, dict_compressed, nx=100, ny=100, noise_level=0.03):
    """
    生成用于测试的伪造 2D 系数图 (Mock Data)。
    当成员 B 的真实数据就绪时，该函数将被弃用。
    """
    print("\n[Mock] 正在生成用于测试的伪造 2D 系数图...")
    num_bases = dict_compressed.shape[2]
    mock_coeff_maps = np.zeros((nx, ny, num_bases), dtype=np.complex128)

    # 设定：左半边画面是灰质 (GM)，右半边是白质 (WM)
    idx_t1_gm, idx_t2_gm = np.argmin(np.abs(t1_grid - 1350)), np.argmin(np.abs(t2_grid - 80))
    idx_t1_wm, idx_t2_wm = np.argmin(np.abs(t1_grid - 850)),  np.argmin(np.abs(t2_grid - 60))

    signal_gm = dict_compressed[idx_t1_gm, idx_t2_gm, :]
    signal_wm = dict_compressed[idx_t1_wm, idx_t2_wm, :]

    # 注入纯净信号并加入复数高斯噪声
    np.random.seed(99)
    for i in range(nx):
        for j in range(ny):
            noise = noise_level * (np.random.randn(num_bases) + 1j * np.random.randn(num_bases))
            if j < ny // 2:
                mock_coeff_maps[i, j, :] = signal_gm + noise
            else:
                mock_coeff_maps[i, j, :] = signal_wm + noise

    print(f"[Mock] 成功伪造系数图! 维度: {mock_coeff_maps.shape}")
    return mock_coeff_maps


# ==========================================
# 模块 2：核心匹配算法 (Template Matching)
# ==========================================
def execute_template_matching(coeff_maps, dict_compressed, t1_grid, t2_grid):
    """
    极速内积最大化匹配算法。
    接收系数图和字典，输出最终的定量参数图。
    """
    print("\n[Match] 开始执行大规模内积匹配 (Template Matching)...")
    start_time = time.time()
    
    nx, ny, num_bases = coeff_maps.shape
    num_t1, num_t2, _ = dict_compressed.shape

    # 1. 字典处理：归一化并拉平为 2D 矩阵 (28160, 5)
    norm_dict = np.linalg.norm(dict_compressed, axis=-1, keepdims=True)
    norm_dict[norm_dict == 0] = np.inf 
    dict_2d_flat = (dict_compressed / norm_dict).reshape(-1, num_bases) 

    # 2. 图像处理：归一化并拉平为 1D 长条 (Nx*Ny, 5)
    norm_maps = np.linalg.norm(coeff_maps, axis=-1, keepdims=True)
    norm_maps[norm_maps == 0] = np.inf
    maps_flat = (coeff_maps / norm_maps).reshape(-1, num_bases)

    # 3. 核心绝杀：矩阵内积 (点乘) 计算相似度
    similarity_matrix = np.abs(maps_flat @ dict_2d_flat.conj().T)
    
    # 4. 寻找最大值
    best_match_indices = np.argmax(similarity_matrix, axis=1)
    print(f"[Match] 匹配完成！{nx*ny} 个像素点比对耗时: {time.time() - start_time:.3f} 秒")

    # 5. 参数映射回物理真实值
    print("[Match] 正在映射定量参数...")
    idx_t1, idx_t2 = np.unravel_index(best_match_indices, (num_t1, num_t2))
    
    T1_map = t1_grid[idx_t1].reshape(nx, ny)
    T2_map = t2_grid[idx_t2].reshape(nx, ny)
    PD_map = norm_maps.reshape(nx, ny)
    
    return T1_map, T2_map, PD_map


# ==========================================
# 模块 3：可视化渲染 (Visualization)
# ==========================================
def plot_quantitative_maps(T1_map, T2_map, PD_map):
    """渲染并显示最终的 T1, T2, PD 图像"""
    print("\n[Render] 渲染最终图像...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(T1_map, cmap='magma', vmin=0, vmax=2000)
    axes[0].set_title('T1 Map (ms)')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(T2_map, cmap='viridis', vmin=0, vmax=150)
    axes[1].set_title('T2 Map (ms)')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(PD_map, cmap='bone')
    axes[2].set_title('PD Map (a.u.)')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle('MR Fingerprinting Quantitative Maps', fontsize=16)
    plt.tight_layout()
    plt.show()


# ==========================================
# 主控制流 (Main Control Flow)
# ==========================================
if __name__ == "__main__":
    print(">>> 启动 MRF 模板匹配流程 <<<")
    
    # 1. 加载字典弹药库 (无论测试还是真实运行，这一步都是必须的)
    print("\n[Init] 正在加载压缩字典...")
    import os

    # 1. 获取当前 template_matching.py 所在的文件夹绝对路径 (比如 .../scripts)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 往上跳一级，回到项目的根目录
    parent_dir = os.path.dirname(current_dir)
    
    # 3. 按照你的多级目录结构，精准向下拼接
    specified_folder = 'data'       # 第一层：指定的文件夹
    dict_sub_folder  = 'processed'  # 第二层：里面存放字典的子文件夹
    
    # 使用 os.path.join 自动跨多级目录拼接，它会自动处理好 Windows 的反斜杠 \
    dict_path = os.path.join(parent_dir, specified_folder, dict_sub_folder, 'mrf_dictionary_data.npz')
    
    print(f"\n[Init] 正在动态追踪多级目录，解析出的绝对路径为: \n{dict_path}")
    
    # 4. 稳妥加载
    dictionary_data = np.load(dict_path)
    dict_compressed = dictionary_data['compressed_dict']  # 👉 换成正确的标签名
    t1_grid = dictionary_data['t1']
    t2_grid = dictionary_data['t2']
    print(f"[Init] 成功加载! 字典维度: {dict_compressed.shape}")

    # 2. 获取前端图像数据 (Mock vs 真实数据 的切换点)
    # -------------------------------------------------------------
    # 【当前状态】：成员 B 还没有发数据，调用刚才拎出来的 Mock 函数
    coeff_maps_input = generate_mock_coeff_maps(t1_grid, t2_grid, dict_compressed)
    
    # 【未来状态】：等成员 B 的数据来了，直接注释掉上面那行，取消下面两行的注释：
    # print("\n[Data] 正在加载成员 B 的真实重建结果...")
    # coeff_maps_input = np.load('member_B_real_data.npy')
    # -------------------------------------------------------------

    # 3. 执行核心算法，获取物理参数图
    T1_result, T2_result, PD_result = execute_template_matching(
        coeff_maps_input, 
        dict_compressed, 
        t1_grid, 
        t2_grid
    )

    # 4. 可视化出图
    plot_quantitative_maps(T1_result, T2_result, PD_result)
    
    print("\n>>> 流程执行完毕 <<<")