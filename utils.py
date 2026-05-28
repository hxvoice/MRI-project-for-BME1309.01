# 导入核心依赖库
import numpy as np  # 数值计算基础库，所有数组运算、矩阵操作的核心
import matplotlib.pyplot as plt  # 数据可视化库，用于绘制轨迹图和体模参数图
import os  # 操作系统接口库，用于文件路径处理和目录创建

# -----------------------------------------------------------------------------
# matplotlib全局配置：解决Windows系统下中文显示乱码问题
# -----------------------------------------------------------------------------
# 指定默认字体为Windows系统自带的黑体（SimHei），支持所有中文汉字
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题，确保坐标轴上的负数能正常显示
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 函数：save_npy
# 功能：将numpy数组保存为.npy格式文件，自动创建不存在的目录
# 项目用途：统一保存轨迹数据、体模参数、k空间数据等所有中间结果
# -----------------------------------------------------------------------------
def save_npy(data, path):
    """
    参数：
        data (np.ndarray): 需要保存的numpy数组（任意维度）
        path (str): 保存文件的完整路径（如：../../data/processed/traj_full_2d.npy）
    """
    # 递归创建文件所在的目录
    # exist_ok=True：如果目录已经存在，不抛出异常，直接继续执行
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 以numpy原生二进制格式保存数组
    # 优点：读写速度快、保留数组的所有维度和数据类型信息、无精度损失
    np.save(path, data)
    
    # 打印保存成功提示，方便调试和确认文件生成
    print(f"已保存: {path}")

# -----------------------------------------------------------------------------
# 函数：plot_traj_2d
# 功能：绘制2D k空间采样轨迹，支持保存为高清图片
# 项目用途：验证螺旋轨迹的正确性，是MRF采样方案的核心可视化工具
# -----------------------------------------------------------------------------
def plot_traj_2d(traj, title="2D螺旋轨迹", save_path=None):
    """
    参数：
        traj (np.ndarray): 轨迹数组，形状为(N, 2)，N为采样点数，每一行是(kx, ky)坐标
        title (str): 图表标题，默认值为"2D螺旋轨迹"
        save_path (str, optional): 图片保存路径，为None时只显示不保存
    """
    # 创建6x6英寸的正方形画布
    # 必须用正方形，否则k空间轨迹会被拉伸变形，无法正确判断螺旋的均匀性
    plt.figure(figsize=(6, 6))
    
    # 绘制散点图，每个点代表一个k空间采样位置
    # s=0.1：点的大小，因为单条轨迹有7000个点，点太大会重叠看不清螺旋结构
    # c='k'：点的颜色为黑色，符合学术论文的绘图规范
    plt.scatter(traj[:, 0], traj[:, 1], s=0.1, c='k')
    
    # 设置坐标轴标签，单位为1/cm（MRI k空间的标准单位）
    plt.xlabel('kx (1/cm)')
    plt.ylabel('ky (1/cm)')
    
    # 设置图表标题
    plt.title(title)
    
    # 强制x轴和y轴的比例相等，确保圆形轨迹显示为正圆形
    # 这是绘制k空间轨迹的关键设置，否则会出现视觉误差
    plt.axis('equal')
    
    # 如果指定了保存路径，则保存图片
    if save_path:
        # 自动创建保存目录，避免因目录不存在导致的保存失败
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存为300DPI的高清PNG图片，适合插入学术报告和论文
        # bbox_inches='tight'：自动裁剪图片边缘的空白区域，减小文件体积
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图片窗口
    plt.show()

# -----------------------------------------------------------------------------
# 函数：plot_phantom
# 功能：绘制脑体模的T1、T2、质子密度(PD)参数分布图
# 项目用途：验证体模参数的合理性，确保仿真使用的组织参数符合3T MRI真实值
# -----------------------------------------------------------------------------
def plot_phantom(t1, t2, pd, save_path=None):
    """
    参数：
        t1 (np.ndarray): T1弛豫时间分布图，形状为(H, W)，单位为毫秒(ms)
        t2 (np.ndarray): T2弛豫时间分布图，形状为(H, W)，单位为毫秒(ms)
        pd (np.ndarray): 质子密度分布图，形状为(H, W)，已归一化到[0,1]区间
        save_path (str, optional): 图片保存路径，为None时只显示不保存
    """
    # 创建12x4英寸的画布，横向排列3张子图
    plt.figure(figsize=(12, 4))
    
    # 子图1：T1弛豫时间图
    plt.subplot(1, 3, 1)  # 1行3列，第1个位置
    plt.imshow(t1, cmap='gray')  # 用灰度图显示，符合医学图像的显示习惯
    plt.title('T1 (ms)')
    plt.colorbar()  # 添加颜色条，显示灰度值对应的T1数值
    
    # 子图2：T2弛豫时间图
    plt.subplot(1, 3, 2)  # 1行3列，第2个位置
    plt.imshow(t2, cmap='gray')
    plt.title('T2 (ms)')
    plt.colorbar()
    
    # 子图3：质子密度(PD)图
    plt.subplot(1, 3, 3)  # 1行3列，第3个位置
    plt.imshow(pd, cmap='gray')
    plt.title('PD (归一化)')
    plt.colorbar()
    
    # 如果指定了保存路径，则保存图片
    if save_path:
        # 自动创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存为300DPI高清PNG
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图片窗口
    plt.show()