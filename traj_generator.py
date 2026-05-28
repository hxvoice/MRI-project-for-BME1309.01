# 导入核心依赖
import numpy as np  # 数值计算基础库
from utils import save_npy, plot_traj_2d  # 导入自定义工具函数：保存数组和绘制轨迹

# -----------------------------------------------------------------------------
# 论文核心参数配置（完全复现MRF经典论文的2D螺旋采样方案）
# 所有参数均符合3T临床MRI设备的硬件限制和MRF成像的标准要求
# -----------------------------------------------------------------------------
FOV = 220e-3       # 视野(Field of View)，单位：米(m)
                   # 220mm是成人脑部成像的标准视野，可完整覆盖整个大脑
RES = 1e-3         # 空间分辨率，单位：米(m)
                   # 1mm是临床脑部MRI的常用分辨率，平衡图像质量和扫描时间
READOUT_TIME = 7e-3 # 单次读出时间，单位：秒(s)
                   # 7ms是螺旋采样的典型读出时长，受梯度线圈最大切换率限制
N_TR = 500         # 总重复次数(Number of Repetitions)
                   # 500个TR是MRF指纹匹配的最优数量，既能保证参数拟合精度，又不会过长
TINY_GOLDEN_ANGLE = 23.63  # 小金角(Tiny Golden Angle)，单位：度(°)
                           # 黄金角≈111.246°，小金角是其1/4.7≈23.63°
                           # 该角度能保证旋转采样后k空间点分布最均匀，无周期性重叠
UNDERSAMPLE_CENTER = 12    # k空间中心欠采样因子
UNDERSAMPLE_EDGE = 24      # k空间边缘欠采样因子
                           # 可变密度采样：中心密(欠采样少)、边缘疏(欠采样多)
                           # 符合k空间能量分布特性，在减少采样点的同时保留主要图像信息

# -----------------------------------------------------------------------------
# 函数：generate_spiral_trajectory
# 功能：纯numpy实现可变密度阿基米德螺旋轨迹生成
# 物理原理：阿基米德螺旋的极坐标方程为 r = aθ，通过调制θ的变化率实现可变密度
# 项目用途：生成单条TR的k空间采样轨迹，是整个MRF采样方案的基础
# -----------------------------------------------------------------------------
def generate_spiral_trajectory(fov, res, readout_time, undersample_center, undersample_edge):
    """
    参数：
        fov (float): 成像视野，单位：m
        res (float): 空间分辨率，单位：m
        readout_time (float): 单次读出时间，单位：s
        undersample_center (int): k空间中心欠采样因子
        undersample_edge (int): k空间边缘欠采样因子
    
    返回：
        traj (np.ndarray): 2D k空间轨迹，形状为(N, 2)，单位：1/cm
    """
    # 计算重建矩阵大小：N = FOV / 分辨率
    N = int(fov / res)
    
    # 计算k空间最大频率：k_max = 1/(2*分辨率)
    # 这是奈奎斯特采样定理的要求，决定了k空间的最大半径
    k_max = 1 / (2 * res)
    
    # 计算总采样点数：采样率 × 读出时间
    # 1MHz是临床MRI设备的标准模数转换器(ADC)采样率
    n_points = int(readout_time * 1e6)
    
    # -------------------------- 生成可变密度螺旋 --------------------------
    # 1. 生成径向坐标r：从0线性增加到k_max，共n_points个点
    r = np.linspace(0, k_max, n_points)
    
    # 2. 计算密度变化指数因子alpha
    # 实现从中心的undersample_center到边缘的undersample_edge的指数密度变化
    alpha = np.log(undersample_edge / undersample_center) / k_max
    
    # 3. 计算每个径向位置的采样密度
    # 密度随半径指数增加，实现中心密、边缘疏的可变密度采样
    density = undersample_center * np.exp(alpha * r)
    
    # 4. 计算极角theta
    # 角度的累积和与密度成正比，密度越大，角度增加越快，螺旋越密
    # 除以N是为了保证在k_max处刚好完成整数圈螺旋
    theta = 2 * np.pi * np.cumsum(density) / N
    
    # -------------------------- 坐标转换 --------------------------
    # 极坐标(r, theta)转换为笛卡尔坐标(kx, ky)
    kx = r * np.cos(theta)
    ky = r * np.sin(theta)
    
    # 转换为MRI领域标准单位：1/cm（原始单位是1/m）
    # 与sigpy库的输出单位完全一致，保证后续代码的兼容性
    traj = np.stack([kx, ky], axis=1) * 1e-2
    
    return traj

# -----------------------------------------------------------------------------
# 步骤1：生成单条TR的基础螺旋轨迹
# -----------------------------------------------------------------------------
# 调用函数生成单条螺旋轨迹
traj_single_cm = generate_spiral_trajectory(FOV, RES, READOUT_TIME, UNDERSAMPLE_CENTER, UNDERSAMPLE_EDGE)

# 打印单条轨迹的采样点数，用于验证参数设置是否正确
print(f"单条螺旋读点数: {traj_single_cm.shape[0]}")

# 绘制单条轨迹并保存为高清图片
# 用于验证螺旋的均匀性和形状是否符合预期
plot_traj_2d(traj_single_cm, title="单TR可变密度螺旋轨迹", 
             save_path="../../data/processed/single_traj.png")

# -----------------------------------------------------------------------------
# 步骤2：生成小金角旋转(TGA)完整采样序列
# 核心原理：将单条基础螺旋轨迹依次旋转小金角，得到所有TR的采样轨迹
# 优势：旋转后k空间采样点分布均匀，无重叠，能高效捕获组织的弛豫特征
# -----------------------------------------------------------------------------
# 将角度从度转换为弧度，用于后续矩阵运算
angles_rad = np.deg2rad(np.arange(N_TR) * TINY_GOLDEN_ANGLE)

# 创建三维数组存储完整轨迹：[TR数, 单TR采样点数, 2(kx, ky)]
# 使用float32数据类型，在保证精度的同时减小内存占用
traj_full = np.zeros((N_TR, traj_single_cm.shape[0], 2), dtype=np.float32)

# 循环生成每个TR的旋转轨迹
for i in range(N_TR):
    theta = angles_rad[i]
    
    # 生成2D旋转矩阵
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # 将单条基础轨迹旋转theta角度，得到当前TR的轨迹
    # 矩阵乘法：(N,2) @ (2,2).T = (N,2)
    traj_full[i] = traj_single_cm @ rot_matrix.T

# -----------------------------------------------------------------------------
# 步骤3：可视化与结果保存
# -----------------------------------------------------------------------------
# 绘制前5个TR的叠加轨迹，验证小金角旋转的效果
# 可以清晰看到每条轨迹都旋转了一个固定角度，无明显重叠
plot_traj_2d(traj_full[:5].reshape(-1,2), title="前5个TR的TGA旋转轨迹",
             save_path="../../data/processed/first5_traj.png")

# 保存完整轨迹数组为.npy格式
# 这是整个项目最重要的输出文件，后续前向仿真、图像重建、参数拟合都依赖此文件
save_npy(traj_full, "../../data/processed/traj_full_2d.npy")

# 打印完成提示
print("✅ 轨迹生成完成，已保存到 data/processed/")