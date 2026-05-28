# src/member-A/__init__.py
"""
Member A 模块：MRF 采样轨迹生成与前向仿真基础工具集
"""

# 从 utils.py 导出通用工具函数
from .utils import (
    save_npy,
    plot_traj_2d,
    plot_phantom
)

# 从 traj_generator.py 导出核心轨迹生成函数
from .traj_generator import generate_spiral_trajectory

# 明确导出列表（其他成员只能导入这里列出的内容）
__all__ = [
    "save_npy",
    "plot_traj_2d",
    "plot_phantom",
    "generate_spiral_trajectory"
]

# 说明：
# 1. phantom_prep.py 和 forward_sim.py 为独立运行脚本，无对外导出函数
# 2. 其他成员使用示例：
#    from src.member_A import generate_spiral_trajectory
#    traj = generate_spiral_trajectory(220e-3, 1e-3, 7e-3, 12, 24)