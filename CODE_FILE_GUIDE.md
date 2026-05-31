# 代码文件作用说明

本文档按仓库目录结构说明每个代码文件的主要作用，方便快速定位项目各模块职责。

## 项目配置

| 文件 | 作用 |
| --- | --- |
| `pyproject.toml` | Python 包构建与测试配置；声明项目依赖 `matplotlib`、`numpy`、`scipy`、`sigpy`，并配置 pytest 的源码路径和测试目录。 |

## 流程脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/generate_traj.py` | 生成 2D tiny-golden-angle 旋转螺旋轨迹，保存到 `data/processed/traj_full_2d.npy`，并绘制前 5 个 TR 的轨迹图。 |
| `scripts/prepare_phantom.py` | 生成简化 2D 脑部参数体模，包含 T1、T2 和 PD 三通道，保存参数图和可视化图片。 |
| `scripts/build_dictionary.py` | 生成 MRF 翻转角序列，构建 EPG 信号字典，对字典做 SVD 子空间压缩，并保存完整字典、压缩字典、子空间基和 T1/T2 网格。 |
| `scripts/run_forward_sim.py` | 读取体模、轨迹和字典，模拟多线圈非笛卡尔 k-space 采集数据，加入噪声后保存 noisy k-space。 |

## Python 包入口

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/__init__.py` | 顶层包声明，导出 `forward`、`recon`、`dictionary` 三个子包名称。 |

## 字典与物理仿真模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/dictionary/__init__.py` | 字典模块公共 API 入口，导出 EPG 仿真器、翻转角序列生成、字典构建和压缩函数。 |
| `src/mri_project/dictionary/epg.py` | 实现 MRF FISP EPG 物理信号仿真；提供默认 T1/T2 网格、翻转角序列生成、完整字典构建和 SVD 子空间压缩。 |

## 前向模拟模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/forward/__init__.py` | 前向模块公共 API 入口，导出轨迹生成和体模准备函数。 |
| `src/mri_project/forward/trajectory.py` | 生成单条 2D 可变密度螺旋轨迹，以及随 TR 旋转的 tiny-golden-angle 轨迹序列。 |
| `src/mri_project/forward/phantom.py` | 构造简化 2D 脑参数图，包含白质、灰质、脑脊液区域的 T1、T2 和 PD 值。 |
| `src/mri_project/forward/simulation.py` | 根据参数图、MRF 字典、轨迹和线圈敏感度图模拟多线圈 k-space 数据，并可加入复高斯白噪声。 |
| `src/mri_project/forward/io.py` | 提供数组保存和可视化辅助函数，包括 `.npy` 保存、轨迹图绘制和体模参数图绘制。 |

## 重建模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/recon/__init__.py` | 重建模块公共 API 入口，导出 NUFFT 子空间算子、迭代重建器和 LLR 正则化函数。 |
| `src/mri_project/recon/nufft_ops.py` | 封装 2D NUFFT 前向与伴随算子；优先使用 SigPy，缺失时用小规模 direct NUDFT fallback，并验证输入输出形状。 |
| `src/mri_project/recon/subspace_ops.py` | 实现 MRF 子空间展开/投影，以及单线圈、多线圈的子空间 NUFFT 前向和伴随算子。 |
| `src/mri_project/recon/iterative.py` | 实现子空间系数图的迭代重建；包含数据一致性梯度下降 `reconstruct_subspace_gd` 和带 LLR 正则的 `reconstruct_subspace_llr`。 |
| `src/mri_project/recon/regularization.py` | 实现局部低秩正则化工具，包括 non-overlapping patch 的奇异值软阈值 `llr_soft_threshold` 和 LLR 核范数计算。 |

## 测试文件

| 文件 | 作用 |
| --- | --- |
| `tests/test_smoke.py` | 冒烟测试核心模块导入、轨迹生成、体模生成、前向 k-space 模拟、EPG 仿真和小字典压缩流程。 |
| `tests/test_nufft_ops.py` | 测试 NUFFT 前向/伴随输出形状、复数有限性，以及前向算子和伴随算子的内积一致性。 |
| `tests/test_subspace_ops.py` | 测试子空间展开/投影、单线圈和多线圈子空间 NUFFT 算子的形状、有限性与伴随一致性。 |
| `tests/test_iterative_recon.py` | 测试子空间梯度下降和 LLR 重建在单线圈、多线圈小规模问题上能降低 loss，并验证无正则路径可运行。 |
| `tests/test_regularization.py` | 测试 LLR 正则化函数的形状、dtype、有限性、零阈值复制行为、核范数下降和非法参数检查。 |

