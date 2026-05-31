# 代码文件作用说明

本文档根据当前仓库文件结构整理每个代码文件的职责，方便快速定位项目模块、运行脚本和测试覆盖范围。

## 当前目录结构

```text
.
├── .gitignore
├── CODE_FILE_GUIDE.md
├── README.md
├── pyproject.toml
├── scripts/
│   ├── build_dictionary.py
│   ├── generate_traj.py
│   ├── prepare_phantom.py
│   ├── run_forward_sim.py
│   └── template_matching.py
├── src/
│   └── mri_project/
│       ├── __init__.py
│       ├── dictionary/
│       │   ├── __init__.py
│       │   └── epg.py
│       ├── forward/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   ├── phantom.py
│       │   ├── simulation.py
│       │   └── trajectory.py
│       ├── quantification/
│       │   ├── __init__.py
│       │   └── template_matching.py
│       └── recon/
│           ├── __init__.py
│           ├── iterative.py
│           ├── nufft_ops.py
│           ├── regularization.py
│           ├── sensitivity.py
│           └── subspace_ops.py
└── tests/
    ├── test_iterative_recon.py
    ├── test_nufft_ops.py
    ├── test_regularization.py
    ├── test_sensitivity.py
    ├── test_smoke.py
    ├── test_subspace_ops.py
    └── test_template_matching.py
```

本地运行时还可能出现 `.numba_cache/`、`.pytest_cache/`、`data/processed/`、`data/output/`、`outputs/` 等目录；这些主要是缓存或生成结果，按当前 `.gitignore` 配置不应提交到 Git。

## 项目配置与文档

| 文件 | 作用 |
| --- | --- |
| `.gitignore` | 忽略 Python 缓存、虚拟环境、测试缓存、本地生成的 `.npy`/`.npz` 数据、`data/processed/`、`data/output/`、`outputs/` 和 `.numba_cache/` 等文件。 |
| `README.md` | 项目总览，说明包结构、安装命令、推荐脚本运行顺序和测试方式。 |
| `pyproject.toml` | Python 包构建与测试配置；声明项目依赖 `matplotlib`、`numpy`、`scipy`、`sigpy`，配置 `src` 布局和 pytest 搜索路径。 |
| `CODE_FILE_GUIDE.md` | 当前文件，按目录说明各代码文件职责。 |

## 流程脚本

这些脚本是面向实验流程的命令行入口，通常从仓库根目录运行。

| 文件 | 作用 |
| --- | --- |
| `scripts/generate_traj.py` | 生成 2D tiny-golden-angle 旋转螺旋轨迹，保存为 `data/processed/traj_full_2d.npy`，并输出前 5 个 TR 的轨迹图 `data/processed/first5_traj.png`。 |
| `scripts/prepare_phantom.py` | 生成简化 2D 脑部参数体模，包含 T1、T2 和 PD 三通道，保存为 `data/processed/brain_param_map_2d.npy` 和 `data/processed/simulated_brain_phantom.png`。 |
| `scripts/build_dictionary.py` | 生成 MRF 翻转角序列，使用 EPG 模型构建信号字典，对字典做 SVD 子空间压缩，并保存 `data/processed/mrf_dictionary_data.npz`。 |
| `scripts/run_forward_sim.py` | 读取体模、轨迹和字典，模拟多线圈非笛卡尔 k-space 采集数据，加入噪声后保存为 `data/output/mrf_kspace_2d_noisy.npy`。 |
| `scripts/template_matching.py` | 模板匹配流程入口；加载 `data/processed/mrf_dictionary_data.npz` 中的压缩字典，调用包内模板匹配函数，当前用 mock 子空间系数图测试流程，并显示 T1、T2、PD 定量图。 |

推荐运行顺序：

```bash
python scripts/generate_traj.py
python scripts/prepare_phantom.py
python scripts/build_dictionary.py
python scripts/run_forward_sim.py
python scripts/template_matching.py
```

其中 `template_matching.py` 当前仍使用 mock 系数图；真实重建系数图接入后，应将输入替换为重建模块输出。

## Python 包入口

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/__init__.py` | 顶层包声明，导出 `forward`、`recon`、`dictionary`、`quantification` 四个子包名称。 |

## 字典与物理仿真模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/dictionary/__init__.py` | 字典模块公共 API 入口，导出 `MRFEPGSimulator`、兼容别名 `MRF_EPG_Simulator`、翻转角序列生成、字典构建和压缩函数。 |
| `src/mri_project/dictionary/epg.py` | 实现 MRF FISP EPG 物理信号仿真；提供默认 T1/T2 网格、翻转角序列生成、完整字典构建、SVD 子空间压缩和一站式字典构建压缩流程。 |

## 前向模拟模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/forward/__init__.py` | 前向模块公共 API 入口，导出轨迹生成函数和体模准备函数。 |
| `src/mri_project/forward/trajectory.py` | 生成单条 2D 可变密度螺旋轨迹，以及随 TR 旋转的 tiny-golden-angle 轨迹序列。 |
| `src/mri_project/forward/phantom.py` | 构造简化 2D 脑参数图，包含白质、灰质、脑脊液等区域的 T1、T2 和 PD 值，并支持保存参数图和可视化图。 |
| `src/mri_project/forward/simulation.py` | 根据参数图、MRF 字典、轨迹和线圈敏感度图模拟多线圈 k-space 数据；可生成默认线圈敏感度并加入复高斯噪声。 |
| `src/mri_project/forward/io.py` | 提供数组保存和可视化辅助函数，包括 `.npy` 保存、2D 轨迹图绘制和体模参数图绘制。 |

## 定量参数估计模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/quantification/__init__.py` | 定量参数估计模块公共 API 入口，导出模板匹配和 mock 子空间系数图生成函数。 |
| `src/mri_project/quantification/template_matching.py` | 实现 MRF 压缩字典模板匹配；通过归一化复内积为每个像素寻找最佳 T1/T2 字典条目，并输出 T1、T2 和 PD 图。 |

## 重建模块

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/recon/__init__.py` | 重建模块公共 API 入口，导出子空间 NUFFT 算子、迭代重建函数、LLR 正则化函数和线圈敏感度估计函数。 |
| `src/mri_project/recon/nufft_ops.py` | 封装 2D NUFFT 前向与伴随算子；优先使用 SigPy，缺失时使用小规模 direct NUDFT fallback，并进行输入输出形状校验。 |
| `src/mri_project/recon/subspace_ops.py` | 实现 MRF 子空间展开/投影，以及单线圈、多线圈的子空间 NUFFT 前向和伴随算子。 |
| `src/mri_project/recon/iterative.py` | 实现子空间系数图的迭代重建；包含数据一致性梯度下降 `reconstruct_subspace_gd` 和带 LLR 正则的 `reconstruct_subspace_llr`。 |
| `src/mri_project/recon/regularization.py` | 实现局部低秩正则化工具，包括 non-overlapping patch 的奇异值软阈值 `llr_soft_threshold` 和 LLR 核范数 `llr_nuclear_norm`。 |
| `src/mri_project/recon/sensitivity.py` | 从多线圈、多 TR 非笛卡尔 k-space 中估计线圈敏感度图；包含时间均值、中心低频样本网格化和 SigPy ESPIRiT 调用。 |

## 测试文件

| 文件 | 作用 |
| --- | --- |
| `tests/test_smoke.py` | 冒烟测试核心模块导入、轨迹生成、体模生成、前向 k-space 模拟、EPG 仿真和小字典压缩流程。 |
| `tests/test_nufft_ops.py` | 测试 NUFFT 前向/伴随输出形状、复数有限性，以及前向算子和伴随算子的内积一致性。 |
| `tests/test_subspace_ops.py` | 测试子空间展开/投影、单线圈和多线圈子空间 NUFFT 算子的形状、有限性与伴随一致性。 |
| `tests/test_iterative_recon.py` | 测试子空间梯度下降和 LLR 重建在单线圈、多线圈小规模问题上能降低 loss，并验证无正则路径可运行。 |
| `tests/test_regularization.py` | 测试 LLR 正则化函数的形状、dtype、有限性、零阈值复制行为、核范数下降和非法参数检查。 |
| `tests/test_sensitivity.py` | 测试多 TR k-space 时间均值、中心校准 k-space 网格化，以及 ESPIRiT 敏感度图估计接口的输出形状和有限性。 |
| `tests/test_template_matching.py` | 测试模板匹配能恢复精确字典条目、mock 系数图生成的形状和确定性、零信号 PD 输出，以及 basis 维度不匹配的参数检查。 |

## 当前数据流

```text
generate_traj.py
    -> data/processed/traj_full_2d.npy

prepare_phantom.py
    -> data/processed/brain_param_map_2d.npy

build_dictionary.py
    -> data/processed/mrf_dictionary_data.npz

run_forward_sim.py
    <- trajectory + phantom + dictionary
    -> data/output/mrf_kspace_2d_noisy.npy

template_matching.py
    <- compressed dictionary
    <- mock coeff maps, later replace with reconstructed coeff maps
    -> displayed T1/T2/PD maps
```

## 文件结构建议

当前结构已经比较清楚：`src/mri_project/` 放可复用包代码，`scripts/` 放流程入口，`tests/` 放测试。后续如果项目继续扩展，可以考虑以下调整：

1. 如果可视化函数继续增多，可以新建 `src/mri_project/visualization/`，把轨迹图、体模图、定量图绘制集中管理；目前规模下保留在 `forward/io.py` 和脚本中也可以接受。
2. 如果要交付完整端到端流程，可以新增 `scripts/run_full_pipeline.py`，按顺序调用轨迹、体模、字典、前向仿真、重建和模板匹配，减少手动运行多个脚本的出错概率。
3. 如果后续要保留少量可复现实验输入，可以使用 `data/examples/` 存放小型示例数据；大体积生成结果继续放在 `data/processed/` 或 `data/output/` 并保持不提交。
4. 如果文档继续增加，可以新建 `docs/`，把本文件、算法说明、实验记录和结果说明集中到文档目录中；当前只有 README 和代码指南时，放在根目录也足够直观。
