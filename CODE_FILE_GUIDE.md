# 代码文件作用说明

本文档按当前仓库结构说明各代码文件的职责、主要输入输出，以及调参后应该从哪一步重新运行。

## 目录结构

```text
.
├── CODE_FILE_GUIDE.md
├── README.md
├── main.py
├── pipeline_config.py
├── pyproject.toml
├── validate_results.py
├── scripts/
│   ├── build_dictionary.py
│   ├── generate_traj.py
│   ├── prepare_phantom.py
│   ├── run_forward_sim.py
│   ├── run_recon.py
│   └── template_matching.py
├── src/
│   └── mri_project/
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

本地运行时还会生成 `.numba_cache/`、`.pytest_cache/`、`data/processed/`、`data/output/` 等目录。这些主要是缓存和实验结果，通常不需要提交到 Git。

## 根目录文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 一键运行完整 2D MRF pipeline：轨迹生成、phantom 准备、字典构建、正演 k-space、LLR 子空间重建、template matching 绘图。会跳过已经存在且形状检查通过的中间结果。 |
| `pipeline_config.py` | 集中保存默认参数，例如图像大小、TR 数、子空间 rank、LLR 参数、轨迹参数、噪声水平等。文件末尾有“改参数后需要重跑哪些脚本”的注释。 |
| `validate_results.py` | 验证脚本。读取 phantom 真值、字典和重建系数图，执行 template matching，按 WM/GM/CSF 输出 T1/T2 与字典值的比较结果，并生成参考图与当前结果图的拼接 PNG。 |
| `README.md` | 项目简要说明，包括安装、推荐运行顺序和测试方式。 |
| `pyproject.toml` | Python 包与测试配置，声明依赖和 `pytest` 的 `src` 搜索路径。 |
| `CODE_FILE_GUIDE.md` | 当前文件，用于快速理解代码结构和运行流程。 |

## 推荐运行方式

完整流程：

```bash
python main.py
```

手动分步运行：

```bash
python scripts/generate_traj.py
python scripts/prepare_phantom.py
python scripts/build_dictionary.py
python scripts/run_forward_sim.py
python scripts/run_recon.py
python scripts/template_matching.py
python validate_results.py
```

只保存图片、不弹窗：

```bash
python scripts/template_matching.py --no-show
python validate_results.py --no-show
```

## 流程脚本

| 文件 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `scripts/generate_traj.py` | 生成 2D tiny-golden-angle spiral 轨迹。 | `pipeline_config.py` 中的轨迹参数 | `data/processed/traj_full_2d.npy`、`data/processed/first5_traj.png` |
| `scripts/prepare_phantom.py` | 生成简化 2D brain phantom，包含 T1、T2、PD 三通道。 | `IMG_SHAPE` | `data/processed/brain_param_map_2d.npy`、`data/processed/simulated_brain_phantom.png` |
| `scripts/build_dictionary.py` | 生成 flip-angle train，使用 EPG 构建 MRF 字典，并用 SVD 压缩到子空间。 | `N_TR`、`SUBSPACE_RANK`、`EPG_NUM_STATES` | `data/processed/mrf_dictionary_data.npz` |
| `scripts/run_forward_sim.py` | 用 phantom、轨迹和字典模拟多线圈非笛卡尔 k-space，并加入噪声。 | phantom、trajectory、dictionary | `data/output/mrf_kspace_2d_noisy.npy` |
| `scripts/run_recon.py` | 估计 coil sensitivity，执行带 LLR 正则的子空间重建。 | k-space、trajectory、dictionary bases | `data/output/reconstructed_coeff_maps.npy` |
| `scripts/template_matching.py` | 读取重建系数图和压缩字典，输出 T1/T2/PD 定量图。 | `reconstructed_coeff_maps.npy`、`mrf_dictionary_data.npz` | 弹窗显示定量图，并保存 `data/output/quantitative_maps.png` |

`scripts/template_matching.py` 支持：

```bash
python scripts/template_matching.py --save-path data/output/quantitative_maps.png
python scripts/template_matching.py --no-show
```

## 验证脚本

`validate_results.py` 用于结果检查，不参与生成原始重建结果。它会：

1. 读取 `data/processed/brain_param_map_2d.npy` 作为 phantom 真值。
2. 读取 `data/processed/mrf_dictionary_data.npz` 中的 T1/T2 字典网格和压缩字典。
3. 读取 `data/output/reconstructed_coeff_maps.npy`，重新执行 template matching。
4. 按 WM、GM、CSF 区域输出 T1/T2 的均值、MAE、exact match 比例和 PD 均值。
5. 保存当前定量图到 `data/output/quantitative_maps.png`。
6. 将参考 PNG 和当前代码输出 PNG 左右拼接，保存为 `data/output/validation_comparison.png`，并默认弹窗显示。

默认运行：

```bash
python validate_results.py
```

常用参数：

```bash
python validate_results.py --no-show
python validate_results.py --reference-png path/to/reference.png
python validate_results.py --generated-png data/output/quantitative_maps.png
python validate_results.py --comparison-png data/output/validation_comparison.png
```

## 参数文件

`pipeline_config.py` 中的主要参数分组如下：

| 参数 | 作用 |
| --- | --- |
| `IMG_SHAPE` | 重建图像大小，也是 phantom 默认大小。 |
| `N_COILS` | 正演模拟的线圈数。 |
| `N_TR` | MRF 时间点数量，会影响轨迹、字典、k-space 和重建。 |
| `SUBSPACE_RANK` | 字典 SVD 压缩后的子空间维度，也是重建系数图最后一维大小。 |
| `N_ITER` | LLR 重建迭代次数。 |
| `LAMBDA_LLR` | LLR 正则化权重 lambda。 |
| `STEP_SIZE` | 重建梯度下降步长。 |
| `PATCH_SHAPE` | LLR patch 大小。 |
| `CENTER_WIDTH`、`CALIB_WIDTH` | sensitivity map 估计时使用的中心 k-space 和 ESPIRiT 校准宽度。 |
| `SPIRAL_*`、`TINY_GOLDEN_ANGLE` | spiral trajectory 相关参数。 |
| `EPG_NUM_STATES` | EPG 仿真状态数。 |
| `NOISE_LEVEL` | 正演模拟加入的噪声水平。 |
| `RANDOM_SEED` | 正演噪声随机种子。 |

调参后建议按 `pipeline_config.py` 文件末尾的 rerun guide 手动运行对应脚本。注意：`main.py` 会跳过已有且 shape 合法的输出文件，所以只改参数后直接 `python main.py` 不一定会刷新旧结果。

## Python 包模块

### Dictionary

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/dictionary/__init__.py` | 导出字典模块公共 API。 |
| `src/mri_project/dictionary/epg.py` | 实现 MRF FISP EPG 信号仿真、flip-angle train 生成、T1/T2 字典构建和 SVD 子空间压缩。 |

### Forward

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/forward/__init__.py` | 导出 forward 模块公共 API。 |
| `src/mri_project/forward/trajectory.py` | 生成单条和多 TR 旋转的 2D variable-density spiral trajectory。 |
| `src/mri_project/forward/phantom.py` | 构造简化 brain phantom，包含 WM、GM、CSF 的 T1/T2/PD 参数。 |
| `src/mri_project/forward/simulation.py` | 根据 phantom、MRF 字典、轨迹和 coil sensitivity 模拟 k-space 数据。 |
| `src/mri_project/forward/io.py` | 保存 `.npy`，绘制轨迹图和 phantom 参数图。 |

### Quantification

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/quantification/__init__.py` | 导出 template matching 相关 API。 |
| `src/mri_project/quantification/template_matching.py` | 使用归一化复内积执行压缩字典 template matching，输出 T1、T2 和 PD map。 |

### Recon

| 文件 | 作用 |
| --- | --- |
| `src/mri_project/recon/__init__.py` | 导出重建模块公共 API。 |
| `src/mri_project/recon/nufft_ops.py` | 封装 2D NUFFT forward/adjoint 算子，优先使用 SigPy，必要时使用 direct NUDFT fallback。 |
| `src/mri_project/recon/subspace_ops.py` | 实现子空间展开/投影，以及单线圈、多线圈子空间 NUFFT 算子。 |
| `src/mri_project/recon/iterative.py` | 实现子空间梯度下降和带 LLR 正则的迭代重建。 |
| `src/mri_project/recon/regularization.py` | 实现 non-overlapping patch 的 LLR soft-threshold 和核范数计算。 |
| `src/mri_project/recon/sensitivity.py` | 从多线圈非笛卡尔 k-space 中估计 sensitivity maps。 |

## 测试文件

| 文件 | 作用 |
| --- | --- |
| `tests/test_smoke.py` | 冒烟测试：核心模块导入、轨迹生成、phantom 生成、正演模拟、EPG 字典构建和压缩。 |
| `tests/test_nufft_ops.py` | 测试 NUFFT forward/adjoint 的形状、有限性和伴随一致性。 |
| `tests/test_subspace_ops.py` | 测试子空间投影/展开，以及单线圈、多线圈子空间 NUFFT 算子。 |
| `tests/test_iterative_recon.py` | 测试子空间梯度下降和 LLR 重建在小规模问题上能降低 loss。 |
| `tests/test_regularization.py` | 测试 LLR 正则化函数的形状、dtype、阈值行为、核范数和非法参数检查。 |
| `tests/test_sensitivity.py` | 测试 k-space 时间平均、中心校准数据 gridding 和 sensitivity map 估计接口。 |
| `tests/test_template_matching.py` | 测试 template matching 能恢复精确字典条目、处理 zero signal，并检查 basis 维度不匹配报错。 |

运行测试：

```bash
pytest
```

## 当前数据流

```text
scripts/generate_traj.py
    -> data/processed/traj_full_2d.npy
    -> data/processed/first5_traj.png

scripts/prepare_phantom.py
    -> data/processed/brain_param_map_2d.npy
    -> data/processed/simulated_brain_phantom.png

scripts/build_dictionary.py
    -> data/processed/mrf_dictionary_data.npz

scripts/run_forward_sim.py
    <- data/processed/brain_param_map_2d.npy
    <- data/processed/traj_full_2d.npy
    <- data/processed/mrf_dictionary_data.npz
    -> data/output/mrf_kspace_2d_noisy.npy

scripts/run_recon.py
    <- data/output/mrf_kspace_2d_noisy.npy
    <- data/processed/traj_full_2d.npy
    <- data/processed/mrf_dictionary_data.npz
    -> data/output/reconstructed_coeff_maps.npy

scripts/template_matching.py
    <- data/output/reconstructed_coeff_maps.npy
    <- data/processed/mrf_dictionary_data.npz
    -> data/output/quantitative_maps.png

validate_results.py
    <- data/processed/brain_param_map_2d.npy
    <- data/processed/mrf_dictionary_data.npz
    <- data/output/reconstructed_coeff_maps.npy
    <- data/processed/simulated_brain_phantom.png
    <- data/output/quantitative_maps.png
    -> data/output/validation_comparison.png
```
