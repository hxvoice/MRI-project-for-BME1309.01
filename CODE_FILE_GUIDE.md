# 代码文件作用说明

本文档用于说明当前仓库中每一个主要文件和目录的功能、用途、输入输出、依赖关系，以及它在整个 2D MRF pipeline 中所处的位置。项目整体目标是完成 MR Fingerprinting 的轻量化实验流程：生成或读取 phantom，生成 MRF 字典，模拟非笛卡尔多线圈 k-space，进行子空间 LLR 重建，再通过 template matching 得到 T1/T2/PD 定量图。

## 一、项目总流程

当前项目有两条主要运行路线：

1. 数学 phantom 路线：运行 `python main.py`。这条路线使用代码生成的三层圆形 brain phantom，适合做算法验证、调参和可复现实验。
2. DICOM-derived phantom 路线：运行 `python main_new.py --dicom-path your_file.dcm --no-show`。这条路线从 DICOM 图像中提取解剖结构和粗略组织分布，再给不同组织赋预设 T1/T2/PD 值，适合测试算法在更真实空间结构上的稳定性。

两条路线共享大部分后续步骤：轨迹生成、字典构建、k-space 正演、敏感度估计、子空间 LLR 重建、template matching 和结果验证。区别主要在 phantom 的来源。

```text
phantom / DICOM phantom
        |
trajectory + dictionary
        |
forward k-space simulation
        |
coil sensitivity estimation
        |
subspace LLR reconstruction
        |
template matching
        |
T1 / T2 / PD quantitative maps
```

## 二、目录结构

```text
.
├── .gitignore
├── 00000020.dcm
├── CODE_FILE_GUIDE.md
├── README.md
├── main.py
├── main_new.py
├── pipeline_config.py
├── pyproject.toml
├── validate_results.py
├── data/
│   ├── processed/
│   └── output/
├── scripts/
│   ├── build_dictionary.py
│   ├── generate_traj.py
│   ├── prepare_dicom_phantom.py
│   ├── prepare_phantom.py
│   ├── run_forward_sim.py
│   ├── run_recon.py
│   └── template_matching.py
├── src/
│   └── mri_project/
│       ├── __init__.py
│       ├── array_backend.py
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
    ├── test_array_backend.py
    ├── test_dicom_phantom.py
    ├── test_iterative_recon.py
    ├── test_nufft_ops.py
    ├── test_regularization.py
    ├── test_sensitivity.py
    ├── test_smoke.py
    ├── test_subspace_ops.py
    └── test_template_matching.py
```

本地运行时还会产生 `.numba_cache/`、`.cupy_cache/`、`.pytest_cache/`、`__pycache__/` 等缓存目录，以及 `data/processed/` 和 `data/output/` 中的实验产物。这些文件多数是可再生成的中间结果，不是核心源代码。

## 三、根目录文件

### `CODE_FILE_GUIDE.md`

当前文件。它不是程序入口，而是代码阅读指南。用途是帮助读者快速理解仓库结构、每个文件的职责、各步骤输入输出以及改参数后应重新运行哪些部分。

### `README.md`

项目简要说明文件。它给出项目定位、基础运行方式、验证脚本说明，以及 DICOM-derived phantom 的使用方法。相比本文件，`README.md` 更偏向给使用者快速启动；本文件更偏向给开发者理解代码。

### `pyproject.toml`

Python 项目配置文件。主要作用如下：

- 声明项目名 `mri-project-bme1309`、版本、Python 版本要求 `>=3.10`。
- 声明基础依赖：`matplotlib`、`numpy`、`pydicom`、`scipy`、`sigpy`。
- 声明测试依赖组 `test = ["pytest"]`。
- 声明 GPU 可选依赖组 `gpu = ["sigpy", "cupy-cuda12x"]`。
- 配置 setuptools 从 `src/` 目录发现 Python 包。
- 配置 pytest 的 `pythonpath = ["src"]` 和测试目录 `tests`。

这个文件的用处是让项目可以被当作标准 Python 包安装和测试，而不是只能靠手动改 `PYTHONPATH`。

### `.gitignore`

Git 忽略规则文件。它包含通用 Python 缓存、虚拟环境、构建产物、测试缓存、IDE 临时文件等忽略规则，并额外忽略本项目运行产生的数据和缓存：

- `.numba_cache/`
- `.cupy_cache/`
- `data/processed/`
- `data/output/`
- `*.npy`
- `*.npz`
- `mrf_dictionary_data.npz`

它的用途是避免把大型中间数据、缓存文件和机器相关文件提交到版本库。

### `00000020.dcm`

示例 DICOM 文件。它不是 Python 源代码，但会被 `scripts/prepare_dicom_phantom.py` 或 `main_new.py` 在未显式传入 `--dicom-path` 时作为候选 DICOM 输入之一。DICOM-derived phantom 路线会读取它的像素数据，用于生成真实解剖结构风格的 phantom。

注意：项目不会从该 DICOM 直接恢复真实临床 T1/T2。DICOM 只提供空间结构和强度分布；T1/T2/PD 是代码按组织标签赋的预设值。

## 四、配置和入口脚本

### `pipeline_config.py`

集中参数配置文件，是整个 pipeline 的默认参数来源。所有流程脚本和两个自动入口都会读取它。

主要参数分组如下：

| 参数组 | 代表参数 | 用途 |
| --- | --- | --- |
| 图像和规模 | `IMG_SHAPE`、`N_COILS`、`N_TR`、`SUBSPACE_RANK` | 决定图像大小、线圈数、MRF 时间点数和低秩子空间维度。 |
| 重建参数 | `N_ITER`、`LAMBDA_LLR`、`STEP_SIZE`、`PATCH_SHAPE` | 控制梯度/FISTA 迭代次数、LLR 正则强度、步长和 patch 大小。 |
| 敏感度估计 | `CENTER_WIDTH`、`CALIB_WIDTH`、`ESPIRIT_*` | 控制中心 k-space gridding 和 ESPIRiT coil sensitivity 估计。 |
| spiral 轨迹 | `SPIRAL_FOV`、`SPIRAL_RES`、`SPIRAL_READOUT_TIME`、`SPIRAL_ADC_RATE`、`TINY_GOLDEN_ANGLE` | 控制 variable-density spiral 和 tiny golden angle 旋转。 |
| EPG 和 flip angle | `EPG_NUM_STATES`、`EPG_TR`、`EPG_TE`、`EPG_TI`、`FA_*` | 控制 MRF FISP EPG 仿真和 flip-angle train。 |
| 字典网格 | `T1_GRID_RANGES`、`T2_GRID_RANGES`、`t1_grid()`、`t2_grid()` | 定义 template matching 使用的 T1/T2 离散字典网格。 |
| 数学 phantom | `PHANTOM_BRAIN_RADIUS`、`PHANTOM_WM_RADIUS`、`PHANTOM_CSF_RADIUS`、`PHANTOM_TISSUES` | 定义圆形 phantom 的组织区域和预设 T1/T2/PD。 |
| 正演噪声 | `NOISE_LEVEL`、`RANDOM_SEED` | 控制模拟 k-space 中加入的复高斯噪声。 |

文件末尾的 rerun guide 说明了改不同参数后要从哪一步重跑。例如只改 `N_ITER` 通常从 `scripts/run_recon.py` 开始即可；改 `N_TR` 则会影响轨迹、字典、正演、重建和 matching。

### `main.py`

数学 phantom 的完整自动流水线入口。运行：

```bash
python main.py
```

它串联 6 个步骤：

1. `scripts/generate_traj.py`
2. `scripts/prepare_phantom.py`
3. `scripts/build_dictionary.py`
4. `scripts/run_forward_sim.py`
5. `scripts/run_recon.py`
6. `scripts/template_matching.py`

它的核心功能不是直接实现算法，而是做流程调度、参数传递、结果检查和跳步判断。

重要内部机制：

- `parse_args()`：允许从命令行覆盖重建迭代次数、LLR lambda、步长、图像大小、patch 大小、CPU/CUDA 设备、batch size、敏感度估计宽度等。
- `FINGERPRINT_PATH = data/processed/pipeline_config_fingerprint.json`：记录每个步骤使用过的关键配置。
- `selected_config()`、`normalize_for_json()`、`config_diff()`：把配置转成可保存的 JSON，并比较当前配置和上次配置。
- `outputs_are_valid()`：检查输出文件是否存在、shape 是否和当前配置匹配。
- `outputs_are_current()`：检查输出是否比输入更新，避免用旧输入生成的结果。
- `run_step()`：统一执行或跳过某个流程脚本。如果输出存在、shape 合法、输入未更新、配置未变化，就跳过该步骤。

它的输入主要是 `pipeline_config.py` 和命令行参数；输出是完整 pipeline 的所有中间文件和最终图像。

### `main_new.py`

DICOM-derived phantom 的完整自动流水线入口。运行示例：

```bash
python main_new.py --dicom-path 00000020.dcm --no-show
```

它与 `main.py` 很像，但第 2 步不是生成数学 phantom，而是调用 `scripts/prepare_dicom_phantom.py` 从 DICOM 图像生成 phantom。

它的 6 个步骤是：

1. 生成 trajectory。
2. 准备 DICOM-derived phantom。
3. 构建 MRF 字典。
4. 正演模拟 k-space。
5. 子空间 LLR 重建。
6. template matching 绘图。

特殊功能：

- `find_dicom_path()`：如果没有传 `--dicom-path`，会在项目根目录递归查找 `.dcm` 文件并使用第一个。
- `DICOM_FINGERPRINT_PATH = data/processed/pipeline_config_fingerprint_dicom.json`：DICOM 路线使用单独的配置指纹文件，避免和数学 phantom 路线互相误判。
- `dicom_phantom_matches()`：检查当前 `brain_param_map_2d.npy` 是否符合 DICOM phantom 的预设组织值和图像 shape。
- 在 fingerprint 中额外记录 `phantom_source = "dicom"` 和 DICOM 文件绝对路径，保证换 DICOM 后能触发重跑。

它适合做更接近真实脑部结构的测试，但仍然不是临床定量工具。

### `validate_results.py`

结果验证和可视化对比脚本。运行：

```bash
python validate_results.py --no-show
```

它不生成 k-space，也不做重建；它读取已经生成的 phantom、dictionary 和 reconstructed coeff maps，然后重新做 template matching 和统计分析。

主要功能：

- 读取 `data/processed/brain_param_map_2d.npy` 作为 phantom 真值。
- 读取 `data/processed/mrf_dictionary_data.npz` 中的 `compressed_dict`、`t1`、`t2`。
- 读取 `data/output/reconstructed_coeff_maps.npy` 并调用 `match_dictionary_indices()`。
- 根据预测字典索引生成 T1/T2 map，并计算 PD map。
- 使用 phantom 的 PD 通道生成 foreground mask，把背景置零。
- 打印 shape 检查、字典网格检查、预测混淆信息和各组织的 T1/T2/PD 统计。
- 渲染当前结果到 `data/output/quantitative_maps.png`。
- 选择参考图：数学 phantom 默认用 `data/processed/simulated_brain_phantom.png`；DICOM phantom 默认用 `data/processed/dicom_phantom_preview.png`。
- 将参考图和当前输出图拼接为 `data/output/validation_comparison.png`。

它的用途是检查结果是否合理、是否落在字典网格上、不同组织是否被匹配到预期附近。

## 五、`scripts/` 流程脚本

这些文件是可单独运行的 pipeline 步骤。它们一般只负责读取磁盘数据、调用 `src/mri_project` 中的算法函数、保存结果。

### `scripts/generate_traj.py`

生成 2D tiny-golden-angle spiral 轨迹。

调用链：

- 读取 `pipeline_config.py` 的 spiral 参数。
- 调用 `mri_project.forward.generate_tga_trajectory()`。
- 调用 `mri_project.forward.io.save_npy()` 保存轨迹。
- 调用 `plot_traj_2d()` 保存前 5 个 TR 的轨迹预览图。

主要输入：

- `N_TR`
- `TINY_GOLDEN_ANGLE`
- `SPIRAL_FOV`
- `SPIRAL_RES`
- `SPIRAL_READOUT_TIME`
- `SPIRAL_UNDERSAMPLE_CENTER`
- `SPIRAL_UNDERSAMPLE_EDGE`
- `SPIRAL_ADC_RATE`

主要输出：

- `data/processed/traj_full_2d.npy`，shape 为 `(N_TR, n_samples, 2)`。
- `data/processed/first5_traj.png`，前 5 个 TR 的 spiral 轨迹图。

### `scripts/prepare_phantom.py`

生成数学 brain phantom。

调用链：

- 读取 `pipeline_config.py` 中的 phantom 参数。
- 调用 `mri_project.forward.prepare_phantom()`。
- 保存 T1/T2/PD 三通道参数图和预览 PNG。

主要输入：

- `IMG_SHAPE`
- `PHANTOM_BRAIN_RADIUS`
- `PHANTOM_WM_RADIUS`
- `PHANTOM_CSF_RADIUS`
- `PHANTOM_TISSUES`

主要输出：

- `data/processed/brain_param_map_2d.npy`，shape 为 `(H, W, 3)`，最后一维依次为 T1、T2、PD。
- `data/processed/simulated_brain_phantom.png`，数学 phantom 的 T1/T2/PD 可视化。

### `scripts/prepare_dicom_phantom.py`

从 DICOM 图像生成 anatomical phantom。

它的作用是把 DICOM 像素图像转换成项目后续流程能使用的 `(H, W, 3)` 参数图。流程包括读取 DICOM、强度归一化、resize、脑区 mask、粗组织分割、赋 T1/T2/PD 值、保存预览。

主要命令：

```bash
python scripts/prepare_dicom_phantom.py --dicom-path 00000020.dcm --overwrite --no-show
```

主要参数：

- `--dicom-path`：DICOM 文件路径；不传时搜索项目下第一个 `.dcm`。
- `--output-path`：输出参数图路径，默认 `data/processed/brain_param_map_2d.npy`。
- `--preview-path`：预览图路径，默认 `data/processed/dicom_phantom_preview.png`。
- `--overwrite`：允许覆盖已有参数图。
- `--no-show`：只保存图，不弹窗。

主要调用：

- `create_dicom_derived_param_map()`
- `save_npy()`
- `save_preview()`

主要输出：

- `data/processed/brain_param_map_2d.npy`
- `data/processed/dicom_phantom_preview.png`

补充说明：如果 `data/processed/mrf_dictionary_data.npz` 已存在，脚本会优先读取里面的 `t1` 和 `t2` 网格，并把组织预设值贴近到最近的字典网格值；否则使用 `pipeline_config.py` 中的网格。

### `scripts/build_dictionary.py`

生成 MRF 字典并压缩到低秩子空间。

调用链：

- `generate_mrf_fa_train_spline()` 生成平滑伪随机 flip-angle train。
- `MRFEPGSimulator()` 创建 EPG 模拟器。
- `build_signal_dictionary()` 在 T1/T2 网格上模拟完整时间序列字典。
- `compress_dictionary()` 对完整字典做 SVD，得到低秩 basis 和 compressed dictionary。
- 使用 `np.savez()` 保存所有字典数据。

主要输入：

- `N_TR`
- `FA_MIN`
- `FA_MAX`
- `FA_NUM_ANCHORS`
- `FA_RANDOM_SEED`
- `EPG_NUM_STATES`
- `EPG_TR`
- `EPG_TE`
- `EPG_TI`
- `T1_GRID_RANGES`
- `T2_GRID_RANGES`
- `SUBSPACE_RANK`

命令行参数：

- `--device cpu|cuda`
- `--gpu-device`
- `--dictionary-batch-size`

主要输出：

- `data/processed/mrf_dictionary_data.npz`

该 `.npz` 内含：

- `signal_dict`：完整 MRF 字典，shape 为 `(n_t1, n_t2, N_TR)`。
- `compressed_dict`：SVD 后的压缩字典，shape 为 `(n_t1, n_t2, SUBSPACE_RANK)`。
- `bases`：时间子空间基，shape 为 `(SUBSPACE_RANK, N_TR)`。
- `t1`：T1 网格。
- `t2`：T2 网格。
- `fa`：flip-angle train。

### `scripts/run_forward_sim.py`

执行正演模拟，把 phantom 参数图转换为多线圈非笛卡尔 k-space。

调用链：

- 读取 `brain_param_map_2d.npy`。
- 读取 `traj_full_2d.npy`。
- 读取 `mrf_dictionary_data.npz`。
- 调用 `mri_project.forward.simulation.simulate_kspace()`。
- 保存 noisy k-space。

主要输入：

- `data/processed/brain_param_map_2d.npy`
- `data/processed/traj_full_2d.npy`
- `data/processed/mrf_dictionary_data.npz`
- `N_COILS`
- `NOISE_LEVEL`
- `RANDOM_SEED`

主要输出：

- `data/output/mrf_kspace_2d_noisy.npy`，shape 为 `(N_COILS, N_TR, n_samples)`。

### `scripts/run_recon.py`

执行子空间 LLR 重建。

调用链：

- `load_inputs()` 读取 noisy k-space、trajectory 和 dictionary basis。
- `estimate_sens_maps_espirit()` 从多线圈非笛卡尔 k-space 中估计 coil sensitivity maps。
- `reconstruct_subspace_llr()` 执行带 LLR 正则的子空间 FISTA 重建。
- 将返回的 coeff maps 从 `(rank, H, W)` 移轴为 `(H, W, rank)`，保存给 template matching 使用。

主要输入：

- `data/output/mrf_kspace_2d_noisy.npy`
- `data/processed/traj_full_2d.npy`
- `data/processed/mrf_dictionary_data.npz`

命令行参数：

- `--n-iter`
- `--lambda-llr`
- `--step-size`
- `--img-shape H W`
- `--patch-shape H W`
- `--device cpu|cuda`
- `--gpu-device`
- `--center-width`
- `--calib-width`

主要输出：

- `data/output/reconstructed_coeff_maps.npy`，shape 为 `(H, W, SUBSPACE_RANK)`。

### `scripts/template_matching.py`

执行 MRF template matching 并绘制最终 T1/T2/PD 定量图。

调用链：

- `load_dictionary()` 读取 `compressed_dict`、`t1`、`t2`。
- 读取 `reconstructed_coeff_maps.npy`。
- 调用 `execute_template_matching()`。
- 调用 `plot_quantitative_maps()` 渲染 T1/T2/PD map。

主要输入：

- `data/processed/mrf_dictionary_data.npz`
- `data/output/reconstructed_coeff_maps.npy`

命令行参数：

- `--save-path`
- `--no-show`
- `--device cpu|cuda`
- `--gpu-device`
- `--batch-size`

主要输出：

- `data/output/quantitative_maps.png`

## 六、`src/mri_project/` 包模块

### `src/mri_project/__init__.py`

包顶层初始化文件。它声明当前包包含四个主要子模块：

- `forward`
- `recon`
- `dictionary`
- `quantification`

它没有复杂逻辑，主要作用是让 `mri_project` 成为可导入的 Python 包，并定义 `__all__`。

### `src/mri_project/array_backend.py`

CPU/CUDA 数组后端抽象层。它的用途是让字典生成、template matching、LLR 正则和重建代码可以在 NumPy 或 CuPy 上运行。

核心内容：

- `_load_cupy()`：惰性导入 CuPy，只有请求 CUDA 时才尝试导入。
- `_normalize_device()`：把 `"cpu"`、`"cuda"`、`"gpu"`、`cuda:0` 或整数设备号规范化。
- `ArrayBackend` dataclass：封装 `device`、`device_id` 和数组模块 `xp`。
- `ArrayBackend.to_device()`：把数组转到 NumPy/CuPy。
- `ArrayBackend.to_cpu()`：把 CuPy 数组拷回 NumPy。
- `ArrayBackend.all_finite()`：CPU 上检查有限性；CUDA 热路径中避免不必要同步。
- `get_array_backend()`：外部最常用入口。请求 CUDA 时会检查 CuPy 是否安装、GPU 是否可用、device id 是否合法。
- `to_cpu()` / `asnumpy_if_needed()`：通用数组转 NumPy 辅助函数。

这个文件的主要价值是隔离 CPU/GPU 差异，避免算法函数到处写 CuPy 判断。

## 七、`dictionary/` 字典模块

### `src/mri_project/dictionary/__init__.py`

字典模块的公共 API 导出文件。它把 `epg.py` 中的核心类和函数导出到 `mri_project.dictionary` 命名空间。

导出的对象包括：

- `MRFEPGSimulator`
- `MRF_EPG_Simulator`，旧名字兼容别名。
- `generate_mrf_fa_train_spline`
- `simulate_mrf_fisp_batch`
- `build_signal_dictionary`
- `compress_dictionary`
- `build_and_compress_dictionary`

用途是让脚本可以写：

```python
from mri_project.dictionary import MRFEPGSimulator, build_signal_dictionary
```

而不必直接引用 `mri_project.dictionary.epg`。

### `src/mri_project/dictionary/epg.py`

MRF FISP EPG 仿真和字典压缩的核心实现文件。

主要类：

- `MRFEPGSimulator`：单组织、单条 flip-angle train 的 Extended Phase Graph 模拟器。

`MRFEPGSimulator` 内部维护：

- `self.f`：横向相干状态，shape 为 `(2, num_states)`，对应 F+ 和 F-。
- `self.z`：纵向状态，shape 为 `(num_states,)`。

主要方法：

- `reset()`：重置 EPG 状态。
- `apply_rf()`：施加 RF 脉冲。
- `apply_relaxation()`：施加 T1/T2 弛豫。
- `apply_shift()`：FISP 相关的相干阶数移动。
- `simulate_mrf_fisp()`：给定 T1、T2、FA train、TR、TE、TI，输出一个组织的复数 MRF 时间信号。

主要函数：

- `generate_mrf_fa_train_spline()`：用随机 anchor 和 cubic spline 生成平滑 flip-angle train。
- `default_t1_grid()` / `default_t2_grid()`：提供默认 T1/T2 网格。
- `simulate_mrf_fisp_batch()`：批量模拟多个 T1/T2 组合，支持 CPU/CUDA backend。
- `build_signal_dictionary()`：在 T1/T2 网格上生成完整字典。只模拟 `T2 <= T1` 的有效组合，其余保持为 0。
- `compress_dictionary()`：把完整时间字典 reshape 成二维矩阵，做 SVD，取前 `rank` 个右奇异向量作为 basis，再投影得到 compressed dictionary。
- `build_and_compress_dictionary()`：一站式构建完整字典并压缩。

它是 `scripts/build_dictionary.py` 的算法核心，也是后续重建和 matching 的基础。

## 八、`forward/` 正演模块

### `src/mri_project/forward/__init__.py`

forward 模块公共 API 导出文件。导出：

- `prepare_phantom`
- `create_dicom_derived_param_map`
- `generate_spiral_trajectory`
- `generate_tga_trajectory`

用途是让脚本可以从 `mri_project.forward` 直接导入常用函数。

### `src/mri_project/forward/trajectory.py`

2D spiral trajectory 生成文件。

主要函数：

- `generate_spiral_trajectory()`：生成单条 variable-density Archimedean spiral。返回坐标 shape 为 `(n_samples, 2)`，坐标顺序是 `(ky, kx)`，单位按 SigPy NUFFT 需要的 image-pixel frequency units。
- `generate_tga_trajectory()`：把单条 spiral 按 tiny golden angle 逐 TR 旋转，得到 shape 为 `(n_tr, n_samples, 2)` 的轨迹栈。

重要参数：

- `fov`
- `res`
- `readout_time`
- `undersample_center`
- `undersample_edge`
- `adc_rate`
- `tiny_golden_angle`

该文件只负责轨迹数学生成，不负责保存文件；保存由 `forward/io.py` 或脚本完成。

### `src/mri_project/forward/phantom.py`

phantom 生成和 DICOM-derived phantom 处理文件。它同时支持数学 phantom 和 DICOM phantom。

数学 phantom 相关：

- `prepare_phantom()`：生成三层圆形 brain phantom，包括 WM、GM、CSF 三类组织区域，输出 `(H, W, 3)` 参数图。
- 参数图最后一维是 T1、T2、PD。
- 可选择保存 `.npy` 和预览 PNG。

DICOM 相关：

- `load_dicom_image()`：使用 `pydicom` 读取 DICOM 像素，应用 `RescaleSlope` 和 `RescaleIntercept`，并确保得到 2D float32 图像。
- `robust_normalize_image()`：用 1% 和 99% 分位数裁剪强度，然后归一化到 `[0, 1]`。
- `resize_image()`：把图像 resize 到 `IMG_SHAPE`。优先用 `skimage.transform.resize`，其次用 `scipy.ndimage.zoom`，最后退化为最近邻。
- `create_brain_mask()`：用 Otsu 或分位数阈值生成粗脑区 mask，并尽可能做小连通域移除、闭运算和孔洞填充。
- `segment_dicom_tissues()`：把 brain mask 内的像素按强度分成三类组织。优先用 `sklearn.cluster.KMeans`，否则用分位数 fallback。
- `assign_tissue_parameters()`：把组织标签映射成预设 T1/T2/PD，并可贴近到给定 T1/T2 字典网格。
- `create_dicom_derived_param_map()`：DICOM phantom 的总入口，串联读取、归一化、resize、mask、分割和参数赋值，并返回参数图和预览用数据。

常量：

- `TISSUE_LABELS`：定义 background、csf_like、wm_like、gm_like 的整数标签。

这个文件是 `scripts/prepare_phantom.py` 和 `scripts/prepare_dicom_phantom.py` 的核心算法来源。

### `src/mri_project/forward/simulation.py`

k-space 正演模拟文件。它把 T1/T2/PD 参数图、MRF 字典、trajectory 和 coil sensitivity 组合起来，生成多线圈非笛卡尔 k-space。

主要函数：

- `_nearest_indices()`：把 phantom 中的连续 T1/T2 值匹配到最近的字典网格索引。
- `_default_sens_maps()`：在没有外部 sensitivity map 时生成一组模拟 coil sensitivity maps。每个 coil 有不同空间中心和相位。
- `simulate_kspace()`：正演核心函数。

`simulate_kspace()` 的流程：

1. 检查 `param_map`、`signal_dict`、`traj` 维度。
2. 把每个像素的 T1/T2 匹配到最近字典条目。
3. 用 PD 乘上对应的 MRF 时间信号，形成 time image series。
4. 对每个 coil、每个 TR，把图像乘以 sensitivity map。
5. 调用 `nufft_forward()` 得到对应非笛卡尔 k-space。
6. 按 `noise_level` 加入复高斯噪声。

输出 shape 为 `(n_coils, n_tr, n_samples)`。

### `src/mri_project/forward/io.py`

小型保存和绘图工具文件。

主要函数：

- `save_npy()`：保存 NumPy 数组并自动创建父目录。
- `plot_traj_2d()`：绘制 2D k-space 坐标散点图。
- `plot_phantom()`：绘制 T1、T2、PD 三张 phantom 图。

它不包含 MRI 算法，只处理文件保存和 matplotlib 可视化。

## 九、`recon/` 重建模块

### `src/mri_project/recon/__init__.py`

重建模块公共 API 导出文件。它把 iterative、regularization、sensitivity、subspace_ops 中常用函数导出到 `mri_project.recon`。

导出内容包括：

- `reconstruct_subspace_gd`
- `reconstruct_subspace_llr`
- `llr_soft_threshold`
- `llr_nuclear_norm`
- `estimate_sens_maps_espirit`
- `grid_center_calibration_kspace`
- `time_average_kspace`
- `subspace_expand`
- `subspace_project`
- `subspace_nufft_forward`
- `subspace_nufft_adjoint`
- `multicoil_subspace_nufft_forward`
- `multicoil_subspace_nufft_adjoint`

### `src/mri_project/recon/nufft_ops.py`

2D NUFFT forward/adjoint 封装文件。

主要职责：

- 惰性导入 SigPy，避免导入项目时立即触发 Numba/SigPy 初始化。
- 在 SigPy 导入前设置可写的 `NUMBA_CACHE_DIR`。
- 对输入 image、coord、kspace、img_shape 做严格 shape 和 dtype 检查。
- 优先调用 `sigpy.nufft()` 和 `sigpy.nufft_adjoint()`。
- 当 SigPy 不存在且使用 CPU 时，提供 direct NUDFT fallback。
- 当请求 CUDA 但 SigPy 不可用时，明确报错。

主要函数：

- `nufft_forward(image, coord, device="cpu", device_id=0)`
- `nufft_adjoint(kspace, coord, img_shape, device="cpu", device_id=0)`
- `_direct_nudft_forward()`
- `_direct_nudft_adjoint()`

这个文件是正演模拟和重建算子的底层 Fourier 算子依赖。

### `src/mri_project/recon/subspace_ops.py`

子空间算子文件，把 MRF 时间子空间和 NUFFT 组合起来。

核心概念：

- `basis` shape 为 `(n_tr, rank)`。
- `coeff_maps` shape 为 `(rank, H, W)`。
- `time_images` shape 为 `(n_tr, H, W)`。

主要函数：

- `subspace_expand()`：执行 `basis @ coeff_maps`，从低秩系数图展开成时间序列图像。
- `subspace_project()`：执行 adjoint projection，把时间序列图像投影回低秩系数图。
- `subspace_nufft_forward()`：单线圈 forward，先子空间展开，再逐 TR 做 NUFFT。
- `subspace_nufft_adjoint()`：单线圈 adjoint，先逐 TR 做 NUFFT adjoint，再投影回子空间。
- `multicoil_subspace_nufft_forward()`：多线圈 forward，对每个 coil 乘 sensitivity map 后逐 TR 做 NUFFT。
- `multicoil_subspace_nufft_adjoint()`：多线圈 adjoint，对每个 coil 做 adjoint 后乘共轭 sensitivity map 并累加，再投影回子空间。

它是 `iterative.py` 中梯度下降和 LLR 重建的数据一致性算子的核心。

### `src/mri_project/recon/regularization.py`

LLR 正则化工具文件。

LLR 是 local low-rank regularization：把重建系数图按空间 patch 切块，每个 patch 视作一个矩阵，对奇异值做 soft-threshold。

主要函数：

- `_validate_patch_shape()`：检查 patch shape。
- `_validate_coeff_maps()`：检查 coeff maps shape、dtype 和有限性。
- `_iter_patch_slices()`：生成 non-overlapping patch 的 slice。
- `llr_soft_threshold()`：对每个 patch 做 SVD，奇异值减去 threshold 后截断到非负，再重建 patch。
- `llr_nuclear_norm()`：计算所有 patch 的核范数总和，用于监控正则项。

它被 `reconstruct_subspace_llr()` 调用，也被测试用于验证阈值行为和核范数下降。

### `src/mri_project/recon/sensitivity.py`

coil sensitivity map 估计文件。

主要流程：

1. `time_average_kspace()`：对多 TR k-space 沿时间维平均，得到每个 coil 的代表性 k-space。
2. `grid_center_calibration_kspace()`：选取中心低频非笛卡尔采样点，用最近邻方式 gridding 到笛卡尔校准数组。
3. `estimate_sens_maps_espirit()`：调用 `sigpy.mri.app.EspiritCalib` 估计 coil sensitivity maps。

主要函数：

- `_ensure_numba_cache_dir()`：为 SigPy/Numba 设置可写缓存路径。
- `_validate_multicoil_time_kspace()`：检查 k-space shape `(n_coils, n_tr, n_samples)`。
- `_validate_coord()`：检查 trajectory shape `(n_tr, n_samples, 2)`。
- `time_average_kspace()`
- `grid_center_calibration_kspace()`
- `estimate_sens_maps_espirit()`

输出 sensitivity maps shape 为 `(n_coils, H, W)`，供 `run_recon.py` 和多线圈子空间算子使用。

### `src/mri_project/recon/iterative.py`

迭代重建核心文件，实现 plain gradient descent 和带 LLR 正则的 FISTA 重建。

主要函数：

- `_validate_inputs()`：统一检查 k-space、basis、coord、img_shape、sens_maps、迭代参数和 dtype。
- `reconstruct_subspace_gd()`：只求解数据一致性项的子空间梯度下降。
- `reconstruct_subspace_llr()`：使用 FISTA + LLR soft-threshold 的子空间重建。

重要细节：

- 两个重建函数都会先用 `kspace_max = max(abs(kspace))` 做全局归一化，让优化数值尺度更稳定。
- loss 打印时再乘回 `kspace_max ** 2`，使日志尺度接近原始物理尺度。
- 输出 `coeff_maps` 时也乘回 `kspace_max`，恢复绝对尺度。
- 如果没有 `sens_maps`，使用单线圈子空间 NUFFT 算子。
- 如果有 `sens_maps`，使用多线圈 `P F S Phi` 算子和它的 adjoint。
- `reconstruct_subspace_llr()` 每轮执行梯度步、LLR soft-threshold、loss 计算和 FISTA momentum 更新。

它是 `scripts/run_recon.py` 的核心算法文件。

## 十、`quantification/` 定量模块

### `src/mri_project/quantification/__init__.py`

定量模块公共 API 导出文件。导出：

- `execute_template_matching`
- `generate_mock_coeff_maps`
- `match_dictionary_indices`

### `src/mri_project/quantification/template_matching.py`

MRF template matching 文件。它把重建得到的低秩系数图匹配到压缩字典条目，输出 T1/T2/PD map。

主要函数：

- `generate_mock_coeff_maps()`：根据给定 compressed dictionary 生成简易模拟 coeff maps，用于测试 template matching。
- `match_dictionary_indices()`：核心匹配函数。先归一化 `coeff_maps` 和 `dict_compressed`，再计算归一化复内积相似度，返回每个像素最匹配的字典扁平索引和 PD map。
- `execute_template_matching()`：完整定量入口。调用 `match_dictionary_indices()` 后，把扁平索引用 `np.unravel_index()` 转成 T1/T2 网格索引，最终返回 T1 map、T2 map、PD map。

匹配依据：

- 相似度为 `abs(normalized_coeff @ normalized_dict.conj().T)`。
- T1/T2 来自最相似字典条目的网格坐标。
- PD map 使用每个像素 coefficient vector 的范数。

支持 CPU/CUDA 后端和 batch matching，适合显存有限时分批处理像素。

## 十一、测试文件

### `tests/test_smoke.py`

冒烟测试文件，用于确认核心模块能以小规模数据跑通。

覆盖内容：

- `generate_tga_trajectory()` 的输出 shape。
- `prepare_phantom()` 的输出 shape。
- `simulate_kspace()` 能生成复数、有限、多线圈 k-space。
- `MRFEPGSimulator.simulate_mrf_fisp()` 能输出复数时间信号。
- 旧参数名 `T1`/`T2` 与新参数名 `t1`/`t2` 兼容。
- `simulate_mrf_fisp_batch()` 与单条 EPG 模拟结果一致。
- `build_signal_dictionary()`、`compress_dictionary()`、`build_and_compress_dictionary()` 的小规模 shape 正确。

### `tests/test_array_backend.py`

CPU/CUDA 后端测试文件。

覆盖内容：

- CPU backend 在没有 CuPy 时也必须可用。
- 请求 CUDA 时，如果不可用，应给出包含 CUDA 或 CuPy 的清晰错误。
- 如果 CUDA 可用，template matching 的 GPU 结果应和 CPU 结果一致。
- 如果 CUDA 可用，LLR soft-threshold 的 GPU 结果应和 CPU 结果一致。
- 如果 CUDA 可用，字典生成和压缩的 GPU 结果应和 CPU 结果在容差内一致。

这个测试文件可以在无 GPU 环境下自动 skip GPU 专项测试。

### `tests/test_dicom_phantom.py`

DICOM phantom 辅助函数测试文件。

覆盖内容：

- `robust_normalize_image()` 输出 float32，范围在 `[0, 1]`。
- `create_brain_mask()` 能生成与输入同 shape 的 mask。
- `segment_dicom_tissues()` 能生成组织标签。
- `assign_tissue_parameters()` 能生成 `(H, W, 3)` 参数图。
- background 标签对应的参数应为 0。
- 参数图应全为有限值，并且 foreground 中 PD 非零。

该测试不依赖真实 DICOM 文件，而是构造一个小型合成图像。

### `tests/test_nufft_ops.py`

NUFFT 算子测试文件。

辅助函数：

- `make_random_phantom()`：生成带随机 blob 和轻微相位的复数 phantom。
- `make_spiral_like_coord()`：生成类似 spiral 的非笛卡尔采样坐标。

测试内容：

- `nufft_forward()` 输出 shape 为 `(n_samples,)`。
- `nufft_adjoint()` 输出 shape 为原图像 shape。
- forward 和 adjoint 都应输出复数且有限。
- 检查伴随一致性：`<Ax, y>` 与 `<x, A*y>` 的相对误差应小于阈值。

### `tests/test_subspace_ops.py`

子空间算子测试文件。

辅助函数：

- `make_complex_basis()`：生成 QR 正交化的复数 basis。
- `make_coeff_maps()`：生成小型复数 coefficient maps。
- `make_subspace_coord()`：生成每个 TR 一条简单 spiral-like 轨迹。
- `make_sens_maps()`：生成模拟 coil sensitivity maps。

测试内容：

- `subspace_expand()` 和 `subspace_project()` shape、复数性、有限性。
- 单线圈 `subspace_nufft_forward()` / `subspace_nufft_adjoint()` shape 和伴随一致性。
- 多线圈 `multicoil_subspace_nufft_forward()` / `multicoil_subspace_nufft_adjoint()` shape 和伴随一致性。

### `tests/test_iterative_recon.py`

迭代重建测试文件。

覆盖内容：

- `reconstruct_subspace_gd()` 在单线圈小问题上 loss 下降。
- `reconstruct_subspace_gd()` 在多线圈 sensitivity map 情况下 loss 下降。
- `reconstruct_subspace_llr()` 在单线圈小问题上 loss 下降。
- `reconstruct_subspace_llr()` 在多线圈情况下 loss 下降。
- `lambda_llr=0.0` 时 LLR 重建仍能运行，相当于无正则的 FISTA 路径。

这些测试使用很小的图像和 TR 数，目的是验证算法方向正确和接口可用，而不是验证最终医学图像质量。

### `tests/test_regularization.py`

LLR 正则测试文件。

覆盖内容：

- `llr_soft_threshold()` 保持 shape、dtype、复数性和有限性。
- threshold 为 0 时返回 coeff maps 的拷贝，数值与输入一致。
- 正 threshold 应降低 patch nuclear norm。
- 非正 patch shape 应触发 `AssertionError`。
- 负 threshold 应触发 `AssertionError`。

### `tests/test_sensitivity.py`

coil sensitivity 估计测试文件。

覆盖内容：

- `time_average_kspace()` 输出 shape 和手动 `np.mean(axis=1)` 一致。
- `grid_center_calibration_kspace()` 能把中心 k-space 样本 gridding 成 `(n_coils, H, W)` 校准数组。
- `estimate_sens_maps_espirit()` 会调用 SigPy ESPIRiT 接口并返回正确 shape。

为了避免测试依赖真实 SigPy ESPIRiT 行为，该文件通过 `install_fake_sigpy()` 注入假的 `sigpy.mri.app.EspiritCalib`，只验证调用路径和接口契约。

### `tests/test_template_matching.py`

template matching 测试文件。

覆盖内容：

- `execute_template_matching()` 能恢复精确的字典条目。
- `match_dictionary_indices()` 返回正确的扁平 argmax 索引。
- `generate_mock_coeff_maps()` 输出 shape 正确且固定 seed 下确定性一致。
- coeff maps 和 compressed dictionary 的 basis 维度不匹配时应报 `ValueError`。
- zero signal 的 PD map 应为 0。

## 十二、数据目录和生成文件

### `data/processed/`

用于保存中间处理结果。常见文件：

| 文件 | 产生者 | 用途 |
| --- | --- | --- |
| `traj_full_2d.npy` | `scripts/generate_traj.py` | 完整 `(N_TR, n_samples, 2)` 非笛卡尔轨迹。 |
| `first5_traj.png` | `scripts/generate_traj.py` | 前 5 个 TR 的轨迹预览图。 |
| `brain_param_map_2d.npy` | `scripts/prepare_phantom.py` 或 `scripts/prepare_dicom_phantom.py` | 后续正演使用的 T1/T2/PD 参数图。 |
| `simulated_brain_phantom.png` | `scripts/prepare_phantom.py` | 数学 phantom 预览图。 |
| `dicom_phantom_preview.png` | `scripts/prepare_dicom_phantom.py` | DICOM-derived phantom 预览图。 |
| `mrf_dictionary_data.npz` | `scripts/build_dictionary.py` | 完整字典、压缩字典、basis、T1/T2 网格和 FA train。 |
| `pipeline_config_fingerprint.json` | `main.py` | 数学 phantom 自动流程的配置指纹。 |
| `pipeline_config_fingerprint_dicom.json` | `main_new.py` | DICOM phantom 自动流程的配置指纹。 |

### `data/output/`

用于保存正演、重建和最终结果。常见文件：

| 文件 | 产生者 | 用途 |
| --- | --- | --- |
| `mrf_kspace_2d_noisy.npy` | `scripts/run_forward_sim.py` | 多线圈 noisy k-space，重建输入。 |
| `reconstructed_coeff_maps.npy` | `scripts/run_recon.py` | 重建得到的低秩系数图，template matching 输入。 |
| `quantitative_maps.png` | `scripts/template_matching.py` 或 `validate_results.py` | T1/T2/PD 定量图。 |
| `validation_comparison.png` | `validate_results.py` | 参考图和当前输出图的拼接对比。 |

### 缓存目录

| 目录 | 用途 |
| --- | --- |
| `.numba_cache/` | SigPy/Numba JIT 缓存。 |
| `.cupy_cache/` | CuPy kernel 缓存。 |
| `.pytest_cache/` | pytest 缓存。 |
| `__pycache__/` | Python 字节码缓存。 |

这些缓存目录可以删除，删除后下次运行会重新生成。

## 十三、推荐运行方式

数学 phantom 完整流程：

```bash
python main.py
python validate_results.py --no-show
```

DICOM-derived phantom 完整流程：

```bash
python main_new.py --dicom-path 00000020.dcm --no-show
python validate_results.py --no-show
```

手动分步运行数学 phantom：

```bash
python scripts/generate_traj.py
python scripts/prepare_phantom.py
python scripts/build_dictionary.py
python scripts/run_forward_sim.py
python scripts/run_recon.py
python scripts/template_matching.py --no-show
python validate_results.py --no-show
```

手动分步运行 DICOM phantom：

```bash
python scripts/generate_traj.py
python scripts/prepare_dicom_phantom.py --dicom-path 00000020.dcm --overwrite --no-show
python scripts/build_dictionary.py
python scripts/run_forward_sim.py
python scripts/run_recon.py
python scripts/template_matching.py --no-show
python validate_results.py --no-show
```

运行测试：

```bash
pytest
```

## 十四、调参后应该重跑什么

| 修改内容 | 建议重跑 |
| --- | --- |
| 只改 `N_ITER`、`LAMBDA_LLR`、`STEP_SIZE`、`PATCH_SHAPE` | `scripts/run_recon.py`、`scripts/template_matching.py`、`validate_results.py` |
| 改 sensitivity 参数 `CENTER_WIDTH`、`CALIB_WIDTH`、`ESPIRIT_*` | `scripts/run_recon.py`、`scripts/template_matching.py`、`validate_results.py` |
| 改 `N_COILS`、`NOISE_LEVEL`、`RANDOM_SEED` | `scripts/run_forward_sim.py`、`scripts/run_recon.py`、`scripts/template_matching.py`、`validate_results.py` |
| 改数学 phantom 参数 `PHANTOM_*` | `scripts/prepare_phantom.py`、`scripts/run_forward_sim.py`、`scripts/run_recon.py`、`scripts/template_matching.py`、`validate_results.py` |
| 换 DICOM 文件 | `scripts/prepare_dicom_phantom.py --overwrite` 及其后所有步骤 |
| 改 trajectory 参数 `SPIRAL_*`、`TINY_GOLDEN_ANGLE` | `scripts/generate_traj.py` 及其后所有步骤 |
| 改字典参数 `N_TR`、`SUBSPACE_RANK`、`EPG_*`、`FA_*`、`T1_GRID_RANGES`、`T2_GRID_RANGES` | 通常从 `scripts/generate_traj.py` 和 `scripts/build_dictionary.py` 开始重跑，并重跑后续所有步骤 |

`main.py` 和 `main_new.py` 会记录配置指纹并自动判断哪些步骤需要重跑。但如果手动替换了中间文件、强行复制旧结果，或者想完全刷新实验，最稳妥的方式是删除 `data/processed/` 和 `data/output/` 中相关产物后重跑。

## 十五、当前数据流总表

```text
scripts/generate_traj.py
    -> data/processed/traj_full_2d.npy
    -> data/processed/first5_traj.png

scripts/prepare_phantom.py
    -> data/processed/brain_param_map_2d.npy
    -> data/processed/simulated_brain_phantom.png

scripts/prepare_dicom_phantom.py
    <- *.dcm
    <- data/processed/mrf_dictionary_data.npz  (optional grid source)
    -> data/processed/brain_param_map_2d.npy
    -> data/processed/dicom_phantom_preview.png

scripts/build_dictionary.py
    <- pipeline_config.py
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
    <- data/processed/simulated_brain_phantom.png or data/processed/dicom_phantom_preview.png
    -> data/output/quantitative_maps.png
    -> data/output/validation_comparison.png
```
