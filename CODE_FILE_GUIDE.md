# 代码文件说明和运行指南

本文档面向第一次打开本项目的人。只要从仓库根目录运行命令，并按本文准备数据，就可以完成两条 MRF 流程：

1. 真实 cMRF scanner 数据流程：读取外部 cMRF 原始数据，提取 k-space、轨迹、线圈敏感度图和字典，再做子空间 LLR 重建和 template matching。
2. 模拟 2D MRF 流程：在本项目中生成 synthetic phantom、spiral trajectory、MRF 字典、模拟 k-space、重建结果和定量图。

所有命令默认都在项目根目录执行：

```powershell
cd "MRI-project-for-BME1309.01"
```

## 一、先安装环境

本项目要求 Python 3.10 或更新版本。建议先创建虚拟环境。

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

如果只想安装主程序依赖，也可以运行：

```powershell
python -m pip install -e .
```

如果要使用 GPU：

```powershell
python -m pip install -e ".[gpu,test]"
```

说明：

- CPU 路线最稳，命令里传 `--device cpu`。
- GPU 路线需要 CUDA、CuPy、SigPy 等环境都可用，命令里传 `--device cuda --gpu-device 0`。
- 真实 cMRF 流程依赖 `torch`、`einops`、`mrpro`，这些已经写在 `pyproject.toml` 的依赖列表里。

安装完成后，先跑测试确认基础代码能导入：

```powershell
python -m pytest -q
```

## 二、最快运行方式

### 方式 A：运行真实 cMRF scanner 数据

先准备外部数据目录，目录结构必须像这样：

```text
CMRF_DATA_ROOT/
  cMRF_fa_705rep.txt
  scanner1/
    cMRF.h5
```

PowerShell 设置数据路径：

```powershell
$env:CMRF_DATA_ROOT="D:\your\path\open_source_cmrf_scanner_comparison"
```

没有 GPU 时用 CPU：

```powershell
python main_cmrf.py --device cpu
```

有 GPU 时用 CUDA：

```powershell
python main_cmrf.py --device cuda --gpu-device 0
```

常用调参运行：

```powershell
python main_cmrf.py --device cuda --gpu-device 0 --n-iter 60 --lambda-llr 0.005 --matching-batch-size 2048
```

真实 cMRF 默认输出：

```text
data/output/reconstructed_coeff_maps.npy
data/output/recon_loss.npy
data/output/quantitative_maps.png
data/output/quantitative_maps.npz
data/output/cmrf_pipeline_fingerprint.json
```

中间 processed 数据默认输出到：

```text
CMRF_DATA_ROOT/scanner1/processed/
```

### 方式 B：运行模拟 2D MRF 数据

模拟流程不需要外部 scanner 数据，直接运行：

```powershell
python main.py --device cpu
```

如果使用 GPU：

```powershell
python main.py --device cuda --gpu-device 0
```

模拟流程默认输出：

```text
data/processed/traj_full_2d.npy
data/processed/first5_traj.png
data/processed/brain_param_map_2d.npy
data/processed/simulated_brain_phantom.png
data/processed/mrf_dictionary_data.npz
data/processed/pipeline_config_fingerprint.json
data/output/mrf_kspace_2d_noisy.npy
data/output/reconstructed_coeff_maps.npy
data/output/quantitative_maps.png
data/output/quantitative_maps.npz
```

模拟流程跑完后，可以验证结果：

```powershell
python validate_results.py --no-show
```

## 三、真实 cMRF 流程详细用法

推荐使用总入口：

```powershell
python main_cmrf.py --device cpu
```

`main_cmrf.py` 会顺序执行三步：

1. `scripts/extract_cmrf_data.py`
2. `scripts/run_recon_cmrf.py`
3. `scripts/template_matching.py`

### main_cmrf.py 参数

| 参数 | 作用 | 示例 |
| --- | --- | --- |
| `--base-dir` | 临时指定 cMRF 数据根目录，覆盖 `CMRF_DATA_ROOT` | `--base-dir D:\data\cmrf` |
| `--scanner` | 指定 scanner 子目录，默认 `scanner1` | `--scanner scanner1` |
| `--scan-name` | 指定 HDF5 原始数据文件名，默认 `cMRF.h5` | `--scan-name cMRF.h5` |
| `--fa-file` | 指定 flip angle 文本文件，默认 `cMRF_fa_705rep.txt` | `--fa-file cMRF_fa_705rep.txt` |
| `--processed-dir` | 指定中间 processed 数据目录 | `--processed-dir data/cmrf_processed` |
| `--output-dir` | 指定最终输出目录，默认 `data/output` | `--output-dir data/output_cmrf` |
| `--n-iter` | LLR 重建迭代次数 | `--n-iter 60` |
| `--lambda-llr` | LLR 正则化权重 | `--lambda-llr 0.005` |
| `--step-size` | 梯度下降步长 | `--step-size 0.001` |
| `--device` | 计算设备，`cpu` 或 `cuda` | `--device cpu` |
| `--gpu-device` | CUDA 设备编号 | `--gpu-device 0` |
| `--matching-batch-size` | template matching 每批处理的像素数 | `--matching-batch-size 1024` |
| `--skip-extract` | 跳过数据提取，复用已有 processed 数据 | `--skip-extract` |
| `--skip-recon` | 跳过重建，复用已有 coefficient maps | `--skip-recon` |
| `--skip-matching` | 跳过 template matching | `--skip-matching` |
| `--force` | 即使 fingerprint 认为结果没过期，也强制重跑未跳过步骤 | `--force` |

### 真实 cMRF 单步运行

如果总入口报错，建议按下面三步单独运行，方便定位问题。

第 1 步：提取 scanner 数据，生成字典和 processed 数组。

```powershell
python scripts/extract_cmrf_data.py `
  --base-dir $env:CMRF_DATA_ROOT `
  --scanner scanner1 `
  --scan-name cMRF.h5 `
  --fa-file cMRF_fa_705rep.txt `
  --output-dir "$env:CMRF_DATA_ROOT\scanner1\processed" `
  --subspace-rank 5
```

第 2 步：重建 coefficient maps。

```powershell
python scripts/run_recon_cmrf.py `
  --input-dir "$env:CMRF_DATA_ROOT\scanner1\processed" `
  --output-dir data/output `
  --device cpu `
  --n-iter 40 `
  --lambda-llr 0.003
```

第 3 步：template matching，生成 T1/T2/PD 定量图。

```powershell
python scripts/template_matching.py `
  --dict-path "$env:CMRF_DATA_ROOT\scanner1\processed\mrf_dictionary_data.npz" `
  --coeff-maps data/output/reconstructed_coeff_maps.npy `
  --save-path data/output/quantitative_maps.png `
  --save-npz data/output/quantitative_maps.npz `
  --device cpu `
  --batch-size 1024
```

## 四、模拟 2D MRF 流程详细用法

推荐使用总入口：

```powershell
python main.py --device cpu
```

`main.py` 会顺序执行六步，并用 `data/processed/pipeline_config_fingerprint.json` 记录每一步的配置。已经存在且没过期的输出会自动跳过。

六步分别是：

1. `scripts/generate_traj.py`
2. `scripts/prepare_phantom.py`
3. `scripts/build_dictionary.py`
4. `scripts/run_forward_sim.py`
5. `scripts/run_recon.py`
6. `scripts/template_matching.py`

### main.py 参数

| 参数 | 作用 | 示例 |
| --- | --- | --- |
| `--n-iter` | LLR 重建迭代次数 | `--n-iter 30` |
| `--lambda-llr` | LLR 正则化权重 | `--lambda-llr 0.001` |
| `--step-size` | 梯度下降步长 | `--step-size 0.001` |
| `--img-shape` | 重建图像大小，高和宽 | `--img-shape 220 220` |
| `--patch-shape` | LLR patch 大小 | `--patch-shape 8 8` |
| `--device` | 计算设备，`cpu` 或 `cuda` | `--device cpu` |
| `--gpu-device` | CUDA 设备编号 | `--gpu-device 0` |
| `--matching-batch-size` | template matching 每批处理的像素数 | `--matching-batch-size 1024` |
| `--dictionary-batch-size` | GPU 字典生成时每批处理的字典条目数 | `--dictionary-batch-size 4096` |
| `--center-width` | ESPIRiT 校准时取中心 k-space 宽度 | `--center-width 24` |
| `--calib-width` | ESPIRiT calibration width | `--calib-width 24` |

### 模拟流程单步运行

第 1 步：生成 spiral trajectory。

```powershell
python scripts/generate_traj.py
```

输出：

```text
data/processed/traj_full_2d.npy
data/processed/first5_traj.png
```

第 2 步：生成 synthetic brain phantom。

```powershell
python scripts/prepare_phantom.py
```

输出：

```text
data/processed/brain_param_map_2d.npy
data/processed/simulated_brain_phantom.png
```

第 3 步：构建并压缩 MRF 字典。

```powershell
python scripts/build_dictionary.py --device cpu
```

输出：

```text
data/processed/mrf_dictionary_data.npz
```

第 4 步：用 phantom、trajectory 和字典模拟多线圈 k-space。

```powershell
python scripts/run_forward_sim.py
```

输出：

```text
data/output/mrf_kspace_2d_noisy.npy
```

第 5 步：执行子空间 LLR 重建。

```powershell
python scripts/run_recon.py `
  --device cpu `
  --n-iter 30 `
  --lambda-llr 0.001 `
  --step-size 0.001 `
  --img-shape 220 220 `
  --patch-shape 8 8
```

输出：

```text
data/output/reconstructed_coeff_maps.npy
```

第 6 步：执行 template matching，生成定量图。

```powershell
python scripts/template_matching.py `
  --dict-path data/processed/mrf_dictionary_data.npz `
  --coeff-maps data/output/reconstructed_coeff_maps.npy `
  --save-path data/output/quantitative_maps.png `
  --save-npz data/output/quantitative_maps.npz `
  --device cpu
```

输出：

```text
data/output/quantitative_maps.png
data/output/quantitative_maps.npz
```

第 7 步：验证模拟结果。

```powershell
python validate_results.py --no-show
```

输出：

```text
data/output/validation_comparison.png
```

## 五、配置文件怎么改

所有主要默认参数集中在 `pipeline_config.py`。

### 真实 cMRF 配置

| 参数 | 含义 |
| --- | --- |
| `CMRF_DATA_ROOT` | 真实 cMRF 数据根目录。也可以用环境变量 `CMRF_DATA_ROOT` 覆盖。 |
| `CMRF_SCANNER` | scanner 子目录名，默认 `scanner1`。 |
| `CMRF_SCAN_NAME` | 原始 HDF5 文件名，默认 `cMRF.h5`。 |
| `CMRF_FA_FILE` | flip angle 文本文件名，默认 `cMRF_fa_705rep.txt`。 |
| `CMRF_PROCESSED_DIR` | processed 中间数组目录。 |
| `CMRF_DICT_PATH` | processed 字典包路径。 |
| `CMRF_KSPACE_PATH` | processed k-space 路径。 |
| `CMRF_TRAJ_PATH` | processed trajectory 路径。 |
| `CMRF_CSM_PATH` | processed coil sensitivity map 路径。 |
| `CMRF_OUTPUT_DIR` | 默认最终输出目录。 |
| `CMRF_COEFF_MAPS_PATH` | 默认重建系数图路径。 |
| `CMRF_RECON_LOSS_PATH` | 默认重建 loss 保存路径。 |
| `CMRF_QUANT_PNG_PATH` | 默认定量图 PNG 路径。 |
| `CMRF_QUANT_NPZ_PATH` | 默认定量图原始数组路径。 |
| `CMRF_FINGERPRINT_PATH` | cMRF 一键流程 fingerprint 路径。 |
| `CMRF_SUBSPACE_RANK` | 字典 SVD 压缩后的子空间 rank。 |
| `CMRF_N_ITER` | cMRF 重建默认迭代次数。 |
| `CMRF_LAMBDA_LLR` | cMRF LLR 正则化权重。 |
| `CMRF_DEVICE` | cMRF 默认设备，当前为 `cuda`。没有 GPU 时命令行传 `--device cpu`。 |
| `CMRF_GPU_DEVICE` | 默认 CUDA 设备编号。 |
| `CMRF_MATCHING_BATCH_SIZE` | cMRF template matching 默认 batch size。 |
| `CMRF_T1_GRID_RANGES` | cMRF 字典 T1 网格。 |
| `CMRF_T2_GRID_RANGES` | cMRF 字典 T2 网格。 |
| `CMRF_T1_VMAX` | T1 图显示上限。 |
| `CMRF_T2_VMAX` | T2 图显示上限。 |

### 模拟 2D MRF 配置

| 参数 | 含义 |
| --- | --- |
| `IMG_SHAPE` | phantom 和重建图像大小。 |
| `N_COILS` | 模拟线圈数。 |
| `N_TR` | MRF 时间点数量。 |
| `SUBSPACE_RANK` | 模拟字典压缩后的子空间 rank。 |
| `N_ITER` | 模拟重建默认迭代次数。 |
| `LAMBDA_LLR` | 模拟重建 LLR 正则化权重。 |
| `STEP_SIZE` | 梯度步长。 |
| `PATCH_SHAPE` | LLR patch 大小。 |
| `CENTER_WIDTH` | 线圈敏感度估计时中心 k-space 宽度。 |
| `CALIB_WIDTH` | ESPIRiT calibration width。 |
| `SPIRAL_FOV` | spiral trajectory FOV。 |
| `SPIRAL_RES` | spiral trajectory 分辨率。 |
| `SPIRAL_READOUT_TIME` | 每个 TR 的 readout 时间。 |
| `SPIRAL_UNDERSAMPLE_CENTER` | trajectory 中心区域 undersampling 设置。 |
| `SPIRAL_UNDERSAMPLE_EDGE` | trajectory 边缘区域 undersampling 设置。 |
| `SPIRAL_ADC_RATE` | ADC 采样率。 |
| `TINY_GOLDEN_ANGLE` | tiny golden angle 旋转角度。 |
| `EPG_NUM_STATES` | EPG 模拟状态数。 |
| `EPG_TR` | 模拟序列 TR。 |
| `EPG_TE` | 模拟序列 TE。 |
| `EPG_TI` | 模拟序列 inversion time。 |
| `FA_MIN`、`FA_MAX` | 模拟 flip angle 曲线范围。 |
| `FA_NUM_ANCHORS` | 生成 flip angle 曲线的控制点数量。 |
| `FA_RANDOM_SEED` | flip angle 随机种子。 |
| `T1_GRID_RANGES` | 模拟字典 T1 网格。 |
| `T2_GRID_RANGES` | 模拟字典 T2 网格。 |
| `PHANTOM_*` | synthetic phantom 的脑区半径和组织参数。 |
| `ESPIRIT_*` | 线圈敏感度估计参数。 |
| `NOISE_LEVEL` | forward simulation 加噪强度。 |
| `RANDOM_SEED` | forward simulation 随机种子。 |

## 六、每个代码文件的功能

### 根目录文件

| 文件 | 功能 | 什么时候用 |
| --- | --- | --- |
| `README.md` | 项目简短说明，告诉读者有哪些流程和最常用命令。 | 第一次打开项目时先读。 |
| `CODE_FILE_GUIDE.md` | 本文件，详细解释代码结构、运行方法、输入输出和排错方法。 | 不知道怎么运行或想理解代码职责时读。 |
| `pyproject.toml` | Python 项目配置，定义包名、依赖、测试依赖、GPU 可选依赖和 `src` 包路径。 | 安装项目、跑测试、检查依赖时用。 |
| `pipeline_config.py` | 中央配置文件，保存真实 cMRF 和模拟 MRF 的所有默认参数。 | 调整图像大小、迭代次数、正则强度、字典网格、路径时改这里。 |
| `main_cmrf.py` | 真实 cMRF 一键入口，负责调度提取、重建、匹配三步，并用 fingerprint 判断是否重跑。 | 有真实 scanner 数据时首选运行它。 |
| `main.py` | 模拟 2D MRF 一键入口，负责调度 trajectory、phantom、dictionary、forward、recon、matching 六步，并用 fingerprint 跳过未变化步骤。 | 没有真实数据，想完整跑通模拟流程时用。 |
| `validate_results.py` | 模拟流程结果验证脚本。读取 phantom、字典、系数图，重新做 template matching，打印 shape 和组织参数误差，并生成对比图。 | 跑完模拟流程后检查结果是否合理。 |
| `.gitignore` | Git 忽略规则。 | 正常不用改。 |

### scripts 目录

| 文件 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `scripts/extract_cmrf_data.py` | 真实 cMRF 第 1 步。读取 `cMRF.h5` 和 flip angle 文件，用 `mrpro` 提取 k-space、trajectory、CSM，构建 EPG 字典，并用 SVD 压缩到子空间。 | `CMRF_DATA_ROOT/cMRF_fa_705rep.txt`、`CMRF_DATA_ROOT/scanner1/cMRF.h5` | `mrf_dictionary_data.npz`、`mrf_kspace_noisy.npy`、`traj_full_2d.npy`、`csm.npy` |
| `scripts/run_recon_cmrf.py` | 真实 cMRF 第 2 步。读取 processed k-space、trajectory、CSM 和 basis，调用子空间 LLR 重建。 | processed 目录下的四个数组文件 | `reconstructed_coeff_maps.npy`，可能还有 `recon_loss.npy` |
| `scripts/template_matching.py` | 真实和模拟都可用的定量图生成脚本。读取重建系数图和字典，执行 flattened dictionary template matching，保存 T1/T2/PD 图和原始数组。 | dictionary npz、coefficient maps | `quantitative_maps.png`、`quantitative_maps.npz` |
| `scripts/generate_traj.py` | 模拟第 1 步。根据 `pipeline_config.py` 生成 tiny-golden-angle spiral trajectory，并保存前 5 个 TR 的轨迹图。 | 配置参数 | `data/processed/traj_full_2d.npy`、`first5_traj.png` |
| `scripts/prepare_phantom.py` | 模拟第 2 步。生成包含 WM、GM、CSF 的 synthetic brain phantom，保存 T1/T2/PD 参数图。 | 配置参数 | `brain_param_map_2d.npy`、`simulated_brain_phantom.png` |
| `scripts/build_dictionary.py` | 模拟第 3 步。生成 flip angle train，运行 EPG 模拟，构建 T1/T2 字典，并做 SVD 子空间压缩。 | 配置参数，可选 GPU 参数 | `data/processed/mrf_dictionary_data.npz` |
| `scripts/run_forward_sim.py` | 模拟第 4 步。根据 phantom、trajectory 和字典，模拟多线圈非笛卡尔 k-space，并加入噪声。 | phantom、trajectory、dictionary | `data/output/mrf_kspace_2d_noisy.npy` |
| `scripts/run_recon.py` | 模拟第 5 步。估计 ESPIRiT 线圈敏感度图，然后调用子空间 LLR 重建。 | 模拟 k-space、trajectory、dictionary basis | `data/output/reconstructed_coeff_maps.npy` |
| `scripts/tips.txt` | 项目笔记或运行提示。 | 无 | 无 |

### src/mri_project 顶层

| 文件 | 功能 |
| --- | --- |
| `src/mri_project/__init__.py` | 声明 `mri_project` 包，并暴露 `forward`、`recon`、`dictionary`、`quantification` 子包名称。 |
| `src/mri_project/array_backend.py` | CPU/CUDA 数组后端封装。统一处理 NumPy/CuPy 数组、设备选择、GPU 可用性检查、数组搬回 CPU 等逻辑。其他模块用它来支持 `--device cpu` 和 `--device cuda`。 |

### src/mri_project/forward

| 文件 | 功能 |
| --- | --- |
| `src/mri_project/forward/__init__.py` | 暴露 forward 子包常用函数：trajectory 生成和 phantom 生成。 |
| `src/mri_project/forward/trajectory.py` | 实现 spiral trajectory 和 tiny-golden-angle 多 TR trajectory 生成。`generate_spiral_trajectory` 生成单条 spiral，`generate_tga_trajectory` 生成整套 MRF trajectory。 |
| `src/mri_project/forward/phantom.py` | 生成 synthetic brain phantom。输出每个像素的 T1、T2、PD 三通道参数图。 |
| `src/mri_project/forward/simulation.py` | forward simulation 核心。根据 phantom 中的 T1/T2 找最近字典条目，生成时间序列图像，再通过 NUFFT 和线圈敏感度模拟 k-space。 |
| `src/mri_project/forward/io.py` | 保存 `.npy` 文件和绘图工具。包括 trajectory 图和 phantom 图的保存。 |

### src/mri_project/dictionary

| 文件 | 功能 |
| --- | --- |
| `src/mri_project/dictionary/__init__.py` | 暴露字典生成相关 API，包括 `MRFEPGSimulator`、`build_signal_dictionary`、`compress_dictionary`。 |
| `src/mri_project/dictionary/epg.py` | MRF 字典核心。包含 EPG 模拟器、flip angle 曲线生成、T1/T2 网格生成、批量 FISP 信号模拟、字典构建和 SVD 子空间压缩。模拟流程的 `scripts/build_dictionary.py` 主要调用这个文件。 |

### src/mri_project/recon

| 文件 | 功能 |
| --- | --- |
| `src/mri_project/recon/__init__.py` | 暴露重建相关 API，包括 LLR 重建、NUFFT 子空间算子、线圈敏感度估计和正则化函数。 |
| `src/mri_project/recon/nufft_ops.py` | NUFFT 前向和伴随算子。优先使用 SigPy；在小规模测试中也提供直接 NUDFT fallback。支持 CPU 和 CUDA。 |
| `src/mri_project/recon/subspace_ops.py` | 子空间算子。负责在 coefficient maps 和 time images 之间展开/投影，并封装单线圈、多线圈的子空间 NUFFT forward/adjoint。 |
| `src/mri_project/recon/sensitivity.py` | 线圈敏感度估计。包括时间平均 k-space、中心 k-space gridding 和 ESPIRiT sensitivity map 估计。 |
| `src/mri_project/recon/regularization.py` | LLR 正则化。实现 patch-wise singular value soft-threshold 和 nuclear norm 计算。 |
| `src/mri_project/recon/iterative.py` | 迭代重建算法。包含普通 gradient descent 子空间重建和带 LLR proximal step 的子空间重建。真实 cMRF 和模拟重建脚本都调用这里。 |

### src/mri_project/quantification

| 文件 | 功能 |
| --- | --- |
| `src/mri_project/quantification/__init__.py` | 暴露 template matching API。 |
| `src/mri_project/quantification/template_matching.py` | 简洁版 template matching 核心函数，主要用于模拟流程和测试。根据 coefficient maps 和压缩字典计算 T1/T2/PD map。 |

### tests 目录

| 文件 | 检查内容 |
| --- | --- |
| `tests/test_smoke.py` | 基础 smoke test。检查核心模块能导入，forward、dictionary 小规模流程能跑。 |
| `tests/test_array_backend.py` | 检查 CPU/CUDA 后端封装、GPU 不可用时报错是否清楚，以及 GPU 可用时结果是否和 CPU 接近。 |
| `tests/test_template_matching.py` | 检查 template matching 是否能恢复字典条目、是否正确处理零信号和 basis 维度不匹配。 |
| `tests/test_nufft_ops.py` | 检查 NUFFT forward/adjoint 的 shape 和伴随关系。 |
| `tests/test_subspace_ops.py` | 检查子空间展开、投影、单线圈和多线圈子空间 NUFFT 算子的 shape 与伴随关系。 |
| `tests/test_sensitivity.py` | 检查 k-space 时间平均、中心 k-space gridding 和 ESPIRiT 调用逻辑。 |
| `tests/test_regularization.py` | 检查 LLR soft-threshold 的 shape、dtype、数值有限性、核范数下降和参数校验。 |
| `tests/test_iterative_recon.py` | 检查 gradient descent 和 LLR 重建能降低 loss，并能处理多线圈输入。 |

## 七、输出数据流

### 真实 cMRF

```text
CMRF_DATA_ROOT/cMRF_fa_705rep.txt
CMRF_DATA_ROOT/scanner1/cMRF.h5
        |
        v
scripts/extract_cmrf_data.py
        |
        v
CMRF_DATA_ROOT/scanner1/processed/mrf_dictionary_data.npz
CMRF_DATA_ROOT/scanner1/processed/mrf_kspace_noisy.npy
CMRF_DATA_ROOT/scanner1/processed/traj_full_2d.npy
CMRF_DATA_ROOT/scanner1/processed/csm.npy
        |
        v
scripts/run_recon_cmrf.py
        |
        v
data/output/reconstructed_coeff_maps.npy
data/output/recon_loss.npy
        |
        v
scripts/template_matching.py
        |
        v
data/output/quantitative_maps.png
data/output/quantitative_maps.npz
```

### 模拟 2D MRF

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
        -> data/output/quantitative_maps.npz

validate_results.py
        <- data/processed/brain_param_map_2d.npy
        <- data/processed/mrf_dictionary_data.npz
        <- data/output/reconstructed_coeff_maps.npy
        -> data/output/validation_comparison.png
```

## 八、常见修改后应该重跑哪些步骤

| 修改内容 | 需要重跑 |
| --- | --- |
| 改 `N_ITER`、`LAMBDA_LLR`、`STEP_SIZE`、`PATCH_SHAPE` | 重建和 matching。模拟可直接 `python main.py`，它会自动判断。 |
| 改 `IMG_SHAPE` | trajectory、phantom、forward、recon、matching 都要重跑。 |
| 改 `N_TR` | trajectory、dictionary、forward、recon、matching 都要重跑。 |
| 改 `T1_GRID_RANGES`、`T2_GRID_RANGES` | dictionary、forward、matching 要重跑；通常 recon 也建议重跑。 |
| 改 `CMRF_T1_GRID_RANGES`、`CMRF_T2_GRID_RANGES` | 真实 cMRF 的 extract、recon、matching 都要重跑。 |
| 改 `NOISE_LEVEL`、`N_COILS` | 模拟 forward、recon、matching 要重跑。 |
| 改真实 cMRF 数据路径 | 重新设置 `CMRF_DATA_ROOT` 或传 `--base-dir`，再运行 `main_cmrf.py`。 |

如果不确定，就使用强制重跑。

真实 cMRF：

```powershell
python main_cmrf.py --device cpu --force
```

模拟流程目前没有 `--force` 参数。如果想完整重跑，可以删除对应输出文件，或直接按单步命令覆盖生成。

## 九、服务器和画图说明

`scripts/template_matching.py` 默认只保存图片，不弹出窗口，适合服务器运行。

如果在本地桌面环境想弹出图像窗口：

```powershell
python scripts/template_matching.py `
  --dict-path data/processed/mrf_dictionary_data.npz `
  --coeff-maps data/output/reconstructed_coeff_maps.npy `
  --save-path data/output/quantitative_maps.png `
  --save-npz data/output/quantitative_maps.npz `
  --show
```

`validate_results.py` 默认可能显示窗口。服务器上请加：

```powershell
python validate_results.py --no-show
```

## 十、常见报错和解决方法

### 找不到 `cMRF.h5`

检查数据目录是否符合：

```text
CMRF_DATA_ROOT/
  cMRF_fa_705rep.txt
  scanner1/
    cMRF.h5
```

PowerShell 重新设置：

```powershell
$env:CMRF_DATA_ROOT="D:\your\path\open_source_cmrf_scanner_comparison"
python main_cmrf.py --device cpu
```

### 找不到 `mrf_dictionary_data.npz`

分两种情况：

- 真实 cMRF：先运行 `scripts/extract_cmrf_data.py`，它会在 `CMRF_DATA_ROOT/scanner1/processed/` 生成字典。
- 模拟 2D MRF：先运行 `scripts/build_dictionary.py --device cpu`，它会在 `data/processed/` 生成字典。

### 没有 GPU 或 CUDA 报错

把命令里的设备改成 CPU：

```powershell
python main_cmrf.py --device cpu
python main.py --device cpu
```

### template matching 内存不够

减小 batch size：

```powershell
python main_cmrf.py --device cpu --matching-batch-size 256
```

或单步运行：

```powershell
python scripts/template_matching.py `
  --dict-path data/processed/mrf_dictionary_data.npz `
  --coeff-maps data/output/reconstructed_coeff_maps.npy `
  --save-path data/output/quantitative_maps.png `
  --save-npz data/output/quantitative_maps.npz `
  --device cpu `
  --batch-size 256
```

### PowerShell 中文乱码

可以临时设置：

```powershell
$env:PYTHONIOENCODING="utf-8"
```

也可以在 PowerShell 中切换 UTF-8 输出：

```powershell
chcp 65001
```

### 不知道某个脚本有哪些参数

所有入口脚本都支持 `--help`：

```powershell
python main_cmrf.py --help
python main.py --help
python scripts/extract_cmrf_data.py --help
python scripts/run_recon_cmrf.py --help
python scripts/run_recon.py --help
python scripts/template_matching.py --help
python validate_results.py --help
```

## 十一、提交或展示前建议检查

至少运行：

```powershell
python -m pytest -q
python -m py_compile pipeline_config.py main.py main_cmrf.py validate_results.py
python -m py_compile scripts/extract_cmrf_data.py scripts/run_recon_cmrf.py scripts/template_matching.py
```

如果改了模拟流程，建议运行：

```powershell
python main.py --device cpu
python validate_results.py --no-show
```

如果改了真实 cMRF 流程，建议运行：

```powershell
python main_cmrf.py --device cpu
```

真实 cMRF 数据通常较大，运行时间会明显长于模拟流程。
