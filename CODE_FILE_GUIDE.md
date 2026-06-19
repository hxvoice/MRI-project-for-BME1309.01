# 代码文件说明

本文档说明当前仓库的主要运行方式、文件职责、输入输出和常用调参位置。

## 推荐流程：真实 cMRF 数据

当前项目推荐使用真实 cMRF scanner 数据流程。数据目录需要满足：

```text
CMRF_DATA_ROOT/
  cMRF_fa_705rep.txt
  scanner1/
    cMRF.h5
```

设置数据根目录有两种方式。

方式一：修改 `pipeline_config.py`：

```python
CMRF_DATA_ROOT = Path(r"/your/path/open_source_cmrf_scanner_comparison")
```

方式二：使用环境变量，不改代码：

```bash
export CMRF_DATA_ROOT=/your/path/open_source_cmrf_scanner_comparison
```

PowerShell：

```powershell
$env:CMRF_DATA_ROOT="D:\your\path\open_source_cmrf_scanner_comparison"
```

然后运行：

```bash
python main_cmrf.py
```

常用调参命令：

```bash
python main_cmrf.py --device cuda --n-iter 60 --lambda-llr 0.005 --matching-batch-size 2048
```

## cMRF 流程文件

| 文件 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `main_cmrf.py` | cMRF 一键入口。按顺序执行数据提取、重建和 template matching，并用 fingerprint 判断是否需要重跑。 | `pipeline_config.py`、原始 cMRF 数据 | `data/output/` 下的最终结果 |
| `scripts/extract_cmrf_data.py` | 读取真实 cMRF scanner 数据，估计 CSM，构建 EPG 字典并做 SVD 压缩。 | `CMRF_DATA_ROOT/cMRF_fa_705rep.txt`、`CMRF_DATA_ROOT/scanner1/cMRF.h5` | `CMRF_DATA_ROOT/scanner1/processed/mrf_dictionary_data.npz`、`mrf_kspace_noisy.npy`、`traj_full_2d.npy`、`csm.npy` |
| `scripts/run_recon_cmrf.py` | 对真实 cMRF 数据执行 subspace LLR 重建。 | processed k-space、trajectory、CSM、dictionary bases | `data/output/reconstructed_coeff_maps.npy`，可选 `recon_loss.npy` |
| `scripts/template_matching.py` | 执行 flattened dictionary template matching，并生成 T1/T2/PD 定量图。 | dictionary npz、重建系数图 | `data/output/quantitative_maps.png`、`data/output/quantitative_maps.npz` |

单步调试：

```bash
python scripts/extract_cmrf_data.py
python scripts/run_recon_cmrf.py
python scripts/template_matching.py
```

`main_cmrf.py` 常用参数：

| 参数 | 用途 |
| --- | --- |
| `--skip-extract` | 跳过数据提取，复用已有 processed 数据。 |
| `--skip-recon` | 跳过重建，复用已有 coefficient maps。 |
| `--skip-matching` | 跳过 template matching。 |
| `--force` | 即使输出看起来是最新的，也强制重跑未跳过的步骤。 |
| `--base-dir` | 临时覆盖 `CMRF_DATA_ROOT`。 |
| `--processed-dir` | 临时覆盖 processed 输入/输出目录。 |
| `--output-dir` | 临时覆盖最终输出目录。 |
| `--device` | 选择 `cpu` 或 `cuda`。 |
| `--n-iter` | 覆盖 LLR 重建迭代次数。 |
| `--lambda-llr` | 覆盖 LLR 正则化权重。 |
| `--matching-batch-size` | 覆盖 template matching batch size。 |

## cMRF 配置项

cMRF 默认配置集中在 `pipeline_config.py`：

| 参数 | 含义 |
| --- | --- |
| `CMRF_DATA_ROOT` | cMRF 数据根目录，包含 `cMRF_fa_705rep.txt` 和 `scanner1/cMRF.h5`；可被环境变量 `CMRF_DATA_ROOT` 覆盖。 |
| `CMRF_SCANNER`、`CMRF_SCAN_NAME`、`CMRF_FA_FILE` | scanner 文件夹和原始数据文件名。 |
| `CMRF_PROCESSED_DIR` | 数据提取后的 processed 数组目录。 |
| `CMRF_OUTPUT_DIR` | 重建和 matching 的最终输出目录。 |
| `CMRF_SUBSPACE_RANK` | cMRF 字典 SVD 压缩后的子空间 rank。 |
| `CMRF_N_ITER`、`CMRF_LAMBDA_LLR` | LLR 重建迭代次数和正则化权重。 |
| `CMRF_DEVICE`、`CMRF_GPU_DEVICE` | 默认计算后端和 CUDA 设备编号。 |
| `CMRF_MATCHING_BATCH_SIZE` | template matching 的 batch size。 |
| `CMRF_T1_GRID_RANGES`、`CMRF_T2_GRID_RANGES` | 真实 cMRF 字典的 T1/T2 网格范围。 |
| `CMRF_T1_VMAX`、`CMRF_T2_VMAX` | 定量图显示上限。 |

## 服务器运行说明

`scripts/template_matching.py` 默认只保存图片，不弹窗，适合服务器运行。

如果在本地桌面环境想显示窗口，可以显式传：

```bash
python scripts/template_matching.py --show
```

`--no-show` 仍然保留，用于兼容旧命令。

## 模拟 2D MRF 流程

原来的 synthetic 2D MRF 流程仍然可用，但没有合并进 `main_cmrf.py`。

完整运行：

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

模拟流程的主要参数也在 `pipeline_config.py`：

| 参数 | 含义 |
| --- | --- |
| `IMG_SHAPE` | phantom 和重建图像大小。 |
| `N_COILS` | 模拟线圈数。 |
| `N_TR` | 时间点数量。 |
| `SUBSPACE_RANK` | 字典子空间 rank。 |
| `N_ITER`、`LAMBDA_LLR`、`STEP_SIZE`、`PATCH_SHAPE` | 模拟流程重建默认参数。 |
| `SPIRAL_*`、`TINY_GOLDEN_ANGLE` | spiral trajectory 参数。 |
| `EPG_*`、`FA_*`、`T1_GRID_RANGES`、`T2_GRID_RANGES` | 模拟字典参数。 |

## 输出数据流

真实 cMRF 流程：

```text
scripts/extract_cmrf_data.py
    <- CMRF_DATA_ROOT/cMRF_fa_705rep.txt
    <- CMRF_DATA_ROOT/scanner1/cMRF.h5
    -> CMRF_DATA_ROOT/scanner1/processed/mrf_dictionary_data.npz
    -> CMRF_DATA_ROOT/scanner1/processed/mrf_kspace_noisy.npy
    -> CMRF_DATA_ROOT/scanner1/processed/traj_full_2d.npy
    -> CMRF_DATA_ROOT/scanner1/processed/csm.npy

scripts/run_recon_cmrf.py
    <- CMRF_DATA_ROOT/scanner1/processed/*
    -> data/output/reconstructed_coeff_maps.npy
    -> data/output/recon_loss.npy

scripts/template_matching.py
    <- data/output/reconstructed_coeff_maps.npy
    <- CMRF_DATA_ROOT/scanner1/processed/mrf_dictionary_data.npz
    -> data/output/quantitative_maps.png
    -> data/output/quantitative_maps.npz
```

模拟流程：

```text
main.py
    -> data/processed/*
    -> data/output/*
```

## 测试和静态检查

运行测试：

```bash
python -m pytest -q
```

修改入口脚本后，建议额外检查：

```bash
python -m py_compile pipeline_config.py main_cmrf.py scripts/extract_cmrf_data.py scripts/run_recon_cmrf.py scripts/template_matching.py
python main_cmrf.py --help
python scripts/extract_cmrf_data.py --help
python scripts/run_recon_cmrf.py --help
python scripts/template_matching.py --help
```
