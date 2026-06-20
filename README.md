# MRI-project-for-BME1309.01

本仓库目前包含两条 MRF 流程：

1. **真实 cMRF scanner 数据流程**：使用已下载的开源 cMRF scanner 数据。
2. **模拟 2D MRF 流程**：生成 synthetic phantom、字典、k-space、重建结果和定量图。

当前项目推荐优先使用真实 cMRF 流程。

## 快速开始：真实 cMRF

先准备数据目录。期望结构如下：

```text
CMRF_DATA_ROOT/
  cMRF_fa_705rep.txt
  scanner1/
    cMRF.h5
```

方式一：在 `pipeline_config.py` 中改一次路径：

```python
CMRF_DATA_ROOT = Path(r"/your/path/open_source_cmrf_scanner_comparison")
```

方式二：服务器上用环境变量覆盖，不改代码：

```bash
export CMRF_DATA_ROOT=/your/path/open_source_cmrf_scanner_comparison
python main_cmrf.py
```

PowerShell：

```powershell
$env:CMRF_DATA_ROOT="D:\your\path\open_source_cmrf_scanner_comparison"
python main_cmrf.py
```

运行完整 cMRF 流程：

```bash
python main_cmrf.py
```

常用调参运行：

```bash
python main_cmrf.py --device cuda --n-iter 60 --lambda-llr 0.005 --matching-batch-size 2048
```

## cMRF 输出

数据提取步骤会把中间数据写到：

```text
CMRF_DATA_ROOT/scanner1/processed/
```

重建和匹配步骤会把最终结果写到：

```text
data/output/reconstructed_coeff_maps.npy
data/output/quantitative_maps.png
data/output/quantitative_maps.npz
data/output/cmrf_pipeline_fingerprint.json
```

## 单步调试

推荐使用 `main_cmrf.py` 一键运行；需要调试时，也可以单独运行每一步：

```bash
python scripts/extract_cmrf_data.py
python scripts/run_recon_cmrf.py
python scripts/template_matching.py
```

常用跳步参数：

```bash
python main_cmrf.py --skip-extract
python main_cmrf.py --skip-recon
python main_cmrf.py --skip-matching
python main_cmrf.py --force
```

## 模拟 2D MRF 流程

原来的模拟流程仍然保留：

```bash
python main.py
```

它的主要默认参数也在 `pipeline_config.py` 中。

## 测试

```bash
python -m pytest -q
```

更详细的文件说明、输入输出和调参建议见 `CODE_FILE_GUIDE.md`。
