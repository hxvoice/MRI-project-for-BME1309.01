# MRI-project-for-BME1309.01

这是 BME1309.01 生医工影像导论期末项目代码库，整理为统一的 Python 包结构，用于 2D MRF 的轨迹生成、体模准备、字典仿真、前向 k-space 仿真和重建算子验证。

## 目录结构

```text
src/mri_project/
  forward/      # 轨迹、体模、前向仿真
  dictionary/   # EPG/MRF 字典生成与子空间压缩
  recon/        # NUFFT 与 subspace 重建算子
scripts/        # 可直接运行的流程入口
tests/          # pytest 测试
data/           # 本地生成数据，不提交大文件
outputs/        # 本地测试或可视化输出，不提交
```

## 安装

```bash
python -m pip install -e ".[test]"
```

## 推荐运行顺序

```bash
python scripts/generate_traj.py
python scripts/prepare_phantom.py
python scripts/build_dictionary.py
python scripts/run_forward_sim.py
```

生成的 `.npy`、`.npz` 和图片会写入 `data/processed/`、`data/output/` 或 `outputs/`，这些目录默认不会提交到 Git。

## 测试

```bash
pytest
```

测试覆盖核心模块导入、轨迹和体模形状、轻量前向仿真、NUFFT 前向/伴随一致性，以及 subspace NUFFT 算子形状和伴随性。
