# MRI-project-for-BME1309.01

这是 BME1309.01 生医工影像导论期末项目代码库，整理为统一的 Python 包结构，用于 2D MRF 的轨迹生成、体模准备、字典仿真、前向 k-space 仿真和重建算子验证。

## 目录结构以及文件功能请详细阅读
**CODE_FILE_GUIDE.py**

## 主要的参数文件
**pipeline_config.py**

每次改参数后需要重跑*scripts*内脚本覆盖旧数据。内有注释说明。

## 执行流程

python main.py即可

## 验证脚本

**validate_results.py**

直接运行可以看结果。

## DICOM-derived anatomical phantom

The default `scripts/prepare_phantom.py` still creates the original rule-based
mathematical phantom. To use a real DICOM slice only as anatomical structure,
generate a DICOM-derived phantom first:

```bash
python scripts/prepare_dicom_phantom.py --dicom-path your_file.dcm --overwrite --no-show
python scripts/build_dictionary.py
python scripts/run_forward_sim.py
python scripts/run_recon.py
python scripts/template_matching.py --no-show
python validate_results.py --no-show
```

If `--dicom-path` is omitted, the script searches for `.dcm` files under the
project root and uses the first one it finds. The output parameter map is saved
to `data/processed/brain_param_map_2d.npy`, and a preview figure is saved to
`data/processed/dicom_phantom_preview.png`.

For the full automated DICOM-derived test pipeline, keep `main.py` for the
original mathematical phantom and run:

```bash
python main_new.py --dicom-path your_file.dcm --no-show
```

Important: this DICOM-derived phantom does not recover true quantitative T1/T2
values from the FLAIR DICOM intensities. The DICOM image provides realistic
brain anatomy and tissue spatial distribution only. The script builds coarse
tissue masks from the normalized image, then assigns preset T1/T2/PD values
that are clipped to the available dictionary grid when needed. This is intended
for testing algorithm stability on more anatomical structure, not for clinical
quantification.
