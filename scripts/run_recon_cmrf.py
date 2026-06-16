import sys
from pathlib import Path

# ==========================================
# 修复路径：自动获取项目根目录，让 Python 能找到 src 和 pipeline_config
# ==========================================
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

import numpy as np
import argparse

# 引入底层子空间 LLR 重建算法
from mri_project.recon.iterative import reconstruct_subspace_llr
import pipeline_config as cfg

def main():
    parser = argparse.ArgumentParser(description="开源心脏数据 cMRF 子空间 LLR 重建脚本")
    parser.add_argument("--n-iter", type=int, default=40, help="FISTA 迭代次数")
    parser.add_argument("--lambda-llr", type=float, default=0.003, help="LLR 正则化参数阈值")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="计算设备")
    args = parser.parse_args()

    # 1. 设定输入与输出路径
    # 下方是完善时需要修改的本地路径
    input_dir = Path(r"C:\Users\Lenovo\Desktop\open_source_cmrf_scanner_comparison\scanner1\processed")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>> 正在加载上一步提取的开源心脏数据矩阵...")
    kspace = np.load(input_dir / "mrf_kspace_noisy.npy")
    traj = np.load(input_dir / "traj_full_2d.npy")
    csm = np.load(input_dir / "csm.npy")
    
    dict_data = np.load(input_dir / "mrf_dictionary_data.npz")
    bases = dict_data["bases"]
    
    # 确保 basis 形状为 (N_TR, rank)
    if bases.shape[0] == 5:
        bases = bases.T  

    print(f"    数据加载成功！")
    print(f"    K-space: {kspace.shape}, Trajectory: {traj.shape}, CSM: {csm.shape}, Bases: {bases.shape}")

    print(f"\n>>> 开始执行子空间 LLR 重建 (设备: {args.device}, 迭代: {args.n_iter}, Lambda: {args.lambda_llr})...")
    
    img_shape = (csm.shape[-2], csm.shape[-1]) 
    
    # 2. 执行迭代求解器 (会耗费较长时间)
    recon_result = reconstruct_subspace_llr(
        kspace=kspace,
        coord=traj,
        basis=bases,
        sens_maps=csm,
        img_shape=img_shape,
        n_iter=args.n_iter,
        lambda_llr=args.lambda_llr,
        patch_shape=getattr(cfg, "PATCH_SHAPE", (8, 8)),
        device=args.device
    )
    
    # ==========================================
    # 终极修复 1：安全拆包元组 (Tuple)
    # ==========================================
    if isinstance(recon_result, tuple):
        coeff_maps = recon_result[0]  # 第一项是系数图
        # 如果有 Loss 数据，顺手保存下来
        if len(recon_result) > 1:
            loss_history = recon_result[1]
            if hasattr(loss_history, "get"):
                loss_history = loss_history.get()
            np.save(output_dir / "recon_loss.npy", loss_history)
    else:
        coeff_maps = recon_result

    # ==========================================
    # 终极修复 2：安全从 GPU (CuPy) 拉回主机内存
    # ==========================================
    if hasattr(coeff_maps, "get"):
        coeff_maps = coeff_maps.get()
    elif hasattr(coeff_maps, "cpu"):
        coeff_maps = coeff_maps.cpu().numpy()
    coeff_maps = np.asarray(coeff_maps)

    # 3. 移轴适配定量模块
    if coeff_maps.shape[0] == 5:
        coeff_maps = np.moveaxis(coeff_maps, 0, -1) # 变为 (128, 128, 5)

    output_path = output_dir / "reconstructed_coeff_maps.npy"
    np.save(output_path, coeff_maps)
    
    print(f"\n>>> 重建全部完成！5张低秩系数图已保存至: {output_path}")

if __name__ == "__main__":
    main()