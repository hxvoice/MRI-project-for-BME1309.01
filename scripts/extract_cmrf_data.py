"""
本地路径版: 提取 cMRF Scanner 1 数据、生成 EPG 字典并进行 SVD 降维
(终极鲁棒版：免疫 API 重构，显式计算 CSM，利用 einops 完美重组 705 TRs 维度)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import scipy.linalg
from pathlib import Path
from einops import rearrange

from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data.CsmData import CsmData

from mrpro.operators.models.EPG import (
    EPGSequence, 
    FispBlock, 
    InversionBlock, 
    T2PrepBlock, 
    DelayBlock, 
    Parameters
)

def generate_subspace_dictionary(full_dict_np, subspace_rank=5):
    print(f">>> 执行 SVD 降维，输入字典维度: {full_dict_np.shape}")
    U, S, Vh = scipy.linalg.svd(full_dict_np, full_matrices=False)
    bases = U[:, :subspace_rank] 
    compressed_dict = bases.conj().T @ full_dict_np 
    print(f">>> 降维完成。Bases 维度: {bases.shape}, 压缩字典维度: {compressed_dict.shape}")
    return bases, compressed_dict

def process_and_intercept_cmrf_data(scanner_dir, scan_name, fa, t1, t2):
    print(f"--- 正在解析 {scanner_dir.name} 的数据 ---")
    
    # 1. 读取原始 k-space 数据
    kdata = KData.from_file(scanner_dir / scan_name, KTrajectoryIsmrmrd())
    
    # ==========================================
    # 修复点 3: 显式计算线圈敏感度图 (CSM)
    # 禁用其内部的自动 CSM，先重建未融合的多线圈图像，再手动调用 Walsh 算法
    # ==========================================
    print(">>> 正在显式计算线圈敏感度图 (CSM)...")
    avg_recon = DirectReconstruction(kdata, csm=None)
    avg_img = avg_recon(kdata)
    csm = CsmData.from_idata_walsh(avg_img) # 直接调用静态方法生成 CsmData
    csm_np = np.squeeze(csm.data.detach().cpu().numpy()) # 降维至 (coils, y, x)

    # ==========================================
    # 修复点 2: 暴力提取 705 TRs 原生数据，无视内部维度排列
    # mrpro 原生维度可能散落在 'other' 或 'k1' 维，利用 einops 泛型匹配 `...`
    # ==========================================
    print(">>> 正在提取并重组 k-space 和 trajectory 数据...")
    k_data_pt = kdata.data.detach().cpu()
    # 强制将除了线圈和采样点之外的所有维度压扁成一条时间序列 (即 705)
    kspace_np = rearrange(k_data_pt.numpy(), '... coils k2 k1 k0 -> coils (... k2 k1) k0')
    
    ky_pt = kdata.traj.ky.detach().cpu()
    kx_pt = kdata.traj.kx.detach().cpu()
    ky_np = rearrange(ky_pt.numpy(), '... 1 k2 k1 k0 -> (... k2 k1) k0')
    kx_np = rearrange(kx_pt.numpy(), '... 1 k2 k1 k0 -> (... k2 k1) k0')
    traj_np = np.stack((ky_np, kx_np), axis=-1)
    
    print(f"    [对齐检查] K-space 形状: {kspace_np.shape} (预期: N_coils, N_TR, N_samples)")
    print(f"    [对齐检查] 轨迹 形状:    {traj_np.shape} (预期: N_TR, N_samples, 2)")
    print(f"    [对齐检查] CSM 形状:     {csm_np.shape} (预期: N_coils, H, W)")

    # ==========================================
    # 修复点 1: 手工组装模块化的 EPG 序列
    # ==========================================
    print(">>> 正在利用 EPGSequence 组装物理演化模型...")
    t1, t2 = torch.broadcast_tensors(t1[None,:], t2[:,None])
    t1_all = t1.flatten().to(dtype=torch.float32)
    t2_all = t2.flatten().to(dtype=torch.float32)
    
    t1_valid = t1_all[t1_all >= t2_all]
    t2_valid = t2_all[t1_all >= t2_all]
    m0 = torch.ones_like(t1_valid)

    # ==========================================
    # 序列时间参数配置
    # ==========================================
    n_rf_pulses_per_block = 47 
    # 暴力展平时间戳，确保得到一个 1D 数组
    acq_t_ms = kdata.header.acq_info.acquisition_time_stamp.flatten().detach().cpu().numpy() * 2.5
    
    delay_between_blocks = [acq_t_ms[n_block*n_rf_pulses_per_block] - acq_t_ms[n_block*n_rf_pulses_per_block-1] for n_block in range(1,3*5)]
    delay_between_blocks.append(delay_between_blocks[-1]) 

    flip_angles = fa
    rf_phases = 0.0
    te = 1.52
    
    # ==========================================
    # 修复点：安全提取 TR 值
    # ==========================================
    tr_raw = kdata.header.tr
    tr_val = tr_raw[0] if isinstance(tr_raw, (list, tuple)) else tr_raw
    tr = float(tr_val.item() if hasattr(tr_val, 'item') else tr_val) * 1000

    inv_prep_ti = [21,None,None,None,None]*3 
    t2_prep_te = [None,None,30,50,100]*3 
    delay_due_to_prep = [0, 30, 50, 100, 21]*3
    delay_after_block = [trig_delay-prep_delay for prep_delay, trig_delay in zip(delay_due_to_prep, delay_between_blocks)]
    
    epg_blocks = []
    n_blocks = 15
    for i in range(n_blocks):
        if inv_prep_ti[i] is not None:
            epg_blocks.append(InversionBlock(inversion_time=inv_prep_ti[i]))
        if t2_prep_te[i] is not None:
            epg_blocks.append(T2PrepBlock(te=t2_prep_te[i]))
            
        start_idx = i * n_rf_pulses_per_block
        end_idx = start_idx + n_rf_pulses_per_block
        epg_blocks.append(FispBlock(
            flip_angles=flip_angles[start_idx:end_idx], rf_phases=rf_phases, te=te, tr=tr
        ))
        
        if delay_after_block[i] > 0:
            epg_blocks.append(DelayBlock(delay_time=delay_after_block[i]))

    epg_sequence = EPGSequence(epg_blocks)
    params = Parameters(m0=m0, t1=t1_valid, t2=t2_valid)
    
    # 执行仿真
    _, signals = epg_sequence(params)
    
    # 将信号按时间维度堆叠: (705, N_dict)
    signal_dictionary = torch.stack(signals, dim=0).abs()

    # 对完整字典在时间维度上进行归一化
    vector_norm = torch.linalg.vector_norm(signal_dictionary, dim=0)
    signal_dictionary /= vector_norm

    # 导出纯 Numpy 格式
    full_dict_np = signal_dictionary.detach().cpu().numpy()
    t1_valid_np = t1_valid.detach().cpu().numpy()
    t2_valid_np = t2_valid.detach().cpu().numpy()
    
    return full_dict_np, t1_valid_np, t2_valid_np, kspace_np, traj_np, csm_np

def main():
    base_dir = Path(r"C:\Users\Lenovo\Desktop\open_source_cmrf_scanner_comparison")
    scanner_dir = base_dir / "scanner1"
    output_dir = scanner_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t1_grid = torch.cat((torch.arange(50, 2000+10, 10), torch.arange(2020, 3000+20, 20), torch.arange(3050,5000+50,50)))
    t2_grid = torch.cat((torch.arange(6, 100+2, 2), torch.arange(105, 200+5, 5), torch.arange(220,500+20,20)))
        
    with open(base_dir / 'cMRF_fa_705rep.txt', "r") as file:
        fa = torch.as_tensor([float(line) for line in file.readlines()])/180 * torch.pi

    # 执行数据提取
    full_dict_np, t1_valid_np, t2_valid_np, kspace_np, traj_np, csm_np = process_and_intercept_cmrf_data(
        scanner_dir=scanner_dir, scan_name='cMRF.h5', fa=fa, t1=t1_grid, t2=t2_grid
    )

    # 执行私有逻辑：SVD 降维
    SUBSPACE_RANK = 5
    bases, compressed_dict = generate_subspace_dictionary(full_dict_np, subspace_rank=SUBSPACE_RANK)

    # 保存数据
    print("\n>>> 正在保存数据到本地...")
    np.savez(
        output_dir / "mrf_dictionary_data.npz",
        signal_dict=full_dict_np, compressed_dict=compressed_dict, bases=bases, t1=t1_valid_np, t2=t2_valid_np
    )
    np.save(output_dir / "mrf_kspace_noisy.npy", kspace_np)
    np.save(output_dir / "traj_full_2d.npy", traj_np)  # 名字改为了与你们工程指引中一致
    np.save(output_dir / "csm.npy", csm_np)
    
    print("    k-space, trajectory, CSM 已分别保存。")
    print("\n>>> 数据提取全部完成！")

if __name__ == "__main__":
    main()