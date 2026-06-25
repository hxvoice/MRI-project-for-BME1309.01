import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import argparse

def plot_scatter_with_regression(ax, ref_mean, cmrf_mean, ref_std, cmrf_std, title, max_val, color):
    """绘制散点图、误差棒，并计算线性回归"""
    # 绘制带误差棒的散点
    ax.errorbar(ref_mean, cmrf_mean, xerr=ref_std, yerr=cmrf_std, fmt="o", color=color, capsize=4, markersize=6)
    
    # 绘制 y=x 理想参考线
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="y = x (Ideal)")
    
    # 计算线性回归
    slope, intercept, r_value, p_value, std_err = linregress(ref_mean, cmrf_mean)
    x_fit = np.array([0, max_val])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='darkorange', label=f"Fit: y={slope:.3f}x+{intercept:.1f}\n$R^2$={r_value**2:.4f}")
    
    # 设置图表属性
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"Reference (ms)", fontsize=12)
    ax.set_ylabel(f"cMRF Subspace LLR (ms)", fontsize=12)
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper left")

# 下面四个路径对应的内容从上到下分别为：cmrf结果图（npz），标准答案（npz），开源数据集scanner1中的mask文件，输出的结果对比图
# 从上至下可修改为：data/output/quantitative_maps.npz
# 任意存放该文件的路径
# CMRF_DATA_ROOT/scanner1/mask.npy
# data/output/accuracy_scatter_plot.png
def main():
    parser = argparse.ArgumentParser(description="Evaluate cMRF accuracy against Reference Spin-Echo maps using 9 tubes ROI mask.")
    parser.add_argument("--cmrf-maps", type=Path, default=Path("data/output/quantitative_maps.npz"), help="Path to your reconstructed cMRF maps.")
    parser.add_argument("--ref-maps", type=Path, default=Path("data/output/reference_maps/reference_maps.npz"), help="Path to generated reference maps.")
    parser.add_argument("--mask", type=Path, default=Path(r"C:\Users\Lenovo\Desktop\open_source_cmrf_scanner_comparison\scanner1\mask.npy"), help="Path to the 9-tube ROI mask.")
    parser.add_argument("--out-png", type=Path, default=Path("data/output/accuracy_scatter_plot.png"), help="Output validation plot.")
    args = parser.parse_args()

    print(">>> 正在加载数据以进行 ROI 精度比对 <<<")
    # 加载 cMRF 和 Reference 数据
    cmrf_data = np.load(args.cmrf_maps)
    ref_data = np.load(args.ref_maps)
    
    # 提取 T1 和 T2 矩阵
    t1_cmrf, t2_cmrf = cmrf_data['t1_map'], cmrf_data['t2_map']
    t1_ref, t2_ref = ref_data['ref_t1_map'], ref_data['ref_t2_map']
    
    # 加载掩膜并去除多余维度
    mask = np.squeeze(np.load(args.mask))

    print(f"数据加载完成。Mask shape: {mask.shape}, 期望试管数量: 9")

    # 统计 9 个试管区域内的均值和标准差
    num_tubes = 9
    t1_mean_cmrf, t1_std_cmrf, t1_mean_ref, t1_std_ref = [], [], [], []
    t2_mean_cmrf, t2_std_cmrf, t2_mean_ref, t2_std_ref = [], [], [], []

    for i in range(1, num_tubes + 1):
        roi_mask = (mask == i)
        
        t1_mean_cmrf.append(np.mean(t1_cmrf[roi_mask]))
        t1_std_cmrf.append(np.std(t1_cmrf[roi_mask]))
        t1_mean_ref.append(np.mean(t1_ref[roi_mask]))
        t1_std_ref.append(np.std(t1_ref[roi_mask]))
        
        t2_mean_cmrf.append(np.mean(t2_cmrf[roi_mask]))
        t2_std_cmrf.append(np.std(t2_cmrf[roi_mask]))
        t2_mean_ref.append(np.mean(t2_ref[roi_mask]))
        t2_std_ref.append(np.std(t2_ref[roi_mask]))

    # 开始绘图
    print(">>> 正在绘制误差散点图并计算线性回归 <<<")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # T1 散点图
    plot_scatter_with_regression(
        axes[0], t1_mean_ref, t1_mean_cmrf, t1_std_ref, t1_std_cmrf, 
        title="T1 Quantification Accuracy", max_val=2000, color="teal"
    )

    # T2 散点图
    plot_scatter_with_regression(
        axes[1], t2_mean_ref, t2_mean_cmrf, t2_std_ref, t2_std_cmrf, 
        title="T2 Quantification Accuracy", max_val=200, color="teal"
    )

    plt.suptitle("Subspace LLR cMRF vs. Spin-Echo Gold Standard", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存并显示
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=300, bbox_inches='tight')
    print(f"\n验证完成！精度比对散点图已保存至: {args.out_png}")
    plt.show()

if __name__ == "__main__":
    main()