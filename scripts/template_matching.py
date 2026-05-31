from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mri_project.quantification import execute_template_matching, generate_mock_coeff_maps


def plot_quantitative_maps(t1_map: np.ndarray, t2_map: np.ndarray, pd_map: np.ndarray) -> None:
    """Render final T1, T2, and PD maps."""
    print("\n[Render] 渲染最终图像...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(t1_map, cmap="magma", vmin=0, vmax=2000)
    axes[0].set_title("T1 Map (ms)")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(t2_map, cmap="viridis", vmin=0, vmax=150)
    axes[1].set_title("T2 Map (ms)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(pd_map, cmap="bone")
    axes[2].set_title("PD Map (a.u.)")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("MR Fingerprinting Quantitative Maps", fontsize=16)
    plt.tight_layout()
    plt.show()


def load_dictionary(dict_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the compressed dictionary package used by template matching."""
    dictionary_data = np.load(dict_path)
    return (
        dictionary_data["compressed_dict"],
        dictionary_data["t1"],
        dictionary_data["t2"],
    )


def main() -> None:
    print(">>> 启动 MRF 模板匹配流程 <<<")

    dict_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "mrf_dictionary_data.npz"
    )
    print(f"\n[Init] 正在加载压缩字典:\n{dict_path}")
    dict_compressed, t1_grid, t2_grid = load_dictionary(dict_path)
    print(f"[Init] 成功加载! 字典维度: {dict_compressed.shape}")

    print("\n[Mock] 正在生成用于测试的伪造 2D 系数图...")
    coeff_maps_input = generate_mock_coeff_maps(t1_grid, t2_grid, dict_compressed)
    print(f"[Mock] 成功伪造系数图! 维度: {coeff_maps_input.shape}")

    # When reconstructed coefficient maps are ready, replace the mock line above
    # with loading the reconstruction output.

    print("\n[Match] 开始执行大规模内积匹配 (Template Matching)...")
    start_time = time.time()
    t1_result, t2_result, pd_result = execute_template_matching(
        coeff_maps_input,
        dict_compressed,
        t1_grid,
        t2_grid,
    )
    num_pixels = coeff_maps_input.shape[0] * coeff_maps_input.shape[1]
    elapsed = time.time() - start_time
    print(f"[Match] 匹配完成！{num_pixels} 个像素点比对耗时: {elapsed:.3f} 秒")

    plot_quantitative_maps(t1_result, t2_result, pd_result)
    print("\n>>> 流程执行完毕 <<<")


if __name__ == "__main__":
    main()
