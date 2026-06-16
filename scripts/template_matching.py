from __future__ import annotations

import os

# ============================================================
# Windows + Anaconda 常见 OpenMP 冲突修复
# 必须放在 numpy / cupy / matplotlib 之前
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flattened-dictionary MRF template matching and render quantitative maps."
    )

    parser.add_argument(
        "--coeff-maps",
        type=Path,
        default=Path("data/output/reconstructed_coeff_maps.npy"),
        help="Input reconstructed coefficient maps. Expected shape [H, W, rank] or [rank, H, W].",
    )

    # 下方是完善时需要修改的本地路径
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=Path(r"C:\Users\Lenovo\Desktop\open_source_cmrf_scanner_comparison\scanner1\processed\mrf_dictionary_data.npz"),
        help="Input dictionary npz file.",
    )

    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("data/output/quantitative_maps.png"),
        help="Path used to save the rendered T1/T2/PD map figure.",
    )

    parser.add_argument(
        "--save-npz",
        type=Path,
        default=Path("data/output/quantitative_maps.npz"),
        help="Path used to save raw T1/T2/PD maps.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening the interactive popup window.",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Array backend used for matching.",
    )

    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device id used when --device cuda is selected.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help=(
            "Number of pixels matched per batch. "
            "Default 1024 is safer for GPU/CPU memory. "
            "Use larger values only if memory is sufficient."
        ),
    )

    parser.add_argument(
        "--t1-vmax",
        type=float,
        default=2000.0,
        help="Upper display limit for T1 map.",
    )

    parser.add_argument(
        "--t2-vmax",
        type=float,
        default=150.0,
        help="Upper display limit for T2 map.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_first_existing_key(npz_data: np.lib.npyio.NpzFile, candidates: list[str], desc: str) -> str:
    available = list(npz_data.keys())
    for key in candidates:
        if key in npz_data:
            return key

    raise KeyError(
        f"在字典文件中找不到 {desc}。\n"
        f"候选 key: {candidates}\n"
        f"当前 npz 中可用 keys: {available}"
    )


def load_dictionary(dict_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 flattened compressed dictionary。

    目标格式：
        dict_compressed.shape = [rank, n_atoms]
        t1_grid.shape         = [n_atoms]
        t2_grid.shape         = [n_atoms]

    你的当前文件中：
        compressed_dict.shape = (5, 22988)
        t1.shape              = (22988,)
        t2.shape              = (22988,)
    """
    if not dict_path.exists():
        raise FileNotFoundError(f"找不到字典文件: {dict_path}")

    data = np.load(dict_path)
    print(f"[Dict] 当前字典 npz keys: {list(data.keys())}")

    # 优先使用 normalized compressed dictionary
    # 因为模板匹配本质是比较方向/相似度，而不是绝对幅度
    dict_key = get_first_existing_key(
        data,
        candidates=[
            "compressed_dict_normalized",
            "compressed_dict",
            "compressed_dictionary",
            "dict_compressed",
            "dictionary_compressed",
        ],
        desc="压缩字典 compressed dictionary",
    )

    t1_key = get_first_existing_key(
        data,
        candidates=["t1", "T1", "t1_grid", "T1_grid", "t1_values", "T1_values"],
        desc="T1 参数数组",
    )

    t2_key = get_first_existing_key(
        data,
        candidates=["t2", "T2", "t2_grid", "T2_grid", "t2_values", "T2_values"],
        desc="T2 参数数组",
    )

    dict_compressed = data[dict_key]
    t1_grid = np.asarray(data[t1_key]).reshape(-1)
    t2_grid = np.asarray(data[t2_key]).reshape(-1)

    if dict_compressed.ndim != 2:
        raise ValueError(
            f"当前新版 matching 需要二维 flattened dictionary，"
            f"即 [rank, n_atoms] 或 [n_atoms, rank]。\n"
            f"但当前 {dict_key}.shape = {dict_compressed.shape}"
        )

    if t1_grid.shape != t2_grid.shape:
        raise ValueError(
            f"T1/T2 数组长度不一致: t1={t1_grid.shape}, t2={t2_grid.shape}"
        )

    if np.iscomplexobj(dict_compressed):
        dict_compressed = dict_compressed.astype(np.complex64, copy=False)
    else:
        dict_compressed = dict_compressed.astype(np.float32, copy=False)

    t1_grid = t1_grid.astype(np.float32, copy=False)
    t2_grid = t2_grid.astype(np.float32, copy=False)

    print(f"[Dict] 使用压缩字典 key: {dict_key}")
    print(f"[Dict] 使用 T1 key: {t1_key}")
    print(f"[Dict] 使用 T2 key: {t2_key}")
    print(f"[Dict] 原始 compressed dictionary shape: {dict_compressed.shape}")
    print(f"[Dict] T1 grid shape: {t1_grid.shape}")
    print(f"[Dict] T2 grid shape: {t2_grid.shape}")

    return dict_compressed, t1_grid, t2_grid


def load_coeff_maps(coeff_maps_path: Path) -> np.ndarray:
    """
    加载重建出的子空间系数图。

    支持：
        [H, W, rank]
        [rank, H, W]
    """
    if not coeff_maps_path.exists():
        raise FileNotFoundError(f"找不到重建系数图文件: {coeff_maps_path}")

    coeff_maps = np.load(coeff_maps_path)

    if coeff_maps.ndim != 3:
        raise ValueError(
            f"reconstructed_coeff_maps.npy 应该是三维数组，"
            f"例如 [H, W, rank] 或 [rank, H, W]，但当前 shape = {coeff_maps.shape}"
        )

    if np.iscomplexobj(coeff_maps):
        coeff_maps = coeff_maps.astype(np.complex64, copy=False)
    else:
        coeff_maps = coeff_maps.astype(np.float32, copy=False)

    return coeff_maps


def standardize_coeff_maps_orientation(coeff_maps: np.ndarray, rank: int) -> np.ndarray:
    """
    统一 coefficient maps 到：

        [H, W, rank]
    """
    print("\n[Check] 正在检查 coefficient maps 方向...")
    print(f"[Check] raw coeff_maps shape = {coeff_maps.shape}")
    print(f"[Check] expected rank = {rank}")

    if coeff_maps.shape[-1] == rank:
        coeff_hwr = coeff_maps
        print("[Check] coeff_maps 格式识别为 [H, W, rank]，无需转置。")

    elif coeff_maps.shape[0] == rank:
        coeff_hwr = np.moveaxis(coeff_maps, 0, -1)
        print("[Check] coeff_maps 格式识别为 [rank, H, W]，已转置为 [H, W, rank]。")

    else:
        raise ValueError(
            f"无法判断 coefficient maps 的 rank 维。\n"
            f"    coeff_maps shape = {coeff_maps.shape}\n"
            f"    expected rank    = {rank}\n"
            f"coefficient maps 应该是 [H, W, rank] 或 [rank, H, W]。"
        )

    return np.ascontiguousarray(coeff_hwr)


def standardize_dictionary_orientation(
    dict_compressed: np.ndarray,
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    rank: int,
) -> np.ndarray:
    """
    统一 compressed dictionary 到：

        [rank, n_atoms]
    """
    n_atoms = t1_grid.size

    print("\n[Check] 正在检查压缩字典方向...")
    print(f"[Check] coeff rank = {rank}")
    print(f"[Check] dictionary raw shape = {dict_compressed.shape}")
    print(f"[Check] number of dictionary atoms from T1/T2 = {n_atoms}")

    if t2_grid.size != n_atoms:
        raise ValueError("T1/T2 grid size mismatch.")

    # 已经是 [rank, n_atoms]
    if dict_compressed.shape[0] == rank and dict_compressed.shape[1] == n_atoms:
        dict_rank_atoms = dict_compressed
        print("[Check] dictionary 格式识别为 [rank, n_atoms]，无需转置。")

    # 是 [n_atoms, rank]
    elif dict_compressed.shape[1] == rank and dict_compressed.shape[0] == n_atoms:
        dict_rank_atoms = dict_compressed.T
        print("[Check] dictionary 格式识别为 [n_atoms, rank]，已转置为 [rank, n_atoms]。")

    else:
        raise ValueError(
            f"无法判断 compressed dictionary 的方向。\n"
            f"    rank = {rank}\n"
            f"    n_atoms = {n_atoms}\n"
            f"    dict shape = {dict_compressed.shape}\n"
            f"dictionary 应该是 [rank, n_atoms] 或 [n_atoms, rank]。"
        )

    return np.ascontiguousarray(dict_rank_atoms)


def normalize_dictionary(dict_rank_atoms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对 dictionary 每个 atom 做 L2 归一化。

    输入：
        [rank, n_atoms]

    输出：
        [rank, n_atoms]
    """
    norms = np.sqrt(np.sum(np.abs(dict_rank_atoms) ** 2, axis=0, keepdims=True))
    norms = np.maximum(norms, eps)
    return dict_rank_atoms / norms


def check_cuda_if_needed(device: str, gpu_device: int) -> None:
    if device != "cuda":
        return

    try:
        import cupy as cp

        n_gpu = cp.cuda.runtime.getDeviceCount()
        if n_gpu <= 0:
            raise RuntimeError("CuPy 没有检测到可用 GPU。")

        props = cp.cuda.runtime.getDeviceProperties(gpu_device)
        gpu_name = props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        print(f"[CUDA] CuPy 可用，检测到 GPU 数量: {n_gpu}")
        print(f"[CUDA] 当前使用 GPU id: {gpu_device}")
        print(f"[CUDA] GPU 名称: {gpu_name}")

    except Exception as exc:
        raise RuntimeError(
            "你设置了 --device cuda，但 CuPy/CUDA 后端不可用。\n"
            "请检查 cupy-cuda11x 或 cupy-cuda12x 是否安装正确，"
            "或者改用 --device cpu。"
        ) from exc


def template_matching_flat_cpu(
    coeff_maps_hwr: np.ndarray,
    dict_rank_atoms: np.ndarray,
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    batch_size: int = 1024,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CPU 版本 flattened dictionary template matching。

    coeff_maps_hwr:
        [H, W, rank]

    dict_rank_atoms:
        [rank, n_atoms]

    逐像素计算：
        argmax | <normalized pixel, normalized dictionary atom> |
    """
    H, W, rank = coeff_maps_hwr.shape
    n_pixels = H * W
    n_atoms = dict_rank_atoms.shape[1]

    coeff_flat = coeff_maps_hwr.reshape(n_pixels, rank)
    dict_norm = normalize_dictionary(dict_rank_atoms, eps=eps)

    t1_out = np.zeros(n_pixels, dtype=np.float32)
    t2_out = np.zeros(n_pixels, dtype=np.float32)
    pd_out = np.zeros(n_pixels, dtype=np.float32)

    print(f"[Match-CPU] n_pixels = {n_pixels}")
    print(f"[Match-CPU] n_atoms  = {n_atoms}")
    print(f"[Match-CPU] rank     = {rank}")
    print(f"[Match-CPU] batch    = {batch_size}")

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        batch = coeff_flat[start:end]

        # 原始 batch 范数
        batch_norm = np.sqrt(np.sum(np.abs(batch) ** 2, axis=1, keepdims=True))
        batch_norm = np.maximum(batch_norm, eps)

        # 投影到归一化字典
        # sim_raw[b, a] = <dict_norm[:, a], batch[b]>
        sim_raw = batch @ np.conj(dict_norm)

        # 用归一化相关性做匹配
        scores = np.abs(sim_raw) / batch_norm

        best_idx = np.argmax(scores, axis=1)

        # PD 估计使用原始投影幅值
        pd = np.abs(sim_raw[np.arange(end - start), best_idx])

        t1_out[start:end] = t1_grid[best_idx]
        t2_out[start:end] = t2_grid[best_idx]
        pd_out[start:end] = pd.astype(np.float32)

        if start == 0 or end == n_pixels or (start // batch_size) % 20 == 0:
            print(f"[Match-CPU] processed {end}/{n_pixels}")

    return (
        t1_out.reshape(H, W),
        t2_out.reshape(H, W),
        pd_out.reshape(H, W),
    )


def template_matching_flat_cuda(
    coeff_maps_hwr: np.ndarray,
    dict_rank_atoms: np.ndarray,
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    gpu_device: int = 0,
    batch_size: int = 1024,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU/CuPy 版本 flattened dictionary template matching。
    """
    import cupy as cp

    H, W, rank = coeff_maps_hwr.shape
    n_pixels = H * W
    n_atoms = dict_rank_atoms.shape[1]

    print(f"[Match-CUDA] n_pixels = {n_pixels}")
    print(f"[Match-CUDA] n_atoms  = {n_atoms}")
    print(f"[Match-CUDA] rank     = {rank}")
    print(f"[Match-CUDA] batch    = {batch_size}")

    with cp.cuda.Device(gpu_device):
        coeff_flat_gpu = cp.asarray(coeff_maps_hwr.reshape(n_pixels, rank))
        dict_gpu = cp.asarray(dict_rank_atoms)

        # dictionary 归一化，按 atom 归一化
        dict_norms = cp.sqrt(cp.sum(cp.abs(dict_gpu) ** 2, axis=0, keepdims=True))
        dict_norms = cp.maximum(dict_norms, eps)
        dict_norm_gpu = dict_gpu / dict_norms

        t1_out = np.zeros(n_pixels, dtype=np.float32)
        t2_out = np.zeros(n_pixels, dtype=np.float32)
        pd_out = np.zeros(n_pixels, dtype=np.float32)

        t1_grid_cpu = np.asarray(t1_grid)
        t2_grid_cpu = np.asarray(t2_grid)

        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)

            batch = coeff_flat_gpu[start:end]

            batch_norm = cp.sqrt(cp.sum(cp.abs(batch) ** 2, axis=1, keepdims=True))
            batch_norm = cp.maximum(batch_norm, eps)

            # sim_raw: [batch, n_atoms]
            sim_raw = batch @ cp.conj(dict_norm_gpu)

            scores = cp.abs(sim_raw) / batch_norm
            best_idx_gpu = cp.argmax(scores, axis=1)

            row_idx_gpu = cp.arange(end - start)
            pd_gpu = cp.abs(sim_raw[row_idx_gpu, best_idx_gpu])

            best_idx = cp.asnumpy(best_idx_gpu)
            pd = cp.asnumpy(pd_gpu).astype(np.float32)

            t1_out[start:end] = t1_grid_cpu[best_idx]
            t2_out[start:end] = t2_grid_cpu[best_idx]
            pd_out[start:end] = pd

            if start == 0 or end == n_pixels or (start // batch_size) % 20 == 0:
                print(f"[Match-CUDA] processed {end}/{n_pixels}")

            # 释放 batch 中间大矩阵
            del sim_raw, scores, best_idx_gpu, pd_gpu
            cp.get_default_memory_pool().free_all_blocks()

    return (
        t1_out.reshape(H, W),
        t2_out.reshape(H, W),
        pd_out.reshape(H, W),
    )


def execute_template_matching_flat(
    coeff_maps_hwr: np.ndarray,
    dict_rank_atoms: np.ndarray,
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    device: str = "cpu",
    gpu_device: int = 0,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一入口：支持 CPU / CUDA 的 flattened dictionary matching。
    """
    if coeff_maps_hwr.ndim != 3:
        raise ValueError(f"coeff_maps_hwr 必须是 [H, W, rank]，当前 {coeff_maps_hwr.shape}")

    if dict_rank_atoms.ndim != 2:
        raise ValueError(f"dict_rank_atoms 必须是 [rank, n_atoms]，当前 {dict_rank_atoms.shape}")

    H, W, rank = coeff_maps_hwr.shape

    if dict_rank_atoms.shape[0] != rank:
        raise ValueError(
            f"coeff rank 和 dictionary rank 不一致。\n"
            f"    coeff rank = {rank}\n"
            f"    dict shape = {dict_rank_atoms.shape}"
        )

    n_atoms = dict_rank_atoms.shape[1]

    if t1_grid.size != n_atoms or t2_grid.size != n_atoms:
        raise ValueError(
            f"T1/T2 数量必须等于 dictionary atom 数。\n"
            f"    n_atoms = {n_atoms}\n"
            f"    t1 size = {t1_grid.size}\n"
            f"    t2 size = {t2_grid.size}"
        )

    if batch_size is None:
        batch_size = H * W

    if device == "cuda":
        return template_matching_flat_cuda(
            coeff_maps_hwr,
            dict_rank_atoms,
            t1_grid,
            t2_grid,
            gpu_device=gpu_device,
            batch_size=batch_size,
        )

    return template_matching_flat_cpu(
        coeff_maps_hwr,
        dict_rank_atoms,
        t1_grid,
        t2_grid,
        batch_size=batch_size,
    )


def plot_quantitative_maps(
    t1_map: np.ndarray,
    t2_map: np.ndarray,
    pd_map: np.ndarray,
    save_path: Path | None = None,
    show: bool = True,
    t1_vmax: float = 2000.0,
    t2_vmax: float = 150.0,
) -> None:
    print("\n[Render] Rendering final quantitative maps...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(t1_map, cmap="magma", vmin=0, vmax=t1_vmax)
    axes[0].set_title("T1 Map (ms)")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(t2_map, cmap="viridis", vmin=0, vmax=t2_vmax)
    axes[1].set_title("T2 Map (ms)")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(pd_map, cmap="bone")
    axes[2].set_title("PD Map (a.u.)")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("MR Fingerprinting Quantitative Maps", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Render] Saved quantitative maps to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.device == "cuda":
        check_cuda_if_needed(args.device, args.gpu_device)

    print("\n>>> Starting REAL cMRF flattened template matching pipeline <<<")

    dict_path = resolve_path(args.dict_path)
    coeff_maps_path = resolve_path(args.coeff_maps)
    save_path = resolve_path(args.save_path)
    save_npz_path = resolve_path(args.save_npz)

    # ------------------------------------------------------------
    # 1. Load dictionary
    # ------------------------------------------------------------
    print(f"\n[Init] Loading compressed dictionary:\n{dict_path}")
    dict_compressed_raw, t1_grid, t2_grid = load_dictionary(dict_path)

    # ------------------------------------------------------------
    # 2. Load coefficient maps
    # ------------------------------------------------------------
    print(f"\n[Data] Loading reconstructed coefficient maps:\n{coeff_maps_path}")
    coeff_maps_raw = load_coeff_maps(coeff_maps_path)
    print(f"[Data] Raw coefficient map shape: {coeff_maps_raw.shape}")

    # ------------------------------------------------------------
    # 3. 自动推断 rank
    # ------------------------------------------------------------
    possible_coeff_ranks = [coeff_maps_raw.shape[-1], coeff_maps_raw.shape[0]]
    dict_dims = dict_compressed_raw.shape

    candidate_ranks = []
    for r in possible_coeff_ranks:
        if r in dict_dims:
            candidate_ranks.append(r)

    candidate_ranks = list(dict.fromkeys(candidate_ranks))

    if len(candidate_ranks) == 0:
        raise ValueError(
            f"无法从 coeff_maps 和 compressed_dict 中推断共同 rank。\n"
            f"    coeff_maps shape = {coeff_maps_raw.shape}\n"
            f"    dict shape       = {dict_compressed_raw.shape}\n"
            f"通常 rank 应该是 5。"
        )

    rank = candidate_ranks[0]

    if len(candidate_ranks) > 1:
        print(
            f"[Warning] 检测到多个可能 rank: {candidate_ranks}，"
            f"默认使用第一个: {rank}"
        )

    coeff_maps_hwr = standardize_coeff_maps_orientation(coeff_maps_raw, rank=rank)

    dict_rank_atoms = standardize_dictionary_orientation(
        dict_compressed_raw,
        t1_grid,
        t2_grid,
        rank=rank,
    )

    print("\n========== [Matching Input Summary] ==========")
    print(f"coeff_maps_hwr shape : {coeff_maps_hwr.shape}  # [H, W, rank]")
    print(f"dict_rank_atoms shape: {dict_rank_atoms.shape}  # [rank, n_atoms]")
    print(f"t1_grid shape        : {t1_grid.shape}")
    print(f"t2_grid shape        : {t2_grid.shape}")
    print(f"device               : {args.device}")
    print(f"batch_size           : {args.batch_size}")
    print("==============================================")

    # ------------------------------------------------------------
    # 4. Flattened template matching
    # ------------------------------------------------------------
    print("\n[Match] Running flattened template matching...")
    start_time = time.time()

    t1_result, t2_result, pd_result = execute_template_matching_flat(
        coeff_maps_hwr,
        dict_rank_atoms,
        t1_grid,
        t2_grid,
        device=args.device,
        gpu_device=args.gpu_device,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - start_time
    num_pixels = coeff_maps_hwr.shape[0] * coeff_maps_hwr.shape[1]

    print(f"\n[Match] Matched {num_pixels} pixels in {elapsed:.3f} seconds.")
    print(f"[Match] T1 map shape: {t1_result.shape}")
    print(f"[Match] T2 map shape: {t2_result.shape}")
    print(f"[Match] PD map shape: {pd_result.shape}")

    print("\n[Match] Quantitative range summary:")
    print(f"    T1 min/max: {np.nanmin(t1_result):.6g} / {np.nanmax(t1_result):.6g}")
    print(f"    T2 min/max: {np.nanmin(t2_result):.6g} / {np.nanmax(t2_result):.6g}")
    print(f"    PD min/max: {np.nanmin(pd_result):.6g} / {np.nanmax(pd_result):.6g}")

    # ------------------------------------------------------------
    # 5. Save raw maps
    # ------------------------------------------------------------
    save_npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_npz_path,
        t1_map=t1_result,
        t2_map=t2_result,
        pd_map=pd_result,
    )

    print(f"\n[Save] Saved raw quantitative maps to:")
    print(f"       {save_npz_path}")

    # ------------------------------------------------------------
    # 6. Render png
    # ------------------------------------------------------------
    plot_quantitative_maps(
        t1_result,
        t2_result,
        pd_result,
        save_path=save_path,
        show=not args.no_show,
        t1_vmax=args.t1_vmax,
        t2_vmax=args.t2_vmax,
    )

    print("\n>>> Template matching pipeline finished <<<")


if __name__ == "__main__":
    main()