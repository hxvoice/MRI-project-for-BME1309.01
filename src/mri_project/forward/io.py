"""Small file and plotting helpers for generated artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_npy(data: np.ndarray, path: str | Path) -> None:
    """Save an array and create the parent directory when needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data)


def plot_traj_2d(traj: np.ndarray, title: str = "2D spiral trajectory", save_path: str | Path | None = None) -> None:
    """Plot 2D k-space coordinates."""

    plt.figure(figsize=(6, 6))
    plt.scatter(traj[:, 0], traj[:, 1], s=0.1, c="k")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.title(title)
    plt.axis("equal")
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_phantom(t1: np.ndarray, t2: np.ndarray, pd: np.ndarray, save_path: str | Path | None = None) -> None:
    """Plot T1, T2, and proton density maps."""

    plt.figure(figsize=(12, 4))
    for index, (image, title) in enumerate([(t1, "T1 (ms)"), (t2, "T2 (ms)"), (pd, "PD")], start=1):
        plt.subplot(1, 3, index)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.colorbar()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
