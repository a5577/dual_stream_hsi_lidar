from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.io import loadmat

from .data import CLASS_NAMES, NUM_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HSI, LiDAR, labels, spectra, and class counts."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("Houston2013"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dual_stream_hsi_lidar/visualizations"),
    )
    parser.add_argument(
        "--rgb-bands",
        type=int,
        nargs=3,
        default=(30, 60, 90),
        metavar=("R", "G", "B"),
        help="Three 1-based HSI bands used for the false-color image.",
    )
    parser.add_argument(
        "--band-grid",
        type=int,
        nargs="+",
        default=(10, 30, 50, 70, 90, 110, 130, 144),
        metavar="BAND",
        help="1-based HSI bands shown in the band-grid figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open Matplotlib windows. Files are always saved.",
    )
    return parser.parse_args()


def load_array(dataset_dir: Path, filename: str, key: str) -> np.ndarray:
    path = dataset_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    contents = loadmat(path)
    if key not in contents:
        raise KeyError(f"{path} does not contain array '{key}'")
    return contents[key]


def normalize_percentile(
    image: np.ndarray,
    valid_mask: np.ndarray | None = None,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    values = image[valid_mask] if valid_mask is not None else image.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lower, upper = np.percentile(values, (low, high))
    if upper <= lower:
        upper = float(values.max())
        lower = float(values.min())
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def make_false_color(
    hsi: np.ndarray,
    bands: Iterable[int],
    valid_mask: np.ndarray,
) -> np.ndarray:
    selected = [band - 1 for band in bands]
    composite = np.stack(
        [
            normalize_percentile(hsi[..., band], valid_mask)
            for band in selected
        ],
        axis=-1,
    )
    return composite


def validate_bands(bands: Iterable[int], band_count: int) -> None:
    invalid = [band for band in bands if band < 1 or band > band_count]
    if invalid:
        raise ValueError(
            f"HSI band numbers must be between 1 and {band_count}; got {invalid}"
        )


def plot_overview(
    hsi: np.ndarray,
    lidar: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    ground_truth: np.ndarray,
    rgb_bands: Tuple[int, int, int],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    color_map = ListedColormap(
        plt.get_cmap("tab20").resampled(NUM_CLASSES)(np.arange(NUM_CLASSES))
    )
    color_map.set_bad("black")
    boundaries = np.arange(0.5, NUM_CLASSES + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, color_map.N)
    label_mask = ground_truth > 0
    display_labels = np.ma.masked_where(~label_mask, ground_truth)
    false_color = make_false_color(hsi, rgb_bands, label_mask)

    sample_canvas = np.zeros((*ground_truth.shape, 3), dtype=np.float32)
    sample_canvas[train_labels > 0] = (0.0, 0.8, 1.0)
    sample_canvas[test_labels > 0] = (1.0, 0.2, 0.2)

    figure, axes = plt.subplots(2, 2, figsize=(16, 7.8))
    axes[0, 0].imshow(false_color)
    axes[0, 0].set_title(
        f"HSI false-color composite | bands {rgb_bands[0]}/{rgb_bands[1]}/"
        f"{rgb_bands[2]} (1-based)"
    )
    axes[0, 1].imshow(lidar, cmap="terrain", vmin=0, vmax=255)
    axes[0, 1].set_title("LiDAR elevation feature map")
    image = axes[1, 0].imshow(
        display_labels,
        cmap=color_map,
        norm=norm,
        interpolation="nearest",
    )
    axes[1, 0].set_title(
        f"Ground truth | {int(label_mask.sum()):,} labeled pixels"
    )
    axes[1, 1].imshow(sample_canvas)
    axes[1, 1].set_title("Official labeled-pixel split")
    axes[1, 1].legend(
        handles=[
            Patch(color=(0.0, 0.8, 1.0), label=f"Train: {int((train_labels > 0).sum()):,}"),
            Patch(color=(1.0, 0.2, 0.2), label=f"Test: {int((test_labels > 0).sum()):,}"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    for axis in axes.ravel():
        axis.axis("off")

    colorbar = figure.colorbar(
        image,
        ax=axes[1, 0],
        ticks=np.arange(1, NUM_CLASSES + 1),
        fraction=0.04,
        pad=0.02,
    )
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("Class ID")
    figure.suptitle("Houston 2013 dataset overview", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_hsi_bands(
    hsi: np.ndarray,
    bands: Iterable[int],
    ground_truth: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = list(bands)
    columns = min(4, len(bands))
    rows = int(np.ceil(len(bands) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 2.5 * rows), squeeze=False
    )
    valid_mask = ground_truth > 0
    for axis, band in zip(axes.ravel(), bands):
        display = normalize_percentile(hsi[..., band - 1], valid_mask)
        axis.imshow(display, cmap="gray")
        axis.set_title(f"HSI band {band}")
        axis.axis("off")
    for axis in axes.ravel()[len(bands) :]:
        axis.axis("off")
    figure.suptitle("Selected HSI spectral bands", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_class_mean_spectra(
    hsi: np.ndarray,
    train_labels: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6.5))
    hsi_float = hsi.astype(np.float32, copy=False)
    band_numbers = np.arange(1, hsi.shape[2] + 1)
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        spectra = hsi_float[train_labels == class_id]
        mean_spectrum = spectra.mean(axis=0)
        axis.plot(
            band_numbers,
            mean_spectrum,
            linewidth=1.2,
            label=f"{class_id}: {class_name}",
        )
    axis.set_xlabel("Spectral band")
    axis.set_ylabel("Mean raw value")
    axis.set_title("Mean training spectra for each class")
    axis.grid(alpha=0.25)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_class_distribution(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_counts = np.bincount(train_labels.ravel(), minlength=NUM_CLASSES + 1)[1:]
    test_counts = np.bincount(test_labels.ravel(), minlength=NUM_CLASSES + 1)[1:]
    positions = np.arange(NUM_CLASSES)
    width = 0.4

    figure, axis = plt.subplots(figsize=(12.5, 5.5))
    train_bars = axis.bar(
        positions - width / 2,
        train_counts,
        width,
        label="Train",
        color="tab:blue",
    )
    test_bars = axis.bar(
        positions + width / 2,
        test_counts,
        width,
        label="Test",
        color="tab:orange",
    )
    axis.set_xticks(positions, labels=range(1, NUM_CLASSES + 1))
    axis.set_xlabel("Class ID")
    axis.set_ylabel("Labeled pixels")
    axis.set_title("Houston 2013 train/test class distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bars in (train_bars, test_bars):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = args.output_dir / ".matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    hsi = load_array(args.dataset_dir, "HSI.mat", "HSI")
    lidar = load_array(args.dataset_dir, "LiDAR.mat", "LiDAR").squeeze()
    train_labels = load_array(
        args.dataset_dir, "TRLabel.mat", "TRLabel"
    ).squeeze()
    test_labels = load_array(
        args.dataset_dir, "TSLabel.mat", "TSLabel"
    ).squeeze()
    ground_truth = load_array(args.dataset_dir, "gt.mat", "gt").squeeze()

    if hsi.ndim != 3:
        raise ValueError(f"Expected a 3-D HSI cube, got shape {hsi.shape}")
    if not (
        hsi.shape[:2]
        == lidar.shape
        == train_labels.shape
        == test_labels.shape
        == ground_truth.shape
    ):
        raise ValueError("HSI, LiDAR, and label spatial dimensions must match")

    validate_bands(args.rgb_bands, hsi.shape[2])
    validate_bands(args.band_grid, hsi.shape[2])

    print(f"HSI:         {hsi.shape}, {hsi.dtype}")
    print(f"LiDAR:       {lidar.shape}, {lidar.dtype}")
    print(
        f"Labels:      train={int((train_labels > 0).sum()):,}, "
        f"test={int((test_labels > 0).sum()):,}, "
        f"total={int((ground_truth > 0).sum()):,}"
    )
    print(f"Output:      {args.output_dir.resolve()}")

    plot_overview(
        hsi,
        lidar,
        train_labels,
        test_labels,
        ground_truth,
        tuple(args.rgb_bands),
        args.output_dir / "dataset_overview.png",
    )
    plot_hsi_bands(
        hsi,
        args.band_grid,
        ground_truth,
        args.output_dir / "hsi_bands.png",
    )
    plot_class_mean_spectra(
        hsi,
        train_labels,
        args.output_dir / "class_mean_spectra.png",
    )
    plot_class_distribution(
        train_labels,
        test_labels,
        args.output_dir / "class_distribution.png",
    )

    print("Saved:")
    for filename in (
        "dataset_overview.png",
        "hsi_bands.png",
        "class_mean_spectra.png",
        "class_distribution.png",
    ):
        print(f"  {args.output_dir / filename}")

    if args.show:
        import matplotlib.pyplot as plt

        for filename in (
            "dataset_overview.png",
            "hsi_bands.png",
            "class_mean_spectra.png",
            "class_distribution.png",
        ):
            image = plt.imread(args.output_dir / filename)
            figure = plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.axis("off")
            plt.title(filename)
        plt.show()


if __name__ == "__main__":
    main()

