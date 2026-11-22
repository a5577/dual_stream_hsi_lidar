from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    CLASS_NAMES,
    NUM_CLASSES,
    HoustonPatchDataset,
    PreparedData,
    prepare_data,
)
from .metrics import classification_metrics
from .model import DualStreamFusionCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dual-stream HSI/LiDAR CNN on Houston 2013."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("Houston2013"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dual_stream_hsi_lidar/runs/fusion"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("dual_stream_hsi_lidar/cache")
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--fusion-mode",
        choices=("fusion", "hsi", "lidar"),
        default="fusion",
        help="Use both modalities (fusion) or run a unimodal ablation.",
    )
    parser.add_argument(
        "--hsi-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to HSI features before concatenation.",
    )
    parser.add_argument(
        "--lidar-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to LiDAR features before concatenation.",
    )
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker processes; keep 0 in restricted sandboxes.",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional limit for smoke tests; zero uses all training pixels.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Optional limit for smoke tests; zero uses all test pixels.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def stratified_limit(
    coordinates: np.ndarray,
    labels: np.ndarray,
    maximum: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if maximum <= 0 or maximum >= len(labels):
        return coordinates, labels
    selected, _ = train_test_split(
        np.arange(len(labels)),
        train_size=maximum,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return coordinates[selected], labels[selected]


def build_loaders(
    data: PreparedData, args: argparse.Namespace
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_coordinates = np.argwhere(data.train_labels > 0)
    train_labels = data.train_labels[data.train_labels > 0]
    train_coordinates, train_labels = stratified_limit(
        train_coordinates, train_labels, args.max_train_samples, args.seed
    )

    fit_indices, val_indices = train_test_split(
        np.arange(len(train_labels)),
        test_size=args.val_ratio,
        stratify=train_labels,
        random_state=args.seed,
        shuffle=True,
    )
    fit_coordinates, fit_labels = train_coordinates[fit_indices], train_labels[fit_indices]
    val_coordinates, val_labels = train_coordinates[val_indices], train_labels[val_indices]

    common = {
        "hsi": data.hsi,
        "lidar": data.lidar,
        "patch_size": args.patch_size,
    }
    train_dataset = HoustonPatchDataset(
        coordinates=fit_coordinates,
        labels=fit_labels,
        augment=True,
        **common,
    )
    val_dataset = HoustonPatchDataset(
        coordinates=val_coordinates,
        labels=val_labels,
        augment=False,
        **common,
    )

    test_coordinates = np.argwhere(data.test_labels > 0)
    test_labels = data.test_labels[data.test_labels > 0]
    test_coordinates, test_labels = stratified_limit(
        test_coordinates, test_labels, args.max_test_samples, args.seed
    )
    test_dataset = HoustonPatchDataset(
        coordinates=test_coordinates,
        labels=test_labels,
        augment=False,
        **common,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, test_loader, test_coordinates, test_labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for hsi, lidar, target in loader:
        hsi = hsi.to(device, non_blocking=True)
        lidar = lidar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(hsi, lidar)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        batch_size = target.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_probabilities: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    for hsi, lidar, target in loader:
        hsi = hsi.to(device, non_blocking=True)
        lidar = lidar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(hsi, lidar)
        loss = criterion(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        all_probabilities.append(probabilities.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        batch_size = target.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    probabilities = np.concatenate(all_probabilities, axis=0)
    encoded_targets = np.concatenate(all_targets, axis=0)
    predictions = probabilities.argmax(axis=1) + 1
    truth = encoded_targets + 1
    average_loss = total_loss / max(total_examples, 1)
    return truth, predictions, probabilities, average_loss


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    truth, prediction, probabilities, loss = predict(
        model, loader, criterion, device
    )
    metrics = classification_metrics(truth, prediction)
    metrics["loss"] = float(loss)
    return metrics, prediction, probabilities


def save_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def plot_training_history(history: Dict[str, List[float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(epochs, history["train_loss"], label="train")
    axes[0, 0].plot(epochs, history["val_loss"], label="validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, history["val_oa"], color="tab:green")
    axes[0, 1].set_title("Validation OA")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[1, 0].plot(epochs, history["val_aa"], color="tab:orange")
    axes[1, 0].set_title("Validation AA")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 1].plot(epochs, history["val_kappa"], color="tab:red")
    axes[1, 1].set_title("Validation Kappa")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0.0, 1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(metrics: Dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(10.5, 9))
    image = axis.imshow(matrix, cmap="Blues")
    ticks = np.arange(NUM_CLASSES)
    axis.set_xticks(ticks, labels=range(1, NUM_CLASSES + 1))
    axis.set_yticks(ticks, labels=range(1, NUM_CLASSES + 1))
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Houston 2013 test confusion matrix")
    threshold = matrix.max() / 2.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = int(matrix[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_class_accuracy(metrics: Dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(metrics["per_class_accuracy"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    positions = np.arange(NUM_CLASSES)
    axis.bar(positions, values, color="tab:blue")
    axis.set_xticks(positions, labels=range(1, NUM_CLASSES + 1))
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_xlabel("Class ID")
    axis.set_title("Per-class test accuracy")
    for position, value in zip(positions, values):
        axis.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_classification_map(
    ground_truth: np.ndarray,
    prediction_map: np.ndarray,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = plt.get_cmap("tab20").resampled(NUM_CLASSES)
    colormap = ListedColormap(colors(np.arange(NUM_CLASSES)))
    colormap.set_bad("black")
    truth_display = np.ma.masked_where(ground_truth == 0, ground_truth)
    prediction_display = np.ma.masked_where(prediction_map == 0, prediction_map)

    figure, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    axes[0].imshow(truth_display, cmap=colormap, vmin=1, vmax=NUM_CLASSES)
    axes[0].set_title("Ground truth")
    axes[1].imshow(prediction_display, cmap=colormap, vmin=1, vmax=NUM_CLASSES)
    axes[1].set_title("Fusion-CNN prediction (labeled pixels)")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


@torch.no_grad()
def create_classification_map(
    model: nn.Module,
    data: PreparedData,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    coordinates = np.argwhere(data.ground_truth > 0)
    dataset = HoustonPatchDataset(
        hsi=data.hsi,
        lidar=data.lidar,
        coordinates=coordinates,
        labels=data.ground_truth[data.ground_truth > 0],
        patch_size=args.patch_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model.eval()
    encoded_predictions: List[np.ndarray] = []
    for hsi, lidar, _ in loader:
        logits = model(hsi.to(device), lidar.to(device))
        encoded_predictions.append(logits.argmax(dim=1).cpu().numpy())
    prediction_map = np.zeros(data.ground_truth.shape, dtype=np.uint8)
    prediction_map[coordinates[:, 0], coordinates[:, 1]] = (
        np.concatenate(encoded_predictions) + 1
    )
    return prediction_map


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = args.output_dir / ".matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    print(f"Device: {device}")
    print(f"Loading and preprocessing {args.dataset_dir} ...")
    data = prepare_data(
        dataset_dir=args.dataset_dir,
        pca_components=args.pca_components,
        seed=args.seed,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    train_loader, val_loader, test_loader, test_coordinates, test_truth = (
        build_loaders(data, args)
    )
    print(
        f"HSI feature shape={data.hsi.shape}, LiDAR shape={data.lidar.shape}, "
        f"train/val/test={len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)}"
    )

    model = DualStreamFusionCNN(
        hsi_channels=args.pca_components,
        num_classes=NUM_CLASSES,
        stem_channels=args.stem_channels,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
        hsi_weight=args.hsi_weight,
        lidar_weight=args.lidar_weight,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.01
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_oa": [],
        "val_aa": [],
        "val_kappa": [],
        "learning_rate": [],
    }
    best_epoch = 0
    best_score = -1.0
    stale_epochs = 0
    best_model_path = args.output_dir / "best_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics, _, _ = evaluate_split(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_oa"].append(float(val_metrics["oa"]))
        history["val_aa"].append(float(val_metrics["aa"]))
        history["val_kappa"].append(float(val_metrics["kappa"]))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        score = float(val_metrics["oa"]) + 0.1 * float(val_metrics["kappa"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "args": vars(args),
                },
                best_model_path,
            )
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} | "
            f"OA {val_metrics['oa']:.4f} | AA {val_metrics['aa']:.4f} | "
            f"Kappa {val_metrics['kappa']:.4f} | "
            f"best {best_epoch:03d}"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} stale epochs.")
            break

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, test_prediction, test_probabilities = evaluate_split(
        model, test_loader, criterion, device
    )
    prediction_map = create_classification_map(model, data, args, device)

    elapsed_seconds = time.time() - start_time
    result = {
        "dataset": "Houston 2013",
        "fusion_mode": args.fusion_mode,
        "class_names": list(CLASS_NAMES),
        "best_epoch": best_epoch,
        "epochs_completed": len(history["train_loss"]),
        "elapsed_seconds": elapsed_seconds,
        "train_pixels": int((data.train_labels > 0).sum()),
        "test_pixels": int((data.test_labels > 0).sum()),
        "validation_pixels": len(val_loader.dataset),
        "test_metrics": test_metrics,
        "config": vars(args),
    }
    serializable_result = json.loads(
        json.dumps(result, default=lambda value: str(value))
    )

    save_json(args.output_dir / "metrics.json", serializable_result)
    save_json(args.output_dir / "history.json", history)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        coordinates=test_coordinates,
        truth=test_truth,
        prediction=test_prediction,
        probabilities=test_probabilities,
    )
    np.savez_compressed(
        args.output_dir / "classification_map.npz",
        prediction=prediction_map,
        ground_truth=data.ground_truth,
        train_mask=data.train_labels,
        test_mask=data.test_labels,
    )
    savemat(
        args.output_dir / "classification_map.mat",
        {
            "prediction": prediction_map,
            "ground_truth": data.ground_truth,
            "train_mask": data.train_labels,
            "test_mask": data.test_labels,
        },
    )
    plot_training_history(history, args.output_dir / "training_history.png")
    plot_confusion_matrix(
        test_metrics, args.output_dir / "confusion_matrix.png"
    )
    plot_class_accuracy(
        test_metrics, args.output_dir / "class_accuracy.png"
    )
    plot_classification_map(
        data.ground_truth,
        prediction_map,
        args.output_dir / "classification_map.png",
    )

    print(
        f"Test | OA {test_metrics['oa']:.4f} | AA {test_metrics['aa']:.4f} | "
        f"Kappa {test_metrics['kappa']:.4f}"
    )
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

