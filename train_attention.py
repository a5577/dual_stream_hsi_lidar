from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.io import savemat
from torch import nn

from .attention_model import AttentionDualStreamCNN
from .data import CLASS_NAMES, NUM_CLASSES, prepare_data
from .train import (
    build_loaders,
    create_classification_map,
    evaluate_split,
    plot_class_accuracy,
    plot_classification_map,
    plot_confusion_matrix,
    plot_training_history,
    resolve_device,
    save_json,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a dual-stream HSI/LiDAR CNN with cross-attention fusion."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("Houston2013"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dual_stream_hsi_lidar/runs/attention_fusion"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("dual_stream_hsi_lidar/cache")
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> AttentionDualStreamCNN:
    return AttentionDualStreamCNN(
        hsi_channels=args.pca_components,
        num_classes=NUM_CLASSES,
        stem_channels=args.stem_channels,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        dropout=args.dropout,
    )


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

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.01,
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
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics, _, _ = evaluate_split(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_oa"].append(float(val_metrics["oa"]))
        history["val_aa"].append(float(val_metrics["aa"]))
        history["val_kappa"].append(float(val_metrics["kappa"]))
        history["learning_rate"].append(
            float(optimizer.param_groups[0]["lr"])
        )

        score = float(val_metrics["oa"]) + 0.1 * float(val_metrics["kappa"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "architecture": "dual_stream_cross_attention",
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
            f"Kappa {val_metrics['kappa']:.4f} | best {best_epoch:03d}"
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
        model,
        test_loader,
        criterion,
        device,
    )
    prediction_map = create_classification_map(model, data, args, device)

    elapsed_seconds = time.time() - start_time
    result = {
        "dataset": "Houston 2013",
        "architecture": "dual_stream_cross_attention",
        "fusion": (
            "bidirectional multi-head cross-attention + gated pooling"
        ),
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
        test_metrics,
        args.output_dir / "confusion_matrix.png",
    )
    plot_class_accuracy(
        test_metrics,
        args.output_dir / "class_accuracy.png",
    )
    plot_classification_map(
        data.ground_truth,
        prediction_map,
        args.output_dir / "classification_map.png",
    )

    print(
        f"Test | OA {test_metrics['oa']:.4f} | "
        f"AA {test_metrics['aa']:.4f} | "
        f"Kappa {test_metrics['kappa']:.4f}"
    )
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
