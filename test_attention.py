from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.io import savemat
from torch import nn
from torch.utils.data import DataLoader

from .attention_model import AttentionDualStreamCNN
from .data import (
    CLASS_NAMES,
    NUM_CLASSES,
    HoustonPatchDataset,
    prepare_data,
)
from .train import (
    create_classification_map,
    evaluate_split,
    plot_class_accuracy,
    plot_classification_map,
    plot_confusion_matrix,
    resolve_device,
    save_json,
    seed_everything,
    stratified_limit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an attention-fused HSI/LiDAR CNN checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save fusion gates and mean cross-attention matrices.",
    )
    return parser.parse_args()


def build_test_loader(
    data,
    patch_size: int,
    batch_size: int,
    workers: int,
    maximum: int,
    seed: int,
):
    coordinates = np.argwhere(data.test_labels > 0)
    labels = data.test_labels[data.test_labels > 0]
    coordinates, labels = stratified_limit(
        coordinates,
        labels,
        maximum,
        seed,
    )
    dataset = HoustonPatchDataset(
        hsi=data.hsi,
        lidar=data.lidar,
        coordinates=coordinates,
        labels=labels,
        patch_size=patch_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return loader, coordinates, labels


@torch.no_grad()
def collect_attention(
    model: AttentionDualStreamCNN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    gates: List[np.ndarray] = []
    hsi_attention: List[np.ndarray] = []
    lidar_attention: List[np.ndarray] = []
    model.eval()
    for hsi, lidar, _ in loader:
        _, details = model(
            hsi.to(device),
            lidar.to(device),
            return_attention=True,
        )
        gates.append(details["fusion_gate"].cpu().numpy())
        hsi_attention.append(
            details["hsi_to_lidar"].mean(dim=0).cpu().numpy()
        )
        lidar_attention.append(
            details["lidar_to_hsi"].mean(dim=0).cpu().numpy()
        )
    return {
        "fusion_gate": np.concatenate(gates, axis=0),
        "mean_hsi_to_lidar_attention": np.mean(hsi_attention, axis=0),
        "mean_lidar_to_hsi_attention": np.mean(lidar_attention, axis=0),
    }


def main() -> None:
    cli = parse_args()
    checkpoint_path = cli.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("architecture") != "dual_stream_cross_attention":
        raise ValueError(
            "Checkpoint is not an AttentionDualStreamCNN model"
        )

    checkpoint_args = dict(checkpoint["args"])
    checkpoint_args["dataset_dir"] = Path(checkpoint_args["dataset_dir"])
    checkpoint_args["output_dir"] = Path(checkpoint_args["output_dir"])
    checkpoint_args["cache_dir"] = Path(checkpoint_args["cache_dir"])
    if cli.dataset_dir is not None:
        checkpoint_args["dataset_dir"] = cli.dataset_dir
    args = argparse.Namespace(**checkpoint_args)

    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    if cli.workers is not None:
        args.workers = cli.workers
    args.output_dir = (
        cli.output_dir
        if cli.output_dir is not None
        else checkpoint_path.parent / "test"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    torch.set_num_threads(max(1, cli.threads))
    device = resolve_device(cli.device)
    data = prepare_data(
        dataset_dir=args.dataset_dir,
        pca_components=args.pca_components,
        seed=args.seed,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    test_loader, coordinates, truth = build_test_loader(
        data=data,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        workers=args.workers,
        maximum=cli.max_test_samples,
        seed=args.seed,
    )

    model = AttentionDualStreamCNN(
        hsi_channels=args.pca_components,
        num_classes=NUM_CLASSES,
        stem_channels=args.stem_channels,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    metrics, prediction, probabilities = evaluate_split(
        model,
        test_loader,
        criterion,
        device,
    )
    prediction_map = create_classification_map(model, data, args, device)

    result = {
        "dataset": "Houston 2013",
        "architecture": "dual_stream_cross_attention",
        "checkpoint": str(checkpoint_path),
        "test_metrics": metrics,
        "class_names": list(CLASS_NAMES),
        "test_pixels": len(truth),
        "config": vars(args),
    }
    serializable_result = json.loads(
        json.dumps(result, default=lambda value: str(value))
    )
    save_json(
        args.output_dir / "test_metrics.json",
        serializable_result,
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        coordinates=coordinates,
        truth=truth,
        prediction=prediction,
        probabilities=probabilities,
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
    plot_confusion_matrix(
        metrics,
        args.output_dir / "confusion_matrix.png",
    )
    plot_class_accuracy(
        metrics,
        args.output_dir / "class_accuracy.png",
    )
    plot_classification_map(
        data.ground_truth,
        prediction_map,
        args.output_dir / "classification_map.png",
    )

    if cli.save_attention:
        attention = collect_attention(model, test_loader, device)
        np.savez_compressed(
            args.output_dir / "attention_analysis.npz",
            **attention,
        )
        print(
            "Mean fusion gate: "
            f"{float(attention['fusion_gate'].mean()):.4f}"
        )

    print(
        f"Test | OA {metrics['oa']:.4f} | "
        f"AA {metrics['aa']:.4f} | Kappa {metrics['kappa']:.4f}"
    )
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
