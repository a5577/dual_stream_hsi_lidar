from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


CLASS_NAMES = (
    "Healthy grass",
    "Stressed grass",
    "Synthetic grass",
    "Trees",
    "Soil",
    "Water",
    "Residential",
    "Commercial",
    "Road",
    "Highway",
    "Railway",
    "Parking Lot 1",
    "Parking Lot 2",
    "Tennis Court",
    "Running Track",
)
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class PreparedData:
    hsi: np.ndarray
    lidar: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    ground_truth: np.ndarray


def _load_single_array(path: Path, expected_key: Optional[str] = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file does not exist: {path}")
    contents = loadmat(path)
    if expected_key is not None and expected_key in contents:
        return contents[expected_key]
    candidates = [
        value
        for key, value in contents.items()
        if not key.startswith("__") and isinstance(value, np.ndarray)
    ]
    if len(candidates) != 1:
        keys = [key for key in contents if not key.startswith("__")]
        raise ValueError(f"Could not choose one array from {path}; keys={keys}")
    return candidates[0]


def _feature_cache_path(
    cache_dir: Path, dataset_dir: Path, components: int, seed: int
) -> Path:
    source_sizes = [
        (dataset_dir / name).stat().st_size
        for name in ("HSI.mat", "LiDAR.mat", "TRLabel.mat")
    ]
    fingerprint = "-".join(str(size) for size in source_sizes)
    return cache_dir / f"features_pca{components}_seed{seed}_{fingerprint}.npz"


def prepare_data(
    dataset_dir: str | Path,
    pca_components: int = 20,
    seed: int = 42,
    cache_dir: Optional[str | Path] = None,
    use_cache: bool = True,
) -> PreparedData:
    """Prepare HSI and LiDAR maps without fitting preprocessing on test labels."""
    dataset_dir = Path(dataset_dir)
    train_labels = _load_single_array(dataset_dir / "TRLabel.mat", "TRLabel").astype(
        np.int64
    )
    test_labels = _load_single_array(dataset_dir / "TSLabel.mat", "TSLabel").astype(
        np.int64
    )
    ground_truth = _load_single_array(dataset_dir / "gt.mat", "gt").astype(np.int64)
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)
    ground_truth = np.squeeze(ground_truth)

    if not (train_labels.shape == test_labels.shape == ground_truth.shape):
        raise ValueError("Training, test, and ground-truth label maps must match")
    if np.any((train_labels > 0) & (test_labels > 0)):
        raise ValueError("Training and test label maps overlap")

    cache_path = None
    if cache_dir is not None and use_cache:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _feature_cache_path(cache_dir, dataset_dir, pca_components, seed)

    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path)
        hsi_features = cached["hsi"]
        lidar_features = cached["lidar"]
    else:
        hsi_raw = _load_single_array(dataset_dir / "HSI.mat", "HSI")
        lidar_raw = _load_single_array(dataset_dir / "LiDAR.mat", "LiDAR")
        lidar_raw = np.squeeze(lidar_raw)
        if hsi_raw.ndim != 3 or hsi_raw.shape[:2] != train_labels.shape:
            raise ValueError(
                f"Unexpected HSI shape {hsi_raw.shape}; label shape is "
                f"{train_labels.shape}"
            )
        if lidar_raw.shape != train_labels.shape:
            raise ValueError(
                f"Unexpected LiDAR shape {lidar_raw.shape}; label shape is "
                f"{train_labels.shape}"
            )
        if pca_components < 1 or pca_components > hsi_raw.shape[2]:
            raise ValueError("pca_components must be within the HSI band count")

        height, width, bands = hsi_raw.shape
        flat_hsi = hsi_raw.reshape(-1, bands).astype(np.float32, copy=False)
        train_mask = train_labels.reshape(-1) > 0
        train_spectra = flat_hsi[train_mask]

        # Fit every data-dependent transform on training pixels only.
        spectral_scaler = StandardScaler()
        scaled_train = spectral_scaler.fit_transform(train_spectra)
        pca = PCA(
            n_components=pca_components,
            svd_solver="randomized",
            random_state=seed,
        )
        pca.fit(scaled_train)

        transformed = np.empty(
            (flat_hsi.shape[0], pca_components), dtype=np.float32
        )
        chunk_size = 65536
        for start in range(0, flat_hsi.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_hsi.shape[0])
            scaled_chunk = spectral_scaler.transform(flat_hsi[start:stop])
            transformed[start:stop] = pca.transform(scaled_chunk).astype(np.float32)

        component_mean = transformed[train_mask].mean(axis=0, keepdims=True)
        component_std = transformed[train_mask].std(axis=0, keepdims=True)
        component_std = np.maximum(component_std, 1e-6)
        hsi_features = ((transformed - component_mean) / component_std).reshape(
            height, width, pca_components
        )
        hsi_features = np.nan_to_num(hsi_features, copy=False).astype(
            np.float32, copy=False
        )

        lidar_features = lidar_raw.astype(np.float32) / 255.0
        lidar_features = np.clip(lidar_features, 0.0, 1.0)[..., None]

        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                hsi=hsi_features,
                lidar=lidar_features,
            )
            cache_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "pca_components": pca_components,
                        "seed": seed,
                        "source": str(dataset_dir.resolve()),
                        "hsi_shape": list(hsi_features.shape),
                        "lidar_shape": list(lidar_features.shape),
                        "explained_variance_ratio_sum": float(
                            pca.explained_variance_ratio_.sum()
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    return PreparedData(
        hsi=hsi_features.astype(np.float32, copy=False),
        lidar=lidar_features.astype(np.float32, copy=False),
        train_labels=train_labels,
        test_labels=test_labels,
        ground_truth=ground_truth,
    )


class HoustonPatchDataset(Dataset):
    """Extract aligned spatial neighborhoods from the two co-registered maps."""

    def __init__(
        self,
        hsi: np.ndarray,
        lidar: np.ndarray,
        coordinates: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
        augment: bool = False,
    ) -> None:
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd number")
        self.hsi = hsi
        self.lidar = lidar
        self.coordinates = coordinates.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.patch_size = patch_size
        self.augment = augment
        half = patch_size // 2
        self.padded_hsi = np.pad(
            hsi,
            ((half, half), (half, half), (0, 0)),
            mode="reflect",
        )
        self.padded_lidar = np.pad(
            lidar,
            ((half, half), (half, half), (0, 0)),
            mode="reflect",
        )

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, column = self.coordinates[index]
        stop = self.patch_size
        hsi_patch = self.padded_hsi[row : row + stop, column : column + stop]
        lidar_patch = self.padded_lidar[row : row + stop, column : column + stop]

        if self.augment:
            rotations = random.randint(0, 3)
            if rotations:
                hsi_patch = np.rot90(hsi_patch, rotations, axes=(0, 1))
                lidar_patch = np.rot90(lidar_patch, rotations, axes=(0, 1))
            if random.random() < 0.5:
                hsi_patch = np.flip(hsi_patch, axis=1)
                lidar_patch = np.flip(lidar_patch, axis=1)
            if random.random() < 0.5:
                hsi_patch = np.flip(hsi_patch, axis=0)
                lidar_patch = np.flip(lidar_patch, axis=0)

        hsi_tensor = torch.from_numpy(
            np.ascontiguousarray(hsi_patch.transpose(2, 0, 1))
        ).float()
        lidar_tensor = torch.from_numpy(
            np.ascontiguousarray(lidar_patch.transpose(2, 0, 1))
        ).float()
        label = torch.tensor(self.labels[index] - 1, dtype=torch.long)
        return hsi_tensor, lidar_tensor, label

