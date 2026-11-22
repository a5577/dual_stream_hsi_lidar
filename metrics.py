from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .data import CLASS_NAMES, NUM_CLASSES


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_names: Iterable[str] = CLASS_NAMES,
) -> Dict[str, object]:
    labels = np.arange(1, NUM_CLASSES + 1)
    matrix = confusion_matrix(truth, prediction, labels=labels)
    supports = matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(matrix),
        supports,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=supports > 0,
    )
    return {
        "oa": float(np.trace(matrix) / matrix.sum()),
        "aa": float(per_class.mean()),
        "kappa": float(cohen_kappa_score(truth, prediction, labels=labels)),
        "support": supports.tolist(),
        "per_class_accuracy": per_class.tolist(),
        "per_class": [
            {
                "class_id": class_id,
                "class_name": class_name,
                "accuracy": float(per_class[class_id - 1]),
                "support": int(supports[class_id - 1]),
                "correct": int(matrix[class_id - 1, class_id - 1]),
            }
            for class_id, class_name in enumerate(class_names, start=1)
        ],
        "confusion_matrix": matrix.tolist(),
    }

