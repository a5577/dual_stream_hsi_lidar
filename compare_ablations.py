from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASS_NAMES


CLASS_NAME_ZH = {
    "Healthy grass": "健康草地",
    "Stressed grass": "受胁迫草地",
    "Synthetic grass": "人工草地",
    "Trees": "树木",
    "Soil": "土壤",
    "Water": "水体",
    "Residential": "住宅",
    "Commercial": "商业区",
    "Road": "道路",
    "Highway": "高速公路",
    "Railway": "铁路",
    "Parking Lot 1": "停车场 1",
    "Parking Lot 2": "停车场 2",
    "Tennis Court": "网球场",
    "Running Track": "跑道",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures and a report for fusion ablations."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--main-run", default="fusion")
    parser.add_argument(
        "--ablation-runs",
        nargs="+",
        default=("hsi_only", "lidar_only"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/comparison"),
    )
    return parser.parse_args()


def load_run(run_dir: Path) -> Dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_metrics = payload["test_metrics"]
    confusion = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
    if confusion.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"Unexpected confusion-matrix shape in {metrics_path}")
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        np.diag(confusion),
        predicted,
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "path": str(run_dir),
        "metrics": test_metrics,
        "confusion": confusion,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "support": support,
        "overall": {
            "oa": float(test_metrics["oa"]),
            "aa": float(test_metrics["aa"]),
            "kappa": float(test_metrics["kappa"]),
            "macro_precision": float(precision.mean()),
            "macro_f1": float(f1.mean()),
            "error_rate": float(1.0 - test_metrics["oa"]),
            "loss": float(test_metrics["loss"]),
        },
        "history": (
            json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            if (run_dir / "history.json").is_file()
            else {}
        ),
    }


def plot_overall_metrics(
    runs: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    labels = list(runs)
    metric_specs = (
        ("OA", "oa", 100.0),
        ("AA", "aa", 100.0),
        ("Kappa", "kappa", 1.0),
        ("Macro precision", "macro_precision", 100.0),
        ("Macro F1", "macro_f1", 100.0),
    )
    figure, axes = plt.subplots(1, len(metric_specs), figsize=(18, 4.8))
    colors = ["tab:green", "tab:blue", "tab:orange"]
    for axis, (title, key, scale) in zip(axes, metric_specs):
        values = [
            float(runs[label]["overall"][key]) * scale
            for label in labels
        ]
        bars = axis.bar(labels, values, color=colors[: len(labels)])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}" if scale == 100.0 else f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if scale == 100.0:
            axis.set_ylim(max(0.0, min(values) - 8.0), 102.0)
        else:
            axis.set_ylim(max(0.0, min(values) - 0.08), 1.02)
    figure.suptitle("HSI-LiDAR fusion ablation comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_per_class_accuracy(
    runs: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    labels = list(runs)
    positions = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(labels)
    colors = ["tab:green", "tab:blue", "tab:orange"]
    figure, axis = plt.subplots(figsize=(17, 7.5))
    for index, label in enumerate(labels):
        values = np.asarray(runs[label]["recall"]) * 100.0
        axis.bar(
            positions + (index - (len(labels) - 1) / 2) * width,
            values,
            width=width,
            label=label,
            color=colors[index % len(colors)],
        )
    axis.set_xticks(positions, labels=CLASS_NAMES, rotation=35, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Recall / accuracy (%)")
    axis.set_title("Per-class test accuracy by input modality")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_horizontal_bars(
    main: Dict[str, object],
    ablation: Dict[str, object],
    ablation_label: str,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    selected_indices = [
        CLASS_NAMES.index(class_name)
        for class_name in class_names
    ]
    main_values = np.asarray(main["recall"])[selected_indices] * 100.0
    ablation_values = np.asarray(ablation["recall"])[selected_indices] * 100.0

    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    positions = np.arange(len(class_names), dtype=np.float64)
    bar_height = 0.36
    figure_height = max(6.5, 0.72 * len(class_names) + 2.1)
    figure, axis = plt.subplots(figsize=(12.5, figure_height))
    main_color = "#E67E22"
    ablation_color = "#20A6A6"
    main_bars = axis.barh(
        positions - bar_height / 2,
        main_values,
        height=bar_height,
        color=main_color,
        label="HSI + LiDAR",
    )
    ablation_bars = axis.barh(
        positions + bar_height / 2,
        ablation_values,
        height=bar_height,
        color=ablation_color,
        label=ablation_label,
    )
    axis.set_yticks(
        positions,
        labels=[CLASS_NAME_ZH[name] for name in class_names],
        fontsize=14 if len(class_names) <= 8 else 12,
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 120)
    axis.set_xticks(np.arange(0, 121, 20))
    axis.tick_params(axis="x", labelsize=14)
    axis.grid(axis="x", color="#DDDDDD", linewidth=1.0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#D0D0D0")

    for bars in (main_bars, ablation_bars):
        for bar in bars:
            value = float(bar.get_width())
            axis.text(
                value + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=13,
                color="#333333",
            )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=15,
        handlelength=0.8,
        handleheight=0.8,
        columnspacing=1.0,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    plt.rcParams["font.family"] = ["DejaVu Sans"]


def plot_fusion_gain_heatmap(
    main: Dict[str, object],
    ablations: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    columns = []
    column_labels = []
    main_recall = np.asarray(main["recall"])
    for name, run in ablations.items():
        columns.append((main_recall - np.asarray(run["recall"])) * 100.0)
        column_labels.append(f"Fusion - {name}")
    best_unimodal = np.max(
        [np.asarray(run["recall"]) for run in ablations.values()],
        axis=0,
    )
    columns.append((main_recall - best_unimodal) * 100.0)
    column_labels.append("Fusion - best unimodal")
    matrix = np.stack(columns, axis=1)
    limit = max(5.0, float(np.abs(matrix).max()))
    figure, axis = plt.subplots(figsize=(8.5, 11))
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    axis.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    axis.set_title("Per-class accuracy change (percentage points)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                f"{matrix[row, col]:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(matrix[row, col]) > 0.55 * limit else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_confusion_difference(
    main_confusion: np.ndarray,
    ablation_confusion: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    difference = main_confusion - ablation_confusion
    limit = max(1.0, float(np.abs(difference).max()))
    figure, axis = plt.subplots(figsize=(10.5, 9))
    image = axis.imshow(
        difference,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ticks = np.arange(len(CLASS_NAMES))
    axis.set_xticks(ticks, labels=range(1, len(CLASS_NAMES) + 1))
    axis.set_yticks(ticks, labels=range(1, len(CLASS_NAMES) + 1))
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    for row in range(difference.shape[0]):
        for col in range(difference.shape[1]):
            value = int(difference[row, col])
            if value:
                axis.text(
                    col,
                    row,
                    f"{value:+d}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_training_curves(
    runs: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = ["tab:green", "tab:blue", "tab:orange"]
    for index, (name, run) in enumerate(runs.items()):
        history = run["history"]
        if not history:
            continue
        color = colors[index % len(colors)]
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label=name, color=color)
        axes[1].plot(
            epochs,
            np.asarray(history["val_oa"]) * 100.0,
            label=name,
            color=color,
        )
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Validation OA")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("OA (%)")
    axes[1].set_ylim(0, 102)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def percentage_points(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def build_summary(
    runs: Dict[str, Dict[str, object]],
    main_name: str,
    output_dir: Path,
) -> Dict[str, object]:
    main = runs[main_name]
    ablation_names = [name for name in runs if name != main_name]
    main_recall = np.asarray(main["recall"])
    best_unimodal = np.max(
        [np.asarray(runs[name]["recall"]) for name in ablation_names],
        axis=0,
    )
    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        record = {
            "class_id": index + 1,
            "class_name": class_name,
            "fusion_accuracy": float(main_recall[index]),
            "best_unimodal_accuracy": float(best_unimodal[index]),
            "gain_vs_best_unimodal": float(
                main_recall[index] - best_unimodal[index]
            ),
        }
        for name in ablation_names:
            record[f"gain_vs_{name}"] = float(
                main_recall[index] - runs[name]["recall"][index]
            )
        per_class.append(record)

    ranked = sorted(
        per_class,
        key=lambda item: item["gain_vs_best_unimodal"],
        reverse=True,
    )
    comparisons = {}
    for name in ablation_names:
        comparisons[name] = {
            key: float(main["overall"][key] - runs[name]["overall"][key])
            for key in (
                "oa",
                "aa",
                "kappa",
                "macro_precision",
                "macro_f1",
                "error_rate",
            )
        }
    return {
        "runs": {
            name: {
                "path": run["path"],
                "overall": run["overall"],
                "per_class_accuracy": np.asarray(run["recall"]).tolist(),
            }
            for name, run in runs.items()
        },
        "main_vs_ablations": comparisons,
        "per_class": per_class,
        "largest_gains_vs_best_unimodal": ranked[:5],
        "largest_losses_vs_best_unimodal": list(reversed(ranked[-5:])),
    }


def write_markdown(
    summary: Dict[str, object],
    main_name: str,
    output_dir: Path,
) -> None:
    runs = summary["runs"]
    comparisons = summary["main_vs_ablations"]
    lines = [
        "# HSI-LiDAR 融合消融对比",
        "",
        f"主实验：`{main_name}`",
        "",
        "## 总体指标",
        "",
        "| 实验 | OA | AA | Kappa | Macro Precision | Macro F1 | 错误率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, record in runs.items():
        values = record["overall"]
        lines.append(
            f"| {name} | {percent(values['oa'])} | "
            f"{percent(values['aa'])} | {values['kappa']:.4f} | "
            f"{percent(values['macro_precision'])} | "
            f"{percent(values['macro_f1'])} | "
            f"{percent(values['error_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 主实验提升",
            "",
        ]
    )
    for ablation, delta in comparisons.items():
        lines.extend(
            [
                f"相对 `{ablation}`：",
                "",
                f"- OA：{percentage_points(delta['oa'])}",
                f"- AA：{percentage_points(delta['aa'])}",
                f"- Kappa：{delta['kappa']:+.4f}",
                f"- Macro Precision："
                f"{percentage_points(delta['macro_precision'])}",
                f"- Macro F1：{percentage_points(delta['macro_f1'])}",
                f"- 错误率变化："
                f"{percentage_points(delta['error_rate'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## 逐类提升",
            "",
            "| 类别 | Fusion | 最佳单模态 | 相对最佳单模态 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for record in sorted(
        summary["per_class"],
        key=lambda item: item["class_id"],
    ):
        lines.append(
            f"| {record['class_name']} | "
            f"{percent(record['fusion_accuracy'])} | "
            f"{percent(record['best_unimodal_accuracy'])} | "
            f"{percentage_points(record['gain_vs_best_unimodal'])} |"
        )

    lines.extend(["", "## 提升最大的类别", ""])
    for record in summary["largest_gains_vs_best_unimodal"]:
        lines.append(
            f"- {record['class_name']}："
            f"{percentage_points(record['gain_vs_best_unimodal'])}"
        )
    lines.extend(["", "## 相对最佳单模态仍下降的类别", ""])
    for record in reversed(summary["largest_losses_vs_best_unimodal"]):
        if record["gain_vs_best_unimodal"] < 0:
            lines.append(
                f"- {record['class_name']}："
                f"{percentage_points(record['gain_vs_best_unimodal'])}"
            )

    lines.extend(
        [
            "",
            "## 图片",
            "",
            "- `overall_metrics_comparison.png`：总体指标对比。",
            "- `per_class_accuracy_comparison.png`：逐类精度对比。",
            "- `selected_classes_horizontal_vs_hsi_only.png`："
            "主实验与 HSI-only 的横向分组条形图。",
            "- `selected_classes_horizontal_vs_lidar_only.png`："
            "主实验与 LiDAR-only 的横向分组条形图。",
            "- `all_classes_horizontal_vs_hsi_only.png`："
            "全部 15 类与 HSI-only 的横向分组条形图。",
            "- `all_classes_horizontal_vs_lidar_only.png`："
            "全部 15 类与 LiDAR-only 的横向分组条形图。",
            "- `fusion_gain_heatmap.png`：相对两个消融及最佳单模态的逐类变化。",
            "- `confusion_difference_vs_*.png`：融合模型减少或增加的混淆数量。",
            "- `training_curves_comparison.png`：验证损失和验证 OA 曲线。",
            "",
        ]
    )
    (output_dir / "comparison_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_names = [args.main_run, *args.ablation_runs]
    runs = {
        name: load_run(args.runs_root / name)
        for name in run_names
    }

    plot_overall_metrics(
        runs,
        args.output_dir / "overall_metrics_comparison.png",
    )
    plot_per_class_accuracy(
        runs,
        args.output_dir / "per_class_accuracy_comparison.png",
    )
    horizontal_labels = {
        "hsi_only": "HSI-only",
        "lidar_only": "LiDAR-only",
    }
    selected_names = [
        "Trees",
        "Road",
        "Residential",
        "Water",
        "Parking Lot 2",
        "Highway",
    ]
    for ablation_name in args.ablation_runs:
        plot_horizontal_bars(
            runs[args.main_run],
            runs[ablation_name],
            horizontal_labels.get(ablation_name, ablation_name),
            selected_names,
            args.output_dir
            / f"selected_classes_horizontal_vs_{ablation_name}.png",
        )
        plot_horizontal_bars(
            runs[args.main_run],
            runs[ablation_name],
            horizontal_labels.get(ablation_name, ablation_name),
            list(CLASS_NAMES),
            args.output_dir
            / f"all_classes_horizontal_vs_{ablation_name}.png",
        )
    plot_fusion_gain_heatmap(
        runs[args.main_run],
        {
            name: runs[name]
            for name in args.ablation_runs
        },
        args.output_dir / "fusion_gain_heatmap.png",
    )
    plot_training_curves(
        runs,
        args.output_dir / "training_curves_comparison.png",
    )
    for name in args.ablation_runs:
        plot_confusion_difference(
            runs[args.main_run]["confusion"],
            runs[name]["confusion"],
            f"Fusion minus {name}: prediction-count difference",
            args.output_dir / f"confusion_difference_vs_{name}.png",
        )

    summary = build_summary(runs, args.main_run, args.output_dir)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, args.main_run, args.output_dir)
    print(f"Comparison report saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
