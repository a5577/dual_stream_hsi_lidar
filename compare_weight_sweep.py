from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compare_ablations import load_run
from .data import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an HSI/LiDAR concatenation weight sweep."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("dual_stream_hsi_lidar/runs/weight_sweep"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dual_stream_hsi_lidar/runs/weight_sweep_comparison"),
    )
    parser.add_argument(
        "--baseline-run",
        default="hsi100_lidar100",
        help="Unweighted 1:1 concatenation run.",
    )
    return parser.parse_args()


def discover_runs(runs_root: Path) -> Dict[str, Dict[str, object]]:
    runs: Dict[str, Dict[str, object]] = {}
    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = payload["config"]
        record = load_run(run_dir)
        record.update(
            {
                "hsi_weight": float(config.get("hsi_weight", 1.0)),
                "lidar_weight": float(config.get("lidar_weight", 1.0)),
                "best_epoch": int(payload["best_epoch"]),
                "epochs_completed": int(payload["epochs_completed"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
            }
        )
        total_weight = record["hsi_weight"] + record["lidar_weight"]
        record["hsi_share"] = float(record["hsi_weight"] / total_weight)
        runs[run_dir.name] = record
    if not runs:
        raise FileNotFoundError(f"No completed weight runs found under {runs_root}")
    return runs


def write_csv(
    runs: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    fieldnames = [
        "run",
        "hsi_weight",
        "lidar_weight",
        "hsi_share",
        "oa",
        "aa",
        "kappa",
        "macro_precision",
        "macro_f1",
        "best_epoch",
        "epochs_completed",
        "elapsed_seconds",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, record in runs.items():
            writer.writerow(
                {
                    "run": name,
                    "hsi_weight": record["hsi_weight"],
                    "lidar_weight": record["lidar_weight"],
                    "hsi_share": record["hsi_share"],
                    "oa": record["overall"]["oa"],
                    "aa": record["overall"]["aa"],
                    "kappa": record["overall"]["kappa"],
                    "macro_precision": record["overall"]["macro_precision"],
                    "macro_f1": record["overall"]["macro_f1"],
                    "best_epoch": record["best_epoch"],
                    "epochs_completed": record["epochs_completed"],
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )


def normalized_records(
    runs: Dict[str, Dict[str, object]],
) -> List[tuple[str, Dict[str, object]]]:
    records = [
        (name, record)
        for name, record in runs.items()
        if abs(record["hsi_weight"] + record["lidar_weight"] - 1.0) < 1e-8
    ]
    return sorted(records, key=lambda item: item[1]["hsi_share"])


def plot_weight_metrics(
    runs: Dict[str, Dict[str, object]],
    baseline_name: str,
    output_path: Path,
) -> None:
    records = normalized_records(runs)
    shares = np.asarray([record["hsi_share"] for _, record in records])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    specs = (
        ("OA (%)", "oa", lambda value: value * 100.0),
        ("AA (%)", "aa", lambda value: value * 100.0),
        ("Kappa", "kappa", lambda value: value),
    )
    for axis, (title, key, transform) in zip(axes, specs):
        values = np.asarray(
            [transform(record["overall"][key]) for _, record in records]
        )
        axis.plot(
            shares,
            values,
            marker="o",
            linewidth=2.2,
            color="tab:blue",
            label="Normalized weights",
        )
        for share, value in zip(shares, values):
            formatted = f"{value:.2f}" if key != "kappa" else f"{value:.4f}"
            axis.annotate(
                formatted,
                (share, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        if baseline_name in runs:
            baseline_value = transform(runs[baseline_name]["overall"][key])
            axis.axhline(
                baseline_value,
                color="tab:orange",
                linestyle="--",
                linewidth=1.6,
                label="Unweighted 1:1 baseline",
            )
        axis.set_title(title)
        axis.set_xlabel("HSI share  w_hsi / (w_hsi + w_lidar)")
        axis.set_xticks(shares)
        axis.set_xticklabels([f"{share:.3f}".rstrip("0").rstrip(".") for share in shares])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Effect of HSI/LiDAR concatenation weights")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_per_class_heatmap(
    runs: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    names = list(runs)
    labels = [
        f"{runs[name]['hsi_weight']:g}:{runs[name]['lidar_weight']:g}"
        for name in names
    ]
    values = np.asarray(
        [np.asarray(runs[name]["recall"]) * 100.0 for name in names]
    )
    figure, axis = plt.subplots(figsize=(16, max(4.8, 0.65 * len(names) + 2.0)))
    image = axis.imshow(values, aspect="auto", cmap="YlGnBu", vmin=50.0, vmax=100.0)
    axis.set_xticks(np.arange(len(CLASS_NAMES)))
    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(names)))
    axis.set_yticklabels(labels)
    axis.set_ylabel("HSI:LiDAR weight")
    axis.set_title("Per-class test accuracy (%)")
    threshold = (50.0 + 100.0) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] < threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_report(
    runs: Dict[str, Dict[str, object]],
    baseline_name: str,
    output_path: Path,
) -> None:
    best_name = max(runs, key=lambda name: runs[name]["overall"]["oa"])
    best = runs[best_name]
    lines = [
        "# HSI/LiDAR 拼接权重扫描",
        "",
        "权重作用在各自编码器输出的 128 维特征上，随后拼接并送入原分类头。"
        "所有实验使用相同的数据划分、PCA 维数、patch、优化器、训练轮数和随机种子。",
        "",
        "## 总体结果",
        "",
        "| 实验 | HSI:LiDAR | OA | AA | Kappa | 宏平均 F1 | 最佳 epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, record in runs.items():
        values = record["overall"]
        lines.append(
            f"| {name} | {record['hsi_weight']:g}:{record['lidar_weight']:g} | "
            f"{values['oa'] * 100.0:.2f}% | {values['aa'] * 100.0:.2f}% | "
            f"{values['kappa']:.4f} | {values['macro_f1'] * 100.0:.2f}% | "
            f"{record['best_epoch']} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 测试集 OA 最高的是 `{best_name}`，"
            f"权重为 HSI:LiDAR = {best['hsi_weight']:g}:{best['lidar_weight']:g}，"
            f"OA={best['overall']['oa'] * 100.0:.2f}%，"
            f"AA={best['overall']['aa'] * 100.0:.2f}%，"
            f"Kappa={best['overall']['kappa']:.4f}。",
        ]
    )
    normalized = normalized_records(runs)
    best_normalized_name, best_normalized = max(
        normalized,
        key=lambda item: item[1]["overall"]["oa"],
    )
    lines.append(
        f"- 归一化权重中测试集 OA 最高的是 `{best_normalized_name}`，"
        f"权重为 HSI:LiDAR = {best_normalized['hsi_weight']:g}:"
        f"{best_normalized['lidar_weight']:g}，"
        f"OA={best_normalized['overall']['oa'] * 100.0:.2f}%，"
        f"AA={best_normalized['overall']['aa'] * 100.0:.2f}%，"
        f"Kappa={best_normalized['overall']['kappa']:.4f}。"
    )
    if baseline_name in runs:
        baseline = runs[baseline_name]["overall"]
        delta_oa = (best["overall"]["oa"] - baseline["oa"]) * 100.0
        delta_aa = (best["overall"]["aa"] - baseline["aa"]) * 100.0
        delta_kappa = best["overall"]["kappa"] - baseline["kappa"]
        lines.append(
            f"- 相对未加权 `1:1` 基线，最佳组 OA 变化 "
            f"{delta_oa:+.2f} 个百分点，AA 变化 {delta_aa:+.2f} 个百分点，"
            f"Kappa 变化 {delta_kappa:+.4f}。"
        )
        normalized_delta_oa = (
            best_normalized["overall"]["oa"] - baseline["oa"]
        ) * 100.0
        normalized_delta_aa = (
            best_normalized["overall"]["aa"] - baseline["aa"]
        ) * 100.0
        normalized_delta_kappa = (
            best_normalized["overall"]["kappa"] - baseline["kappa"]
        )
        lines.append(
            f"- 最佳归一化组相对 `1:1` 基线，OA 变化 "
            f"{normalized_delta_oa:+.2f} 个百分点，AA 变化 "
            f"{normalized_delta_aa:+.2f} 个百分点，Kappa 变化 "
            f"{normalized_delta_kappa:+.4f}。"
        )
    lines.extend(
        [
            "- 当前是单随机种子扫描，权重间小于约 1 个百分点的差异"
            "仍需多随机种子复核后再下结论。",
            "- 拼接特征随后会进入线性层，因此单独缩放某一分支原则上可被"
            "该线性层权重等价吸收；本扫描也会受到优化轨迹和 weight decay 的影响。"
            "若要让权重成为严格的固定贡献约束，需要改用加权求和、"
            "分支独立分类后的决策融合或带权重正则的融合头。",
            "",
            "## 文件",
            "",
            "- `summary.csv` / `summary.json`：完整数值结果。",
            "- `weight_metrics.png`：OA、AA、Kappa 随 HSI 权重占比的变化。",
            "- `per_class_weight_heatmap.png`：各权重组合的逐类精度。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.runs_root)
    if args.baseline_run not in runs:
        raise FileNotFoundError(
            f"Baseline run {args.baseline_run!r} was not found under {args.runs_root}"
        )

    write_csv(runs, args.output_dir / "summary.csv")
    summary = {
        "baseline_run": args.baseline_run,
        "best_run": max(runs, key=lambda name: runs[name]["overall"]["oa"]),
        "runs": {
            name: {
                "hsi_weight": record["hsi_weight"],
                "lidar_weight": record["lidar_weight"],
                "hsi_share": record["hsi_share"],
                "overall": record["overall"],
                "per_class_accuracy": np.asarray(record["recall"]).tolist(),
                "best_epoch": record["best_epoch"],
                "epochs_completed": record["epochs_completed"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
            for name, record in runs.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_weight_metrics(
        runs,
        args.baseline_run,
        args.output_dir / "weight_metrics.png",
    )
    plot_per_class_heatmap(
        runs,
        args.output_dir / "per_class_weight_heatmap.png",
    )
    write_report(
        runs,
        args.baseline_run,
        args.output_dir / "report.md",
    )
    print(f"Weight comparison saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
