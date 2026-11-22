# HSI-LiDAR 融合消融对比

主实验：`fusion`

## 总体指标

| 实验 | OA | AA | Kappa | Macro Precision | Macro F1 | 错误率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fusion | 88.17% | 89.64% | 0.8717 | 87.95% | 87.85% | 11.83% |
| hsi_only | 83.62% | 85.68% | 0.8221 | 86.68% | 85.43% | 16.38% |
| lidar_only | 71.70% | 72.48% | 0.6930 | 71.51% | 71.12% | 28.30% |

## 主实验提升

相对 `hsi_only`：

- OA：+4.55 pp
- AA：+3.96 pp
- Kappa：+0.0495
- Macro Precision：+1.28 pp
- Macro F1：+2.42 pp
- 错误率变化：-4.55 pp

相对 `lidar_only`：

- OA：+16.47 pp
- AA：+17.16 pp
- Kappa：+0.1787
- Macro Precision：+16.45 pp
- Macro F1：+16.73 pp
- 错误率变化：-16.47 pp

## 逐类提升

| 类别 | Fusion | 最佳单模态 | 相对最佳单模态 |
| --- | ---: | ---: | ---: |
| Healthy grass | 78.35% | 76.64% | +1.71 pp |
| Stressed grass | 87.22% | 85.15% | +2.07 pp |
| Synthetic grass | 97.43% | 98.42% | -0.99 pp |
| Trees | 87.03% | 73.96% | +13.07 pp |
| Soil | 99.62% | 99.62% | +0.00 pp |
| Water | 95.80% | 88.81% | +6.99 pp |
| Residential | 79.38% | 85.26% | -5.88 pp |
| Commercial | 89.36% | 92.88% | -3.51 pp |
| Road | 89.71% | 70.44% | +19.26 pp |
| Highway | 66.70% | 70.17% | -3.47 pp |
| Railway | 100.00% | 96.20% | +3.80 pp |
| Parking Lot 1 | 92.60% | 92.51% | +0.10 pp |
| Parking Lot 2 | 82.46% | 82.81% | -0.35 pp |
| Tennis Court | 100.00% | 100.00% | +0.00 pp |
| Running Track | 98.94% | 95.56% | +3.38 pp |

## 提升最大的类别

- Road：+19.26 pp
- Trees：+13.07 pp
- Water：+6.99 pp
- Railway：+3.80 pp
- Running Track：+3.38 pp

## 相对最佳单模态仍下降的类别

- Parking Lot 2：-0.35 pp
- Synthetic grass：-0.99 pp
- Highway：-3.47 pp
- Commercial：-3.51 pp
- Residential：-5.88 pp

## 图片

- `overall_metrics_comparison.png`：总体指标对比。
- `per_class_accuracy_comparison.png`：逐类精度对比。
- `selected_classes_horizontal_vs_hsi_only.png`：主实验与 HSI-only 的横向分组条形图。
- `selected_classes_horizontal_vs_lidar_only.png`：主实验与 LiDAR-only 的横向分组条形图。
- `all_classes_horizontal_vs_hsi_only.png`：全部 15 类与 HSI-only 的横向分组条形图。
- `all_classes_horizontal_vs_lidar_only.png`：全部 15 类与 LiDAR-only 的横向分组条形图。
- `fusion_gain_heatmap.png`：相对两个消融及最佳单模态的逐类变化。
- `confusion_difference_vs_*.png`：融合模型减少或增加的混淆数量。
- `training_curves_comparison.png`：验证损失和验证 OA 曲线。
