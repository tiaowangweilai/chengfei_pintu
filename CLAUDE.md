# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

超声 C 扫描图像拼接系统，用于无损检测。将机械臂扫查的多幅超声图像/体数据，根据位姿信息拼接为全景 C 扫描图或 3D 点云。

## Pipeline

- **`1_generate_poses.py`** — 根据检测区域尺寸规划扫查路径，生成位姿 JSON (`pose_data.json`)。主扫描带（垂直条带）+ 横向填充带的路径规划。
- **`2_stitch_from_poses.py`** — 2D 图像拼接。读取 `object/*.png` 并根据位姿将切片拼接到画布上，使用 alpha 混合 + 均值填充后处理。
- **`3_stitch_3d_from_poses.py`** — 3D 体数据拼接。读取 `xdt/*.txt` 三维超声体数据，沿步进轴切分，3D 空间位姿放置，输出 PLY 彩色点云 + C 扫投影图。
- **`lidar_all.py`** — 一体化脚本（位姿生成 + 图像拼接），独立版本。
- **`picture.py`** — 无位姿的固定网格拼接（上 N-2 张平分，下 2 张旋转放置），羽化融合。
- **`lidar_dataget.py`** — 工具脚本：将单张 C 扫描图按位姿数等分切片，保存到 `output/`。
- **`lidar_picture.py`** — 简化版 2D 拼接，从 `output/` 读取预切分图片。
- **`analyze_vol.py`** — 体数据快速分析（值分布、阈值统计）。

## Data Flow

```
object/*.png  →  preprocess_and_split()  →  子切片  →  warpAffine()  →  融合  →  2D 拼接图
xdt/*.txt     →  load_volume_from_txt()  →  split_volume()  →  3D 位姿放置  →  .ply 点云 + C 扫投影
```

## Key Parameters

- 位姿: `(x, y, angle)` — 模拟扫查器位置和旋转，带随机噪声
- 切片尺寸: `target_w=360, target_h=80` (px) — 所有切片强制缩放到此尺寸
- 主带间距: `main_strip_gap=350` (mm 单位)
- 切片间距: `pose_gap=70` (mm 单位)

## Dependencies

- Python 3, OpenCV (`cv2`), NumPy

## Commands

1. 生成位姿:
   ```
   python 1_generate_poses.py
   ```
2. 2D 拼接 (需要 `object/` 目录下有图片 + `pose_data.json`):
   ```
   python 2_stitch_from_poses.py
   ```
3. 3D 拼接 (需要 `xdt/` 目录下有 txt 体数据 + `pose_data.json`):
   ```
   python 3_stitch_3d_from_poses.py
   ```

## Code Architecture Notes

- 所有脚本是独立可执行的（没有公共模块导入关系），每个脚本包含完整的预处理 -> 拼接 -> 后处理逻辑。
- `preprocess_and_split()` 在多个脚本中重复实现（`lidar_dataget.py`, `2_stitch_from_poses.py`, `lidar_all.py`），逻辑相同：去底边背景行 → 等分切片 → 补白。
- 位姿偏移补偿在主调度函数中硬编码（主带 y += 140, 横向带 x ±= 170/180）。
- 3D 拼接使用累加器 + 亚像素致密填充（`fill_factor` 方式），2D 拼接使用 alpha 混合或加权均值融合。
- `stitched_3d.npy` / `stitched_3d.ply` / `stitched_3d_defects.ply` 是大文件（GB 级），已在 `.gitignore` 中屏蔽。
