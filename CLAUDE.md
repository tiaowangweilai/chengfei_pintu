# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

超声 C 扫描图像拼接系统，用于无损检测。将机械臂扫查的多幅超声图像/体数据，根据位姿信息拼接为全景 C 扫描图或 3D 点云。

## Pipeline Selection Guide

| 脚本 | 用途 | 输入 |
|------|------|------|
| `1_generate_poses.py` → `2_stitch_from_poses.py` | **主 2D 流水线**：先生成位姿再拼接 | `object/*.png` + `pose_data.json` |
| `1_generate_poses.py` → `3_stitch_3d_from_poses.py` | **主 3D 流水线**：体数据拼接 | `xdt/*.txt` + `pose_data.json` |
| `2_stitch_from_poses.py` → `4_extract_defect_3d.py` | **缺陷深度提取**：将 2D 缺陷映射到 3D 体数据中提取 Z 深度 | `xdt/*.txt` + `pose_data.json` + `perfect_aligned_c_scan_raw.jpg` |
| `analyze_vol.py` | 体数据幅值分布快速分析 | `xdt/*.txt` |
| `tradition3d_BEV.py` | 独立 3D 缺陷分析（含 TGC 补偿+连通域） | 单个体数据文件 |

## Active Pipeline

### 2D 拼接 (`2_stitch_from_poses.py`)
```
object/*.png → preprocess_and_split() → 子切片 → resize(360x80) → warpAffine() → alpha混合累加 → inpaint → 裁剪 → 黑白均值填充 → Canny缺陷检测
```
- 读取 `object/*.png`（已伪彩的 C 扫条带图）
- 去底边白/透明行 → 等分切片 → 补白边 → resize 到统一尺寸
- 按位姿旋转平移 → 加权均值融合 → 形态学闭运算填充间隙 → 裁剪 → 黑白填充
- 后处理独立调用 `detect_defects_canny()`：蓝度图分析 → Canny 边缘检测 → 轮廓筛选 → 矩形标注

### 3D 体数据拼接 (`3_stitch_3d_from_poses.py`)
```
xdt/*.txt → load_volume_from_txt() → split_volume() → stitch_3d_to_points() → voxel_grid_filter() → HTML/PLY/C扫投影
```
- 读取 `xdt/*.txt` 三维超声体数据（步进×扫查×声程），自动缓存 `.npz`（含阈值过滤+压缩）
- 沿步进轴切分，按位姿变换到世界坐标（自适应亚像素填充确保 ~1mm 间距）
- 体素网格降采样（各网格保留最大幅值点）
- 输出：PLY 点云 / Plotly HTML 交互式 3D / C 扫投影图
- C 扫投影：深度方向最大幅值投影 → `cv2.resize` → `warpAffine` 密集投影（不经阈值筛选）

### 2D→3D 缺陷深度提取 (`4_extract_defect_3d.py`)
```
perfect_aligned_c_scan_raw.jpg → Canny检测 → 缺陷bbox列表
  → 反向映射到3D体数据(世界坐标→位姿→子体数据→深度剖面)
  → 每个缺陷的(X,Y,Z中心+Z范围) → 密集3D体素重建 → Plotly HTML + CSV
```
- 对 `perfect_aligned_c_scan_raw.jpg` 执行 Canny 检测（与 2D 拼接的后处理一致）
- 依赖 `_crop_bounds.json`（由 `2_stitch_from_poses.py` 自动保存）精确还原缺陷的世界坐标
- 每个缺陷：中心周围采样网格 → 逆位姿变换找到对应子体数据 → 提取深度方向幅值剖面 → 确定 Z 中位数和 Z 范围
- 使用原始 3D 体数据中超过 `amplitude_threshold` 的体素来确定有效深度
- 当原始 3D 点稀疏时，自动生成密集体素重建缺陷区域（中心高幅值、边缘渐弱）
- 输出：`defect_depth_report.csv`（深度报告）+ `defect_3d_render.html`（Plotly 3D）

### 位姿生成 (`1_generate_poses.py`)
- 根据检测区域尺寸规划扫查路径 → 输出 `pose_data.json`
- 主扫描带（垂直条带）+ 横向填充带路径规划
- 位姿格式：`{x, y, angle}` 坐标 + 角度

### 体数据快速分析 (`analyze_vol.py`)
- 单文件体数据读取，输出幅值分布统计（>0, >1, >3, >5, >8, >10, >15, >20, >30, >50 的百分比）
- 用于确定 `defect_threshold` 的合理取值

## Legacy / Alternative Scripts (非主线)

这些脚本是早期版本或实验性替代方案，在主流水线稳定后不再维护：

| 脚本 | 说明 |
|------|------|
| `lidar_all.py` | 位姿生成+拼接一体化（硬编码切分参数，功能被 `1_generate_poses.py`+`2_stitch_from_poses.py` 替代） |
| `lidar_dataget.py` | 雷达风格位姿生成+比例校正拼接（单条主带，无填充带逻辑） |
| `lidar_picture.py` | 预处理拆分（`lidar_all.py` 的预处理模块提取版） |
| `picture.py` | 不使用位姿的静态全景拼接（固定布局、无旋转补偿） |
| `test_premul.py` | 仅 `import cv2, numpy; print('ok')`，测试脚本 |
| `2_stitch_from_poses_backup1.py` | `2_stitch_from_poses.py` 历史备份 |
| `2_stitch_from_poses_backup2.py` | `2_stitch_from_poses.py` 历史备份 |

## Standalone 3D Defect Analysis (`tradition3d_BEV.py`)

独立的 3D 超声缺陷检测引擎，与主流水线不共享代码。具备完整的物理建模流程：

```
原始数据 → 头信息动态解析(ny,nx,nz) → 张量重塑 → 智能极性翻转(背景255反转) → 归一化 → TGC指数补偿(α=1.4) → Gamma增强(γ=1.5) → 硬阈值去噪 → 盲区保护 → 形态学闭运算(Z轴定向聚类) → 连通域标记 → 物理包围盒过滤(>L*0.4丢弃) → Plotly三联屏(切片滑块+3D体+量化表)
```

关键参数：
- 物理尺寸：`PHYSICAL_LENGTH_X=1100`, `PHYSICAL_WIDTH_Y=2600`, `PHYSICAL_DEPTH_Z=30` (mm)
- TGC 指数：`alpha_tgc=1.4`，越深放得越大
- Gamma：`1.5`，压制底噪凸显主反射
- 硬阈值：`<0.15` 置零，高置信度阈值 `>0.45`
- 过滤：物理尺寸 `<3mm` 或 `>40%` 物理空间 → 丢弃
- 输出：`PhysBEV_Ultimate_Dashboard.html`（Plotly 交互式仪表盘）

用法：修改 `FILE_NAME` 为数据路径，运行 `python tradition3d_BEV.py`。

## Key Parameters

| 参数 | 值 | 说明 |
|------|-----|------|
| `target_w, target_h` | 360, 80 | 切片物理尺寸 (mm) |
| `main_strip_gap` | 350mm | 主带间距 |
| `pose_gap` | 70mm | 切片间距（位姿步长） |
| `y_offset` | 74 | 主带 Y 偏移（原 140，已调至 74 消除与填充带的间隙） |
| `x_offset` | ±170 | 填充带 X 偏移 |
| `defect_threshold` | 2.0 | 缺陷幅值阈值（3D 体素过滤） |
| `voxel_size` | 5.0mm | 体素网格降采样尺寸 |
| `fill_spacing` | =voxel_size | 自适应填充目标间距 |
| `physical_depth_mm` | 20.0 | 声程方向物理厚度 |
| `depth_range` | (20, 310) | 声程有效体素范围 |
| `vol_stride_scan` | True | BEV 风格体数据预降采样（目标 ~100 扫描列） |

## Pose Offsets Logic

所有脚本共用同一位姿偏移补偿逻辑。`split_configs` 最后 2 组为填充带：

```python
for i, num_splits in enumerate(split_configs):
    if i < num_main_strips:          # 主带 → y += 74
        global_poses[pose_idx]['y'] += 74.0
    elif i == total_images - 2:      # 填充带1 → x -= 170
        global_poses[pose_idx]['x'] -= 170.0
    elif i == total_images - 1:      # 填充带2 → x += 170
        global_poses[pose_idx]['x'] += 170.0
```

## Pose Data Format (`pose_data.json`)

```json
{
  "total_images": N,
  "split_configs": [n1, n2, ..., nN],
  "poses": [
    {"x": float, "y": float, "angle": float},
    ...
  ]
}
```

- `total_images`: 总图数 = 主带数 + 2（填充带）
- `split_configs`: 每张图切分的子切片数，长度 = total_images
- `poses`: 所有子切片的位姿，长度 = sum(split_configs)
- `split_configs` 最后 2 组是填充带，偏移补偿不同

## Data Reduction & Caching

- **TXT → NPZ**: `load_volume_from_txt()` 首次加载后自动缓存压缩 `.npz`（`np.savez_compressed`），含 `prethreshold=2.0` 背景置零。重读快 ~200x。
- **Volume stride**: `vol_stride_scan=True` 在 split 前对扫描方向做 strided 降采样（目标 ~100 列）。
- **Adaptive fill**: `fill_spacing=voxel_size` 控制生成点密度，避免产生后续会被过滤掉的中间点。
- **Voxel grid filter**: 拼接后用 `voxel_grid_filter(coords, values, voxel_size=5.0)` 保最大幅值点，136M→~165k。

## Canny Defect Detection

`2_stitch_from_poses.py` 中的 `detect_defects_canny()` 在拼接完成后独立运行（不修改原图）：

```
C扫JPG → 蓝度图(B-RG) → 阈值剔除 → 增益 → Canny → 膨胀 → 外轮廓 → 面积/周长筛选 → MinAreaRect → 分类标注
```

参数（在 `main()` 中调整）：
- `blue_thresh=40`：蓝度阈值，越高只检更深蓝区域
- `gain=1`：蓝度增益
- Canny 阈值：low=5, high=20
- 过滤：面积 <50px 或 >40% 图像面积 → 丢弃

## Dependencies

- Python 3, OpenCV (`cv2`), NumPy
- Plotly（可选，用于 3D HTML 可视化）
- scipy（可选，`tradition3d_BEV.py` 形态学运算用）

## Commands

```bash
# 1. 生成位姿（调整 1_generate_poses.py 中 area_width/area_length 后运行）
python 1_generate_poses.py

# 2. 2D 拼接 + Canny 缺陷检测（需要 object/*.png + pose_data.json）
python 2_stitch_from_poses.py

# 3. 3D 拼接 + HTML 可视化 + C 扫投影（需要 xdt/*.txt + pose_data.json）
python 3_stitch_3d_from_poses.py

# 4. 2D→3D 缺陷深度提取（需要先运行 2_stitch_from_poses.py）
python 4_extract_defect_3d.py

# 5. 体数据快速分析（调整文件路径后运行）
python analyze_vol.py

# 6. 独立 3D 缺陷分析（先修改文件中的 FILE_NAME 路径）
python tradition3d_BEV.py
```

## Output Files

| 文件 | 来源 | 说明 |
|------|------|------|
| `perfect_aligned_c_scan.jpg` | 2D | 拼接+后处理的最终 C 扫图 |
| `perfect_aligned_c_scan_raw.jpg` | 2D | 黑白填充前的原始拼接数据 |
| `perfect_aligned_c_scan_defects.jpg` | 2D | Canny 缺陷标注图 |
| `perfect_aligned_c_scan_enhanced.jpg` | 2D | 蓝度预处理图（调试用） |
| `perfect_aligned_c_scan_edges.jpg` | 2D | Canny 边缘图（调试用） |
| `stitched_3d.html` | 3D | Plotly 交互式 3D 点云 |
| `stitched_c_scan_from_3d_aligned.jpg` | 3D | 3D 拼接的 C 扫投影 |
| `stitched_3d_defects.ply` | 3D | PLY 点云（可关闭） |
| `xdt/*.npz` | 3D | TXT 缓存（压缩+阈值过滤） |
| `pose_data.json` | 位姿生成 | 位姿+切分配置 |
| `PhysBEV_Ultimate_Dashboard.html` | tradition3d_BEV | Plotly 交互式缺陷分析 |
| `defect_depth_report.csv` | 4_extract_defect_3d | 缺陷深度报告（世界XY + Z中心/Z范围 + 置信度） |
| `defect_3d_render.html` | 4_extract_defect_3d | Plotly 3D 缺陷重建可视化 |
| `_crop_bounds.json` | 2_stitch_from_poses | 裁剪边界缓存（供 4_ 使用） |

## Code Architecture Notes

- 所有主流水线脚本是独立可执行的（没有公共模块导入关系），每个脚本包含完整的预处理→拼接→后处理逻辑。
- 2D 和 3D 拼接共用 `pose_data.json` 中的位姿和 `split_configs`，偏移逻辑完全一致。
- `split_volume()` 在 3D 代码中：先剔除全零尾部，有效层不足时用均值层填充（替代零填充），确保每块都有数据。
- 文件数 mismatch 自动处理：当 `object/*.png` 或 `xdt/*.txt` 数量 < `total_images` 时，自动截断 `split_configs` 和 poses 适配。
- 位姿生成在 `1_generate_poses.py` 中，通过 `area_width`/`area_length` 推导主带数和步进路径，`2_stitch_from_poses.py` 和 `3_stitch_3d_from_poses.py` 从 `pose_data.json` 读取。
- `.gitignore` 屏蔽了 `.ply`、`.npy`、`.txt`、`.jpg`、`.png`——注意这也会屏蔽输入数据文件。
- `.claude/settings.local.json` 包含已批准的 Bash 命令权限白名单。
