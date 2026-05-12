import numpy as np
import math
import os
import re
import json
import struct
import cv2
import base64

# ==========================================
# Plotly 可选导入（用于 HTML 3D 可视化，参照 tradition3d_BEV.py）
# ==========================================
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==========================================
# 1. 解析 txt 体数据（BEV 风格：一次性读取→reshape，比逐行解析快）
# ==========================================
def load_volume_from_txt(filepath, prethreshold=2.0):
    """
    从 txt 文件读取三维超声体数据。
    自动缓存压缩后的 .npz（含阈值过滤，远小于原始 npy/txt）。
    prethreshold: 缓存时低于此值的体素置零（默认 2.0，与 defect_threshold 一致）。
    """
    npz_path = filepath.rsplit('.', 1)[0] + '.npz'
    if os.path.exists(npz_path):
        return np.load(npz_path)['vol'].astype(np.float32)

    # 旧版 .npy 缓存兼容
    npy_path = filepath.rsplit('.', 1)[0] + '.npy'
    if os.path.exists(npy_path):
        vol = np.load(npy_path).astype(np.float32)
        return vol

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # 分离头信息与数据体
    parts = re.split(r'={5,}', content)
    header_str = parts[0] if len(parts) >= 2 else ""
    data_str  = parts[1] if len(parts) >= 2 else content

    nums = re.findall(r'\d+', header_str)
    n_step, n_scan, n_depth = int(nums[0]), int(nums[1]), int(nums[2])
    total = n_step * n_scan * n_depth

    # 剔除分隔线，一次性解析所有数值
    cleaned = re.sub(r'[-=]{3,}.*', '', data_str)
    normalized = cleaned.replace('\t', ' ').replace('\r\n', ' ').replace('\n', ' ')
    all_vals = np.fromstring(normalized, dtype=np.int32, sep=' ')
    if len(all_vals) < total // 2:
        tokens = [int(x) for x in cleaned.split() if x.strip()]
        all_vals = np.array(tokens, dtype=np.int32)
    if len(all_vals) < total:
        padded = np.zeros(total, dtype=np.int32)
        padded[:len(all_vals)] = all_vals
        all_vals = padded
    elif len(all_vals) > total:
        all_vals = all_vals[:total]

    volume = all_vals.reshape((n_step, n_scan, n_depth)).astype(np.float32)

    # 阈值过滤：背景置零，压缩时体积大幅减小
    if prethreshold > 0:
        volume[volume <= prethreshold] = 0.0

    # 写入压缩缓存 (.npz 比 .npy 小数倍)
    np.savez_compressed(npz_path, vol=volume)
    return volume

# ==========================================
# 2. 沿步进轴切分：有效数据不足时自动复制填充（避免空块）
# ==========================================
def split_volume(volume, num_poses):
    """
    先剔除尾部的传感器空跑/补零数据，再等分。
    若有效步进层数 < num_poses，自动复制已有数据填满，确保每块都有内容。
    """
    # 1. 计算每一层的绝对值总和，识别出"全零"的空扫描区域
    layer_sums = np.sum(np.abs(volume), axis=(1, 2))

    # 2. 寻找真实的有效数据边界 (使用极小的阈值抵抗底噪)
    valid_indices = np.where(layer_sums > 1e-3)[0]

    if len(valid_indices) > 0:
        crop_bottom_idx = valid_indices[-1] + 1
    else:
        crop_bottom_idx = volume.shape[0]

    cropped_vol = volume[:crop_bottom_idx, ...]
    new_height = cropped_vol.shape[0]

    if new_height == 0:
        return [np.zeros((1, volume.shape[1], volume.shape[2]), dtype=volume.dtype) for _ in range(num_poses)]

    # 3. 每份高度 + 有效层均值填充不足部分（比 tile 复制更平滑，同样快）
    sub_height = math.ceil(new_height / num_poses)
    target = sub_height * num_poses
    if new_height < target:
        mean_layer = cropped_vol.mean(axis=0, keepdims=True)  # (1, w, d)
        n_pad = target - new_height
        padding = np.repeat(mean_layer, n_pad, axis=0)
        padded_vol = np.vstack((cropped_vol, padding))
    else:
        padded_vol = cropped_vol

    # 4. 执行物理等分
    sub_volumes = []
    for i in range(num_poses):
        start_y = i * sub_height
        end_y = start_y + sub_height
        sub_volumes.append(padded_vol[start_y:end_y, ...])

    return sub_volumes

# ==========================================
# 3. 体素网格降采样 (降数据量核心，参照 tradition3d_BEV.py)
# ==========================================
def voxel_grid_filter(coords, values, voxel_size=5.0):
    """
    体素网格降采样 —— 对标 tradition3d_BEV.py 的 strided 降采样策略。
    每个体素网格内仅保留最大幅值的点，大幅减少数据量同时保持关键缺陷信息。
    """
    if len(coords) == 0 or voxel_size <= 0:
        return coords, values

    min_c = coords.min(axis=0)
    indices = np.floor((coords - min_c) / voxel_size).astype(np.int64)

    # 编码唯一体素 key
    range_yx = indices[:, 1].max() + 1
    range_z  = indices[:, 2].max() + 1
    keys = indices[:, 0] * (range_yx * range_z) + indices[:, 1] * range_z + indices[:, 2]

    sort_idx = np.argsort(keys)
    keys_sorted = keys[sort_idx]
    coords_sorted = coords[sort_idx]
    values_sorted = values[sort_idx]

    # unique 直接返回每组边界和大小，避免 Python while 逐元素扫描
    unique_keys, start_idx, counts = np.unique(keys_sorted,
                                               return_index=True,
                                               return_counts=True)
    n = len(unique_keys)
    result = np.empty((n, 3), dtype=coords.dtype)
    result_v = np.empty(n, dtype=values.dtype)

    for i in range(n):
        s = start_idx[i]
        e = s + counts[i]
        best = s + np.argmax(values_sorted[s:e])
        result[i] = coords_sorted[best]
        result_v[i] = values_sorted[best]

    return result, result_v

# ==========================================
# 4. 3D 物理拼接 (添加 fill_factor 参数控制数据密度)
# ==========================================
def stitch_3d_to_points(all_sub_volumes, all_poses, target_w, target_h,
                        amplitude_threshold=10.0, depth_range=None,
                        voxel_scale_z=0.1, fill_factor=1,
                        adaptive_fill=True, fill_spacing=1.0):
    """
    将子体数据中超过阈值的体素变换到世界坐标，生成点云。

    fill_factor: 固定亚像素填充倍数（adaptive_fill=False 时生效）。
    adaptive_fill: 为 True 时根据 scale_x/scale_y 自动计算填充倍数。
    fill_spacing:  自适应填充的目标采样间距 (mm)，越大则点越少。
                   设为 voxel_size 可避免生成无用点。
    """
    if not all_sub_volumes:
        return np.empty((0, 3)), np.empty(0)

    n_depth = all_sub_volumes[0].shape[2]
    z0, z1 = (depth_range if depth_range else (0, n_depth))
    total = len(all_sub_volumes)

    all_x = [p['x'] for p in all_poses]
    all_y = [p['y'] for p in all_poses]
    margin = max(target_w, target_h)
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin

    offset_x = -min_x
    offset_y = max_y

    all_coords = []
    all_values = []

    for idx, (vol, pose) in enumerate(zip(all_sub_volumes, all_poses)):
        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"    placing block {idx+1}/{total}...")

        angle_rad = np.radians(pose['angle'] - 90.0)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        cx = pose['x'] + offset_x
        cy = offset_y - pose['y']

        h_orig, w_orig = vol.shape[0], vol.shape[1]
        scale_x = target_w / w_orig
        scale_y = target_h / h_orig

        vol_slice = vol[:, :, z0:z1]
        pos = np.where(vol_slice > amplitude_threshold)
        if len(pos[0]) == 0: continue

        dy_arr, dx_arr, dz_arr = pos[0], pos[1], pos[2]
        amp_arr = vol_slice[pos]

        # 自适应填充：间距 ≈ fill_spacing (设为 voxel_size 可避免生成无用中间点)
        if adaptive_fill:
            fy_max = max(1, int(math.ceil(scale_y / fill_spacing)))
            fx_max = max(1, int(math.ceil(scale_x / fill_spacing)))
        else:
            fy_max = fx_max = fill_factor

        for fy in range(fy_max):
            for fx in range(fx_max):
                offset_y_sub = fy / fy_max if fy_max > 1 else 0.5
                offset_x_sub = fx / fx_max if fx_max > 1 else 0.5

                ly = (dy_arr.astype(np.float64) + offset_y_sub) * scale_y
                lx = (dx_arr.astype(np.float64) + offset_x_sub) * scale_x

                rx = lx - target_w / 2.0
                ry = ly - target_h / 2.0

                gx = cx + cos_a * rx + sin_a * ry
                gy = cy - sin_a * rx + cos_a * ry
                gz = (dz_arr + z0).astype(np.float64) * voxel_scale_z

                all_coords.append(np.column_stack([gx, gy, gz]))
                all_values.append(amp_arr)

    if not all_coords:
        return np.empty((0, 3)), np.empty(0)

    return np.vstack(all_coords), np.concatenate(all_values)

# ==========================================
# 4b. 密集 C 扫投影（warpAffine 方式，与 2D 拼接完全一致）
# ==========================================
def project_c_scan_dense(all_sub_volumes, all_poses, target_w, target_h, depth_range=None):
    """
    使用 warpAffine 进行密集 C 扫投影：
      1. 每块子体数据沿深度取最大幅值投影 (max_proj)
      2. resize 到 target_w × target_h（铺满物理宽度，消除空隙）
      3. 仿射变换旋转移到画布坐标
      4. 跨块最大幅值累积
    结果与 2D warpAffine 拼接在覆盖密度上完全一致。
    """
    if not all_sub_volumes:
        return None, None

    n_depth = all_sub_volumes[0].shape[2]
    z0, z1 = depth_range if depth_range else (0, n_depth)
    total = len(all_sub_volumes)

    all_x = [p['x'] for p in all_poses]
    all_y = [p['y'] for p in all_poses]
    margin = max(target_w, target_h)
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin
    offset_x = -min_x
    offset_y = max_y

    canvas_w = int(math.ceil(max_x - min_x))
    canvas_h = int(math.ceil(max_y - min_y))

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    footprint = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for idx, (vol, pose) in enumerate(zip(all_sub_volumes, all_poses)):
        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"    cscan block {idx+1}/{total}...")

        h_orig, w_orig = vol.shape[0], vol.shape[1]

        # 深度方向最大幅值投影
        vol_slice = vol[:, :, z0:z1]
        max_proj = vol_slice.max(axis=2).astype(np.float32)

        # 缩放到 target_w × target_h — 铺满物理宽度，消除空隙
        max_proj_resized = cv2.resize(max_proj, (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)

        # 仿射变换（与 2D 拼接的 warpAffine 一致）
        cx = pose['x'] + offset_x
        cy = offset_y - pose['y']

        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0),
                                    pose['angle'] - 90.0, 1.0)
        M[0, 2] += (cx - target_w / 2.0)
        M[1, 2] += (cy - target_h / 2.0)

        warped = cv2.warpAffine(max_proj_resized, M, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR, borderValue=0)

        # 最大幅值累积
        np.maximum(canvas, warped, out=canvas)
        footprint[warped > 0] = 1

    return canvas, footprint.astype(bool)

# ==========================================
# 5. 导出 PLY
# ==========================================
def export_points_to_ply(coords, values, output_path, binary=True):
    n_points = len(values)
    v_min, v_max = values.min(), values.max()
    norm = (values - v_min) / (v_max - v_min) if v_max > v_min else np.ones_like(values)

    r = (np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0, 1) * 255).astype(np.uint8)
    g = (np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0, 1) * 255).astype(np.uint8)

    if binary:
        with open(output_path, 'wb') as f:
            header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n_points}\n"
                      f"property float x\nproperty float y\nproperty float z\n"
                      f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      f"property float amplitude\n"
                      f"end_header\n").encode('ascii')
            f.write(header)

            data = np.empty(n_points, dtype=[('v', '3f4'), ('c', '3u1'), ('a', 'f4')])
            data['v'] = coords.astype(np.float32)
            data['c'] = np.column_stack([r, g, b])
            data['a'] = values.astype(np.float32)
            f.write(data.tobytes())
    else:
        with open(output_path, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {n_points}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "property float amplitude\nend_header\n")
            for i in range(n_points):
                f.write(f"{coords[i,0]:.2f} {coords[i,1]:.2f} {coords[i,2]:.2f} "
                        f"{r[i]} {g[i]} {b[i]} {values[i]:.1f}\n")
    print(f"  [OK] PLY 保存成功: {output_path} ({n_points} 点)")

# ==========================================
# 6. 导出 HTML 3D 可视化 (参照 tradition3d_BEV.py)
# ==========================================
def export_to_html(coords, values, output_path, c_scan_path=None,
                   title="3D 超声点云拼接结果"):
    """
    使用 Plotly 生成交互式 3D HTML 可视化。
    参考 tradition3d_BEV.py 的 plotly Volume + Scatter3d 策略：
      - 显示降采样控制点数量在 ~5 万以内
      - 深色模板 + Jet 伪彩
      - auto_open 自动在浏览器中打开
    """
    if not HAS_PLOTLY:
        print("[!] plotly 未安装，跳过 HTML 生成。执行: pip install plotly")
        return
    if len(coords) == 0:
        print("[!] 无点云数据，跳过 HTML 生成")
        return

    print(f"  [+] 正在生成 HTML 3D 可视化...")

    n_points = len(values)
    target_display = 50000
    step = max(1, n_points // target_display) if n_points > target_display else 1

    if step > 1:
        idx = np.arange(0, n_points, step)
        coords_disp = coords[idx]
        values_disp = values[idx]
        print(f"    显示降采样: {n_points} → {len(values_disp)} 点 (步长={step})")
    else:
        coords_disp = coords
        values_disp = values

    fig = go.Figure()

    # 3D 散点图 — Jet 伪彩映射，对标 BEV 的 go.Volume 配色
    fig.add_trace(go.Scatter3d(
        x=coords_disp[:, 0], y=coords_disp[:, 1], z=coords_disp[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=values_disp,
            colorscale='Jet',
            colorbar=dict(title=dict(text="幅值", font=dict(color='white'))),
            showscale=True
        ),
        name="点云"
    ))

    # 嵌入 C 扫投影图（如有）
    images = []
    if c_scan_path and os.path.exists(c_scan_path):
        with open(c_scan_path, "rb") as f:
            data_uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        images.append(dict(
            source=data_uri,
            xref="paper", yref="paper",
            x=0.7, y=0, sizex=0.28, sizey=0.28,
            xanchor="left", yanchor="bottom",
            layer="below", opacity=0.95
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=18)),
        scene=dict(
            xaxis=dict(title='X (mm)', color='white', showbackground=False,
                       gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Y (mm)', color='white', showbackground=False,
                       gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(title='Z (mm)', color='white', showbackground=False,
                       gridcolor='rgba(255,255,255,0.1)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        template='plotly_dark',
        images=images,
        margin=dict(l=10, r=10, b=10, t=50),
    )

    fig.write_html(output_path, auto_open=True)
    print(f"  [OK] HTML 3D 可视化已保存: {output_path}")

# ==========================================
# 7. 主程序
# ==========================================
def main():
    xdt_dir = "./xdt"
    pose_json_path = "pose_data.json"
    output_ply = "stitched_3d_defects.ply"
    output_html = "stitched_3d.html"
    output_cscan = "stitched_c_scan_from_3d_aligned.jpg"

    defect_threshold = 2.0
    depth_range = (20, 310)
    physical_depth_mm = 200.0   # 声程方向物理厚度 (mm)，参照 tradition3d_BEV.PHYSICAL_DEPTH_Z
    fill_factor = 1            # 亚像素填充倍数 (1=不填充，大幅降数据量)
    voxel_size = 5.0           # 体素网格降采样 (mm, 0=跳过)，参照 tradition3d_BEV.py 的 strided 策略
    vol_stride_scan = True     # 类似 BEV 对体数据做 strided 降采样 (step_x = max(1, nx//80))
    use_ply = False
    use_html = True
    use_binary_ply = True

    if not os.path.exists(pose_json_path):
        print(f"[X] 找不到位姿文件: {pose_json_path}")
        return

    with open(pose_json_path, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)

    split_configs = pose_data['split_configs']
    global_poses = pose_data['poses']

    total_images = len(split_configs)
    num_main_strips = total_images - 2
    pose_idx = 0
    for i, num_splits in enumerate(split_configs):
        if i < num_main_strips:
            for _ in range(num_splits):
                global_poses[pose_idx]['y'] += 74.0
                pose_idx += 1
        elif i == total_images - 2:
            for _ in range(num_splits):
                global_poses[pose_idx]['x'] -= 170.0
                pose_idx += 1
        elif i == total_images - 1:
            for _ in range(num_splits):
                global_poses[pose_idx]['x'] += 170.0
                pose_idx += 1

    files = [f for f in os.listdir(xdt_dir) if f.lower().endswith('.txt')]
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else 0)

    all_subs = []
    n_depth_vals = []
    for i, filename in enumerate(files):
        print(f"  [+] 处理 {filename}...", end=" ")
        vol = load_volume_from_txt(os.path.join(xdt_dir, filename), prethreshold=defect_threshold)
        n_depth_vals.append(vol.shape[2])

        # BEV 风格 strided 降采样：扫描方向 → 目标 ~100 样本（深度不动，depth_range 索引不变）
        if vol_stride_scan:
            s_scan = max(1, vol.shape[1] // 100)
            if s_scan > 1:
                vol = vol[:, ::s_scan, :]
                print(f"降采样(×{s_scan})", end=" ")

        subs = split_volume(vol, split_configs[i])

        all_subs.extend(subs)
        print(f"切分为 {len(subs)} 块")

    # -----------------------------------------------------------
    # 动态计算 Z 轴物理分辨率（参照 tradition3d_BEV.py 的 PHYSICAL_DEPTH_Z / nz）
    # -----------------------------------------------------------
    n_depth = max(n_depth_vals)
    voxel_scale_z = physical_depth_mm / n_depth
    print(f"\n  [Z] 声程方向: {n_depth} voxel → {physical_depth_mm} mm (分辨率={voxel_scale_z:.4f} mm/voxel)")

    # target_w: 扫描方向长度 (mm), target_h: 步进方向宽度 (mm)
    target_w, target_h = 360, 80
    print("\n[*] 正在进行 3D 物理拼接...")
    coords, values = stitch_3d_to_points(
        all_subs, global_poses, target_w, target_h,
        amplitude_threshold=defect_threshold,
        depth_range=depth_range,
        voxel_scale_z=voxel_scale_z,
        fill_factor=fill_factor,
        fill_spacing=voxel_size if voxel_size > 0 else 1.0
    )

    if len(values) == 0:
        print("[!] 未发现超过阈值的点")
        return

    # -----------------------------------------------
    # 体素网格降采样 (降数据量，参照 tradition3d_BEV.py)
    # -----------------------------------------------
    if voxel_size > 0:
        print(f"\n[*] 体素网格降采样 (voxel_size={voxel_size} mm)...")
        n_before = len(coords)
        coords, values = voxel_grid_filter(coords, values, voxel_size)
        print(f"    点数: {n_before} → {len(coords)} (减少 {n_before - len(coords)})")

    # -----------------------------------------------
    # 导出 PLY
    # -----------------------------------------------
    if use_ply:
        export_points_to_ply(coords, values, output_ply, binary=use_binary_ply)

    # -----------------------------------------------
    # 导出 HTML 3D 可视化 (参照 tradition3d_BEV.py)
    # -----------------------------------------------
    if use_html and HAS_PLOTLY:
        export_to_html(coords, values, output_html,
                       c_scan_path=None)  # C 扫在之后生成，首次不嵌入
    elif use_html and not HAS_PLOTLY:
        print("[!] 需安装 plotly 以生成 HTML: pip install plotly")

    # ---------------------------------------------------------
    # 渲染密集 C 扫图（不经阈值筛选，与 2D 拼接一致）
    # ---------------------------------------------------------
    print("\n[*] 正在生成密集 C 扫投影（不经阈值筛选）...")
    c_scan_img, valid_mask = project_c_scan_dense(
        all_subs, global_poses, target_w, target_h,
        depth_range=depth_range
    )

    if c_scan_img is None:
        print("[!] C 扫投影失败")
        return

    img_h, img_w = c_scan_img.shape
    print(f"    投影分辨率: {img_w} x {img_h}")

    max_amp = c_scan_img.max() if c_scan_img.max() > 0 else 80.0
    img_norm = np.clip(c_scan_img * (255.0 / max_amp), 0, 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

    # 有数据区域均值填充（消除未覆盖像素的纯蓝底）
    if np.any(valid_mask):
        avg_color = np.mean(img_color[valid_mask], axis=0)
        blank = ~valid_mask

        # 画 footprint（所有位姿覆盖的区域）
        footprint_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        all_x = [p['x'] for p in global_poses]
        all_y = [p['y'] for p in global_poses]
        margin = max(target_w, target_h)
        min_x_pose, max_x_pose = min(all_x) - margin, max(all_x) + margin
        min_y_pose, max_y_pose = min(all_y) - margin, max(all_y) + margin
        offset_x_pose = -min_x_pose
        offset_y_pose = max_y_pose

        for pose in global_poses:
            angle_rad = np.radians(pose['angle'] - 90.0)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            cx_ = pose['x'] + offset_x_pose
            cy_ = offset_y_pose - pose['y']
            corners = np.array([
                [-target_w/2.0, -target_h/2.0], [ target_w/2.0, -target_h/2.0],
                [ target_w/2.0,  target_h/2.0], [-target_w/2.0,  target_h/2.0]
            ])
            rx_ = corners[:, 0] * cos_a + corners[:, 1] * sin_a
            ry_ = -corners[:, 0] * sin_a + corners[:, 1] * cos_a
            # 画布坐标 = 中心画布坐标 + 旋转偏移（与 project_c_scan_dense 的 warpAffine 一致）
            pts = np.column_stack((cx_ + rx_, cy_ + ry_)).astype(np.int32)
            cv2.fillConvexPoly(footprint_mask, pts, 1)

        # 被扫过但无投影数据的像素用均值填充
        img_color[(footprint_mask == 1) & blank] = avg_color
        img_color[footprint_mask == 0] = [255, 255, 255]

    cv2.imwrite(output_cscan, img_color)
    print(f"  [OK] C 扫投影图已保存: {output_cscan}")

    # -----------------------------------------------
    # C 扫已生成，更新 HTML 嵌入该图（重写）
    # -----------------------------------------------
    if use_html and HAS_PLOTLY and os.path.exists(output_cscan):
        export_to_html(coords, values, output_html,
                       c_scan_path=output_cscan,
                       title="3D 超声点云拼接结果 (含 C 扫投影)")

    print("\n[OK] 任务完成！")

if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"  总耗时: {time.time()-_t0:.1f}s")
