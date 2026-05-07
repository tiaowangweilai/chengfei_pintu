import numpy as np
import math
import os
import re
import json
import struct
import cv2

# ==========================================
# 1. 解析 txt 体数据
# ==========================================
def load_volume_from_txt(filepath):
    """从 txt 文件读取三维超声体数据"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    header = lines[0].strip()
    nums = re.findall(r'\d+', header)
    n_step, n_scan, n_depth = int(nums[0]), int(nums[1]), int(nums[2])

    blocks = []
    current_block = []
    for line in lines[2:]:
        stripped = line.strip()
        if stripped == '' or stripped.startswith('---') or stripped.startswith('==='):
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            vals = [int(v) for v in stripped.split('\t') if v.strip()]
            current_block.append(vals)
    if current_block:
        blocks.append(current_block)

    volume = np.array(blocks, dtype=np.float32)
    return volume

# ==========================================
# 2. 沿步进轴切分 (核心修复：精准动态切除尾部废数据)
# ==========================================
def split_volume(volume, num_poses):
    """
    直击根源：先剔除 txt 尾部的传感器空跑/补零数据，再进行等分！
    这确保了有效数据会被正确地分配到所有的位姿上，消除中间的断层间隙。
    """
    # 1. 计算每一层的绝对值总和，识别出“全零”的空扫描区域
    layer_sums = np.sum(np.abs(volume), axis=(1, 2))
    
    # 2. 寻找真实的有效数据边界 (使用极小的阈值抵抗底噪)
    valid_indices = np.where(layer_sums > 1e-3)[0]
    
    if len(valid_indices) > 0:
        crop_bottom_idx = valid_indices[-1] + 1  # 真实数据的最后一行
    else:
        crop_bottom_idx = volume.shape[0]
        
    cropped_vol = volume[:crop_bottom_idx, ...]
    new_height = cropped_vol.shape[0]
    
    # 3. 计算每份的理想高度，并向上取整
    sub_height = math.ceil(new_height / num_poses)
    
    # 4. 补齐到能被完全等分的大小 (防止最后一份形状残缺报错)
    pad_height = (sub_height * num_poses) - new_height
    if pad_height > 0:
        pad_shape = list(cropped_vol.shape)
        pad_shape[0] = pad_height
        padding = np.zeros(pad_shape, dtype=cropped_vol.dtype)
        padded_vol = np.vstack((cropped_vol, padding))
    else:
        padded_vol = cropped_vol
        
    # 5. 执行物理等分
    sub_volumes = []
    for i in range(num_poses):
        start_y = i * sub_height
        end_y = start_y + sub_height
        sub_volumes.append(padded_vol[start_y:end_y, ...])
        
    return sub_volumes

# ==========================================
# 3. 3D 物理拼接
# ==========================================
def stitch_3d_to_points(all_sub_volumes, all_poses, target_w, target_h,
                        amplitude_threshold=10.0, depth_range=None,
                        voxel_scale_z=0.1):
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

        # 亚像素致密膨胀填充
        fill_factor_y = max(1, int(math.ceil(scale_y))) + 1
        fill_factor_x = max(1, int(math.ceil(scale_x))) + 1
        
        for fy in range(fill_factor_y):
            for fx in range(fill_factor_x):
                offset_y_sub = fy / fill_factor_y
                offset_x_sub = fx / fill_factor_x
                
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
# 4. 导出 PLY
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
# 5. 主程序
# ==========================================
def main():
    xdt_dir = "./xdt"
    pose_json_path = "pose_data.json"
    output_ply = "stitched_3d_defects.ply"

    defect_threshold = 2.0   
    depth_range = (20, 310)   
    voxel_scale_z = 0.1       
    downsample = 1            
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
                global_poses[pose_idx]['y'] += 140.0
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
    for i, filename in enumerate(files):
        print(f"  [+] 处理 {filename}...", end=" ")
        vol = load_volume_from_txt(os.path.join(xdt_dir, filename))
        
        # 调用修复后的动态切分函数
        subs = split_volume(vol, split_configs[i])
        
        all_subs.extend(subs)
        print(f"切分为 {len(subs)} 块")

    target_w, target_h = 360, 80 
    print("\n[*] 正在进行 3D 物理拼接...")
    coords, values = stitch_3d_to_points(
        all_subs, global_poses, target_w, target_h,
        amplitude_threshold=defect_threshold,
        depth_range=depth_range,
        voxel_scale_z=voxel_scale_z
    )

    if len(values) == 0:
        print("[!] 未发现超过阈值的点")
        return

    if downsample > 1:
        idx = np.arange(0, len(values), downsample)
        coords, values = coords[idx], values[idx]

    export_points_to_ply(coords, values, output_ply, binary=use_binary_ply)

    # ---------------------------------------------------------
    # 渲染正确的 C 扫图，自动绘制边界并均值填充
    # ---------------------------------------------------------
    print("\n[*] 正在生成 3D 投影 C 扫图片(纯净对齐版)...")
    
    all_x = [p['x'] for p in global_poses]
    all_y = [p['y'] for p in global_poses]
    margin = max(target_w, target_h)
    
    min_x_pose, max_x_pose = min(all_x) - margin, max(all_x) + margin
    min_y_pose, max_y_pose = min(all_y) - margin, max(all_y) + margin
    offset_x = -min_x_pose
    offset_y = max_y_pose

    all_gx, all_gy = [], []
    for pose in global_poses:
        angle_rad = np.radians(pose['angle'] - 90.0)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy = pose['x'] + offset_x, offset_y - pose['y']
        
        corners = np.array([
            [-target_w/2.0, -target_h/2.0], [ target_w/2.0, -target_h/2.0],
            [ target_w/2.0,  target_h/2.0], [-target_w/2.0,  target_h/2.0]
        ])
        rx = corners[:, 0] * cos_a + corners[:, 1] * sin_a
        ry = -corners[:, 0] * sin_a + corners[:, 1] * cos_a
        all_gx.extend(cx + rx)
        all_gy.extend(cy + ry)

    x_min = min(np.min(coords[:, 0]) if len(coords) > 0 else float('inf'), min(all_gx))
    x_max = max(np.max(coords[:, 0]) if len(coords) > 0 else float('-inf'), max(all_gx))
    y_min = min(np.min(coords[:, 1]) if len(coords) > 0 else float('inf'), min(all_gy))
    y_max = max(np.max(coords[:, 1]) if len(coords) > 0 else float('-inf'), max(all_gy))
    
    img_w = int(math.ceil(x_max - x_min)) + 1
    img_h = int(math.ceil(y_max - y_min)) + 1
    print(f"    投影分辨率: {img_w} x {img_h}")

    c_scan_img = np.zeros((img_h, img_w), dtype=np.float32)
    footprint_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for pose in global_poses:
        angle_rad = np.radians(pose['angle'] - 90.0)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy = pose['x'] + offset_x, offset_y - pose['y']
        
        corners = np.array([
            [-target_w/2.0, -target_h/2.0], [ target_w/2.0, -target_h/2.0],
            [ target_w/2.0,  target_h/2.0], [-target_w/2.0,  target_h/2.0]
        ])
        rx = corners[:, 0] * cos_a + corners[:, 1] * sin_a
        ry = -corners[:, 0] * sin_a + corners[:, 1] * cos_a
        
        gx = cx + rx - x_min
        gy = cy + ry - y_min
        pts = np.column_stack((gx, gy)).astype(np.int32)
        cv2.fillConvexPoly(footprint_mask, pts, 1)

    if len(coords) > 0:
        ix = (coords[:, 0] - x_min).astype(np.int32)
        iy = (coords[:, 1] - y_min).astype(np.int32)
        np.maximum.at(c_scan_img, (iy, ix), values)

    max_amp = c_scan_img.max() if c_scan_img.max() > 0 else 80.0
    img_norm = np.clip(c_scan_img * (255.0 / max_amp), 0, 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

    # 计算有数据区域的均值色
    mapped_mask = (c_scan_img > 0)
    if np.any(mapped_mask):
        avg_color = np.mean(img_color[mapped_mask], axis=0)
        # 用均值色填充没有信号但被扫过的区域 (消除纯蓝底或黑洞)
        img_color[(footprint_mask == 1) & (~mapped_mask)] = avg_color

    # 没有被扫查过的地方变为纯白背景
    img_color[footprint_mask == 0] = [255, 255, 255]
    
    output_img = "stitched_c_scan_from_3d_aligned.jpg"
    cv2.imwrite(output_img, img_color)
    print(f"  [OK] 终极对齐版 C 扫投影图已保存: {output_img}")
    print("\n[OK] 任务完成！")

if __name__ == "__main__":
    main()