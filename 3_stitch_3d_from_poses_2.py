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
    """从 txt 文件读取三维超声体数据，返回 (step, scan, depth)"""
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
# 2. 沿步进轴切分 (严格对齐 2D 的去底边逻辑)
# ==========================================
def split_volume(volume, num_poses, defect_threshold=1.0):
    """
    沿步进轴 (axis=0) 切分。
    对齐 2D 逻辑：先剔除尾部没有数据的空扫区域，再进行等分！
    """
    # 找到每一层的最大振幅
    max_amp = np.max(volume, axis=(1, 2))
    
    # 寻找最后有有效信号的索引 (对应 2D 脚本中的去除白边/透明边)
    valid_indices = np.where(max_amp > defect_threshold)[0]
    if len(valid_indices) > 0:
        crop_bottom_idx = valid_indices[-1] + 1
    else:
        crop_bottom_idx = volume.shape[0]
        
    cropped_vol = volume[:crop_bottom_idx, ...]
    new_height = cropped_vol.shape[0]
    
    # 按照 2D 的逻辑计算子块高度并进行零填充补齐
    sub_height = math.ceil(new_height / num_poses)
    pad_height = (sub_height * num_poses) - new_height
    
    if pad_height > 0:
        pad_shape = list(cropped_vol.shape)
        pad_shape[0] = pad_height
        padding = np.zeros(pad_shape, dtype=cropped_vol.dtype)
        padded_vol = np.vstack((cropped_vol, padding))
    else:
        padded_vol = cropped_vol
        
    # 完美切分成指定的份数
    sub_volumes = []
    for i in range(num_poses):
        start_y = i * sub_height
        end_y = start_y + sub_height
        sub_volumes.append(padded_vol[start_y:end_y, ...])
        
    return sub_volumes

# ==========================================
# 3. 3D 物理拼接 (严苛对齐 2D cv2.warpAffine 并致密膨胀)
# ==========================================
def stitch_3d_to_points(all_sub_volumes, all_poses, target_w, target_h,
                        amplitude_threshold=10.0, depth_range=None,
                        voxel_scale_z=0.1):
    """将 3D 体素映射到与 2D 画布完全一致的物理空间坐标"""
    if not all_sub_volumes:
        return np.empty((0, 3)), np.empty(0)

    n_depth = all_sub_volumes[0].shape[2]
    z0, z1 = (depth_range if depth_range else (0, n_depth))
    total = len(all_sub_volumes)

    # ---------------------------------------------------------
    # 核心对齐：计算与 2D 脚本完全一致的画布偏移 (offset_x, offset_y)
    # ---------------------------------------------------------
    all_x = [p['x'] for p in all_poses]
    all_y = [p['y'] for p in all_poses]
    margin = max(target_w, target_h)
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin
    
    offset_x = -min_x
    offset_y = max_y
    # ---------------------------------------------------------

    all_coords = []
    all_values = []

    for idx, (vol, pose) in enumerate(zip(all_sub_volumes, all_poses)):
        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"    placing block {idx+1}/{total}...")

        # cv2.getRotationMatrix2D 角度计算公式
        angle_rad = np.radians(pose['angle'] - 90.0)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 严格复刻 2D 中心映射逻辑 (Y轴反转)
        cx = pose['x'] + offset_x
        cy = offset_y - pose['y']

        # h 对应 step(axis=0), w 对应 scan(axis=1)
        h_orig, w_orig = vol.shape[0], vol.shape[1]
        scale_x = target_w / w_orig
        scale_y = target_h / h_orig

        vol_slice = vol[:, :, z0:z1]
        pos = np.where(vol_slice > amplitude_threshold)
        if len(pos[0]) == 0: continue

        dy_arr, dx_arr, dz_arr = pos[0], pos[1], pos[2]
        amp_arr = vol_slice[pos]

        # --- 膨胀/拉伸到真实物理大小 (+1 确保致密覆盖，消除孔洞割裂) ---
        fill_factor_y = max(1, int(math.ceil(scale_y))) + 1
        fill_factor_x = max(1, int(math.ceil(scale_x))) + 1
        
        for fy in range(fill_factor_y):
            for fx in range(fill_factor_x):
                # 亚像素均匀散布插值
                offset_y_sub = fy / fill_factor_y
                offset_x_sub = fx / fill_factor_x
                
                ly = (dy_arr.astype(np.float64) + offset_y_sub) * scale_y
                lx = (dx_arr.astype(np.float64) + offset_x_sub) * scale_x

                # 相对于目标 2D 切片的旋转中心 (target_w/2, target_h/2)
                rx = lx - target_w / 2.0
                ry = ly - target_h / 2.0

                # 严格按照 OpenCV warpAffine 变换公式
                gx = cx + cos_a * rx + sin_a * ry
                gy = cy - sin_a * rx + cos_a * ry  
                gz = (dz_arr + z0).astype(np.float64) * voxel_scale_z

                all_coords.append(np.column_stack([gx, gy, gz]))
                all_values.append(amp_arr)

    if not all_coords:
        return np.empty((0, 3)), np.empty(0)

    return np.vstack(all_coords), np.concatenate(all_values)

# ==========================================
# 4. 导出函数 (保持不变)
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

    # --- 严格同步 2D 的偏移补偿逻辑 ---
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
        # 传入较低的阈值 (1.0) 用于安全剔除尾部空数据
        subs = split_volume(vol, split_configs[i], defect_threshold=1.0)
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

    print("\n[*] 正在生成 3D 投影 C 扫图片(完美对齐 2D)...")
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    img_w = int(math.ceil(x_max - x_min)) + 1
    img_h = int(math.ceil(y_max - y_min)) + 1
    print(f"    投影分辨率: {img_w} x {img_h}")

    c_scan_img = np.zeros((img_h, img_w), dtype=np.float32)

    ix = (coords[:, 0] - x_min).astype(np.int32)
    iy = (coords[:, 1] - y_min).astype(np.int32)
    
    np.maximum.at(c_scan_img, (iy, ix), values)

    # 动态归一化色彩映射
    max_amp = c_scan_img.max() if c_scan_img.max() > 0 else 80.0
    img_norm = np.clip(c_scan_img * (255.0 / max_amp), 0, 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)
    
    # ---------------------------------------------------------
    # 对齐 2D 脚本：全局均值填充逻辑 (让背景是白的，缝隙是均值色)
    # ---------------------------------------------------------
    mapped_mask = (c_scan_img > 0)
    coords_y, coords_x = np.where(mapped_mask)
    if len(coords_y) > 0:
        # 获取有效扫描的整体包围盒
        y_min_bb, y_max_bb = coords_y.min(), coords_y.max()
        x_min_bb, x_max_bb = coords_x.min(), coords_x.max()
        
        bb_mask = np.zeros_like(mapped_mask)
        bb_mask[y_min_bb:y_max_bb+1, x_min_bb:x_max_bb+1] = True
        
        # 计算扫描区域的平均颜色
        avg_color = np.mean(img_color[mapped_mask], axis=0)
        
        # 包围盒内部的空隙填充平均色
        unmapped_inside_bb = bb_mask & ~mapped_mask
        img_color[unmapped_inside_bb] = avg_color
        
        # 包围盒外部填充纯白
        img_color[~bb_mask] = [255, 255, 255]
    # ---------------------------------------------------------

    output_img = "stitched_c_scan_from_3d_aligned.jpg"
    cv2.imwrite(output_img, img_color)
    print(f"  [OK] C 扫投影图已保存: {output_img}")
    print("\n[OK] 任务完成！")

if __name__ == "__main__":
    main()