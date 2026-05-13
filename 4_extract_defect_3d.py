import cv2
import numpy as np
import math
import os
import re
import json

# ==========================================
# 1. Canny 缺陷检测
# ==========================================
def detect_defects_from_raw(raw_image_path, blue_thresh=40, gain=1):
    img = cv2.imread(raw_image_path)
    if img is None:
        print(f"[X] 无法读取: {raw_image_path}")
        return []

    h, w = img.shape[:2]
    b = img[:, :, 0].astype(np.float32)
    rg_max = np.maximum(img[:, :, 1], img[:, :, 2]).astype(np.float32)
    blue_score = np.clip(b - rg_max, 0, 255)
    enhanced = blue_score.copy()
    enhanced[enhanced < blue_thresh] = 0
    enhanced = np.clip(enhanced * gain, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 5, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        if length < 1 or area < length * 1.5 or area < 50 or area > h * w * 0.4:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        rect_w, rect_h = rect[1]
        cx, cy = int(rect[0][0]), int(rect[0][1])
        defect_size = max(rect_w, rect_h)
        if defect_size < 20:
            label = "small"
        elif defect_size < 50:
            label = "medium"
        else:
            label = "large"
        defects.append({
            'id': len(defects) + 1,
            'center_px': (cx, cy),
            'size_px': int(defect_size),
            'size_w_px': int(rect_w),
            'size_h_px': int(rect_h),
            'label': label
        })
    return defects


# ==========================================
# 2. 坐标系统
# ==========================================
def load_crop_info(fallback_poses, target_w=360, target_h=80):
    crop_path = "_crop_bounds.json"
    canvas_path = "_canvas_params.json"
    if os.path.exists(crop_path) and os.path.exists(canvas_path):
        with open(crop_path) as f:
            cb = json.load(f)
        with open(canvas_path) as f:
            cp = json.load(f)
        return cb['x1'], cb['y1'], cb['x2'], cb['y2'], cp['min_x'], cp['max_y']

    print("[!] 未找到裁剪信息，从位姿推算...")
    all_x = [p['x'] for p in fallback_poses]
    all_y = [p['y'] for p in fallback_poses]
    margin = max(target_w, target_h)
    min_x = min(all_x) - margin
    max_y = max(all_y) + margin
    offset_x, offset_y = -min_x, max_y

    cw = int((max(all_x) + margin) - min_x)
    ch = int((max(all_y) + margin) - (min(all_y) - margin))

    mask = np.zeros((ch, cw), dtype=np.uint8)
    for pose in fallback_poses:
        cx = int(pose['x'] + offset_x)
        cy = int(offset_y - pose['y'])
        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0),
                                    pose['angle'] - 90.0, 1.0)
        M[0, 2] += (cx - target_w / 2.0)
        M[1, 2] += (cy - target_h / 2.0)
        src = np.ones((target_h, target_w), dtype=np.uint8) * 255
        warped = cv2.warpAffine(src, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = np.maximum(mask, warped)

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    x, y, w, h = cv2.boundingRect(closed)
    x1, y1, x2, y2 = x, y, x + w, y + h
    while x1 < x2 and y1 < y2:
        te = closed[y1, x1:x2]; be = closed[y2 - 1, x1:x2]
        le = closed[y1:y2, x1]; re = closed[y1:y2, x2 - 1]
        st = np.any(te == 0); sb = np.any(be == 0)
        sl = np.any(le == 0); sr = np.any(re == 0)
        if not (st or sb or sl or sr): break
        if st: y1 += 1; continue
        if sb: y2 -= 1; continue
        if sl: x1 += 1; continue
        if sr: x2 -= 1
    return x1, y1, x2, y2, min_x, max_y


# ==========================================
# 3. 体数据加载与深度提取
# ==========================================
def load_volume_from_txt(filepath, prethreshold=2.0, cache_tag=''):
    npz_path = filepath.rsplit('.', 1)[0] + cache_tag + '.npz'
    if os.path.exists(npz_path):
        return np.load(npz_path)['vol'].astype(np.float32)
    if not cache_tag:
        old = filepath.rsplit('.', 1)[0] + '.npz'
        if os.path.exists(old):
            return np.load(old)['vol'].astype(np.float32)

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    parts = re.split(r'={5,}', content)
    nums = re.findall(r'\d+', parts[0] if len(parts) >= 2 else "")
    data_str = parts[1] if len(parts) >= 2 else content
    n_step, n_scan, n_depth = int(nums[0]), int(nums[1]), int(nums[2])
    total = n_step * n_scan * n_depth
    cleaned = re.sub(r'[-=]{3,}.*', '', data_str)
    normalized = cleaned.replace('\t', ' ').replace('\r\n', ' ').replace('\n', ' ')
    all_vals = np.fromstring(normalized, dtype=np.int32, sep=' ')
    if len(all_vals) < total // 2:
        tokens = [int(x) for x in cleaned.split() if x.strip()]
        all_vals = np.array(tokens, dtype=np.int32)
    if len(all_vals) < total:
        p = np.zeros(total, dtype=np.int32); p[:len(all_vals)] = all_vals; all_vals = p
    elif len(all_vals) > total:
        all_vals = all_vals[:total]
    volume = all_vals.reshape((n_step, n_scan, n_depth)).astype(np.float32)
    if prethreshold > 0:
        volume[volume <= prethreshold] = 0.0
    np.savez_compressed(npz_path, vol=volume)
    return volume


def split_volume(volume, num_poses):
    layer_sums = np.sum(np.abs(volume), axis=(1, 2))
    valid = np.where(layer_sums > 1e-3)[0]
    crop_idx = valid[-1] + 1 if len(valid) > 0 else volume.shape[0]
    cropped = volume[:crop_idx, ...]
    nh = cropped.shape[0]
    if nh == 0:
        return [np.zeros((1, volume.shape[1], volume.shape[2]), dtype=volume.dtype) for _ in range(num_poses)]
    sh = math.ceil(nh / num_poses)
    target = sh * num_poses
    if nh < target:
        ml = cropped.mean(axis=0, keepdims=True)
        padding = np.repeat(ml, target - nh, axis=0)
        padded = np.vstack((cropped, padding))
    else:
        padded = cropped
    subs = []
    for i in range(num_poses):
        subs.append(padded[i * sh:(i + 1) * sh, ...])
    return subs


def apply_pose_offset(pose_data):
    split_configs = pose_data['split_configs']
    global_poses = [dict(p) for p in pose_data['poses']]
    total = len(split_configs)
    n_main = total - 2
    pi = 0
    for i, ns in enumerate(split_configs):
        if i < n_main:
            for _ in range(ns):
                global_poses[pi]['y'] += 74.0; pi += 1
        elif i == total - 2:
            for _ in range(ns):
                global_poses[pi]['x'] -= 170.0; pi += 1
        else:
            for _ in range(ns):
                global_poses[pi]['x'] += 170.0; pi += 1
    return global_poses, split_configs


# ==========================================
# 4. 主程序
# ==========================================
def main():
    xdt_dir = "./xdt"
    pose_json_path = "pose_data.json"
    raw_image_path = "perfect_aligned_c_scan_raw.jpg"
    output_txt = "defects.txt"

    target_w, target_h = 360, 80
    depth_range = (20, 310)
    physical_depth_mm = 30.0    # 与 tradition3d_BEV.py 一致
    area_x_mm = 1100.0          # 检测区域 X 物理尺寸 (mm)
    area_y_mm = 2600.0          # 检测区域 Y 物理尺寸 (mm)
    area_z_mm = 30.0            # 检测区域 Z 物理尺寸 (mm)
    # 缺陷尺寸缩放：拼接时子图从 sub_h px 拉伸到 target_h px，使缺陷偏大
    # 矫正系数 = 原始物理高度 / 画布高度 = (area_y/num_splits) / target_h
    # 主带 split=38: 68.4/80 = 0.855；填充带 split=8: 325/80 = 4.06
    # 取主带系数作为默认，用户可按需调整
    size_correction_y = 0.855
    size_correction_x = 1.0
    amplitude_threshold = 2.0
    blue_thresh = 40

    if not os.path.exists(raw_image_path) or not os.path.exists(pose_json_path):
        print("[X] 缺少输入文件，请先运行 1_generate_poses.py 和 2_stitch_from_poses.py")
        return

    print("=" * 60)
    print("  缺陷检测与深度提取 → defects.txt")
    print("=" * 60)

    # ---- 加载位姿 ----
    with open(pose_json_path, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
    global_poses, split_configs = apply_pose_offset(pose_data)
    used_pose_count = sum(split_configs)
    poses_used = global_poses[:used_pose_count]

    # ---- 2D 缺陷检测 ----
    print("\n[1] Canny 缺陷检测...")
    defects = detect_defects_from_raw(raw_image_path, blue_thresh)
    if not defects:
        print(" 无缺陷")
        return
    print(f" 找到 {len(defects)} 个缺陷")

    # ---- 坐标系统 ----
    print("\n[2] 坐标变换...")
    x1, y1, x2, y2, min_x, max_y = load_crop_info(poses_used, target_w, target_h)
    print(f" 裁剪: ({x1},{y1})-({x2},{y2})  世界原点: min_x={min_x:.1f}, max_y={max_y:.1f}")

    # 根据检测区域(1100x2600mm)计算像素→物理尺寸分辨率
    raw_img = cv2.imread(raw_image_path)
    raw_h, raw_w = raw_img.shape[:2]
    res_x = area_x_mm / raw_w
    res_y = area_y_mm / raw_h
    print(f" C-scan 图: {raw_w}x{raw_h}px, 检测区域 {area_x_mm:.0f}x{area_y_mm:.0f}mm "
          f"(分辨率 {res_x:.3f}x{res_y:.3f} mm/px, 左下角=0,0)")

    # ---- 3D 体数据深度提取 ----
    print("\n[3] 加载 3D 体数据并提取深度...")
    files = [f for f in os.listdir(xdt_dir) if f.lower().endswith('.txt')]
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else 0)

    if len(files) < len(split_configs):
        eff = split_configs[:len(files)]
        used_poses = poses_used[:sum(eff)]
    else:
        eff = split_configs
        used_poses = poses_used

    all_subs = []
    n_depth_vals = []
    for i, fn in enumerate(files):
        vol = load_volume_from_txt(os.path.join(xdt_dir, fn), prethreshold=0.5, cache_tag='_depth')
        n_depth_vals.append(vol.shape[2])
        all_subs.extend(split_volume(vol, eff[i]))

    n = min(len(all_subs), len(used_poses))
    all_subs = all_subs[:n]
    used_poses = used_poses[:n]

    n_depth = max(n_depth_vals)
    voxel_scale_z = physical_depth_mm / n_depth
    z0, z1 = depth_range

    # 跳过表面波区（顶部 3mm）和底波区（底部 3mm），只在中间找缺陷峰值
    skip_mm = 3.0
    skip_vox = int(skip_mm / voxel_scale_z)
    z_search_start = z0 + skip_vox
    z_search_end   = z1 - skip_vox
    if z_search_end <= z_search_start:
        z_search_start, z_search_end = z0, z1  # 兜底

    print(f" Z 分辨率: {voxel_scale_z:.4f} mm/voxel")
    print(f" 深度搜索范围: {z_search_start * voxel_scale_z:.1f}~{z_search_end * voxel_scale_z:.1f}mm "
          f"(跳过表面/底波各 {skip_mm:.0f}mm)")

    vol_scales = [{'sx': target_w / v.shape[1], 'sy': target_h / v.shape[0]} for v in all_subs]

    # ---- 每个缺陷提取深度 ----
    strip_defects = []
    for defect in defects:
        cx_px, cy_px = defect['center_px']

        # 输出坐标：左下角为(0,0)，X向右，Y向上，以物理 mm 为单位
        out_x = cx_px * res_x
        out_y = (raw_h - 1 - cy_px) * res_y

        # 拼接器世界坐标（用于反向位姿映射，深度提取用）
        swx = cx_px + x1 + min_x
        swy = max_y - (cy_px + y1)

        # 采样网格深度
        sample_r = max(2, defect['size_px'] // 3)
        best_z = None
        best_amp_total = 0

        for dy in range(-sample_r, sample_r + 1, max(1, sample_r // 3)):
            for dx in range(-sample_r, sample_r + 1, max(1, sample_r // 3)):
                if dx * dx + dy * dy > sample_r * sample_r:
                    continue
                spx, spy = cx_px + dx, cy_px + dy
                # 采样点也用拼接器世界坐标
                s_swx = spx + x1 + min_x
                s_swy = max_y - (spy + y1)

                for vi, (vol, pose) in enumerate(zip(all_subs, used_poses)):
                    angle_rad = np.radians(pose['angle'] - 90.0)
                    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
                    cx = pose['x'] + (-min_x)
                    cy = max_y - pose['y']
                    ddx = s_swx - cx
                    ddy = s_swy - cy
                    lx = ca * ddx - sa * ddy + target_w / 2.0
                    ly = sa * ddx + ca * ddy + target_h / 2.0
                    if lx < 0 or lx >= target_w or ly < 0 or ly >= target_h:
                        continue
                    si = int(ly / vol_scales[vi]['sy'])
                    sj = int(lx / vol_scales[vi]['sx'])
                    if si < 0 or si >= vol.shape[0] or sj < 0 or sj >= vol.shape[1]:
                        continue
                    dc = vol[si, sj, z_search_start:z_search_end]
                    ma = dc.max()
                    if ma > best_amp_total:
                        best_amp_total = ma
                        best_z = (dc.argmax() + z_search_start) * voxel_scale_z

        if best_z is None:
            # 最近位姿整体深度（也跳过表面/底波区）
            for vi, (vol, pose) in enumerate(zip(all_subs, used_poses)):
                dist = (pose['x'] - swx) ** 2 + (pose['y'] - swy) ** 2
                if dist < 1e8:
                    vs = vol[:, :, z_search_start:z_search_end]
                    mi = vs.argmax()
                    if vs.flat[mi] > 0:
                        dz = np.unravel_index(mi, vs.shape)[2]
                        best_z = (dz + z_search_start) * voxel_scale_z
                        break

        if best_z is None:
            best_z = (z0 + (z1 - z0) / 2) * voxel_scale_z

        half_len = max(defect['size_w_px'] / 2 * res_x * size_correction_x, 1)
        half_wid = max(defect['size_h_px'] / 2 * res_y * size_correction_y, 1)
        half_ht = 2.0  # 固定半高

        strip_defects.append({
            'x': out_x, 'y': out_y, 'z': best_z,
            'sx': half_len, 'sy': half_wid, 'sz': half_ht,
            'label': defect['label']
        })
        print(f"  #{defect['id']} [{defect['label']:6s}] "
              f"({out_x:7.1f}, {out_y:7.1f}, {best_z:5.1f})mm "
              f"半长={half_len:.0f} 半宽={half_wid:.0f}")

    # ---- 输出 txt ----
    print("\n[4] 写入 defects.txt...")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("# 绘图区域物理尺寸 (单位: mm)\n")
        f.write("# Area 长度(X) 宽度(Y) 厚度(Z)\n")
        f.write(f"Area {area_x_mm:.0f} {area_y_mm:.0f} {area_z_mm:.0f}\n\n")
        f.write("# 缺陷配置 (全部为长条形 strip)\n")
        f.write("# 格式: 类型 中心X 中心Y 中心Z 半长(sx) 半宽(sy) 半高(sz) Roll Pitch Yaw\n")

        for i, d in enumerate(strip_defects):
            comment = ""
            if d['label'] == 'small':
                comment = "  # 较小缺陷"
            elif d['label'] == 'medium':
                comment = "  # 中等缺陷"
            else:
                comment = "  # 较大缺陷"
            f.write(f"strip {d['x']:.1f} {d['y']:.1f} {d['z']:.2f} "
                    f"{d['sx']:.0f} {d['sy']:.0f} {d['sz']:.0f} 0 0 0{comment}\n")

    print(f"\n  已保存: {output_txt}")
    print(f"  共 {len(strip_defects)} 个缺陷")
    print("=" * 60)


if __name__ == "__main__":
    main()
