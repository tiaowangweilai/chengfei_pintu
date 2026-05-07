import cv2
import numpy as np
import math
import os
import random
import re

# ==========================================
# 1. 通用图像预处理
# ==========================================
def preprocess_and_split(img, num_poses):
    """读取已加载的图像，去底边，等分提取切片"""
    if img is None: return []

    height = img.shape[0]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    white_threshold = 250 
    crop_bottom_idx = height

    for i in range(height - 1, -1, -1):
        row = img[i]
        if channels == 4:
            is_white = np.all(row[:, :3] >= white_threshold, axis=1)
            is_transparent = row[:, 3] < 10
            is_bg_row = np.all(is_white | is_transparent)
        else:
            is_bg_row = np.all(row >= white_threshold)

        if not is_bg_row:
            crop_bottom_idx = i + 1
            break

    cropped_img = img[:crop_bottom_idx, ...]
    new_height = cropped_img.shape[0]

    sub_height = math.ceil(new_height / num_poses)
    pad_height = (sub_height * num_poses) - new_height

    if pad_height > 0:
        pad_shape = list(cropped_img.shape)
        pad_shape[0] = pad_height 
        white_padding = np.full(tuple(pad_shape), 255, dtype=img.dtype)
        if channels == 4: white_padding[:, :, 3] = 0 
        padded_img = np.vstack((cropped_img, white_padding))
    else:
        padded_img = cropped_img

    sub_images = []
    for i in range(num_poses):
        start_y = i * sub_height
        end_y = start_y + sub_height
        sub_images.append(padded_img[start_y:end_y, ...])

    return sub_images

# ==========================================
# 2. 扫查规则定义 (包含特定补偿与修复)
# ==========================================
def generate_main_strip(strip_index, num_points):
    """前 n-2 张图的主扫描带"""
    poses = []
    base_x_offset = strip_index * 350.0
    shift_y = 300.0 # 整体向上偏移留出空间
    
    for i in range(num_points):
        base_x = base_x_offset
        base_y = 1440.0 - (i * 70.0) + shift_y 
        base_angle = 90.0
        
        poses.append({
            "x": base_x + random.uniform(-5.0, 5.0),
            "y": base_y + random.uniform(-3.0, 3.0),
            "angle": base_angle + random.uniform(-3.0, 3.0)
        })
    return poses

def generate_reverse_strip(base_x_start, num_points):
    """第 n-1 张图的横向反向填充 (向 0 靠拢)"""
    poses = []
    base_y = 170.0 # 固定 Y 轴位置补偿旋转中心
    shift_x = 180.0 # 补偿 180度旋转导致的 X 轴偏置
    
    for i in range(num_points):
        base_x = base_x_start - (i * 70.0) + shift_x
        base_angle = 180.0 
        
        poses.append({
            "x": base_x + random.uniform(-5.0, 5.0),
            "y": base_y + random.uniform(-3.0, 3.0),
            "angle": base_angle + random.uniform(-3.0, 3.0)
        })
    return poses

def generate_forward_strip(num_points):
    """第 n 张图的横向正向填充 (从 0 向右)"""
    poses = []
    base_y = 170.0 # 固定 Y 轴位置补偿旋转中心
    shift_x = -180.0 # 补偿 180度旋转导致的 X 轴偏置
    
    for i in range(num_points):
        base_x = 40.0 + (i * 70.0) + shift_x
        base_angle = 180.0 
        
        poses.append({
            "x": base_x + random.uniform(-5.0, 5.0),
            "y": base_y + random.uniform(-3.0, 3.0),
            "angle": base_angle + random.uniform(-3.0, 3.0)
        })
    return poses

# ==========================================
# 3. 全局拼接器 (比例拉伸 + 融合剪裁)
# ==========================================
def run_global_stitcher(all_images, all_poses, output_path):
    if not all_images: return

    target_w, target_h = 360, 80 
    processed_images = []
    
    for img in all_images:
        if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        processed_images.append(cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR))

    all_x = [p['x'] for p in all_poses]
    all_y = [p['y'] for p in all_poses]
    
    margin = max(target_w, target_h)
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin
    
    canvas_w, canvas_h = int(max_x - min_x), int(max_y - min_y)
    offset_x, offset_y = -min_x, max_y
    
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for img, pose in zip(processed_images, all_poses):
        cx = int(pose['x'] + offset_x)
        cy = int(offset_y - pose['y']) 
        
        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), pose['angle'] - 90.0, 1.0)
        M[0, 2] += (cx - target_w / 2.0)
        M[1, 2] += (cy - target_h / 2.0)
        
        warped = cv2.warpAffine(img, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
        
        alpha = (warped[:, :, 3] / 255.0)[:, :, np.newaxis]
        canvas[:, :, :3] = (1.0 - alpha) * canvas[:, :, :3] + alpha * warped[:, :, :3]
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], warped[:, :, 3])

    y_idx, x_idx = np.where(canvas[:, :, 3] > 0)
    if len(y_idx) > 0:
        cropped = canvas[np.min(y_idx):np.max(y_idx)+1, np.min(x_idx):np.max(x_idx)+1]
        cv2.imwrite(output_path, cropped)
        print(f"✅ 复杂规则拼接完成！图像已保存至: {output_path}")
    else:
        cv2.imwrite(output_path, canvas)

# ==========================================
# 4. 主控调度 (完全独立的切分数量)
# ==========================================
def main():
    input_dir = "./object"   
    
    # ==========================================
    # --- 核心切分参数控制区 ---
    main_splits_count = 23    # 前 n-2 张图片（主扫描带）切分数
    n_minus_1_splits = 10      # 第 n-1 张图片（横向反向）专属切分数
    n_splits = 8              # 第 n 张图片（横向正向）专属切分数
    # ==========================================

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else 0)
    n = len(files)
    
    if n < 3:
        print("需要至少 3 张图片来执行此逻辑。")
        return

    global_images = []
    global_poses = []

    print(f"启动拼接流。共检测到 {n} 张源图。")

    for i, filename in enumerate(files):
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)
        
        if i < n - 2:
            # 规则 1：主扫描带
            print(f"处理主带 [{filename}] -> 切分 {main_splits_count} 份, 角度 90°")
            slices = preprocess_and_split(img, main_splits_count)
            poses = generate_main_strip(strip_index=i, num_points=main_splits_count)
            
        elif i == n - 2:
            # 规则 2：横向反向填充带 (第 n-1 张)
            print(f"处理反向带 [{filename}] -> 切分 {n_minus_1_splits} 份, 角度 180°, 独立补偿")
            slices = preprocess_and_split(img, n_minus_1_splits)
            
            # 反向起始点基于主带的最后一条线
            start_x = (n - 3) * 350.0 
            poses = generate_reverse_strip(base_x_start=start_x-40, num_points=n_minus_1_splits)
            
        elif i == n - 1:
            # 规则 3：横向正向填充带 (第 n 张)
            print(f"处理正向带 [{filename}] -> 切分 {n_splits} 份, 角度 180°, 独立补偿")
            slices = preprocess_and_split(img, n_splits)
            poses = generate_forward_strip(num_points=n_splits)

        global_images.extend(slices)
        global_poses.extend(poses)

    run_global_stitcher(global_images, global_poses, "perfect_aligned_c_scan.png")

if __name__ == "__main__":
    main()