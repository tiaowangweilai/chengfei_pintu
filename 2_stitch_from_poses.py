import cv2
import numpy as np
import math
import os
import re
import json

# ==========================================
# 1. 通用图像预处理 (保持不变)
# ==========================================
def preprocess_and_split(img, num_poses):
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
# 2. 全局拼接器 (新增：全黑全白区域全局均值填充)
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
    
    # 累加器
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    for img, pose in zip(processed_images, all_poses):
        cx = int(pose['x'] + offset_x)
        cy = int(offset_y - pose['y']) 
        
        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), pose['angle'] - 90.0, 1.0)
        M[0, 2] += (cx - target_w / 2.0)
        M[1, 2] += (cy - target_h / 2.0)
        
        warped = cv2.warpAffine(img, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
        
        # 严格过滤插值产生的半透明边缘
        alpha_mask = (warped[:, :, 3] > 150).astype(np.uint8)
        kernel_erode = np.ones((3, 3), np.uint8)
        strict_mask = cv2.erode(alpha_mask, kernel_erode, iterations=1) > 0
        
        canvas_sum[strict_mask] += warped[strict_mask, :3]
        canvas_count[strict_mask, 0] += 1

    # --- 开始后处理：修复与裁剪 ---
    print("正在进行周围像素推测与边缘修整，请稍候...")
    
    valid_mask_bool = canvas_count[:, :, 0] > 0
    final_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    avg_color = np.zeros_like(canvas_sum)
    np.divide(canvas_sum, canvas_count, out=avg_color, where=valid_mask_bool[:, :, np.newaxis])
    final_canvas[valid_mask_bool] = np.clip(avg_color[valid_mask_bool], 0, 255).astype(np.uint8)

    valid_mask_uint8 = (valid_mask_bool * 255).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8) 
    closed_mask = cv2.morphologyEx(valid_mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    gap_mask = cv2.bitwise_xor(closed_mask, valid_mask_uint8)

    if np.any(gap_mask):
        final_canvas = cv2.inpaint(final_canvas, gap_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    x, y, w, h = cv2.boundingRect(closed_mask)
    x1, y1, x2, y2 = x, y, x + w, y + h

    while x1 < x2 and y1 < y2:
        top_edge = closed_mask[y1, x1:x2]
        bottom_edge = closed_mask[y2-1, x1:x2]
        left_edge = closed_mask[y1:y2, x1]
        right_edge = closed_mask[y1:y2, x2-1]

        shrink_top = np.sum(top_edge == 0) > len(top_edge) * 0.05
        shrink_bot = np.sum(bottom_edge == 0) > len(bottom_edge) * 0.05
        shrink_lft = np.sum(left_edge == 0) > len(left_edge) * 0.05
        shrink_rgt = np.sum(right_edge == 0) > len(right_edge) * 0.05

        if not (shrink_top or shrink_bot or shrink_lft or shrink_rgt):
            break 

        if shrink_top: y1 += 1
        if shrink_bot: y2 -= 1
        if shrink_lft: x1 += 1
        if shrink_rgt: x2 -= 1

    # 保存裁剪信息供 4_extract_defect_3d.py 使用
    with open("_crop_bounds.json", "w") as f:
        json.dump({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, f)

    cropped_final = final_canvas[y1:y2, x1:x2]

    # Save raw version before black/white fill (for defect detection)
    cv2.imwrite(output_path.replace('.jpg', '_raw.jpg'), cropped_final)

    if cropped_final.size > 0:
        # ==========================================
        # black/white fill
        # ==========================================
        # 识别纯黑像素 (RGB全部低于15) 和纯白像素 (RGB全部高于240)
        is_black = np.all(cropped_final < 15, axis=2)
        is_white = np.all(cropped_final > 240, axis=2)

        # 找到属于正常纹理的像素集
        valid_pixels_mask = ~(is_black | is_white)

        # 只要画面中还有正常颜色，就计算它们的全局平均 BGR 值
        if np.any(valid_pixels_mask):
            global_avg_color = np.mean(cropped_final[valid_pixels_mask], axis=0)

            # 将算出的平均色直接像油漆桶一样填入纯黑和纯白的区域
            cropped_final[is_black] = global_avg_color
            cropped_final[is_white] = global_avg_color
        # ==========================================

        cv2.imwrite(output_path, cropped_final)
        print(f" 拼接、裁剪及黑白块均值填充完成！最终图像 ({cropped_final.shape[1]}x{cropped_final.shape[0]}) 已保存至: {output_path}")

    else:
        print(" 警告：裁剪过度导致图像为空。退回保存完整画布。")
        cv2.imwrite(output_path, final_canvas)

# ==========================================
# 3. Canny 缺陷检测（参考 C# 代码移植）
# ==========================================
def detect_defects_canny(img, output_path, blue_thresh=40, gain=1):
    b = img[:, :, 0].astype(np.float32)
    rg_max = np.maximum(img[:, :, 1], img[:, :, 2]).astype(np.float32)
    blue_score = np.clip(b - rg_max, 0, 255)

    enhanced = blue_score.copy()
    enhanced[enhanced < blue_thresh] = 0
    enhanced = np.clip(enhanced * gain, 0, 255).astype(np.uint8)

    prep_path = output_path.replace('_defects.jpg', '_enhanced.jpg')
    cv2.imwrite(prep_path, enhanced)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

    low_thresh = 5
    high_thresh = 20
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    debug_path = output_path.replace('.jpg', '_edges.jpg')
    cv2.imwrite(debug_path, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    defect_count = 0
    h, w = img.shape[:2]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        if length < 1:
            continue
        if area < length * 1.5:
            continue
        if area < 50 or area > h * w * 0.4:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        xs = box[:, 0]
        ys = box[:, 1]
        min_x = max(0, int(xs.min()) - 1)
        max_x = min(w, int(xs.max()) + 2)
        min_y = max(0, int(ys.min()) - 1)
        max_y = min(h, int(ys.max()) + 2)

        defect_count += 1

        rect_w = rect[1][0]
        rect_h = rect[1][1]
        defect_size = max(rect_w, rect_h)
        if defect_size < 20:
            label = "small"
            color = (0, 255, 255)
        elif defect_size < 50:
            label = "medium"
            color = (0, 165, 255)
        else:
            label = "large"
            color = (0, 0, 255)

        cv2.drawContours(result, [box], 0, color, 2)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.putText(result, f"#{defect_count} {label}", (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(result, f"{defect_size:.0f}mm", (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        print(f"  Defect #{defect_count}: center=({cx},{cy}) size={defect_size:.0f}px area={area:.0f} class={label}")

    cv2.imwrite(output_path, result)
    print(f"  Canny done, {defect_count} defects, saved: {output_path}")

# ==========================================
# 4. 主控调度 (保留中心偏移逻辑)
# ==========================================
def main():
    input_dir = "./object"   
    pose_json_path = "pose_data.json"
    
    if not os.path.exists(pose_json_path):
        print(f"错误：找不到 {pose_json_path} 文件。请先运行位姿规划脚本。")
        return

    with open(pose_json_path, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
        
    split_configs = pose_data['split_configs']
    global_poses = pose_data['poses']

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else 0)
    n = len(files)

    total_images_orig = len(split_configs)
    if n != total_images_orig:
        print(f"  提示: 配置 {total_images_orig} 组位姿, 实际 {n} 个文件")
        effective_configs = split_configs[:n]
        used_pose_count = sum(effective_configs)
    else:
        effective_configs = split_configs
        used_pose_count = len(global_poses)

    total_images = len(split_configs)  # 保留原始组数用于偏移判断
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

    global_images = []
    print(f"启动拼接流。共加载 {n} 个文件, {len(split_configs)} 组位姿...")

    for i, filename in enumerate(files):
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)
        num_poses_for_this_img = split_configs[i]
        
        slices = preprocess_and_split(img, num_poses_for_this_img)
        global_images.extend(slices)

    if len(global_images) != used_pose_count:
        print(f"严重错误：切片 {len(global_images)}  vs 位姿 {used_pose_count}")
        return

    run_global_stitcher(global_images, global_poses[:used_pose_count], "perfect_aligned_c_scan.jpg")

    # Canny defect detection (independent post-process, does not modify original)
    stitched = cv2.imread("perfect_aligned_c_scan_raw.jpg")
    if stitched is not None:
        detect_defects_canny(stitched, "perfect_aligned_c_scan_defects.jpg", blue_thresh=40, gain=1)

if __name__ == "__main__":
    main()