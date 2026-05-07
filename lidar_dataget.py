import cv2
import numpy as np
import os
import random

# ==========================================
# 1. 位姿生成
# ==========================================
def generate_radar_poses(start_y=1440.0, num_points=18, step_y=70.0):
    poses = []
    for i in range(num_points):
        base_x = 0.0
        base_y = start_y - (i * step_y)
        base_angle = 90.0
        
        error_y = random.uniform(-3.0, 3.0) 
        error_x = random.uniform(-5.0, 5.0)
        error_angle = random.uniform(-3.0, 3.0)
        
        poses.append({
            "id": i + 1,
            "x": round(base_x + error_x, 2),
            "y": round(base_y + error_y, 2),
            "angle": round(base_angle + error_angle, 2)
        })
    return poses

# ==========================================
# 2. 比例校正与拼接剪裁
# ==========================================
def stitch_and_crop_c_scan_scaled(image_dir, poses, output_path="final_stitched_c_scan_scaled.png"):
    # 物理参数定义
    PHYSICAL_WIDTH_MM = 360.0  
    STEP_Y_MM = 80.0           
    
    # 设定全局目标分辨率：1 毫米 = 1 个像素
    # 这样物理世界的 350mm x 80mm 的切片，在程序里就被强制拉伸成 350px x 80px 的图像
    TARGET_RES = 1.0 # px/mm
    
    # 计算目标切片在统一分辨率下的像素尺寸
    target_slice_w = int(PHYSICAL_WIDTH_MM * TARGET_RES) # 350 px
    target_slice_h = int(STEP_Y_MM * TARGET_RES)         # 80 px

    images = []
    valid_poses = []
    
    # 2.1 读取并立刻拉伸切片
    for pose in poses:
        img_name = f"pose_{pose['id']:03d}.png"
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        # 统一转为 RGBA 格式
        if img.shape[2] == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
        # ================= 核心修改：图像拉伸 =================
        # 不管原图是 682x37 还是多少，强制拉伸到反映物理尺寸的比例 (350x80)
        # 建议使用 INTER_LINEAR 用于一般图像，INTER_NEAREST 适合保留数据边缘
        img_resized = cv2.resize(img, (target_slice_w, target_slice_h), interpolation=cv2.INTER_LINEAR)
        # =======================================================
        
        images.append(img_resized)
        valid_poses.append(pose)

    if not images:
        print("错误：未找到图片。")
        return

    # 由于现在 1mm = 1px，物理坐标直接就是像素坐标
    all_x_px = [p['x'] * TARGET_RES for p in valid_poses]
    all_y_px = [p['y'] * TARGET_RES for p in valid_poses]
    
    # 留出边距
    margin = max(target_slice_w, target_slice_h)
    min_x_px, max_x_px = min(all_x_px) - margin, max(all_x_px) + margin
    min_y_px, max_y_px = min(all_y_px) - margin, max(all_y_px) + margin
    
    canvas_w = int(max_x_px - min_x_px)
    canvas_h = int(max_y_px - min_y_px)
    
    offset_x = -min_x_px
    offset_y = max_y_px 
    
    print(f"创建等比例物理画布... 尺寸: {canvas_w}x{canvas_h}")

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    # 2.3 融合拉伸后的切片
    for img, pose in zip(images, valid_poses):
        cx = int(pose['x'] * TARGET_RES + offset_x)
        cy = int(offset_y - pose['y'] * TARGET_RES) 
        
        rotation_angle = pose['angle'] - 90.0
        # 旋转中心使用拉伸后的尺寸
        center = (target_slice_w / 2.0, target_slice_h / 2.0)
        
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        M[0, 2] += (cx - target_slice_w / 2.0)
        M[1, 2] += (cy - target_slice_h / 2.0)
        
        warped_img = cv2.warpAffine(img, M, (canvas_w, canvas_h), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0, 0))
        
        alpha_mask = (warped_img[:, :, 3] / 255.0)[:, :, np.newaxis]
        canvas[:, :, :3] = (1.0 - alpha_mask) * canvas[:, :, :3] + alpha_mask * warped_img[:, :, :3]
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], warped_img[:, :, 3])

    # 2.4 自动剪裁
    alpha_channel = canvas[:, :, 3]
    non_transparent_points = np.where(alpha_channel > 0)
    
    if len(non_transparent_points[0]) > 0:
        y_coords, x_coords = non_transparent_points
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        cropped_canvas = canvas[min_y:max_y+1, min_x:max_x+1]
        cv2.imwrite(output_path, cropped_canvas)
        print(f"✅ 比例校正与剪裁完成。目标物理比例图像已保存。")
        print(f"最终图像尺寸应该约为 {int(PHYSICAL_WIDTH_MM)} x {int(18 * STEP_Y_MM)} 像素。")
    else:
        cv2.imwrite(output_path, canvas)

if __name__ == "__main__":
    poses = generate_radar_poses()
    stitch_and_crop_c_scan_scaled("./output", poses)