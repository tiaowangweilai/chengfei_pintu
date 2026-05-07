import cv2
import numpy as np
import math
import os

def preprocess_c_scan(image_path, num_poses, output_dir="output"):
    # 1. 保留原格式读取图像 (保留 4 通道)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径: {image_path}")

    height = img.shape[0]
    width = img.shape[1]
    # 判断通道数：如果是灰度图维度为2，否则获取第三个维度的值
    channels = img.shape[2] if len(img.shape) > 2 else 1

    # 2. 去除底部连续的背景行（纯白 或 透明）
    white_threshold = 250 
    crop_bottom_idx = height

    # 从图像最底部一行开始向上遍历
    for i in range(height - 1, -1, -1):
        row = img[i]
        
        # 判断这一行是否全为"背景"
        if channels == 4:
            # RGBA 图像：分离 RGB 和 Alpha 通道
            rgb = row[:, :3]
            alpha = row[:, 3]
            
            # 条件1：像素是白色的 (R, G, B 都 >= 250)
            is_white = np.all(rgb >= white_threshold, axis=1)
            # 条件2：像素是透明的 (Alpha 值极低，比如小于10)
            is_transparent = alpha < 10
            
            # 只要满足是白色或透明其中之一，就算作背景像素
            is_bg_pixel = is_white | is_transparent
            # 如果整行都是背景像素，才算作背景行
            is_bg_row = np.all(is_bg_pixel)
            
        elif channels == 3:
            # 普通 RGB 图像
            is_bg_row = np.all(row >= white_threshold)
        else:
            # 单通道灰度图
            is_bg_row = np.all(row >= white_threshold)

        # 如果发现这一行不全是背景，说明碰到了真正的雷达扫描数据
        if not is_bg_row:
            crop_bottom_idx = i + 1
            break

    # 截取有效区域
    cropped_img = img[:crop_bottom_idx, ...]
    new_height = cropped_img.shape[0]

    # 3. 计算等分并补白
    sub_height = math.ceil(new_height / num_poses)
    target_height = sub_height * num_poses 
    pad_height = target_height - new_height

    if pad_height > 0:
        pad_shape = list(cropped_img.shape)
        pad_shape[0] = pad_height 
        
        # 填充 255。对于 4 通道图像，这会生成 [255,255,255,255] 的纯白且不透明背景
        white_padding = np.full(tuple(pad_shape), 255, dtype=img.dtype)
        padded_img = np.vstack((cropped_img, white_padding))
    else:
        padded_img = cropped_img

    # 4. 按照雷达位姿等分图像
    sub_images = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_poses):
        start_y = i * sub_height
        end_y = start_y + sub_height
        
        sub_img = padded_img[start_y:end_y, ...]
        sub_images.append(sub_img)
        
        save_path = os.path.join(output_dir, f"pose_{i+1:03d}.png")
        cv2.imwrite(save_path, sub_img)

    return sub_images

# --- 运行示例 ---
if __name__ == "__main__":
    radar_poses_count = 18
    result = preprocess_c_scan("./object/1.png", radar_poses_count)
    print(f"成功将图像分割为 {len(result)} 份，每份尺寸为: {result[0].shape}")