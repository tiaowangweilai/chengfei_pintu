import cv2
import numpy as np
import os
import re

# ============================================================
# 动态 n 张图交错全景拼接（终极完整版：宽平分拉伸 + 高拉伸 + 蓝边在外）
# ============================================================
def stitch_dynamic_n_images(img_dir, target_w, target_h, overlap_px=30, bottom_h=300, edge_crop=15):
    # 读取并按数字大小排序图片 (如 1.png, 2.png ... 5.png)
    files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files.sort(key=lambda x: int(re.search(r'\d+', x).group() if re.search(r'\d+', x) else 0))
    
    n = len(files)
    if n < 3:
        print(f"错误: 至少需要 3 张图片，但只找到了 {n} 张。")
        return None

    print(f"找到 {n} 张图片，正在按 {target_w}x{target_h} 的尺寸进行规划...")

    # 画布初始化
    canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_map = np.zeros((target_h, target_w, 1), dtype=np.float32)

    # 交界线的位置 (例如 2600 - 300 = 2300)
    base_top_h = target_h - bottom_h
    layout_info = []

    # ---------------------------------------------------------
    # 1. 规划上半部分 (⭐ 宽平分 n-2 份，高拉满，整体强制压缩/拉伸)
    # ---------------------------------------------------------
    top_n = n - 2
    # ⭐ 宽度逻辑：直接把目标总宽平分给 n-2 张图，算出一份的基准宽度
    step_x = target_w // top_n  
    
    for i in range(top_n):
        img = cv2.imread(os.path.join(img_dir, files[i]))
        if img is None: continue
        
        # 切掉原始图片的自带黑边/白边
        if edge_crop > 0:
            img = img[edge_crop:-edge_crop, edge_crop:-edge_crop]
            
        # 计算每张图该放在 X 轴的哪里
        x_pos = i * step_x
        # ⭐ 宽度分配：每张图占据自己的一份宽度，再加上 overlap_px 用来与右边融合
        w_curr = step_x + overlap_px if i < top_n - 1 else target_w - x_pos
        
        # 高度分配：起点固定在 0，高度占满整个上半区
        y_start = 0
        h_curr = base_top_h + overlap_px  

        layout_info.append({
            'img': img, 'x': x_pos, 'y': y_start, 'w': w_curr, 'h': h_curr
        })

    # ---------------------------------------------------------
    # 2. 规划下半部分 (高度固定 300，蓝色边缘放在最左和最右)
    # ---------------------------------------------------------
    half_w = target_w // 2

    # 倒数第一张 (5.png)：放左侧，逆时针旋转。原图顶部(蓝)转到最左侧边缘。
    img_n1 = cv2.imread(os.path.join(img_dir, files[-1]))
    if img_n1 is not None:
        if edge_crop > 0:
            img_n1 = img_n1[edge_crop:-edge_crop, edge_crop:-edge_crop]
        img_n1 = cv2.rotate(img_n1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        layout_info.append({
            'img': img_n1, 'x': 0, 'y': base_top_h - overlap_px,
            'w': half_w + overlap_px, 'h': bottom_h + overlap_px
        })

    # 倒数第二张 (4.png)：放右侧，顺时针旋转。原图顶部(蓝)转到最右侧边缘。
    img_n2 = cv2.imread(os.path.join(img_dir, files[-2]))
    if img_n2 is not None:
        if edge_crop > 0:
            img_n2 = img_n2[edge_crop:-edge_crop, edge_crop:-edge_crop]
        img_n2 = cv2.rotate(img_n2, cv2.ROTATE_90_CLOCKWISE)
        layout_info.append({
            'img': img_n2, 'x': half_w - overlap_px, 'y': base_top_h - overlap_px,
            'w': target_w - half_w + overlap_px, 'h': bottom_h + overlap_px
        })

    # ---------------------------------------------------------
    # 3. 执行变形、裁剪与智能羽化贴图
    # ---------------------------------------------------------
    for info in layout_info:
        img = info['img']
        x, y, w, h = info['x'], info['y'], info['w'], info['h']

        # ⭐⭐ 强制缩放：这句代码会同时把“宽”和“高”拉伸或压缩，完美塞进我们分配好的格子里
        img_resized = cv2.resize(img, (w, h))

        # 计算在画布上的实际可见范围 (防越界)
        c_y_start = max(0, y)
        c_y_end = min(target_h, y + h)
        c_x_start = max(0, x)
        c_x_end = min(target_w, x + w)

        if c_y_end <= c_y_start or c_x_end <= c_x_start:
            continue

        # 计算在图像本身上对应的裁剪区域
        i_y_start = c_y_start - y
        i_y_end = i_y_start + (c_y_end - c_y_start)
        i_x_start = c_x_start - x
        i_x_end = i_x_start + (c_x_end - c_x_start)

        img_cropped = img_resized[i_y_start:i_y_end, i_x_start:i_x_end]
        h_actual, w_actual = img_cropped.shape[:2]
        
        mask = np.ones((h_actual, w_actual, 1), dtype=np.float32)

        if overlap_px > 0:
            # 智能羽化：只在内部交界处羽化，画板外边缘不羽化
            if c_x_start > 0:
                for col in range(min(overlap_px, w_actual)):
                    t = col / float(overlap_px)
                    mask[:, col, 0] = np.minimum(mask[:, col, 0], 0.5 - 0.5 * np.cos(np.pi * t))
            
            if c_x_end < target_w:
                for col in range(min(overlap_px, w_actual)):
                    t = col / float(overlap_px)
                    mask[:, w_actual - 1 - col, 0] = np.minimum(mask[:, w_actual - 1 - col, 0], 0.5 - 0.5 * np.cos(np.pi * t))

            if c_y_start > 0:
                for row in range(min(overlap_px, h_actual)):
                    t = row / float(overlap_px)
                    mask[row, :, 0] = np.minimum(mask[row, :, 0], 0.5 - 0.5 * np.cos(np.pi * t))

            if c_y_end < target_h:
                for row in range(min(overlap_px, h_actual)):
                    t = row / float(overlap_px)
                    mask[h_actual - 1 - row, :, 0] = np.minimum(mask[h_actual - 1 - row, :, 0], 0.5 - 0.5 * np.cos(np.pi * t))

        # 加权贴图
        canvas[c_y_start:c_y_end, c_x_start:c_x_end] += img_cropped * mask
        weight_map[c_y_start:c_y_end, c_x_start:c_x_end] += mask

    # 归一化权重
    weight_map[weight_map == 0] = 1
    final_canvas = canvas / weight_map
    
    return final_canvas.astype(np.uint8)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 自动获取当前脚本绝对路径，拼上 object 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FOLDER = os.path.join(current_dir, "object")
    
    TARGET_WIDTH = 1100   
    TARGET_HEIGHT = 2600  
    BOTTOM_HEIGHT = 300 
    OVERLAP = 30  
    
    # 切掉原图四周的像素，消除原图自带的黑框或白边（如果切得不够，可以改大点）
    EDGE_CROP = 15  

    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"已创建测试文件夹 {INPUT_FOLDER}，请放入测试图片后重试。")
    else:
        result_img = stitch_dynamic_n_images(
            img_dir=INPUT_FOLDER,
            target_w=TARGET_WIDTH,
            target_h=TARGET_HEIGHT,
            overlap_px=OVERLAP,
            bottom_h=BOTTOM_HEIGHT,
            edge_crop=EDGE_CROP
        )
        
        if result_img is not None:
            cv2.imwrite("Dynamic_C_Scan_Panorama.png", result_img)
            print(f"✅ 拼接成功！已按检测区约束生成 {TARGET_WIDTH}x{TARGET_HEIGHT} 的全景图。")