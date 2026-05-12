import math
import random
import json

# ==========================================
# 1. 扫查规则定义
# ==========================================
def generate_main_strip(base_x, start_y, num_points, pose_gap):
    """生成单条主扫描带的位姿"""
    poses = []
    for i in range(num_points):
        base_y = start_y - (i * pose_gap)
        base_angle = 90.0
        poses.append({
            "x": base_x + random.uniform(-5.0, 5.0),
            "y": base_y + random.uniform(-3.0, 3.0),
            "angle": base_angle + random.uniform(-3.0, 3.0)
        })
    return poses

def generate_reverse_strip(start_x, fixed_y, num_points, pose_gap):
    """横向反向填充 (从外侧向左靠拢)"""
    poses = []
    shift_x = 180.0 
    for i in range(num_points):
        base_x = start_x - (i * pose_gap) + shift_x
        base_angle = 180.0 
        poses.append({
            "x": base_x + random.uniform(-5.0, 5.0),
            "y": fixed_y + random.uniform(-3.0, 3.0),
            "angle": base_angle + random.uniform(-3.0, 3.0)
        })
    return poses

def generate_forward_strip(start_x, fixed_y, num_points, pose_gap):
    """横向正向填充 (从左向外侧展开)"""
    poses = []
    shift_x = -180.0 
    for i in range(num_points):
        base_x = start_x + (i * pose_gap) + shift_x
        base_angle = 180.0 
        poses.append({
            "x": base_x + random.uniform(-5.0, 5.0),
            "y": fixed_y + random.uniform(-3.0, 3.0),
            "angle": base_angle + random.uniform(-3.0, 3.0)
        })
    return poses

# ==========================================
# 2. 核心大脑：固定步进 + 边界封口
# ==========================================
def main():
    output_json = "pose_data.json"
    
    # ==========================================
    # [用户物理配置区]
    # ==========================================
    area_width = 1000.0   # 👈 在这里填入检测区域总宽 (试试 1000 或 1100)
    area_length =1000.0   # 👈 检测区域总长
    
    edge_margin_x = 170.0 # 左右两侧探头缩进量
    main_strip_gap = 350.0 # 主带之间的固定物理步进间距
    
    pose_gap = 70.0       # 切片之间的间距 (长宽方向通用)
    margin_y = 140.0      # Y轴扫查的顶部越界余量
    horizontal_fixed_y = 170.0 # 横向带固定的 Y 坐标
    # ==========================================

    # --- 数学推导：主带 X 坐标 (按 350 累加) ---
    first_x = edge_margin_x
    last_x = area_width - edge_margin_x
    
    main_x_coords = [first_x]
    current_x = first_x + main_strip_gap
    
    # 只要加上 350 后还没有越过右边界，就继续加
    while current_x < last_x:
        # 防浮点误差：如果极其接近右边界，直接跳出，让下面统一补位
        if last_x - current_x < 1.0: 
            break
        main_x_coords.append(current_x)
        current_x += main_strip_gap

    # 最后一条带死死锁在右边界上
    if last_x > first_x and main_x_coords[-1] != last_x:
        main_x_coords.append(last_x)

    num_main_strips = len(main_x_coords)

    # --- 数学推导：切分数量 ---
    main_splits_count = math.ceil(area_length / pose_gap) 
    
    total_horiz_points = math.ceil(area_width / pose_gap) + 1 
    n_minus_1_splits = math.ceil(total_horiz_points / 2)      
    n_splits = total_horiz_points - n_minus_1_splits          
    if n_splits <= 0: n_splits = 1 
    
    total_images = num_main_strips + 2 # 总图数 = 算出来的主带数 + 2

    print(f"📊 规划开始：检测区域 {area_width} x {area_length}")
    print(f" -> 步进策略：左边缘 {first_x}, 每次步进 {main_strip_gap}, 右边缘封口 {last_x}")
    print(f" -> 实际推导的主带坐标：{[round(x, 1) for x in main_x_coords]}")
    print(f" -> 预计规划 {num_main_strips} 条主带，每条带切分 {main_splits_count} 份")
    print(f" -> 预计规划 2 条横向带，分别切分 {n_minus_1_splits} 和 {n_splits} 份")
    print(f" -> 📸 结论：机器人需要拍摄并提供 {total_images} 张源图！\n")

    global_poses = []
    split_configs = [] 
    start_y = area_length + margin_y

    # 1. 写入主带
    for i, base_x in enumerate(main_x_coords):
        print(f" [写入] 主带 {i+1}/{num_main_strips} | Base X: {base_x:.1f} | 切分: {main_splits_count}")
        poses = generate_main_strip(base_x, start_y, main_splits_count, pose_gap)
        split_configs.append(main_splits_count)
        global_poses.extend(poses)
        
    # 2. 写入横向反向带 (从最右侧开始，向左靠拢)
    reverse_start_x = area_width - 40.0 
    print(f" [写入] 反向横向带  | Start X: {reverse_start_x:.1f} | 切分: {n_minus_1_splits}")
    poses_rev = generate_reverse_strip(reverse_start_x, horizontal_fixed_y, n_minus_1_splits, pose_gap)
    split_configs.append(n_minus_1_splits)
    global_poses.extend(poses_rev)
    
    # 3. 写入横向正向带 (从最左侧开始向右展开)
    forward_start_x = 40.0
    print(f" [写入] 正向横向带  | Start X: {forward_start_x:.1f} | 切分: {n_splits}")
    poses_fwd = generate_forward_strip(forward_start_x, horizontal_fixed_y, n_splits, pose_gap)
    split_configs.append(n_splits)
    global_poses.extend(poses_fwd)

    # 导出 JSON
    export_data = {
        "total_images": total_images,
        "split_configs": split_configs,
        "poses": global_poses
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4)
        
    print(f"\n✅ 位姿生成完毕！共 {len(global_poses)} 个控制点。数据已保存至: {output_json}")

if __name__ == "__main__":
    main()