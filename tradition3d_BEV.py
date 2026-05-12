# # =====================================================================
# # 脚本名称 : tradition3d.py
# # 修改日期 : 2026-05-07
# # 作    者 : Antigravity & USER
# # 脚本作用 : 核心超声缺陷检测引擎。对真实的超声原始采集数据进行自动极性判定、形态学闭运算和3D连通域缺陷量化分析。
# # =====================================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
from scipy.ndimage import label, find_objects, binary_closing
import matplotlib.pyplot as plt
import base64

FILE_NAME = "raw_ultrasound_data.txt"

# ==========================================
# 🌟 全局物理尺寸指定区 (单位: 毫米 mm)
# ==========================================
PHYSICAL_LENGTH_X = 1100.0  # 扫查方向物理长度 (Scan X)
PHYSICAL_WIDTH_Y  = 2600.0  # 步进方向物理宽度 (Step Y)
PHYSICAL_DEPTH_Z  = 30.0    # 声程方向物理厚度 (Depth Z)

def parse_render_and_quantify_raw(filepath):
    print(f"📂 [PhysBEV Engine] 正在解析真实采集卡数据: {filepath} ...")
    
    if not os.path.exists(filepath):
        print("❌ 找不到文件，请确保文件名正确！")
        return

    # 1. 强力读取数据
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError: continue

    if not content: return

    # 2. 动态切分头信息与数据体
    parts = re.split(r'={5,}', content)
    header_str = parts[0] if len(parts) >= 2 else ""
    data_str = parts[1] if len(parts) >= 2 else content

    nums = re.findall(r'\d+', header_str)
    if len(nums) >= 3:
        ny, nx, nz = int(nums[0]), int(nums[1]), int(nums[2])
    else:
        ny, nx, nz = 20, 300, 339

    # ==========================================
    # 🌟 核心引擎 1：动态推导物理分辨率
    # ==========================================
    Lx, Ly, Lz = PHYSICAL_LENGTH_X, PHYSICAL_WIDTH_Y, PHYSICAL_DEPTH_Z
    res_x = Lx / nx
    res_y = Ly / ny
    res_z = Lz / nz
    print(f"📐 物理空间锁定: {Lx}x{Ly}x{Lz} mm")
    print(f"🔍 自动标定分辨率: X={res_x:.3f}, Y={res_y:.3f}, Z={res_z:.3f} mm/voxel")

    # 3. 提取底层信号
    print("⏳ 正在执行底层信号极性判定与张量重构...")
    cleaned_data = data_str.strip()
    data_list = [float(item) for item in cleaned_data.split() if item.replace('.','',1).lstrip('-').isdigit()]
            
    raw_values = np.array(data_list, dtype=np.float32)
    
    total_expected = nx * ny * nz
    if len(raw_values) < total_expected:
        padded = np.zeros(total_expected)
        padded[:len(raw_values)] = raw_values
        raw_values = padded
    elif len(raw_values) > total_expected:
        raw_values = raw_values[:total_expected]

    # 张量重塑为 (X, Y, Z) 的物理空间分布
    volume_3d = raw_values.reshape((ny, nx, nz)).transpose((1, 0, 2)) 
    
    # ==========================================
    # 🌟 核心引擎 2：物理补偿与 TGC 重构
    # ==========================================
    print("🌊 启动 TGC 深度增益补偿与 Gamma 增强...")
    
    # 智能极性翻转 (如果背景是255，反转它)
    v_min, v_max, v_median = np.min(volume_3d), np.max(volume_3d), np.median(volume_3d)
    if v_median > (v_min + v_max) / 2.0:
        volume_3d = v_max - volume_3d
        
    # 强制拉伸至 0-1，准备物理补偿
    v_min, v_max = np.min(volume_3d), np.max(volume_3d)
    volume_3d = (volume_3d - v_min) / (v_max - v_min + 1e-5)

    # TGC 指数补偿 (越深的地方放得越大)
    alpha_tgc = 1.4 
    z_gain = np.exp(alpha_tgc * (np.arange(nz) / nz)).reshape(1, 1, nz)
    volume_3d = np.clip(volume_3d * z_gain, 0, 1.0)
    
    # Gamma 增强 (压制低频底噪，凸显高能主反射)
    volume_3d = np.power(volume_3d, 1.5) 
    volume_3d[volume_3d < 0.15] = 0.0 # 物理硬阈值去噪

    # ==========================================
    # 🌟 核心引擎 3：形态学智能目标提取
    # ==========================================
    print("📊 执行 3D 形态学闭运算与连通域量化...")
    
    # 盲区保护
    surface_blind_mm = 5.0 
    z_start_idx = int(surface_blind_mm / res_z)
    z_end_idx = nz - int(3.0 / res_z) 
    
    internal_raw = np.zeros_like(volume_3d)
    internal_raw[:, :, z_start_idx:z_end_idx] = volume_3d[:, :, z_start_idx:z_end_idx]
    
    binary_mask = internal_raw > 0.45 # 高置信度阈值
    
    # Z轴定向聚类，修复超声切片间的断层
    z_struct = np.zeros((3, 3, 3))
    z_struct[1, 1, :] = 1  
    binary_mask = binary_closing(binary_mask, structure=z_struct, iterations=4)
    
    labeled_array, num_features = label(binary_mask)
    defects_info = []
    valid_count = 0
    target_slice_z = int(nz * 0.8) # 默认截面深度
    
    if num_features > 0:
        objects = find_objects(labeled_array)
        for i, obj in enumerate(objects):
            slice_x, slice_y, slice_z = obj
            phys_lx = (slice_x.stop - slice_x.start) * res_x
            phys_ly = (slice_y.stop - slice_y.start) * res_y
            phys_lz = (slice_z.stop - slice_z.start) * res_z
            
            cx = (slice_x.start + slice_x.stop) / 2 * res_x
            cy = (slice_y.start + slice_y.stop) / 2 * res_y
            cz = (slice_z.start + slice_z.stop) / 2 * res_z

            # 过滤微小噪点和巨型假阳性层
            if phys_lx < 3.0 and phys_ly < 3.0: continue
            if phys_lx > (Lx * 0.4) or phys_ly > (Ly * 0.4): continue
                
            valid_count += 1
            defects_info.append({
                'id': valid_count, 'cx': cx, 'cy': cy, 'cz': cz,
                'lx': phys_lx, 'ly': phys_ly, 'lz': phys_lz,
                'x_range': (slice_x.start * res_x, slice_x.stop * res_x),
                'y_range': (slice_y.start * res_y, slice_y.stop * res_y),
                'z_range': (slice_z.start * res_z, slice_z.stop * res_z)
            })
            target_slice_z = int((slice_z.start + slice_z.stop) / 2)
            
    print(f"✅ 成功捕获 {valid_count} 处独立实体缺陷。")

    # ==========================================
    # 🌟 核心引擎 4：动态切片与工业大屏渲染
    # ==========================================
    slice_dir = "slices_dashboard"
    if not os.path.exists(slice_dir): os.makedirs(slice_dir)
        
    print(f"🚀 正在构建动态切片交互矩阵...")
    steps = []
    initial_data_uri = ""
    
    # 渲染所有切片为图片
    for i in range(nz):
        depth_val = i * res_z
        slice_data = volume_3d[:, :, i].T 
        img_path = f"{slice_dir}/slice_{i}.png"
        # 伪彩渲染 0~1 的重建场
        plt.imsave(img_path, slice_data, cmap='jet', format='png', origin='upper', vmin=0.0, vmax=1.0)
        
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        data_uri = f"data:image/png;base64,{encoded_string}"
        
        if i == target_slice_z: initial_data_uri = data_uri
            
        steps.append(dict(
            method="relayout",
            args=[{"images[0].source": data_uri}, {"annotations[0].text": f"2D C-Scan Tomography (Z={depth_val:.1f}mm)"}],
            label=f"{depth_val:.1f}"
        ))

    sliders = [dict(
        active=target_slice_z, currentvalue={"prefix": "🔪 当前截面深度: ", "suffix": " mm"},
        pad={"t": 30}, x=0.01, y=-0.15, xanchor='left', yanchor='top', len=0.30, steps=steps, font=dict(color='white')
    )]
    # 3D 渲染场降采样与网格生成 (严格采用物理坐标)
    step_x, step_y, step_z = max(1, nx//80), max(1, ny//100), 1
    vol_downsampled = volume_3d[::step_x, ::step_y, ::step_z].copy() # 加上 copy 防止污染原数据
    
    # 🛡️ 核心修复：底波视觉抑制 (Back-wall Dampening)
    # 物理意义：抵消 TGC 在底面的过曝效应，将底部 3mm 的显示能量强行压制，使其回归冷色调基底。
    bottom_voxels = max(1, int(3.0 / (res_z * step_z))) # 计算底部 3mm 对应的体素层数
    vol_downsampled[:, :, -bottom_voxels:] *= 0.25 # 抑制 75% 的视觉能量
    
    grid_x, grid_y, grid_z = np.mgrid[0:Lx:complex(vol_downsampled.shape[0]), 
                                      0:Ly:complex(vol_downsampled.shape[1]), 
                                      0:Lz:complex(vol_downsampled.shape[2])]
    # 组装三联屏
    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.20, 0.45, 0.35], 
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "table"}]],
        subplot_titles=(f"2D C-Scan Tomography (Z={target_slice_z*res_z:.1f}mm)", "3D Physical Field with Bounding Boxes", "AI 自动量化检测报告")
    )

    # 左屏：动态切片 (注入物理边界)
    fig.add_trace(go.Scatter(
        x=[0, Lx, Lx, 0, 0], y=[0, 0, Ly, Ly, 0],
        mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    fig.add_layout_image(dict(
        source=initial_data_uri, xref="x1", yref="y1",
        x=0, y=0, xanchor="left", yanchor="top",
        sizex=Lx, sizey=Ly, sizing="stretch", opacity=1, layer="below"
    ))

    # 中屏：厚重的 3D 体素图
    fig.add_trace(go.Volume(
        x=grid_x.flatten(), y=grid_y.flatten(), z=grid_z.flatten(),
        value=vol_downsampled.flatten(),
        isomin=0.2, isomax=1.0, opacity=0.35, surface_count=12, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title=dict(text="Energy", font=dict(color='white')), x=0.68)
    ), row=1, col=2)

    # 绘制高亮洋红物理包围盒
    for d in defects_info:
        x0, x1 = d['x_range']
        y0, y1 = d['y_range']
        z0, z1 = d['z_range']
        bx = [x0, x1, x1, x0, x0, x0, x1, x1, x0, x0, x1, x1, x1, x1, x0, x0]
        by = [y0, y0, y1, y1, y0, y0, y0, y1, y1, y0, y0, y0, y1, y1, y1, y1]
        bz = [z0, z0, z0, z0, z0, z1, z1, z1, z1, z1, z1, z0, z0, z1, z1, z0]
        
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines', line=dict(color='magenta', width=5, dash='solid'), name=f"Box {d['id']}"
        ), row=1, col=2)

    # 右屏：数据量化表格
    table_headers = ['ID', '中心 (X,Y,Z) mm', '尺寸 L*W*H mm', '状态']
    table_rows = []
    for d in defects_info:
        coord_str = f"{d['cx']:.1f}, {d['cy']:.1f}, {d['cz']:.1f}"
        size_str = f"{d['lx']:.1f} x {d['ly']:.1f} x {d['lz']:.1f}"
        status = "🔴 危急" if max(d['lx'], d['ly']) > 25 else "🟡 警告"
        table_rows.append([d['id'], coord_str, size_str, status])
    if not table_rows: table_rows = [["-"]*4]
    
    fig.add_trace(go.Table(
        columnwidth=[0.6, 2.4, 2.0, 1.0], 
        header=dict(values=table_headers, fill_color='#1f77b4', font=dict(color='white', size=13), align='center'),
        cells=dict(values=list(zip(*table_rows)), fill_color='#2c3e50', font=dict(color='white', size=12), align='center', height=30)
    ), row=1, col=3)

    # 全局排版：约束长宽比，反转Y轴
    fig.update_layout(
        title=dict(text='PhysBEV 工业级多模态探伤与量化系统 (TGC+形态学增强版)', font=dict(color='white', size=22)),
        sliders=sliders,
        scene=dict(
            xaxis=dict(title='Scan X (mm)', color='white', showbackground=False),
            yaxis=dict(title='Step Y (mm)', color='white', showbackground=False),
            zaxis=dict(title='Depth Z (mm)', color='white', showbackground=False, autorange='reversed'),
            # 🎯 极其核心：强制比例，绝不变形！
            aspectratio=dict(x=1, y=Ly/Lx, z=0.4), 
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))
        ),
        dragmode='orbit', template='plotly_dark', margin=dict(l=20, r=10, b=100, t=80)
    )
    # 确保左侧2D图像也匹配工业视觉的Y轴方向
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    
    output_html = "PhysBEV_Ultimate_Dashboard.html"
    fig.write_html(output_html, auto_open=True)
    print(f"✅ 生成完毕！请双击查看终极融合的完美物理大屏: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    parse_render_and_quantify_raw(FILE_NAME)