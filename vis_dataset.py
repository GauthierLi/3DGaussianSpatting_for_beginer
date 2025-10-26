import os
import numpy as np
import plotly.graph_objects as go

from PIL import Image
from common.scene_tools import getWorld2View2
from dataset.scene_dataset import SceneDataset

scene = SceneDataset()
fig = go.Figure()

translate = scene.scene_info.nerf_normalization['translate']
scale = 1.0 / scene.scene_info.nerf_normalization['radius']

all_x = []
all_y = []
all_z = []

if scene.scene_info.point_cloud is not None:
    points = scene.scene_info.point_cloud.points
    colors = scene.scene_info.point_cloud.colors
    # 采样点云 
    sample_step = 10
    sampled_points = points[::sample_step]
    sampled_colors = colors[::sample_step]
    sampled_points_norm = (sampled_points + translate) * scale
    sampled_colors_rgb = (sampled_colors * 255).astype(int)
    fig.add_trace(go.Scatter3d(
        x=sampled_points_norm[:, 0],
        y=sampled_points_norm[:, 1],
        z=sampled_points_norm[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=['rgb({}, {}, {})'.format(r, g, b) for r, g, b in sampled_colors_rgb],
            opacity=0.8
        ),
        name='Point Cloud'
    ))
    all_x.extend(sampled_points_norm[:, 0])
    all_y.extend(sampled_points_norm[:, 1])
    all_z.extend(sampled_points_norm[:, 2])

# 限制绘制相机，避免过多 (每隔一定间隔显示)
step = 10  # 间隔步长
max_display = 12  # 最大显示数量
selected_indices = range(0, min(len(scene.scene_info.train_cameras), max_display * step), step)
for i in selected_indices:
    cam = scene.scene_info.train_cameras[i]
    W2C = getWorld2View2(cam.R, cam.T)
    C2W = np.linalg.inv(W2C)
    pos_raw = C2W[:3, 3]
    pos = (pos_raw + translate) * scale
    R = C2W[:3, :3]
    
    # 顶点 
    size = 0.05  
    cuboid_vertices = np.array([
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ])
    rotated_cuboid = (R @ cuboid_vertices.T).T + pos
    
    # 计算远裁剪面点 (4个点)
    zfar_vis = 1.0  
    tanHalfFovX = np.tan(cam.FovX / 2)
    tanHalfFovY = np.tan(cam.FovY / 2)
    far_corners_cam = np.array([
        [-tanHalfFovX * zfar_vis, -tanHalfFovY * zfar_vis, zfar_vis],
        [tanHalfFovX * zfar_vis, -tanHalfFovY * zfar_vis, zfar_vis],
        [tanHalfFovX * zfar_vis, tanHalfFovY * zfar_vis, zfar_vis],
        [-tanHalfFovX * zfar_vis, tanHalfFovY * zfar_vis, zfar_vis]
    ])
    far_corners_world = (R @ far_corners_cam.T).T + pos_raw
    far_corners = (far_corners_world + translate) * scale
    
    # 所有顶点 (12个点)
    all_vertices = np.vstack([rotated_cuboid, far_corners])
    
    all_x.extend(all_vertices[:, 0])
    all_y.extend(all_vertices[:, 1])
    all_z.extend(all_vertices[:, 2])
    
    # 绘制长方体线 (蓝色)
    cuboid_edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for edge in cuboid_edges:
        fig.add_trace(go.Scatter3d(
            x=[all_vertices[edge[0], 0], all_vertices[edge[1], 0]],
            y=[all_vertices[edge[0], 1], all_vertices[edge[1], 1]],
            z=[all_vertices[edge[0], 2], all_vertices[edge[1], 2]],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    
    # 绘制梯台线 (红色) 
    frustum_edges = [(4,8), (5,9), (6,10), (7,11)]
    for edge in frustum_edges:
        fig.add_trace(go.Scatter3d(
            x=[all_vertices[edge[0], 0], all_vertices[edge[1], 0]],
            y=[all_vertices[edge[0], 1], all_vertices[edge[1], 1]],
            z=[all_vertices[edge[0], 2], all_vertices[edge[1], 2]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # 绘制远裁剪面线 (红色)
    far_plane_edges = [(8,9), (9,10), (10,11), (11,8)]
    for edge in far_plane_edges:
        fig.add_trace(go.Scatter3d(
            x=[all_vertices[edge[0], 0], all_vertices[edge[1], 0]],
            y=[all_vertices[edge[0], 1], all_vertices[edge[1], 1]],
            z=[all_vertices[edge[0], 2], all_vertices[edge[1], 2]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # 计算heading并添加文本
    forward = R @ np.array([0, 0, 1])
    heading = np.arctan2(forward[1], forward[0]) * 180 / np.pi
    fig.add_trace(go.Scatter3d(
        x=[pos[0]],
        y=[pos[1]],
        z=[pos[2]],
        mode='text',
        text=[f'{heading:.1f}°'],
        textposition="top center",
        showlegend=False
    ))
    
    # 在梯台底面渲染图片 (近裁剪面)
    if os.path.exists(cam.image_path):
        img = Image.open(cam.image_path)
        img = img.resize((128, 128))  
        img_array = np.array(img) / 255.0
        
        aspect_ratio = img.width / img.height
        
        znear_vis = -1.01
        tanHalfFovX = np.tan(cam.FovX / 2)
        tanHalfFovY = np.tan(cam.FovY / 2)
        base_x = tanHalfFovX * abs(znear_vis) * 2
        base_y = tanHalfFovY * abs(znear_vis) * 2
        
        if aspect_ratio > 1:  
            x_range = base_x * aspect_ratio
            y_range = base_y
        else:  
            x_range = base_x
            y_range = base_y / aspect_ratio
        
        nx, ny = 128, 128  
        x_vals = np.linspace(-x_range/2, x_range/2, nx)
        y_vals = np.linspace(-y_range/2, y_range/2, ny)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.full_like(X, znear_vis)
        
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        rotated_grid = (R @ grid_points.T).T + pos_raw
        rotated_grid_norm = (rotated_grid + translate) * scale
        
        vertexcolor = []
        for i in range(ny):
            for j in range(nx):
                r, g, b = img_array[i, j]
                vertexcolor.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
        
        triangles = []
        for i in range(ny-1):
            for j in range(nx-1):
                p0 = i*nx + j
                p1 = i*nx + j+1
                p2 = (i+1)*nx + j
                p3 = (i+1)*nx + j+1
                triangles.append([p0, p1, p2])
                triangles.append([p1, p3, p2])
        
        i_list, j_list, k_list = zip(*triangles)
        
        fig.add_trace(go.Mesh3d(
            x=rotated_grid_norm[:, 0],
            y=rotated_grid_norm[:, 1],
            z=rotated_grid_norm[:, 2],
            i=i_list,
            j=j_list,
            k=k_list,
            vertexcolor=vertexcolor,
            showlegend=False,
            hoverinfo='skip'
        ))

if all_x and all_y and all_z:
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    aspectratio = dict(x=x_range, y=y_range, z=z_range)
else:
    aspectratio = dict(x=1, y=1, z=1)

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='manual',
        aspectratio=aspectratio,
        camera=dict(projection=dict(type='orthographic'))
    ),
    width=1200,
    height=1000,
    title=f'Interactive Camera Positions with Wireframe Cuboids and Frustums (Selected {len(selected_indices)} Cameras, Step {step})'
)

fig.show() 
