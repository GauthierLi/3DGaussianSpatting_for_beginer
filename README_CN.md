
# 3D Gaussian Splatting 辅助工具（COLMAP 解析与可视化）

本目录包含用于解析 COLMAP 稀疏重建并进行可视化的轻量工具，目的是帮助在研究与调试过程中理解 3D Gaussian Splatting（3dgs）相关的几何与外观表示。

## 依赖

安装用于可视化和数学计算的最小依赖：

```bash
pip install numpy scipy matplotlib
```

## 球谐函数可视化（`spherical_harmonics.py`）

### 主要功能解析
- 使用 scipy 的 sph_harm 计算球谐基函数 Y_l^m(theta, phi)。
- 提供实部、虚部、模和相位等常用量的接口，便于用于光照或基函数分析。
- 基于 matplotlib 的交互式 3D 渲染：支持通过滑块调整 l / m，并切换可视化模式（模、实部、虚部、相位）。
- 支持将当前视图保存为图片，便于记录实验结果或生成报告图。

### 快速上手（Quick start）

命令行运行（打开交互式窗口）：

```bash
python spherical_harmonics.py
```

编程调用示例：

```python
from spherical_harmonics import SphericalHarmonics, SphericalHarmonicsVisualizer

# 计算单点球谐值
sh = SphericalHarmonics(l=2, m=1)
value = sh(theta=0.5, phi=1.0)

# 打开交互式可视化
vis = SphericalHarmonicsVisualizer(l=2, m=1, resolution=80)
vis.visualize(interactive=True)
```

——

## 点云可视化（`point_cloud_visualizer.py`）

### 主要功能解析
- 解析 COLMAP 的二进制输出（`cameras.bin`, `images.bin`, `points3D.bin`），并暴露相机、图像和三维点数据结构。
- 交互式点云渲染：支持颜色显示、点采样率调整、点大小与透明度设置。
- 相机可视化：显示相机位置、坐标轴和视锥（frustums），并支持采样显示以提高性能。
- 输出基础统计信息：点数、相机数、重投影误差等，便于快速评估重建质量。

### 快速上手（Quick start）

可视化 COLMAP sparse 文件夹：

```bash
python point_cloud_visualizer.py path/to/sparse_folder
```

常用命令行选项：

```bash
# 基本可视化（默认使用 10% 点云采样）
python point_cloud_visualizer.py sparse/0

# 使用更多点（5% 采样）和更大的点尺寸
python point_cloud_visualizer.py sparse/0 --sample-rate 0.05 --point-size 2

# 只显示相机，最少点云
python point_cloud_visualizer.py sparse/0 --sample-rate 0.001

# 隐藏相机视锥体
python point_cloud_visualizer.py sparse/0 --no-frustums

# 隐藏相机，只显示点云
python point_cloud_visualizer.py sparse/0 --no-cameras

# 显示所有相机（默认只显示 20 个采样相机）
python point_cloud_visualizer.py sparse/0 --max-cameras 0

# 调整视锥体大小
python point_cloud_visualizer.py sparse/0 --frustum-scale 0.2

# 只查看统计信息（不显示可视化）
python point_cloud_visualizer.py sparse/0 --stats-only
```

编程调用示例：

```python
from point_cloud_visualizer import PointCloudVisualizer

v = PointCloudVisualizer('path/to/sparse_folder')
v.load_data()
v.print_statistics()
v.interactive_visualize(point_sample_rate=0.1, show_cameras=True)
```

### 命令行参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `sparse_path` | str | - | COLMAP 稀疏重建文件夹路径 |
| `--sample-rate` | float | 0.1 | 点云采样率 (0.0-1.0) |
| `--no-cameras` | flag | False | 隐藏相机位置 |
| `--no-frustums` | flag | False | 隐藏相机视锥体 |
| `--max-cameras` | int | 20 | 最大显示相机数量 (0 表示全部) |
| `--point-size` | float | 1.0 | 点渲染大小 |
| `--frustum-scale` | float | 0.5 | 相机视锥体缩放因子 |
| `--stats-only` | flag | False | 只打印统计信息，不显示可视化 |

### 使用提示
- 若在 SSH 下无法显示 GUI，请使用 X11 转发或在本地带显示的机器上运行。
- 对于非常大的点云，降低采样率（例如 0.01）能显著提升交互响应。

---

*注：这些工具为轻量级调试和研究检查设计，不适用于生产渲染。*
