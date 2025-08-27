# 3D Gaussian Splatting Tools (COLMAP parsing & visualization)

This folder contains lightweight utilities for parsing COLMAP sparse reconstructions and performing visualization. The purpose is to help inspect and understand geometric and appearance representations used in 3D Gaussian Splatting (3dgs) research and related pipelines during research and debugging.

## Dependencies

Install minimal dependencies for visualization and mathematical computation:

```bash
pip install numpy scipy matplotlib
```

## Spherical Harmonics visualizer

### Main features
- Compute spherical harmonics Y_l^m for arbitrary degree l and order m using scipy's special functions.
- Expose real, imaginary, magnitude and phase of basis functions.
- Interactive 3D rendering with sliders for l and m and a mode selector (magnitude / real / imag / phase).
- Save current view as an image.

### Quick start

Main dependency (if not installed):
```bash
pip install numpy scipy matplotlib
```

Run the interactive visualizer:
```bash
python spherical_harmonics.py
```

Programmatic quick example:
```python
from spherical_harmonics import SphericalHarmonics, SphericalHarmonicsVisualizer

sh = SphericalHarmonics(l=2, m=1)
value = sh(0.5, 1.0)  # theta, phi in radians

vis = SphericalHarmonicsVisualizer(l=2, m=1, resolution=80)
vis.visualize(interactive=True)
```

## Point cloud visualizer (`point_cloud_visualizer.py`)

### Main features
- Parse COLMAP binary outputs (`cameras.bin`, `images.bin`, `points3D.bin`) and expose camera, image, and 3D point data structures.
- Interactive point cloud rendering: supports color display, sampling rate adjustment, point size and transparency settings.
- Camera visualization: display camera positions, coordinate axes and frustums, with sampling display for better performance.
- Output basic statistics: number of points, cameras, reprojection errors, etc., for quick reconstruction quality assessment.

### Quick start

Visualize a COLMAP sparse folder:

```bash
python point_cloud_visualizer.py path/to/sparse_folder
```

Common command line options:

```bash
# Basic visualization (10% point sampling by default)
python point_cloud_visualizer.py sparse/0

# Use more points (5% sampling) with larger point size
python point_cloud_visualizer.py sparse/0 --sample-rate 0.05 --point-size 2

# Show only cameras, minimal point cloud
python point_cloud_visualizer.py sparse/0 --sample-rate 0.001

# Hide camera frustums
python point_cloud_visualizer.py sparse/0 --no-frustums

# Hide cameras, show only point cloud
python point_cloud_visualizer.py sparse/0 --no-cameras

# Display all cameras (default shows 20 sampled cameras)
python point_cloud_visualizer.py sparse/0 --max-cameras 0

# Adjust frustum size
python point_cloud_visualizer.py sparse/0 --frustum-scale 0.2

# Statistics only (no visualization)
python point_cloud_visualizer.py sparse/0 --stats-only
```

Programmatic usage:

```python
from point_cloud_visualizer import PointCloudVisualizer

v = PointCloudVisualizer('path/to/sparse_folder')
v.load_data()
v.print_statistics()
v.interactive_visualize(point_sample_rate=0.1, show_cameras=True)
```

### Command line parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sparse_path` | str | - | Path to COLMAP sparse reconstruction folder |
| `--sample-rate` | float | 0.1 | Point cloud sampling rate (0.0-1.0) |
| `--no-cameras` | flag | False | Hide camera positions |
| `--no-frustums` | flag | False | Hide camera frustums |
| `--max-cameras` | int | 20 | Maximum cameras to display (0 for all) |
| `--point-size` | float | 1.0 | Point rendering size |
| `--frustum-scale` | float | 0.5 | Camera frustum scale factor |
| `--stats-only` | flag | False | Only print statistics, no visualization |

### Usage tips
- If GUI doesn't display over SSH, use X11 forwarding or run locally on a machine with display.
- For very large point clouds, reduce sampling rate (e.g. 0.01) to significantly improve interaction response.

---

*Note: These tools are designed for lightweight debugging and research inspection, not for production rendering.*
