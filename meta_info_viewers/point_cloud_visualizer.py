import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict, Any
import os
import argparse

from colmap_bin_parser import ColmapBinParser, ColmapReconstruction, Camera, Image, Point3D


class PointCloudVisualizer:
    """Interactive point cloud visualization tool"""
    
    def __init__(self, sparse_folder_path: str):
        """
        Initialize visualizer
        
        Args:
            sparse_folder_path: Path to COLMAP sparse reconstruction folder
        """
        self.sparse_folder_path = sparse_folder_path
        self.parser = ColmapBinParser(sparse_folder_path)
        self.reconstruction: Optional[ColmapReconstruction] = None
        
    def load_data(self) -> None:
        """Load COLMAP reconstruction data"""
        print(f"Loading COLMAP reconstruction from: {self.sparse_folder_path}")
        self.reconstruction = self.parser.parse()
        print(f"Loaded {self.reconstruction.num_points3D} 3D points, "
              f"{self.reconstruction.num_cameras} cameras, "
              f"{self.reconstruction.num_images} images")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reconstruction statistics"""
        if self.reconstruction is None:
            return {}
        
        bounds = self.reconstruction.point_cloud_bounds
        return {
            'num_points3D': self.reconstruction.num_points3D,
            'num_cameras': self.reconstruction.num_cameras,
            'num_images': self.reconstruction.num_images,
            'mean_reprojection_error': self.reconstruction.mean_reprojection_error,
            'point_cloud_bounds': bounds,
            'total_observations': self.reconstruction.total_observations
        }
    
    def sample_point_cloud(self, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample point cloud to improve rendering performance
        
        Args:
            sample_rate: Sampling rate, 1.0 means using all points
            
        Returns:
            Sampled point coordinates and colors
        """
        if self.reconstruction is None:
            raise ValueError("Please call load_data() first to load data")
        
        points = self.reconstruction.point_cloud_array
        colors = self.reconstruction.point_colors_array
        
        if sample_rate < 1.0 and len(points) > 1000:
            n_samples = int(len(points) * sample_rate)
            indices = np.random.choice(len(points), n_samples, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        return points, colors
    
    def create_camera_frustum(self, camera: Camera, image: Image, 
                             frustum_scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Create 3D wireframe of camera frustum"""
        K = camera.intrinsic_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        w, h = camera.width, camera.height
        
        cam_center = np.array([0, 0, 0])
        
        corners_cam = np.array([
            [(0 - cx) / fx, (0 - cy) / fy, 1],      
            [(w - cx) / fx, (0 - cy) / fy, 1],       
            [(w - cx) / fx, (h - cy) / fy, 1],       
            [(0 - cx) / fx, (h - cy) / fy, 1]        
        ]) * frustum_scale
        
        frustum_points_cam = np.vstack([cam_center, corners_cam])
        
        R = image.rotation_matrix
        t = image.tvec
        
        camera_world_pos = -R.T @ t
        
        frustum_points_world = []
        for point in frustum_points_cam:
            world_point = R.T @ point + camera_world_pos
            frustum_points_world.append(world_point)
        
        frustum_points_world = np.array(frustum_points_world)
        
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  
            [1, 2], [2, 3], [3, 4], [4, 1]   
        ]
        
        return frustum_points_world, lines
    
    def visualize(self, point_sample_rate: float = 0.1, 
                 show_cameras: bool = True,
                 show_frustums: bool = True,
                 max_cameras: int = 20,
                 point_size: float = 1.0,
                 frustum_scale: float = 0.1) -> None:
        """
        Create basic 3D visualization
        
        Args:
            point_sample_rate: Point cloud sampling rate
            show_cameras: Whether to show camera positions
            show_frustums: Whether to show camera frustums
            max_cameras: Maximum number of cameras to display, 0 for all
            point_size: Point size for rendering
            frustum_scale: Frustum scale factor
        """
        if self.reconstruction is None:
            raise ValueError("Please call load_data() first to load data")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 显示点云
        points, colors = self.sample_point_cloud(point_sample_rate)
        colors_normalized = colors / 255.0
        
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=colors_normalized, s=point_size, alpha=0.6)
        
        print(f"Displaying {len(points)} points (sampled at {point_sample_rate:.1%})")
        
        if show_cameras:
            image_items = list(self.reconstruction.images.items())
            total_cameras = len(image_items)
            
            if max_cameras <= 0 or total_cameras <= max_cameras:
                selected_items = image_items
                print(f"Displaying all {len(selected_items)} cameras")
            else:
                np.random.seed(42)  
                selected_indices = np.random.choice(total_cameras, max_cameras, replace=False)
                selected_items = [image_items[i] for i in sorted(selected_indices)]
                print(f"Displaying {len(selected_items)} cameras (sampled from {total_cameras})")
            
            camera_positions = []
            
            first_camera = True
            for image_id, image in selected_items:
                camera = self.reconstruction.cameras[image.camera_id]
                R = image.rotation_matrix
                t = image.tvec
                cam_pos = -R.T @ t
                camera_positions.append(cam_pos)
                
                ax.scatter(*cam_pos, color='red', s=point_size, marker='o', 
                          label='Camera' if first_camera else "")
                
                if show_frustums:
                    frustum_points, lines = self.create_camera_frustum(camera, image, frustum_scale)
                    for line_idx in lines:
                        p1, p2 = frustum_points[line_idx[0]], frustum_points[line_idx[1]]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               'k--', alpha=0.7, linewidth=1)
                
                first_camera = False
            
            camera_positions = np.array(camera_positions)
            print(f"Camera position bounds:")
            print(f"  X: [{camera_positions[:, 0].min():.3f}, {camera_positions[:, 0].max():.3f}]")
            print(f"  Y: [{camera_positions[:, 1].min():.3f}, {camera_positions[:, 1].max():.3f}]")
            print(f"  Z: [{camera_positions[:, 2].min():.3f}, {camera_positions[:, 2].max():.3f}]")
        
        ax.set_axis_off()  
        
        ax.set_title(f'COLMAP 3D Reconstruction\n'
                    f'{self.reconstruction.num_points3D} points, '
                    f'{self.reconstruction.num_images} cameras\n'
                    f'Mean reprojection error: {self.reconstruction.mean_reprojection_error:.4f}')
        
        bounds = self.reconstruction.point_cloud_bounds
        max_range = np.max(bounds['max'] - bounds['min']) / 2.0
        center = bounds['center']
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.legend()
        
        ax.mouse_init()  
        
        ax.set_autoscale_on(True)  
        
        def zoom_callback(event):
            if event.inaxes == ax:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                z_center = (zlim[0] + zlim[1]) / 2
                
                x_range = xlim[1] - xlim[0]
                y_range = ylim[1] - ylim[0] 
                z_range = zlim[1] - zlim[0]
                
                if event.button == 'up':
                    scale = 0.8
                elif event.button == 'down':
                    scale = 1.25
                else:
                    return
                
                new_x_range = x_range * scale
                new_y_range = y_range * scale
                new_z_range = z_range * scale
                
                ax.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
                ax.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
                ax.set_zlim(z_center - new_z_range/2, z_center + new_z_range/2)
                
                plt.draw()
        
        fig.canvas.mpl_connect('scroll_event', zoom_callback)
        
        plt.tight_layout()
        
        plt.draw()
        plt.show(block=False)  
    
    def interactive_visualize(self, **kwargs) -> None:
        """Create interactive 3D visualization"""
        print("Creating interactive 3D visualization...")
        print("=== Interactive Controls ===")
        print("🖱️  Rotate view: Left click + drag")
        print("🖱️  Pan view: Right click + drag") 
        print("🖱️  Zoom: Mouse wheel (up to zoom in, down to zoom out)")
        print("🖱️  Alternative zoom: Middle click + drag up/down")
        print("⌨️  Reset view: Press 'r' key")
        print("❌ Close window: Click window's X button")
        print("=" * 30)
        
        plt.ion()
        
        self.visualize(**kwargs)
        
        try:
            input("Window opened. You can interact with the visualization. Press Enter to close...")
        except KeyboardInterrupt:
            pass
        finally:
            plt.close('all')
    
    def print_statistics(self) -> None:
        """Print reconstruction statistics"""
        if self.reconstruction is None:
            print("No data loaded. Please call load_data() first.")
            return
        
        stats = self.get_statistics()
        bounds = stats['point_cloud_bounds']
        
        print("\n" + "="*50)
        print("COLMAP RECONSTRUCTION STATISTICS")
        print("="*50)
        print(f"Number of 3D points: {stats['num_points3D']:,}")
        print(f"Number of cameras: {stats['num_cameras']}")
        print(f"Number of images: {stats['num_images']}")
        print(f"Total observations: {stats['total_observations']:,}")
        print(f"Mean reprojection error: {stats['mean_reprojection_error']:.4f}")
        
        print(f"\nPoint cloud bounds:")
        print(f"  X: [{bounds['min'][0]:.3f}, {bounds['max'][0]:.3f}]")
        print(f"  Y: [{bounds['min'][1]:.3f}, {bounds['max'][1]:.3f}]") 
        print(f"  Z: [{bounds['min'][2]:.3f}, {bounds['max'][2]:.3f}]")
        print(f"  Center: [{bounds['center'][0]:.3f}, {bounds['center'][1]:.3f}, {bounds['center'][2]:.3f}]")
        
        model_dist = self.reconstruction.get_camera_model_distribution()
        print(f"\nCamera models:")
        for model, count in model_dist.items():
            print(f"  {model}: {count}")
        
        print("="*50)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='COLMAP Point Cloud Visualizer')
    parser.add_argument('sparse_path', type=str,
                       help='Path to COLMAP sparse reconstruction folder')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                       help='Point cloud sampling rate (default: 0.1)')
    parser.add_argument('--no-cameras', action='store_true',
                       help='Hide camera positions')
    parser.add_argument('--no-frustums', action='store_true', 
                       help='Hide camera frustums')
    parser.add_argument('--max-cameras', type=int, default=20,
                       help='Maximum number of cameras to display (default: 20, 0 for all)')
    parser.add_argument('--point-size', type=float, default=1.0,
                       help='Point size for rendering (default: 1.0)')
    parser.add_argument('--frustum-scale', type=float, default=0.5,
                       help='Camera frustum scale factor (default: 0.5)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics, do not visualize')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    if not os.path.exists(args.sparse_path):
        print(f"Error: Path '{args.sparse_path}' does not exist")
        return 1
    
    try:
        visualizer = PointCloudVisualizer(args.sparse_path)
        visualizer.load_data()
        
        visualizer.print_statistics()
        
        if not args.stats_only:
            viz_params = {
                'point_sample_rate': args.sample_rate,
                'show_cameras': not args.no_cameras,
                'show_frustums': not args.no_frustums,
                'max_cameras': args.max_cameras,
                'point_size': args.point_size,
                'frustum_scale': args.frustum_scale
            }
            
            visualizer.interactive_visualize(**viz_params)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
