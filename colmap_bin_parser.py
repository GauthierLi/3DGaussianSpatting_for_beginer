import os
import struct
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union

@dataclass
class Camera:
    """Camera parameters structure"""
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray
    
    def __post_init__(self):
        """Ensure params is numpy array"""
        if not isinstance(self.params, np.ndarray):
            self.params = np.array(self.params, dtype=np.float64)
    
    @property
    def focal_length(self) -> float:
        """Get focal length"""
        return float(self.params[0]) if len(self.params) > 0 else 0.0
    
    @property
    def principal_point(self) -> np.ndarray:
        """Get principal point coordinates"""
        if self.model_id == 0:  # SIMPLE_PINHOLE
            return self.params[1:3]
        elif self.model_id in [1, 4, 5, 6]:  # PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV
            return self.params[2:4]
        return np.array([0.0, 0.0])
    
    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get intrinsic matrix K"""
        K = np.zeros((3, 3))
        if self.model_id == 0:  # SIMPLE_PINHOLE
            f, cx, cy = self.params[0], self.params[1], self.params[2]
            K = np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0, 1]])
        elif self.model_id == 1:  # PINHOLE
            fx, fy, cx, cy = self.params[0], self.params[1], self.params[2], self.params[3]
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
        return K

@dataclass  
class Point2D:
    """2D feature point structure"""
    x: float
    y: float
    point3D_id: int
    
    @property
    def is_triangulated(self) -> bool:
        """Whether triangulated"""
        return self.point3D_id != -1
    
    @property
    def coords(self) -> np.ndarray:
        """Return coordinates as numpy array"""
        return np.array([self.x, self.y])

@dataclass
class Image:
    """Image information structure"""
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    points2D: List[Point2D]
    
    def __post_init__(self):
        """Ensure vectors are numpy arrays"""
        if not isinstance(self.qvec, np.ndarray):
            self.qvec = np.array(self.qvec, dtype=np.float64)
        if not isinstance(self.tvec, np.ndarray):
            self.tvec = np.array(self.tvec, dtype=np.float64)
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to rotation matrix - using numpy vectorized operations"""
        qw, qx, qy, qz = self.qvec
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        return R
    
    @property
    def pose_matrix(self) -> np.ndarray:
        """Get 4x4 pose matrix"""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.tvec
        return T
    
    @property
    def num_triangulated_points(self) -> int:
        """Number of triangulated feature points - using numpy vectorized operations"""
        point3D_ids = np.array([p.point3D_id for p in self.points2D])
        return np.sum(point3D_ids != -1)
    
    @property
    def points2D_coords(self) -> np.ndarray:
        """Return all 2D point coordinates as numpy array (N, 2)"""
        return np.array([[p.x, p.y] for p in self.points2D])
    
    @property
    def triangulated_mask(self) -> np.ndarray:
        """Return boolean mask of triangulated points"""
        return np.array([p.is_triangulated for p in self.points2D])

@dataclass
class TrackElement:
    """Track element structure"""
    image_id: int
    point2D_idx: int

@dataclass
class Point3D:
    """3D point structure"""
    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[TrackElement]
    
    def __post_init__(self):
        """Ensure coordinates and colors are numpy arrays"""
        if not isinstance(self.xyz, np.ndarray):
            self.xyz = np.array(self.xyz, dtype=np.float64)
        if not isinstance(self.rgb, np.ndarray):
            self.rgb = np.array(self.rgb, dtype=np.uint8)
    
    @property
    def track_length(self) -> int:
        """Track length (number of observations)"""
        return len(self.track)
    
    @property
    def position(self) -> np.ndarray:
        """Get 3D position"""
        return self.xyz
    
    @property
    def color_normalized(self) -> np.ndarray:
        """Get normalized color values [0, 1]"""
        return self.rgb.astype(np.float32) / 255.0

@dataclass
class ColmapReconstruction:
    """COLMAP reconstruction result structure"""
    cameras: Dict[int, Camera]
    images: Dict[int, Image]  
    points3D: Dict[int, Point3D]
    
    @property
    def num_cameras(self) -> int:
        return len(self.cameras)
    
    @property  
    def num_images(self) -> int:
        return len(self.images)
    
    @property
    def num_points3D(self) -> int:
        return len(self.points3D)
    
    @property
    def total_observations(self) -> int:
        track_lengths = np.array([point.track_length for point in self.points3D.values()])
        return int(np.sum(track_lengths))
    
    @property
    def mean_reprojection_error(self) -> float:
        """Mean reprojection error - using numpy vectorized operations"""
        if not self.points3D:
            return 0.0
        errors = np.array([point.error for point in self.points3D.values()])
        return float(np.mean(errors))
    
    @property
    def point_cloud_bounds(self) -> Dict[str, np.ndarray]:
        """Get point cloud bounds"""
        if not self.points3D:
            return {'min': np.array([0, 0, 0]), 'max': np.array([0, 0, 0])}
        
        points = np.array([point.xyz for point in self.points3D.values()])
        return {
            'min': np.min(points, axis=0),
            'max': np.max(points, axis=0),
            'center': np.mean(points, axis=0),
            'std': np.std(points, axis=0)
        }
    
    @property
    def point_cloud_array(self) -> np.ndarray:
        """Return numpy array of all 3D points (N, 3)"""
        return np.array([point.xyz for point in self.points3D.values()])
    
    @property
    def point_colors_array(self) -> np.ndarray:
        """Return numpy array of all point colors (N, 3)"""
        return np.array([point.rgb for point in self.points3D.values()])
    
    @property
    def reprojection_errors_array(self) -> np.ndarray:
        """Return numpy array of all reprojection errors (N,)"""
        return np.array([point.error for point in self.points3D.values()])
    
    def get_camera_model_distribution(self) -> Dict[str, int]:
        """Get camera model distribution"""
        model_names = [camera.model_name for camera in self.cameras.values()]
        unique_models, counts = np.unique(model_names, return_counts=True)
        return dict(zip(unique_models, counts))
    
    def get_cameras_array(self) -> Dict[str, np.ndarray]:
        """Return camera parameters as numpy arrays"""
        if not self.cameras:
            return {}
        
        # Get intrinsic matrices of all cameras
        intrinsics = []
        focal_lengths = []
        principal_points = []
        
        for camera in self.cameras.values():
            intrinsics.append(camera.intrinsic_matrix)
            focal_lengths.append(camera.focal_length)
            principal_points.append(camera.principal_point)
        
        return {
            'intrinsics': np.array(intrinsics),  # (N, 3, 3)
            'focal_lengths': np.array(focal_lengths),  # (N,)
            'principal_points': np.array(principal_points)  # (N, 2)
        }
    
    def get_poses_array(self) -> Dict[str, np.ndarray]:
        """Return numpy arrays of all image poses"""
        if not self.images:
            return {}
        
        pose_matrices = []
        rotations = []
        translations = []
        quaternions = []
        
        for image in self.images.values():
            pose_matrices.append(image.pose_matrix)
            rotations.append(image.rotation_matrix)
            translations.append(image.tvec)
            quaternions.append(image.qvec)
        
        return {
            'poses': np.array(pose_matrices),  # (N, 4, 4)
            'rotations': np.array(rotations),  # (N, 3, 3)
            'translations': np.array(translations),  # (N, 3)
            'quaternions': np.array(quaternions)  # (N, 4)
        }

class ColmapBinParser:
    """COLMAP binary file parser"""
    
    CAMERA_MODELS = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE", 
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE"
    }
    
    def __init__(self, bin_folder: str):
        self.bin_folder = bin_folder
        self.reconstruction = None

    def parse(self) -> ColmapReconstruction:
        """Parse COLMAP binary files and return structured data"""
        cameras = self._parse_cameras_bin(os.path.join(self.bin_folder, 'cameras.bin'))
        images = self._parse_images_bin(os.path.join(self.bin_folder, 'images.bin'))
        points3D = self._parse_points3D_bin(os.path.join(self.bin_folder, 'points3D.bin'))
        
        self.reconstruction = ColmapReconstruction(cameras, images, points3D)
        return self.reconstruction

    def _parse_cameras_bin(self, path: str) -> Dict[int, Camera]:
        """Parse camera binary file - optimized with numpy"""
        cameras = {}
        if not os.path.exists(path):
            return cameras
            
        with open(path, 'rb') as f:
            num_cameras = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack('<I', f.read(4))[0]
                model_id = struct.unpack('<i', f.read(4))[0]
                width = struct.unpack('<Q', f.read(8))[0]
                height = struct.unpack('<Q', f.read(8))[0]
                num_params = self._get_num_params(model_id)
                
                params_data = f.read(8 * num_params)
                params = np.frombuffer(params_data, dtype='<f8')
                
                model_name = self.CAMERA_MODELS.get(model_id, f"UNKNOWN_MODEL_{model_id}")
                
                cameras[camera_id] = Camera(
                    camera_id=camera_id,
                    model_id=model_id,
                    model_name=model_name,
                    width=width,
                    height=height,
                    params=params
                )
        return cameras

    def _parse_images_bin(self, path: str) -> Dict[int, Image]:
        """Parse image binary file - optimized with numpy"""
        images = {}
        if not os.path.exists(path):
            return images
            
        with open(path, 'rb') as f:
            num_images = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack('<I', f.read(4))[0]
                
                qvec_data = f.read(32)
                qvec = np.frombuffer(qvec_data, dtype='<f8')
                
                tvec_data = f.read(24)
                tvec = np.frombuffer(tvec_data, dtype='<f8')
                
                camera_id = struct.unpack('<I', f.read(4))[0]
                
                name = b''
                while True:
                    c = f.read(1)
                    if c == b'\x00':
                        break
                    name += c
                name = name.decode('utf-8')
                
                num_points2D = struct.unpack('<Q', f.read(8))[0]
                if num_points2D > 0:
                    points2D = []
                    for _ in range(num_points2D):
                        x, y, point3D_id = struct.unpack('<2dQ', f.read(24))
                        if point3D_id == 18446744073709551615:
                            point3D_id = -1
                        points2D.append(Point2D(x, y, point3D_id))
                else:
                    points2D = []
                
                images[image_id] = Image(
                    image_id=image_id,
                    name=name,
                    camera_id=camera_id,
                    qvec=qvec,
                    tvec=tvec,
                    points2D=points2D
                )
        return images

    def _parse_points3D_bin(self, path: str) -> Dict[int, Point3D]:
        """Parse 3D points binary file - optimized with numpy"""
        points3D = {}
        if not os.path.exists(path):
            return points3D
            
        with open(path, 'rb') as f:
            num_points = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack('<Q', f.read(8))[0]
                
                xyz_data = f.read(24)
                xyz = np.frombuffer(xyz_data, dtype='<f8')
                
                rgb_data = f.read(3)
                rgb = np.frombuffer(rgb_data, dtype=np.uint8)
                
                error = struct.unpack('<d', f.read(8))[0]
                track_length = struct.unpack('<Q', f.read(8))[0]
                
                if track_length > 0:
                    track_data = f.read(8 * track_length)
                    track_array = np.frombuffer(track_data, dtype='<u4').reshape(-1, 2)
                    
                    track = []
                    for i in range(track_length):
                        image_id, point2D_idx = track_array[i]
                        track.append(TrackElement(int(image_id), int(point2D_idx)))
                else:
                    track = []
                
                points3D[point3D_id] = Point3D(
                    point3D_id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    track=track
                )
        return points3D

    def _get_num_params(self, model_id: int) -> int:
        """Get number of parameters based on camera model ID"""
        model_params = {
            0: 3,   # SIMPLE_PINHOLE: f, cx, cy
            1: 4,   # PINHOLE: fx, fy, cx, cy
            2: 4,   # SIMPLE_RADIAL: f, cx, cy, k1
            3: 5,   # RADIAL: f, cx, cy, k1, k2
            4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
            6: 12,  # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            7: 5,   # FOV: fx, fy, cx, cy, omega
            8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k1
            9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
            10: 8,  # THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2
        }
        return model_params.get(model_id, 4)
    
    @property
    def cameras(self) -> Optional[Dict[int, Any]]:
        """Backward compatible camera attribute"""
        if self.reconstruction is None:
            return None
        return {cam_id: {
            'model_id': cam.model_id,
            'width': cam.width, 
            'height': cam.height,
            'params': cam.params.tolist()
        } for cam_id, cam in self.reconstruction.cameras.items()}
    
    @property
    def images(self) -> Optional[Dict[int, Any]]:
        """Backward compatible image attribute"""
        if self.reconstruction is None:
            return None
        return {img_id: {
            'name': img.name,
            'camera_id': img.camera_id,
            'qvec': img.qvec.tolist(),
            'tvec': img.tvec.tolist(),
            'points2D': [(p.x, p.y, p.point3D_id) for p in img.points2D]
        } for img_id, img in self.reconstruction.images.items()}
    
    @property 
    def points3D(self) -> Optional[Dict[int, Any]]:
        """Backward compatible 3D points attribute"""
        if self.reconstruction is None:
            return None
        return {pt_id: {
            'xyz': pt.xyz.tolist(),
            'rgb': pt.rgb.tolist(),
            'error': pt.error,
            'track': [(t.image_id, t.point2D_idx) for t in pt.track]
        } for pt_id, pt in self.reconstruction.points3D.items()}

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse COLMAP binary files and output reconstruction statistics')
    parser.add_argument('sparse_path', type=str, 
                       help='COLMAP sparse reconstruction folder path (containing cameras.bin, images.bin, points3D.bin)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Display verbose output information')
    
    return parser.parse_args()

def validate_path(sparse_path: str) -> None:
    """Validate the sparse reconstruction path and required files"""
    import os
    
    if not os.path.exists(sparse_path):
        print(f"Error: Path '{sparse_path}' does not exist")
        exit(1)
    
    if not os.path.isdir(sparse_path):
        print(f"Error: '{sparse_path}' is not a directory")
        exit(1)
    
    required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(sparse_path, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        print(f"Error: Missing files in path '{sparse_path}': {', '.join(missing_files)}")
        exit(1)

def print_reconstruction_statistics(reconstruction: ColmapReconstruction, verbose: bool = False, parser: 'ColmapBinParser' = None) -> None:
    """Print reconstruction statistics"""
    print(f"Reconstruction Statistics:")
    print(f"  Number of cameras: {reconstruction.num_cameras}")
    print(f"  Number of images: {reconstruction.num_images}")
    print(f"  Number of 3D points: {reconstruction.num_points3D}")
    print(f"  Mean reprojection error: {reconstruction.mean_reprojection_error:.4f}")
    print(f"  Camera model distribution: {reconstruction.get_camera_model_distribution()}")
    
    if verbose:
        point_cloud = reconstruction.point_cloud_array
        point_colors = reconstruction.point_colors_array
        errors = reconstruction.reprojection_errors_array
        bounds = reconstruction.point_cloud_bounds
        
        print(f"\nPoint Cloud Statistics:")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Point cloud bounds: min={bounds['min']}, max={bounds['max']}")
        print(f"  Point cloud center: {bounds['center']}")
        print(f"  Reprojection error: min={np.min(errors):.4f}, max={np.max(errors):.4f}, std={np.std(errors):.4f}")
        
        camera_arrays = reconstruction.get_cameras_array()
        pose_arrays = reconstruction.get_poses_array()
        
        if camera_arrays:
            print(f"\nCamera Arrays:")
            print(f"  Intrinsic matrix shape: {camera_arrays['intrinsics'].shape}")
            print(f"  Focal length range: {np.min(camera_arrays['focal_lengths']):.2f} - {np.max(camera_arrays['focal_lengths']):.2f}")
        
        if pose_arrays:
            print(f"\nPose Arrays:")
            print(f"  Pose matrix shape: {pose_arrays['poses'].shape}")
            print(f"  Rotation matrix shape: {pose_arrays['rotations'].shape}")
            print(f"  Translation vector range: {np.min(pose_arrays['translations'], axis=0)} - {np.max(pose_arrays['translations'], axis=0)}")
        
        for camera_id, camera in reconstruction.cameras.items():
            print(f"\nCamera {camera_id}: {camera.model_name}")
            print(f"  Focal length: {camera.focal_length:.2f}")
            print(f"  Principal point: {camera.principal_point}")
            print(f"  Intrinsic matrix:\n{camera.intrinsic_matrix}")
            break
        
        for image_id, image in reconstruction.images.items():
            print(f"\nImage {image_id}: {image.name}")
            print(f"  Triangulated points: {image.num_triangulated_points}")
            print(f"  Rotation matrix shape: {image.rotation_matrix.shape}")
            print(f"  2D point coordinates shape: {image.points2D_coords.shape}")
            break
        
        if parser:
            print(f"\nBackward Compatible API:")
            print(f"  Number of cameras: {len(parser.cameras)}")
            print(f"  Number of images: {len(parser.images)}")
            print(f"  Number of 3D points: {len(parser.points3D)}")

def main():
    """Main function to run the COLMAP binary parser"""
    args = parse_arguments()
    validate_path(args.sparse_path)
    
    print(f"Parsing COLMAP reconstruction files: {args.sparse_path}")
    
    try:
        parser = ColmapBinParser(args.sparse_path)
        reconstruction = parser.parse()
    except Exception as e:
        print(f"Error: Exception occurred while parsing files: {e}")
        exit(1)
    
    print_reconstruction_statistics(reconstruction, args.verbose, parser)

# Usage example:
if __name__ == "__main__":
    main()