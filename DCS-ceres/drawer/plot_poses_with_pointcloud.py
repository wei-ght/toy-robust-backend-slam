#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def read_g2o_file(g2o_path):
    """
    Read g2o file and extract poses and point clouds
    
    Returns:
        poses: dict {id: [x, y, theta]}
        points: dict {pose_id: [(x1, y1), (x2, y2), ...]}  
        edges: list of edge connections
    """
    poses = {}
    points = {}
    edges = []
    
    try:
        with open(g2o_path, 'r') as f:
            current_pose_id = None
            reading_points = False
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                if parts[0] == 'VERTEX_SE2':
                    # VERTEX_SE2 id x y theta
                    if len(parts) >= 5:
                        vertex_id = int(parts[1])
                        x, y, theta = float(parts[2]), float(parts[3]), float(parts[4])
                        poses[vertex_id] = [x, y, theta]
                        current_pose_id = vertex_id
                        reading_points = False
                
                elif parts[0] == 'VERTEX_XY':
                    # VERTEX_XY id x y (point landmark)
                    if len(parts) >= 4:
                        point_id = int(parts[1])
                        x, y = float(parts[2]), float(parts[3])
                        if 'standalone' not in points:
                            points['standalone'] = []
                        points['standalone'].append((x, y))
                
                elif parts[0] == 'EDGE_SE2':
                    # EDGE_SE2 id1 id2 dx dy dtheta info(6)
                    if len(parts) >= 5:
                        id1, id2 = int(parts[1]), int(parts[2])
                        edges.append((id1, id2))
                
                elif line.startswith('# POINTCLOUD_DATA_START'):
                    # Extract pose ID from comment
                    if len(parts) >= 3:
                        current_pose_id = int(parts[2])
                    reading_points = True
                
                elif line.startswith('# POINTS_DATA') and current_pose_id is not None:
                    # Parse point data: # POINTS_DATA x1 y1 x2 y2 x3 y3 ...
                    if len(parts) >= 3:
                        point_coords = [float(x) for x in parts[2:]]
                        if len(point_coords) % 2 == 0:  # Make sure we have pairs
                            pose_points = []
                            for i in range(0, len(point_coords), 2):
                                pose_points.append((point_coords[i], point_coords[i+1]))
                            
                            if current_pose_id not in points:
                                points[current_pose_id] = []
                            points[current_pose_id].extend(pose_points)
                        
    except Exception as e:
        print(f"Error reading g2o file {g2o_path}: {e}")
    
    return poses, points, edges

def read_poses_file(poses_path):
    """
    Read poses from text file (format: id x y theta)
    
    Returns:
        dict {id: [x, y, theta]}
    """
    poses = {}
    try:
        with open(poses_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    vertex_id = int(parts[0])
                    x, y, theta = float(parts[1]), float(parts[2]), float(parts[3])
                    poses[vertex_id] = [x, y, theta]
    except Exception as e:
        print(f"Error reading poses file {poses_path}: {e}")
    
    return poses

def transform_points_to_global(poses, local_points):
    """
    Transform local points to global coordinate system using optimized poses
    
    Args:
        poses: dict {pose_id: [x, y, theta]}
        local_points: dict {pose_id: [(x1, y1), (x2, y2), ...]}
    
    Returns:
        global_points: list of (global_x, global_y) tuples
    """
    global_points = []
    
    for pose_id, points_list in local_points.items():
        if pose_id == 'standalone':
            # Standalone points (already in global coordinates)
            global_points.extend(points_list)
        elif pose_id in poses:
            pose_x, pose_y, pose_theta = poses[pose_id]
            
            # Transform points from local to global coordinates
            cos_theta = np.cos(pose_theta)
            sin_theta = np.sin(pose_theta)
            
            for local_x, local_y in points_list:
                global_x = pose_x + local_x * cos_theta - local_y * sin_theta
                global_y = pose_y + local_x * sin_theta + local_y * cos_theta
                global_points.append((global_x, global_y))
    
    return global_points

def plot_poses_with_pointcloud(g2o_path=None, initial_poses_path=None, optimized_poses_path=None, 
                              output_path=None, show_pointcloud=True, show_edges=True):
    """
    Plot optimized poses with pointcloud overlay
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Read original g2o data
    original_poses = {}
    original_points = {}
    edges = []
    
    if g2o_path and os.path.exists(g2o_path):
        print(f"Reading original g2o file: {g2o_path}")
        original_poses, original_points, edges = read_g2o_file(g2o_path)
        print(f"Found {len(original_poses)} poses, {len(original_points)} points, {len(edges)} edges")
    
    # Read initial poses if available
    initial_poses = {}
    if initial_poses_path and os.path.exists(initial_poses_path):
        print(f"Reading initial poses: {initial_poses_path}")
        initial_poses = read_poses_file(initial_poses_path)
        print(f"Found {len(initial_poses)} initial poses")
    
    # Read optimized poses
    optimized_poses = {}
    if optimized_poses_path and os.path.exists(optimized_poses_path):
        print(f"Reading optimized poses: {optimized_poses_path}")
        optimized_poses = read_poses_file(optimized_poses_path)
        print(f"Found {len(optimized_poses)} optimized poses")
    
    # Plot pose trajectories
    if initial_poses:
        pose_ids = sorted(initial_poses.keys())
        init_x = [initial_poses[pid][0] for pid in pose_ids]
        init_y = [initial_poses[pid][1] for pid in pose_ids]
        ax.plot(init_x, init_y, 'g-', linewidth=2, alpha=0.7, label='Initial Trajectory')
    
    if optimized_poses:
        pose_ids = sorted(optimized_poses.keys())
        opt_x = [optimized_poses[pid][0] for pid in pose_ids]
        opt_y = [optimized_poses[pid][1] for pid in pose_ids]
        ax.plot(opt_x, opt_y, 'b-', linewidth=2, alpha=0.8, label='Optimized Trajectory')
        
        # Draw pose orientations (arrows)
        for i, pid in enumerate(pose_ids[::max(1, len(pose_ids)//20)]):  # Show every N-th pose
            x, y, theta = optimized_poses[pid]
            dx = 2.0 * np.cos(theta)  # Arrow length
            dy = 2.0 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, 
                    fc='blue', ec='blue', alpha=0.6, linewidth=1)
    
    # Plot edges (loop closures)
    if show_edges and edges and optimized_poses:
        print(f"Drawing {len(edges)} edges")
        for id1, id2 in edges:
            if id1 in optimized_poses and id2 in optimized_poses:
                x1, y1 = optimized_poses[id1][0], optimized_poses[id1][1] 
                x2, y2 = optimized_poses[id2][0], optimized_poses[id2][1]
                # Only draw if not consecutive (likely loop closure)
                if abs(id1 - id2) > 1:
                    ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.3, linewidth=0.5)
    
    # Plot pointcloud
    if show_pointcloud and original_points:
        total_points = sum(len(points_list) for points_list in original_points.values())
        print(f"Transforming and plotting {total_points} points from {len(original_points)} poses")
        
        # Use optimized poses if available, otherwise original poses
        transform_poses = optimized_poses if optimized_poses else original_poses
        
        if transform_poses:
            # Transform points to global coordinates using optimized poses
            global_points = transform_points_to_global(transform_poses, original_points)
            
            if global_points:
                point_x = [p[0] for p in global_points]
                point_y = [p[1] for p in global_points]
                ax.scatter(point_x, point_y, c='red', s=0.5, alpha=0.6, 
                          label=f'Point Cloud ({len(global_points)} points)')
    
    # Formatting
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('SLAM Results: Poses and Point Cloud')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    # Add statistics text
    stats_text = []
    if original_poses:
        stats_text.append(f"Original poses: {len(original_poses)}")
    if optimized_poses:
        stats_text.append(f"Optimized poses: {len(optimized_poses)}")  
    if original_points:
        total_points = sum(len(points_list) for points_list in original_points.values())
        stats_text.append(f"Point cloud: {total_points} points")
    if edges:
        stats_text.append(f"Edges: {len(edges)}")
    
    if stats_text:
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot SLAM poses with pointcloud overlay')
    parser.add_argument('--g2o', help='Path to original g2o file')
    parser.add_argument('--initial-poses', help='Path to initial poses file')
    parser.add_argument('--optimized-poses', help='Path to optimized poses file')
    parser.add_argument('--output', '-o', help='Output PNG file path')
    parser.add_argument('--no-pointcloud', action='store_true', help='Skip pointcloud visualization')
    parser.add_argument('--no-edges', action='store_true', help='Skip edge visualization')
    
    args = parser.parse_args()
    
    # Default paths if not specified
    if not args.g2o and not args.initial_poses and not args.optimized_poses:
        print("Please specify at least one of: --g2o, --initial-poses, --optimized-poses")
        return 1
    
    plot_poses_with_pointcloud(
        g2o_path=args.g2o,
        initial_poses_path=args.initial_poses,
        optimized_poses_path=args.optimized_poses,
        output_path=args.output,
        show_pointcloud=not args.no_pointcloud,
        show_edges=not args.no_edges
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())