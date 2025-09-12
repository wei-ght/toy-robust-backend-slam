#!/usr/bin/env python3
"""
METHOD 5 Results Plotter
- Single global optimization trajectory
- Initial vs Optimized comparison
- Switch variables visualization
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

def load_poses(file_path):
    """Load poses from file"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
        
    poses = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        idx = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        theta = float(parts[3])
                        poses.append((idx, x, y, theta))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    return poses

def load_switches(file_path):
    """Load switch variables from file"""
    if not os.path.exists(file_path):
        return None
        
    switches = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        edge_id = parts[0] + "-" + parts[1]  # "i-j"
                        switch_val = float(parts[2])
                        switches[edge_id] = switch_val
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    
    return switches

def plot_trajectory(poses, label, color='blue', alpha=0.8, linewidth=2):
    """Plot a trajectory"""
    if poses is None or len(poses) == 0:
        return
        
    x_vals = [pose[1] for pose in poses]
    y_vals = [pose[2] for pose in poses]
    
    plt.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=linewidth, label=label)
    plt.scatter(x_vals[0], y_vals[0], color=color, s=100, marker='o', alpha=0.9)  # Start
    plt.scatter(x_vals[-1], y_vals[-1], color=color, s=100, marker='s', alpha=0.9)  # End

def main():
    parser = argparse.ArgumentParser(description='Plot METHOD 5 SLAM results')
    parser.add_argument('--save_path', required=True, help='Path to results directory')
    parser.add_argument('--output', default='method5_result.png', help='Output image filename')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    print(f"METHOD 5 Plotter - Processing: {args.save_path}")
    
    # File paths
    init_poses_file = os.path.join(args.save_path, 'init_nodes.txt')
    opt_poses_file = os.path.join(args.save_path, 'opt_nodes.txt')
    switches_file = os.path.join(args.save_path, 'switches.txt')
    
    # Load trajectories
    init_poses = load_poses(init_poses_file)
    opt_poses = load_poses(opt_poses_file)
    switches = load_switches(switches_file)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot trajectories
    if init_poses is not None:
        plot_trajectory(init_poses, 'Initial (Odometry)', color='gray', alpha=0.5, linewidth=1)
    
    if opt_poses is not None:
        plot_trajectory(opt_poses, 'Optimized (METHOD 5)', color='red', alpha=0.9, linewidth=2)
    
    plt.title('METHOD 5: Single Global Optimization with Switchable Constraints', fontsize=14)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add statistics text
    info_text = f"METHOD 5: SimpleLayerManager2\\n"
    info_text += f"- Single global problem\\n"
    info_text += f"- Switchable Constraints for robustness\\n"
    info_text += f"- Online incremental optimization"
    
    if switches:
        active_switches = sum(1 for v in switches.values() if v > 0.5)
        total_switches = len(switches)
        info_text += f"\\n- Active constraints: {active_switches}/{total_switches}"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             verticalalignment='top', fontsize=10, family='monospace')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {args.output}")
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()