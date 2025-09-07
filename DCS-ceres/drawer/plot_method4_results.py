#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_trajectory(poses, title, color='blue', alpha=1.0):
    """Plot a single trajectory"""
    x = poses[:, 0]
    y = poses[:, 1]
    plt.plot(x, y, color=color, linewidth=2, alpha=alpha, label=title)
    plt.scatter(x[0], y[0], color=color, s=100, marker='o', alpha=alpha)  # start
    plt.scatter(x[-1], y[-1], color=color, s=100, marker='s', alpha=alpha)  # end

def load_poses(filename):
    """Load poses from file"""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    try:
        poses = np.genfromtxt(filename, usecols=(1, 2))
        return poses
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def load_stats(filename):
    """Load method4 statistics"""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    try:
        # Skip header line and load data
        data = np.genfromtxt(filename, skip_header=1, dtype=None, encoding='utf-8')
        # Handle single row case (0-d array)
        if data.ndim == 0:
            data = np.array([data])
        elif data.ndim == 1 and len(data.shape) == 1:
            data = data.reshape(1, -1) if data.dtype.names else np.array([data])
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Plot METHOD 4 results')
    parser.add_argument('--save_path', default='../save', help='Save directory path')
    parser.add_argument('--output', default='method4_comparison.png', help='Output image filename')
    parser.add_argument('--no-show', action='store_true', help='Save only and do not display the window')
    
    args = parser.parse_args()
    
    # File paths
    init_poses_file = os.path.join(args.save_path, 'init_nodes.txt')
    best_poses_file = os.path.join(args.save_path, 'opt_nodes.txt')
    most_visited_file = os.path.join(args.save_path, 'opt_nodes_most_visited.txt')
    most_edges_file = os.path.join(args.save_path, 'opt_nodes_most_edges.txt')
    stats_file = os.path.join(args.save_path, 'method4_stats.txt')
    
    # Load trajectories
    init_poses = load_poses(init_poses_file)
    best_poses = load_poses(best_poses_file)
    most_visited_poses = load_poses(most_visited_file)
    most_edges_poses = load_poses(most_edges_file)
    
    # Load statistics
    stats = load_stats(stats_file)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main trajectory comparison plot
    plt.subplot(2, 3, (1, 4))
    
    if init_poses is not None:
        plot_trajectory(init_poses, 'Initial', color='gray', alpha=0.5)
    
    if best_poses is not None:
        plot_trajectory(best_poses, 'Best (Normalized Reward)', color='red', alpha=0.9)
    
    if most_visited_poses is not None:
        plot_trajectory(most_visited_poses, 'Most Visited', color='blue', alpha=0.7)
    
    if most_edges_poses is not None:
        plot_trajectory(most_edges_poses, 'Most Edges', color='green', alpha=0.7)
    
    plt.title('METHOD 4: Layer Comparison', fontsize=16)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Statistics plots if available
    if stats is not None and len(stats) > 0:
        try:
            # Ensure stats is at least 2D
            if stats.ndim == 1:
                stats = stats.reshape(1, -1)
            
            # Extract statistics safely
            layer_ids = [str(row[0]) for row in stats]
            visits = [int(row[1]) for row in stats]
            total_rewards = [float(row[2]) for row in stats]
            normalized_rewards = [float(row[4]) for row in stats]
            edge_counts = [int(row[5]) for row in stats]
        except Exception as e:
            print(f"Error processing stats: {e}")
            stats = None
    
    if stats is not None:
        # Visits bar chart
        plt.subplot(2, 3, 2)
        plt.bar(range(len(visits)), visits, alpha=0.7, color='blue')
        plt.title('Layer Visits')
        plt.xlabel('Layer Index')
        plt.ylabel('Visits')
        plt.xticks(range(len(visits)), [f"L{i}" for i in range(len(visits))], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Normalized rewards scatter
        plt.subplot(2, 3, 3)
        plt.scatter(edge_counts, normalized_rewards, s=np.array(visits)*5, alpha=0.6, c=total_rewards, cmap='viridis')
        plt.xlabel('Total Edges')
        plt.ylabel('Normalized Reward')
        plt.title('Reward vs Edges (size=visits)')
        plt.colorbar(label='Total Reward')
        plt.grid(True, alpha=0.3)
        
        # Total reward vs normalized reward
        plt.subplot(2, 3, 5)
        plt.scatter(total_rewards, normalized_rewards, s=50, alpha=0.7, color='purple')
        plt.xlabel('Total Reward')
        plt.ylabel('Normalized Reward')
        plt.title('Total vs Normalized Reward')
        plt.grid(True, alpha=0.3)
        
        # Edge distribution
        plt.subplot(2, 3, 6)
        plt.hist(edge_counts, bins=10, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Edge Count')
        plt.ylabel('Number of Layers')
        plt.title('Edge Count Distribution')
        plt.grid(True, alpha=0.3)
        
        # Print summary statistics
        print(f"\n=== METHOD 4 ANALYSIS ===")
        print(f"Total layers: {len(stats)}")
        print(f"Max visits: {max(visits)} (Layer {layer_ids[visits.index(max(visits))]})")
        print(f"Max edges: {max(edge_counts)} (Layer {layer_ids[edge_counts.index(max(edge_counts))]})")
        print(f"Best normalized reward: {max(normalized_rewards):.4f} (Layer {layer_ids[normalized_rewards.index(max(normalized_rewards))]})")
        print(f"Average edges per layer: {np.mean(edge_counts):.1f}")
        print(f"Average visits per layer: {np.mean(visits):.1f}")
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(args.save_path, args.output)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Show the plot unless suppressed or likely headless
    if not args.no_show and os.environ.get('DISPLAY', ''):
        plt.show()

if __name__ == '__main__':
    main()
