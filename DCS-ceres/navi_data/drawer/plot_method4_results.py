#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
    parser.add_argument('--auto_refresh_sec', type=float, default=0.0,
                        help='>0 enables periodic refresh every N seconds (interactive only)')
    parser.add_argument('--save_on_refresh', action='store_true',
                        help='If set, saves to --output on each refresh (interactive only)')
    
    args = parser.parse_args()
    
    # File paths
    init_poses_file = os.path.join(args.save_path, 'init_nodes.txt')
    best_poses_file = os.path.join(args.save_path, 'opt_nodes.txt')
    most_visited_file = os.path.join(args.save_path, 'opt_nodes_most_visited.txt')
    most_edges_file = os.path.join(args.save_path, 'opt_nodes_most_edges.txt')
    stats_file = os.path.join(args.save_path, 'method4_stats.txt')
    
    # Build a refreshable plotting routine
    fig = plt.figure(figsize=(20, 12))

    # define axes grid once
    ax_main = plt.subplot(2, 3, (1, 4))
    ax_vis = plt.subplot(2, 3, 2)
    ax_scatter = plt.subplot(2, 3, 3)
    ax_tr = plt.subplot(2, 3, 5)
    ax_hist = plt.subplot(2, 3, 6)

    def draw_once():
        # load
        init_poses = load_poses(init_poses_file)
        best_poses = load_poses(best_poses_file)
        most_visited_poses = load_poses(most_visited_file)
        most_edges_poses = load_poses(most_edges_file)
        stats = load_stats(stats_file)

        # clear axes
        ax_main.clear(); ax_vis.clear(); ax_scatter.clear(); ax_tr.clear(); ax_hist.clear()

        # main traj
        if init_poses is not None:
            ax_main.plot(init_poses[:, 0], init_poses[:, 1], color='gray', linewidth=2, alpha=0.5, label='Initial')
            ax_main.scatter(init_poses[0,0], init_poses[0,1], color='gray', s=50)
        if best_poses is not None:
            ax_main.plot(best_poses[:, 0], best_poses[:, 1], color='red', linewidth=2, alpha=0.9, label='Best')
        if most_visited_poses is not None:
            ax_main.plot(most_visited_poses[:, 0], most_visited_poses[:, 1], color='blue', linewidth=2, alpha=0.7, label='Most Visited')
        if most_edges_poses is not None:
            ax_main.plot(most_edges_poses[:, 0], most_edges_poses[:, 1], color='green', linewidth=2, alpha=0.7, label='Most Edges')
        ax_main.set_title('METHOD 4: Layer Comparison', fontsize=16)
        ax_main.set_xlabel('X (m)'); ax_main.set_ylabel('Y (m)')
        ax_main.grid(True, alpha=0.3); ax_main.axis('equal'); ax_main.legend(loc='best')

        # stats
        if stats is not None and len(stats) > 0:
            try:
                if stats.ndim == 1:
                    stats = stats.reshape(1, -1)
                layer_ids = [str(row[0]) for row in stats]
                visits = [int(row[1]) for row in stats]
                total_rewards = [float(row[2]) for row in stats]
                normalized_rewards = [float(row[4]) for row in stats]
                edge_counts = [int(row[5]) for row in stats]
            except Exception as e:
                print(f"Error processing stats: {e}")
                stats_local = None
            else:
                ax_vis.bar(range(len(visits)), visits, alpha=0.7, color='blue')
                ax_vis.set_title('Layer Visits'); ax_vis.set_xlabel('Layer Index'); ax_vis.set_ylabel('Visits')
                ax_vis.set_xticks(range(len(visits)))
                ax_vis.set_xticklabels([f"L{i}" for i in range(len(visits))], rotation=45)
                ax_vis.grid(True, alpha=0.3)

                sc = ax_scatter.scatter(edge_counts, normalized_rewards, s=np.array(visits)*5, alpha=0.6, c=total_rewards, cmap='viridis')
                ax_scatter.set_xlabel('Total Edges'); ax_scatter.set_ylabel('Normalized Reward'); ax_scatter.set_title('Reward vs Edges (size=visits)')
                fig.colorbar(sc, ax=ax_scatter, label='Total Reward')
                ax_scatter.grid(True, alpha=0.3)

                ax_tr.scatter(total_rewards, normalized_rewards, s=50, alpha=0.7, color='purple')
                ax_tr.set_xlabel('Total Reward'); ax_tr.set_ylabel('Normalized Reward'); ax_tr.set_title('Total vs Normalized Reward')
                ax_tr.grid(True, alpha=0.3)

                ax_hist.hist(edge_counts, bins=10, alpha=0.7, color='orange', edgecolor='black')
                ax_hist.set_xlabel('Edge Count'); ax_hist.set_ylabel('Number of Layers'); ax_hist.set_title('Edge Count Distribution')
                ax_hist.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.06, 1, 1])

    # initial draw
    draw_once()

    # Save once (initial)
    output_path = os.path.join(args.save_path, args.output)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    # Add refresh button and auto-refresh if interactive
    if not args.no_show and os.environ.get('DISPLAY', ''):
        # Add a refresh button beneath the plots
        btn_ax = plt.axes([0.4, 0.01, 0.2, 0.04])
        btn = Button(btn_ax, 'Refresh', hovercolor='0.85')

        def on_refresh(event=None):
            draw_once()
            if args.save_on_refresh:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Refreshed and saved to: {output_path}")
            plt.draw()

        btn.on_clicked(on_refresh)

        # Auto refresh with timer if requested
        if args.auto_refresh_sec and args.auto_refresh_sec > 0:
            timer = fig.canvas.new_timer(interval=int(args.auto_refresh_sec * 1000))
            timer.add_callback(on_refresh)
            timer.start()

        plt.show()

if __name__ == '__main__':
    main()
