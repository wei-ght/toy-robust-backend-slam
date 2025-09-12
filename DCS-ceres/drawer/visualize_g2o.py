#!/usr/bin/env python3
"""
G2O Dataset Visualizer
Reads g2o files and visualizes nodes and edge connections
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import math

class G2OVisualizer:
    def __init__(self):
        self.nodes = {}  # index -> (x, y, theta)
        self.odometry_edges = []  # [(from_idx, to_idx, dx, dy, dtheta)]
        self.closure_edges = []   # [(from_idx, to_idx, dx, dy, dtheta)]
        self.bogus_edges = []     # [(from_idx, to_idx, dx, dy, dtheta)]
        self.scans = {}           # index -> Nx2 array (points in node frame)
        
    def parse_g2o(self, filename):
        """Parse g2o format file"""
        print(f"Reading g2o file: {filename}")
        last_node_id = None
        pending_pts_expected = None
        pending_pts_vals = []

        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # parse optional scan comment blocks
                if line.startswith('#'):
                    parts = line[1:].strip().split()
                    if len(parts) >= 2 and parts[0] == 'POINTS_COUNT':
                        try:
                            pending_pts_expected = int(parts[1])
                            pending_pts_vals = []
                        except ValueError:
                            print(f"Warning: invalid POINTS_COUNT at line {line_num}")
                    elif len(parts) >= 2 and parts[0] == 'POINTS_DATA':
                        # append floats in this line
                        try:
                            floats = list(map(float, parts[1:]))
                            pending_pts_vals.extend(floats)
                        except ValueError:
                            print(f"Warning: invalid POINTS_DATA at line {line_num}")
                        # if enough values are gathered, store to current vertex
                        if pending_pts_expected is not None and last_node_id is not None:
                            needed = pending_pts_expected * 2
                            if len(pending_pts_vals) >= needed:
                                arr = np.array(pending_pts_vals[:needed], dtype=float).reshape(-1, 2)
                                self.scans[last_node_id] = arr
                                pending_pts_expected = None
                                pending_pts_vals = []
                    # move to next line
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                try:
                    if parts[0] in ['VERTEX_SE2', 'VERTEX2']:
                        # Parse vertex: VERTEX_SE2 id x y theta
                        node_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        theta = float(parts[4])
                        self.nodes[node_id] = (x, y, theta)
                        last_node_id = node_id
                        
                    elif parts[0] in ['EDGE_SE2', 'EDGE2']:
                        # Parse edge: EDGE_SE2 id1 id2 dx dy dtheta info(6)
                        from_id = int(parts[1])
                        to_id = int(parts[2])
                        dx = float(parts[3])
                        dy = float(parts[4])
                        dtheta = float(parts[5])
                        
                        # Classify edge type based on node indices
                        if abs(from_id - to_id) <= 5:  # Odometry edges (sequential)
                            self.odometry_edges.append((from_id, to_id, dx, dy, dtheta))
                        else:  # Loop closure edges
                            self.closure_edges.append((from_id, to_id, dx, dy, dtheta))
                            
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    print(f"Error: {e}")
                    continue
        
        print(f"Loaded {len(self.nodes)} nodes, {len(self.odometry_edges)} odometry edges, {len(self.closure_edges)} closure edges")
        if self.scans:
            print(f"Loaded scan data for {len(self.scans)} nodes")
        
    def plot_pose_graph(self, show_odometry=True, show_closures=True, show_nodes=True, 
                       node_size=20, edge_alpha=0.6, figsize=(12, 10),
                       show_scans=False, scan_alpha=0.3, scan_size=1.0):
        """Plot the pose graph"""
        
        if not self.nodes:
            print("No nodes to plot!")
            return
            
        # Extract node positions
        node_ids = list(self.nodes.keys())
        node_ids.sort()
        
        x_coords = [self.nodes[nid][0] for nid in node_ids]
        y_coords = [self.nodes[nid][1] for nid in node_ids]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot odometry edges (trajectory backbone)
        if show_odometry and self.odometry_edges:
            for from_id, to_id, _, _, _ in self.odometry_edges:
                if from_id in self.nodes and to_id in self.nodes:
                    x1, y1, _ = self.nodes[from_id]
                    x2, y2, _ = self.nodes[to_id]
                    plt.plot([x1, x2], [y1, y2], 'b-', alpha=edge_alpha, linewidth=1, 
                            label='Odometry' if from_id == self.odometry_edges[0][0] else "")
        
        # Plot closure edges (loop closures)
        if show_closures and self.closure_edges:
            for from_id, to_id, _, _, _ in self.closure_edges:
                if from_id in self.nodes and to_id in self.nodes:
                    x1, y1, _ = self.nodes[from_id]
                    x2, y2, _ = self.nodes[to_id]
                    plt.plot([x1, x2], [y1, y2], 'r-', alpha=edge_alpha, linewidth=0.8,
                            label='Loop Closure' if from_id == self.closure_edges[0][0] else "")
        
        # Plot bogus edges if any
        if self.bogus_edges:
            for from_id, to_id, _, _, _ in self.bogus_edges:
                if from_id in self.nodes and to_id in self.nodes:
                    x1, y1, _ = self.nodes[from_id]
                    x2, y2, _ = self.nodes[to_id]
                    plt.plot([x1, x2], [y1, y2], 'm--', alpha=edge_alpha, linewidth=0.6,
                            label='Bogus' if from_id == self.bogus_edges[0][0] else "")
        
        # Plot nodes
        if show_nodes:
            plt.scatter(x_coords, y_coords, s=node_size, c='black', alpha=0.7, zorder=5)
            
            # Highlight start and end nodes
            if node_ids:
                start_id = min(node_ids)
                end_id = max(node_ids)
                
                start_x, start_y, _ = self.nodes[start_id]
                end_x, end_y, _ = self.nodes[end_id]
                
                plt.scatter([start_x], [start_y], s=node_size*3, c='green', marker='o', 
                           zorder=10, label='Start')
                plt.scatter([end_x], [end_y], s=node_size*3, c='red', marker='s', 
                           zorder=10, label='End')
        
        plt.title(f'G2O Pose Graph Visualization\n'
                 f'Nodes: {len(self.nodes)}, Odometry: {len(self.odometry_edges)}, '
                 f'Closures: {len(self.closure_edges)}')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # Plot scan points, transformed to world frame
        if show_scans and self.scans:
            xs_all = []
            ys_all = []
            for nid, pts in self.scans.items():
                if nid not in self.nodes:
                    continue
                x, y, th = self.nodes[nid]
                c = math.cos(th)
                s = math.sin(th)
                # node->world transform
                wx = c * pts[:, 0] - s * pts[:, 1] + x
                wy = s * pts[:, 0] + c * pts[:, 1] + y
                xs_all.append(wx)
                ys_all.append(wy)
            if xs_all:
                xs_cat = np.concatenate(xs_all)
                ys_cat = np.concatenate(ys_all)
                plt.scatter(xs_cat, ys_cat, s=scan_size, c='gray', alpha=scan_alpha, linewidths=0)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()
    
    def plot_edge_statistics(self, figsize=(15, 5)):
        """Plot edge statistics and distributions"""
        if not (self.odometry_edges or self.closure_edges):
            print("No edges to analyze!")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Edge length distribution
        odometry_lengths = []
        closure_lengths = []
        
        for from_id, to_id, dx, dy, _ in self.odometry_edges:
            length = math.sqrt(dx*dx + dy*dy)
            odometry_lengths.append(length)
            
        for from_id, to_id, dx, dy, _ in self.closure_edges:
            length = math.sqrt(dx*dx + dy*dy)
            closure_lengths.append(length)
            
        axes[0].hist(odometry_lengths, bins=30, alpha=0.7, label='Odometry', color='blue')
        axes[0].hist(closure_lengths, bins=30, alpha=0.7, label='Loop Closure', color='red')
        axes[0].set_xlabel('Edge Length (m)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Edge Length Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Node degree distribution
        node_degrees = {}
        for nid in self.nodes:
            node_degrees[nid] = 0
            
        for from_id, to_id, _, _, _ in self.odometry_edges + self.closure_edges:
            if from_id in node_degrees:
                node_degrees[from_id] += 1
            if to_id in node_degrees:
                node_degrees[to_id] += 1
                
        degrees = list(node_degrees.values())
        axes[1].hist(degrees, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Node Degree')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Node Degree Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Edge span distribution (for closure edges)
        if self.closure_edges:
            spans = [abs(from_id - to_id) for from_id, to_id, _, _, _ in self.closure_edges]
            axes[2].hist(spans, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[2].set_xlabel('Edge Span (node index difference)')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Loop Closure Span Distribution')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No closure edges', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Loop Closure Span Distribution')
        
        plt.tight_layout()
        return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize G2O pose graph files')
    parser.add_argument('g2o_file', help='Path to g2o file')
    parser.add_argument('--output', help='Output image file (optional)')
    parser.add_argument('--opt-poses', help='Optional optimized poses file (index x y theta) to override g2o nodes')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots')
    parser.add_argument('--stats', action='store_true', help='Show edge statistics')
    parser.add_argument('--hide-odometry', action='store_true', help='Hide odometry edges')
    parser.add_argument('--hide-closures', action='store_true', help='Hide closure edges')
    parser.add_argument('--hide-nodes', action='store_true', help='Hide nodes')
    parser.add_argument('--node-size', type=int, default=20, help='Node size for plotting')
    parser.add_argument('--edge-alpha', type=float, default=0.6, help='Edge transparency')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 10], help='Figure size [width height]')
    parser.add_argument('--scan', action='store_true', help='Overlay raw scan points if present in g2o comments')
    parser.add_argument('--scan-alpha', type=float, default=0.3, help='Alpha for scan points')
    parser.add_argument('--scan-size', type=float, default=1.0, help='Marker size for scan points')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.g2o_file):
        print(f"Error: File {args.g2o_file} not found!")
        return
    
    # Create visualizer and parse file
    viz = G2OVisualizer()
    viz.parse_g2o(args.g2o_file)

    # Optionally override node poses with optimized ones
    if args.opt_poses and os.path.exists(args.opt_poses):
        try:
            with open(args.opt_poses, 'r') as fp:
                cnt = 0
                for line in fp:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    try:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        th = float(parts[3])
                        if nid in viz.nodes:
                            viz.nodes[nid] = (x, y, th)
                            cnt += 1
                    except ValueError:
                        continue
            print(f"Overrode {cnt} node poses from optimized file: {args.opt_poses}")
        except Exception as e:
            print(f"Warning: failed to load opt poses {args.opt_poses}: {e}")
    
    # Plot main graph
    fig1 = viz.plot_pose_graph(
        show_odometry=not args.hide_odometry,
        show_closures=not args.hide_closures,
        show_nodes=not args.hide_nodes,
        node_size=args.node_size,
        edge_alpha=args.edge_alpha,
        figsize=tuple(args.figsize),
        show_scans=args.scan,
        scan_alpha=args.scan_alpha,
        scan_size=args.scan_size
    )
    
    if args.output:
        base_name = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1] or '.png'
        main_output = base_name + ext
        fig1.savefig(main_output, dpi=300, bbox_inches='tight')
        print(f"Saved main plot to: {main_output}")
    
    # Plot statistics if requested
    if args.stats:
        fig2 = viz.plot_edge_statistics()
        if args.output:
            stats_output = base_name + '_stats' + ext
            fig2.savefig(stats_output, dpi=300, bbox_inches='tight')
            print(f"Saved statistics plot to: {stats_output}")
    
    # Show plots if not suppressed
    if not args.no_show and os.environ.get('DISPLAY', ''):
        plt.show()
    else:
        print("Plot display suppressed (use --no-show=false to show)")

if __name__ == '__main__':
    main()
