#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import sys

def load_statistics(filepath):
    """Load statistics data from CSV file"""
    try:
        # Read CSV file, skipping comment lines and specify column names
        column_names = ['method_name', 'time_step', 'node_count', 'cumulative_distance', 'layer_count', 'processing_time_ms']
        df = pd.read_csv(filepath, comment='#', names=column_names, header=None)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_statistics(stats_files, output_path="dcs_statistics_plot.png"):
    """Create plots from DCS statistics files"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DCS Methods Statistics Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Load all statistics files
    all_data = {}
    for i, filepath in enumerate(stats_files):
        if os.path.exists(filepath):
            df = load_statistics(filepath)
            if df is not None:
                method_name = os.path.basename(filepath).replace('_statistics.txt', '')
                all_data[method_name] = df
                print(f"Loaded {method_name}: {len(df)} data points")
        else:
            print(f"Warning: File not found: {filepath}")
    
    if not all_data:
        print("No valid data files found!")
        return
    
    # Plot 1: Node count over time steps
    ax1 = axes[0, 0]
    for i, (method, df) in enumerate(all_data.items()):
        ax1.plot(df['time_step'].values, df['node_count'].values, 
                label=method, color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Node Count')
    ax1.set_title('Node Count vs Time Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distance over time steps
    ax2 = axes[0, 1]
    for i, (method, df) in enumerate(all_data.items()):
        ax2.plot(df['time_step'].values, df['cumulative_distance'].values, 
                label=method, color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Distance (m)')
    ax2.set_title('Cumulative Distance vs Time Step')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Processing time over time steps
    ax3 = axes[1, 0]
    for i, (method, df) in enumerate(all_data.items()):
        ax3.plot(df['time_step'].values, df['processing_time_ms'].values, 
                label=method, color=colors[i % len(colors)], linewidth=2, alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Processing Time (ms)')
    ax3.set_title('Processing Time vs Time Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Node count vs cumulative distance
    ax4 = axes[1, 1]
    for i, (method, df) in enumerate(all_data.items()):
        ax4.plot(df['cumulative_distance'].values, df['node_count'].values, 
                label=method, color=colors[i % len(colors)], linewidth=2)
    ax4.set_xlabel('Cumulative Distance (m)')
    ax4.set_ylabel('Node Count')
    ax4.set_title('Node Count vs Cumulative Distance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()

def print_summary_statistics(stats_files):
    """Print summary statistics for all methods"""
    print("\n=== Summary Statistics ===")
    print(f"{'Method':<20} {'Total Steps':<12} {'Final Nodes':<12} {'Final Dist(m)':<15} {'Total Time(ms)':<15}")
    print("-" * 80)
    
    for filepath in stats_files:
        if os.path.exists(filepath):
            df = load_statistics(filepath)
            if df is not None:
                method_name = os.path.basename(filepath).replace('_statistics.txt', '')
                total_steps = len(df)
                final_nodes = df['node_count'].iloc[-1] if not df.empty else 0
                final_dist = df['cumulative_distance'].iloc[-1] if not df.empty else 0
                total_time = df['processing_time_ms'].sum() if not df.empty else 0
                
                print(f"{method_name:<20} {total_steps:<12} {final_nodes:<12} {final_dist:<15.3f} {total_time:<15.1f}")

def main():
    parser = argparse.ArgumentParser(description='Plot DCS statistics from CSV files')
    parser.add_argument('--save-dir', default='save', help='Directory containing statistics files')
    parser.add_argument('--output', default='dcs_statistics_plot.png', help='Output plot filename')
    parser.add_argument('--files', nargs='+', help='Specific statistics files to plot')
    
    args = parser.parse_args()
    
    if args.files:
        # Use specific files provided
        stats_files = args.files
    else:
        # Auto-detect statistics files in save directory
        save_dir = args.save_dir
        potential_files = [
            f"{save_dir}/method0_baseline_statistics.txt",
            f"{save_dir}/method1_dcs_statistics.txt", 
            f"{save_dir}/method2_sc_statistics.txt",
            f"{save_dir}/method_statistics.txt",
            f"{save_dir}/method_statistics_online.txt"
        ]
        
        # Filter existing files
        stats_files = [f for f in potential_files if os.path.exists(f)]
        
        if not stats_files:
            print(f"No statistics files found in {save_dir}/")
            print("Expected files:")
            for f in potential_files:
                print(f"  {f}")
            sys.exit(1)
    
    print(f"Found {len(stats_files)} statistics file(s):")
    for f in stats_files:
        print(f"  {f}")
    
    # Print summary statistics
    print_summary_statistics(stats_files)
    
    # Create plots
    plot_statistics(stats_files, args.output)

if __name__ == "__main__":
    main()