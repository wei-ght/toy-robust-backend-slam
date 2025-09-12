#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
import os

def plot_optimization_timing(timing_file_path, output_dir=None):
    """
    Plot optimization timing data from optimization_timing.txt
    
    Args:
        timing_file_path: path to optimization_timing.txt
        output_dir: directory to save plots (default: same as timing file)
    """
    
    if not os.path.exists(timing_file_path):
        print(f"Error: File {timing_file_path} not found")
        return
    
    # Read the data
    try:
        df = pd.read_csv(timing_file_path, sep=' ', comment='#',
                        names=['step_counter', 'k', 'edge_a', 'edge_b', 'edge_type', 
                              'selected_layer', 'optimization_type', 'duration_ms'])
        print(f"Loaded {len(df)} optimization records")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    if len(df) == 0:
        print("No data found in timing file")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(timing_file_path)
    
    # Separate data by optimization type
    temp_layer_data = df[df['optimization_type'] == 'temp_layer']
    topk_best_data = df[df['optimization_type'] == 'topk_best']
    parent_data = df[df['optimization_type'] == 'parent']
    
    print(f"Found {len(temp_layer_data)} temp_layer optimizations")
    print(f"Found {len(topk_best_data)} topk_best optimizations")
    print(f"Found {len(parent_data)} parent optimizations")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimization Timing Analysis', fontsize=16)
    
    # 1. Time series plot for temp_layer vs topk_best
    ax1 = axes[0, 0]
    if len(temp_layer_data) > 0:
        ax1.plot(temp_layer_data['step_counter'], temp_layer_data['duration_ms'], 
                'o-', label='temp_layer', alpha=0.7, markersize=3)
    if len(topk_best_data) > 0:
        ax1.plot(topk_best_data['step_counter'], topk_best_data['duration_ms'], 
                's-', label='topk_best', alpha=0.7, markersize=3)
    if len(parent_data) > 0:
        ax1.plot(parent_data['step_counter'], parent_data['duration_ms'], 
                '^-', label='parent', alpha=0.7, markersize=3)
    
    ax1.set_xlabel('Step Counter')
    ax1.set_ylabel('Duration (ms)')
    ax1.set_title('Optimization Time vs Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    ax2 = axes[0, 1]
    bins = np.logspace(0, np.log10(max(df['duration_ms']) + 1), 30)
    
    if len(temp_layer_data) > 0:
        ax2.hist(temp_layer_data['duration_ms'], bins=bins, alpha=0.6, 
                label=f'temp_layer (n={len(temp_layer_data)})', density=True)
    if len(topk_best_data) > 0:
        ax2.hist(topk_best_data['duration_ms'], bins=bins, alpha=0.6, 
                label=f'topk_best (n={len(topk_best_data)})', density=True)
    if len(parent_data) > 0:
        ax2.hist(parent_data['duration_ms'], bins=bins, alpha=0.6, 
                label=f'parent (n={len(parent_data)})', density=True)
    
    ax2.set_xlabel('Duration (ms)')
    ax2.set_ylabel('Density')
    ax2.set_title('Optimization Time Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = axes[1, 0]
    data_for_boxplot = []
    labels = []
    
    if len(temp_layer_data) > 0:
        data_for_boxplot.append(temp_layer_data['duration_ms'])
        labels.append('temp_layer')
    if len(topk_best_data) > 0:
        data_for_boxplot.append(topk_best_data['duration_ms'])
        labels.append('topk_best')
    if len(parent_data) > 0:
        data_for_boxplot.append(parent_data['duration_ms'])
        labels.append('parent')
    
    if data_for_boxplot:
        ax3.boxplot(data_for_boxplot, labels=labels)
        ax3.set_ylabel('Duration (ms)')
        ax3.set_title('Optimization Time Distribution (Box Plot)')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. Statistics by node k
    ax4 = axes[1, 1]
    if len(df) > 0:
        # Group by k and calculate mean duration for each optimization type
        k_stats = df.groupby(['k', 'optimization_type'])['duration_ms'].mean().unstack(fill_value=0)
        
        if 'temp_layer' in k_stats.columns:
            ax4.plot(k_stats.index, k_stats['temp_layer'], 'o-', label='temp_layer', markersize=4)
        if 'topk_best' in k_stats.columns:
            ax4.plot(k_stats.index, k_stats['topk_best'], 's-', label='topk_best', markersize=4)
        if 'parent' in k_stats.columns:
            ax4.plot(k_stats.index, k_stats['parent'], '^-', label='parent', markersize=4)
        
        ax4.set_xlabel('Node k')
        ax4.set_ylabel('Mean Duration (ms)')
        ax4.set_title('Mean Optimization Time by Node')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'optimization_timing_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print statistics
    print("\n=== TIMING STATISTICS ===")
    for opt_type in ['temp_layer', 'topk_best', 'parent']:
        data = df[df['optimization_type'] == opt_type]['duration_ms']
        if len(data) > 0:
            print(f"\n{opt_type}:")
            print(f"  Count: {len(data)}")
            print(f"  Mean:  {data.mean():.2f} ms")
            print(f"  Median: {data.median():.2f} ms")
            print(f"  Std:   {data.std():.2f} ms")
            print(f"  Min:   {data.min():.2f} ms")
            print(f"  Max:   {data.max():.2f} ms")
    
    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot optimization timing data')
    parser.add_argument('timing_file', help='Path to optimization_timing.txt file')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    
    args = parser.parse_args()
    
    plot_optimization_timing(args.timing_file, args.output_dir)

if __name__ == '__main__':
    main()