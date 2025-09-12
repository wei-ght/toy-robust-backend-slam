#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def load_method_timing_data(data_dir):
    """
    Load timing data from all available method files
    
    Returns:
        dict: method_name -> DataFrame
    """
    data_dir = Path(data_dir)
    method_data = {}
    
    # Method 0, 1, 2 timing files (from main.cpp)
    for method_num in [0, 1, 2]:
        method_names = {
            0: "method0_baseline",
            1: "DCS", 
            2: "SC"
        }
        timing_file = data_dir / f"{method_names[method_num]}_optimization_timing.txt"
        
        if timing_file.exists():
            try:
                df = pd.read_csv(timing_file, sep=' ', comment='#',
                               names=['k', 'edge_type', 'num_edges', 'optimization_type', 'duration_ms'])
                method_data[method_names[method_num]] = df
                print(f"Loaded {len(df)} records from {method_names[method_num]}")
            except Exception as e:
                print(f"Error loading {timing_file}: {e}")
    
    # Method 4 timing file (from simple_layer_manager.cpp)
    method4_file = data_dir / "optimization_timing.txt"
    if method4_file.exists():
        try:
            df = pd.read_csv(method4_file, sep=' ', comment='#',
                           names=['step_counter', 'k', 'edge_a', 'edge_b', 'edge_type', 
                                 'selected_layer', 'optimization_type', 'duration_ms'])
            method_data['MCTS'] = df
            print(f"Loaded {len(df)} records from MCTS")
        except Exception as e:
            print(f"Error loading {method4_file}: {e}")
    
    return method_data

def plot_per_file_optimization_times(data_dir, output_dir=None, show=False, save=True):
    """Plot only optimization time series for each available timing file.

    - For methods 0/1/2 (baseline/DCS/SC): duration_ms vs k
    - For method 4 (MCTS): duration_ms vs step_counter (fallback to index)
    Saves one PNG per file in output_dir (defaults to data_dir).
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)

    method_data = load_method_timing_data(data_dir)
    if not method_data:
        print("No timing data found!")
        return

    colors = {
        'method0_baseline': '#1f77b4',
        'DCS': '#ff7f0e',
        'SC': '#2ca02c',
        'MCTS': '#d62728'
    }

    for method_name, df in method_data.items():
        if len(df) == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        if method_name == 'MCTS':
            x = df['step_counter'] if 'step_counter' in df.columns else np.arange(len(df))
            ax.plot(x, df['duration_ms'], 'o-', color=colors.get(method_name, 'C0'), markersize=3, alpha=0.9)
            ax.set_xlabel('Step counter')
        else:
            ax.plot(df['k'], df['duration_ms'], 'o-', color=colors.get(method_name, 'C0'), markersize=3, alpha=0.9)
            ax.set_xlabel('Node k')
        ax.set_ylabel('Duration (ms)')
        ax.set_title(f'{method_name}: optimization time (n={len(df)})')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        if save:
            out_name = f'{method_name}_optimization_times.png'
            out_path = output_dir / out_name
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f'Saved: {out_path}')
        if show:
            plt.show()
        plt.close(fig)

def plot_unified_timing_analysis(data_dir, output_dir=None):
    """
    Create unified timing analysis plots for all available methods
    """
    method_data = load_method_timing_data(data_dir)
    
    if not method_data:
        print("No timing data found!")
        return
    
    if output_dir is None:
        output_dir = data_dir
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Unified Optimization Timing Analysis - All Methods', fontsize=16)
    
    # Define colors for each method
    colors = {
        'method0_baseline': '#1f77b4',
        'DCS': '#ff7f0e', 
        'SC': '#2ca02c',
        'MCTS': '#d62728'
    }
    
    # 1. Time series by k (node index)
    ax1 = axes[0, 0]
    for method_name, df in method_data.items():
        if method_name == 'MCTS':
            # For method4, group by k and calculate mean duration
            k_stats = df.groupby('k')['duration_ms'].mean()
            ax1.plot(k_stats.index, k_stats.values, 'o-', 
                    label=f'{method_name} (mean)', color=colors[method_name], alpha=0.7, markersize=4)
        else:
            # For method1/2, plot directly
            if len(df) > 0:
                ax1.plot(df['k'], df['duration_ms'], 'o-', 
                        label=method_name, color=colors[method_name], alpha=0.7, markersize=4)
    
    ax1.set_xlabel('Node k')
    ax1.set_ylabel('Duration (ms)')
    ax1.set_title('Optimization Time vs Node Index')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Distribution comparison (histogram)
    ax2 = axes[0, 1]
    all_durations = []
    labels = []
    for method_name, df in method_data.items():
        if len(df) > 0:
            all_durations.append(df['duration_ms'].values)
            labels.append(f'{method_name} (n={len(df)})')
    
    if all_durations:
        max_duration = max([max(d) for d in all_durations])
        bins = np.logspace(0, np.log10(max_duration + 1), 30)
        
        for i, (durations, label) in enumerate(zip(all_durations, labels)):
            method_name = labels[i].split(' ')[0]
            ax2.hist(durations, bins=bins, alpha=0.6, label=label, 
                    density=True, color=colors.get(method_name, f'C{i}'))
    
    ax2.set_xlabel('Duration (ms)')
    ax2.set_ylabel('Density')
    ax2.set_title('Optimization Time Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = axes[0, 2]
    box_data = []
    box_labels = []
    box_colors = []
    
    for method_name, df in method_data.items():
        if len(df) > 0:
            box_data.append(df['duration_ms'].values)
            box_labels.append(method_name.replace('_', '\n'))
            box_colors.append(colors[method_name])
    
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax3.set_ylabel('Duration (ms)')
        ax3.set_title('Optimization Time Distribution (Box Plot)')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Method4 specific analysis (optimization types)
    ax4 = axes[1, 0]
    if 'MCTS' in method_data:
        df_m4 = method_data['MCTS']
        opt_types = df_m4['optimization_type'].unique()
        
        for opt_type in opt_types:
            data = df_m4[df_m4['optimization_type'] == opt_type]
            if len(data) > 0:
                ax4.scatter(data['k'], data['duration_ms'], 
                           label=f'{opt_type} (n={len(data)})', alpha=0.7, s=30)
        
        ax4.set_xlabel('Node k')
        ax4.set_ylabel('Duration (ms)')
        ax4.set_title('Method4: Optimization Types vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No Method4 data available', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Method4: No Data')
    
    # 5. Cumulative timing analysis
    ax5 = axes[1, 1]
    for method_name, df in method_data.items():
        if len(df) > 0:
            if method_name == 'MCTS':
                # For method4, sort by step_counter and calculate cumulative
                df_sorted = df.sort_values('step_counter')
                cumulative_time = df_sorted['duration_ms'].cumsum()
                ax5.plot(df_sorted['step_counter'], cumulative_time, 
                        label=method_name, color=colors[method_name], linewidth=2)
            else:
                # For method1/2, sort by k and calculate cumulative
                df_sorted = df.sort_values('k')
                cumulative_time = df_sorted['duration_ms'].cumsum()
                ax5.plot(df_sorted['k'], cumulative_time, 
                        label=method_name, color=colors[method_name], linewidth=2)
    
    ax5.set_xlabel('Processing Step/Node')
    ax5.set_ylabel('Cumulative Time (ms)')
    ax5.set_title('Cumulative Optimization Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics summary table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats_data = []
    for method_name, df in method_data.items():
        if len(df) > 0:
            stats_data.append([
                method_name.replace('_', ' '),
                len(df),
                f"{df['duration_ms'].mean():.1f}",
                f"{df['duration_ms'].median():.1f}",
                f"{df['duration_ms'].std():.1f}",
                f"{df['duration_ms'].sum():.0f}"
            ])
    
    if stats_data:
        table = ax6.table(cellText=stats_data,
                         colLabels=['Method', 'Count', 'Mean(ms)', 'Median(ms)', 'Std(ms)', 'Total(ms)'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the rows
        for i, (method_name, _) in enumerate(method_data.items()):
            if len(method_data[method_name]) > 0:
                color = colors[method_name]
                for j in range(6):  # 6 columns
                    table[(i+1, j)].set_facecolor(color)
                    table[(i+1, j)].set_alpha(0.3)
    
    ax6.set_title('Statistics Summary', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / 'unified_timing_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Unified plot saved to: {output_path}")
    
    # Print detailed statistics
    print("\n=== UNIFIED TIMING STATISTICS ===")
    for method_name, df in method_data.items():
        if len(df) > 0:
            print(f"\n{method_name.upper()}:")
            print(f"  Count:      {len(df)}")
            print(f"  Mean:       {df['duration_ms'].mean():.2f} ms")
            print(f"  Median:     {df['duration_ms'].median():.2f} ms")
            print(f"  Std:        {df['duration_ms'].std():.2f} ms")
            print(f"  Min:        {df['duration_ms'].min():.2f} ms")
            print(f"  Max:        {df['duration_ms'].max():.2f} ms")
            print(f"  Total:      {df['duration_ms'].sum():.0f} ms")
            
            if method_name == 'MCTS':
                print(f"  Optimization types:")
                for opt_type in df['optimization_type'].unique():
                    count = len(df[df['optimization_type'] == opt_type])
                    mean_time = df[df['optimization_type'] == opt_type]['duration_ms'].mean()
                    print(f"    {opt_type}: {count} times, avg {mean_time:.2f} ms")
    
    # Show plot
    plt.show()

def plot_combined_three_with_table(data_dir, output_dir=None, show=True, save=False, bucket_size=1000, scatter_all=False):
    """One figure with 3 plots (time-series, histogram, boxplot)
    plus a table of per-1000-node bucket means and MCTS speedups.
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)

    method_data = load_method_timing_data(data_dir)
    if not method_data:
        print("No timing data found!")
        return

    colors = {
        'method0_baseline': '#1f77b4',
        'DCS': '#ff7f0e',
        'SC': '#2ca02c',
        'MCTS': '#d62728'
    }

    # Aggregate MCTS per k (mean) for fair comparison
    mcts_k = None
    if 'MCTS' in method_data and len(method_data['MCTS']) > 0:
        mcts_k = method_data['MCTS'].groupby('k', as_index=False)['duration_ms'].mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Optimization Times (All Methods) + MCTS Speedups', fontsize=16)

    # 1) Time series
    ax1 = axes[0, 0]
    for method_name, df in method_data.items():
        if len(df) == 0:
            continue
        if scatter_all:
            # scatter raw points vs k for all methods
            x = df['k'] if 'k' in df.columns else np.arange(len(df))
            ax1.scatter(x, df['duration_ms'], s=10, alpha=0.5,
                        label=method_name, color=colors.get(method_name, None))
        else:
            if method_name == 'MCTS':
                if mcts_k is not None:
                    ax1.plot(mcts_k['k'], mcts_k['duration_ms'], 'o-', label='MCTS (mean per k)',
                             color=colors['MCTS'], markersize=3, alpha=0.9)
            else:
                ax1.plot(df['k'], df['duration_ms'], 'o-', label=method_name,
                         color=colors.get(method_name, 'C0'), markersize=3, alpha=0.9)
    ax1.set_xlabel('Node k')
    ax1.set_ylabel('Duration (ms)')
    ax1.set_title('Time series')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend()

    # 2) Histogram
    ax2 = axes[0, 1]
    for method_name, df in method_data.items():
        if method_name == 'MCTS':
            data = mcts_k['duration_ms'].values if mcts_k is not None else df['duration_ms'].values
        else:
            data = df['duration_ms'].values
        if len(data) == 0:
            continue
        maxv = np.max(data)
        bins = np.logspace(0, np.log10(maxv + 1), 30) if maxv > 0 else 30
        ax2.hist(data, bins=bins, alpha=0.6, density=True,
                 label=f"{method_name} (n={len(data)})", color=colors.get(method_name, None))
    ax2.set_xlabel('Duration (ms)')
    ax2.set_ylabel('Density')
    ax2.set_title('Histogram')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Box plot
    ax3 = axes[1, 0]
    box_data, box_labels, box_colors = [], [], []
    for method_name, df in method_data.items():
        if method_name == 'MCTS':
            data = mcts_k['duration_ms'].values if mcts_k is not None else df['duration_ms'].values
        else:
            data = df['duration_ms'].values
        if len(data) == 0:
            continue
        box_data.append(data)
        box_labels.append(method_name.replace('_', '\n'))
        box_colors.append(colors.get(method_name, 'C0'))
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax3.set_ylabel('Duration (ms)')
    ax3.set_title('Box plot')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    # 4) Table: per bucket mean and MCTS speedups
    ax4 = axes[1, 1]
    ax4.axis('off')

    bucket_means = {}
    all_buckets = set()
    for method_name, df in method_data.items():
        if method_name == 'MCTS':
            if mcts_k is None or len(mcts_k) == 0:
                continue
            tmp = mcts_k.copy()
        else:
            if len(df) == 0:
                continue
            tmp = df[['k', 'duration_ms']].copy()
        tmp['bucket'] = (tmp['k'] // bucket_size).astype(int)
        g = tmp.groupby('bucket')['duration_ms'].mean()
        bucket_means[method_name] = g
        all_buckets.update(g.index.tolist())

    rows = []
    header = [f'bucket({bucket_size})', 'MCTS(ms)', 'method0(ms)', 'DCS(ms)', 'SC(ms)', 'm0/mcts', 'DCS/mcts', 'SC/mcts']
    for b in sorted(all_buckets):
        mcts_v = bucket_means.get('MCTS', pd.Series()).get(b, np.nan)
        m0_v   = bucket_means.get('method0_baseline', pd.Series()).get(b, np.nan)
        dcs_v  = bucket_means.get('DCS', pd.Series()).get(b, np.nan)
        sc_v   = bucket_means.get('SC', pd.Series()).get(b, np.nan)
        def ratio(x, y):
            return (x / y) if (pd.notnull(x) and pd.notnull(y) and y > 0) else np.nan
        rows.append([
            b,
            f"{mcts_v:.1f}" if pd.notnull(mcts_v) else '-',
            f"{m0_v:.1f}"   if pd.notnull(m0_v) else '-',
            f"{dcs_v:.1f}"  if pd.notnull(dcs_v) else '-',
            f"{sc_v:.1f}"   if pd.notnull(sc_v) else '-',
            f"{ratio(m0_v, mcts_v):.2f}" if pd.notnull(ratio(m0_v, mcts_v)) else '-',
            f"{ratio(dcs_v, mcts_v):.2f}" if pd.notnull(ratio(dcs_v, mcts_v)) else '-',
            f"{ratio(sc_v, mcts_v):.2f}"  if pd.notnull(ratio(sc_v, mcts_v)) else '-',
        ])

    if rows:
        tbl = ax4.table(cellText=rows, colLabels=header, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        ax4.set_title('Per-bucket mean and MCTS speedup (×)', pad=10)

    plt.tight_layout()
    if save:
        out_path = Path(output_dir) / 'combined_three_with_table.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved combined figure: {out_path}')
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot optimization time (per-file) or combined 1-figure dashboard')
    parser.add_argument('data_dir', help='Directory containing timing files')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots instead of only saving')
    parser.add_argument('--no-save', action='store_true', help='Do not save PNGs (show only)')
    parser.add_argument('--combined', action='store_true', help='Show one figure: 3 plots + MCTS speedup table')
    parser.add_argument('--bucket', type=int, default=1000, help='Bucket size for speedup table (default: 1000)')
    parser.add_argument('--scatter-all', action='store_true', help='Scatter all points (no per-k aggregation) in time-series')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found")
        return 1
    if args.combined:
        plot_combined_three_with_table(
            args.data_dir,
            args.output_dir,
            show=args.show or args.no_save or True,
            save=not args.no_save,
            bucket_size=args.bucket,
            scatter_all=args.scatter_all,
        )
    else:
        plot_per_file_optimization_times(
            args.data_dir,
            args.output_dir,
            show=args.show or args.no_save,
            save=not args.no_save,
            # Use scatter for per-file too if requested
            scatter_all=args.scatter_all,
        )
    return 0

if __name__ == '__main__':
    sys.exit(main())
