#!/bin/bash

# Enhanced plotting script with pointcloud support
# Usage: ./plot_with_pointcloud.sh [DATASET_NAME] [OUTPUT_DIR]

# Set default paths
SAVE_DIR=${2:-"../save"}
DATA_DIR=${3:-"../data"}
DATASET_NAME=${1:-"INTEL"}

echo "=== SLAM Visualization with Point Cloud ==="
echo "Dataset: $DATASET_NAME"
echo "Save directory: $SAVE_DIR"
echo "Data directory: $DATA_DIR"

# Check if we have the required files
G2O_FILE="${DATA_DIR}/${DATASET_NAME}.g2o"
INIT_POSES="${SAVE_DIR}/init_nodes.txt"
OPT_POSES="${SAVE_DIR}/opt_nodes.txt"

echo ""
echo "Checking for files:"
echo "  G2O file: $G2O_FILE"
if [ -f "$G2O_FILE" ]; then
    echo "    ✓ Found"
else
    echo "    ✗ Not found"
fi

echo "  Initial poses: $INIT_POSES"
if [ -f "$INIT_POSES" ]; then
    echo "    ✓ Found"
else
    echo "    ✗ Not found"
fi

echo "  Optimized poses: $OPT_POSES"
if [ -f "$OPT_POSES" ]; then
    echo "    ✓ Found"
else
    echo "    ✗ Not found"
fi

echo ""

# Create output directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Generate visualizations
echo "=== Generating Visualizations ==="

# 1. Standard pose trajectory plot
echo "1. Creating standard trajectory plot..."
if [ -f "$INIT_POSES" ] && [ -f "$OPT_POSES" ]; then
    python3 plot_results.py \
        --initial_poses "$INIT_POSES" \
        --optimized_poses "$OPT_POSES" \
        --output "${SAVE_DIR}/trajectory_comparison.png"
    echo "   Saved: ${SAVE_DIR}/trajectory_comparison.png"
else
    echo "   Skipped: Missing pose files"
fi

# 2. Enhanced plot with pointcloud
echo "2. Creating enhanced plot with pointcloud..."
if [ -f "$G2O_FILE" ]; then
    python3 plot_poses_with_pointcloud.py \
        --g2o "$G2O_FILE" \
        --initial-poses "$INIT_POSES" \
        --optimized-poses "$OPT_POSES" \
        --output "${SAVE_DIR}/slam_with_pointcloud.png"
    echo "   Saved: ${SAVE_DIR}/slam_with_pointcloud.png"
else
    echo "   Skipped: Missing G2O file"
fi

# 3. Pointcloud only (using optimized poses)
echo "3. Creating pointcloud-only visualization..."
if [ -f "$G2O_FILE" ] && [ -f "$OPT_POSES" ]; then
    python3 plot_poses_with_pointcloud.py \
        --g2o "$G2O_FILE" \
        --optimized-poses "$OPT_POSES" \
        --no-edges \
        --output "${SAVE_DIR}/pointcloud_optimized.png"
    echo "   Saved: ${SAVE_DIR}/pointcloud_optimized.png"
else
    echo "   Skipped: Missing required files"
fi

# 4. Check for method-specific results
echo "4. Checking for method-specific results..."

if [ -f "${SAVE_DIR}/method4_stats.txt" ]; then
    echo "   Found METHOD 4 results, creating enhanced visualization..."
    python3 plot_method4_results.py --save_path "$SAVE_DIR"
    echo "   METHOD 4 plot created"
fi

# 5. Create timing analysis if available
echo "5. Creating timing analysis..."
TIMING_FILES=(
    "${SAVE_DIR}/method0_baseline_optimization_timing.txt"
    "${SAVE_DIR}/method1_dcs_optimization_timing.txt" 
    "${SAVE_DIR}/method2_sc_optimization_timing.txt"
    "${SAVE_DIR}/optimization_timing.txt"
)

FOUND_TIMING=false
for timing_file in "${TIMING_FILES[@]}"; do
    if [ -f "$timing_file" ]; then
        FOUND_TIMING=true
        break
    fi
done

if [ "$FOUND_TIMING" = true ]; then
    echo "   Found timing data, creating unified timing analysis..."
    python3 plot_all_methods_timing.py "$SAVE_DIR" --output-dir "$SAVE_DIR"
    echo "   Timing analysis plot created"
else
    echo "   No timing data found, skipping timing analysis"
fi

echo ""
echo "=== Visualization Complete ==="
echo "Output files saved in: $SAVE_DIR"
echo ""
echo "Generated visualizations:"
[ -f "${SAVE_DIR}/trajectory_comparison.png" ] && echo "  ✓ trajectory_comparison.png - Standard pose trajectories"
[ -f "${SAVE_DIR}/slam_with_pointcloud.png" ] && echo "  ✓ slam_with_pointcloud.png - Full SLAM visualization with pointcloud"
[ -f "${SAVE_DIR}/pointcloud_optimized.png" ] && echo "  ✓ pointcloud_optimized.png - Pointcloud with optimized poses only"
[ -f "${SAVE_DIR}/unified_timing_analysis.png" ] && echo "  ✓ unified_timing_analysis.png - Optimization timing analysis"

echo ""
echo "Usage examples:"
echo "  ./plot_with_pointcloud.sh INTEL ../save ../data"
echo "  ./plot_with_pointcloud.sh M3500 /path/to/results /path/to/data"