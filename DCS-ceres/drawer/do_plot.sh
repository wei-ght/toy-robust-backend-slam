# Check if METHOD 4 results exist
if [ -f "../save/method4_stats.txt" ]; then
    echo "Found METHOD 4 results, creating enhanced visualization..."
    python3 plot_method4_results.py --save_path ../save
else
    echo "Using standard visualization..."
    python3 plot_results.py \
        --initial_poses ../save/init_nodes.txt \
        --optimized_poses ../save/opt_nodes.txt
fi
