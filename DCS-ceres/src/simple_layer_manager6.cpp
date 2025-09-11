#include "simple_layer_manager6.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>
#include <sstream>
#include <iomanip>

SimpleLayerManager6::SimpleLayerManager6(ReadG2O& g, const std::string& save_path, const SimpleLayer6Config& cfg)
    : g2o_(g), save_path_(save_path), config_(cfg)
{
    // Anchor first pose (add parameter blocks explicitly)
    if (!g2o_.nNodes.empty()) {
        problem_.AddParameterBlock(&g2o_.nNodes[0]->p[0], 1);
        problem_.AddParameterBlock(&g2o_.nNodes[0]->p[1], 1);
        problem_.AddParameterBlock(&g2o_.nNodes[0]->p[2], 1);
        problem_.SetParameterBlockConstant(&g2o_.nNodes[0]->p[0]);
        problem_.SetParameterBlockConstant(&g2o_.nNodes[0]->p[1]);
        problem_.SetParameterBlockConstant(&g2o_.nNodes[0]->p[2]);
        anchor_added_ = true;
    }
}

void SimpleLayerManager6::log_line(const std::string& s)
{
    std::cout << s << std::endl;
}

void SimpleLayerManager6::save_nodes(const std::string& filepath)
{
    std::ofstream fp(filepath);
    for (size_t i = 0; i < g2o_.nNodes.size(); ++i) {
        double* p = g2o_.nNodes[i]->p;
        fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
}

void SimpleLayerManager6::run_online()
{
    log_line("[method6] online starting (PoseGraph2dErrorTerm, no SC/DCS)");

    ceres::LossFunction* loss = new ceres::HuberLoss(config_.huber_delta);

    int N = static_cast<int>(g2o_.nNodes.size());
    for (int k = 1; k < N; ++k) {
        step_counter_++;
        int add_odo = 0, add_loop = 0, add_bogus = 0;

        auto add_edge = [&](Edge* e){
            // Build sqrt information from edge info matrix
            Eigen::Matrix3d info;
            info << e->I11, e->I12, e->I13,
                    e->I12, e->I22, e->I23,
                    e->I13, e->I23, e->I33;
            Eigen::LLT<Eigen::Matrix3d> llt(info);
            Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();
            if (llt.info() == Eigen::Success) sqrt_info = llt.matrixL();

            // Add residual with 1D blocks: x_a, y_a, yaw_a, x_b, y_b, yaw_b
            ceres::CostFunction* cost = PoseGraph2dErrorTerm::Create(
                e->x, e->y, e->theta, sqrt_info);
            problem_.AddResidualBlock(cost, loss,
                &e->a->p[0], &e->a->p[1], &e->a->p[2],
                &e->b->p[0], &e->b->p[1], &e->b->p[2]);
        };

        // Add odometry with max(a,b) == k
        for (auto* e : g2o_.nEdgesOdometry) {
            int ia = e->a->index, ib = e->b->index;
            if (std::max(ia, ib) == k) { add_edge(e); add_odo++; }
        }
        // Add closure
        for (auto* e : g2o_.nEdgesClosure) {
            int ia = e->a->index, ib = e->b->index;
            if (std::max(ia, ib) == k) { 
                add_edge(e); add_loop++; 
                b_add_loop_edges_ = true;
            }
        }
        // Add bogus
        for (auto* e : g2o_.nEdgesBogus) {
            int ia = e->a->index, ib = e->b->index;
            if (std::max(ia, ib) == k) { 
                add_edge(e); add_bogus++; 
                b_add_loop_edges_ = true;
            }
        }

        log_line("[method6] k=" + std::to_string(k) +
                 ", odom=" + std::to_string(add_odo) +
                 ", loop=" + std::to_string(add_loop) +
                 ", bogus=" + std::to_string(add_bogus));

        if(b_add_loop_edges_) {
            // Short solve at each step
            ceres::Solver::Options opts;
            opts.max_num_iterations = std::max(1, config_.iters_per_step);
            opts.minimizer_progress_to_stdout = false;
            opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            opts.num_threads = std::thread::hardware_concurrency() != 0
                ? std::thread::hardware_concurrency()
                : 4;
            ceres::Solver::Summary sum;
            ceres::Solve(opts, &problem_, &sum);
            b_add_loop_edges_ = false;
        }
        
        // Snapshot using Method 4-style API
        if (config_.snapshot_every > 0 && (step_counter_ % config_.snapshot_every == 0)) {
            save_snapshot(step_counter_, k);
        }
    }

    // Save final
    save_nodes(save_path_ + "/opt_nodes.txt");
}

void SimpleLayerManager6::save_snapshot(int step_index, int k_cap)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path snap_dir = fs::path(save_path_) / "snapshots";
    fs::create_directories(snap_dir, ec);

    std::ostringstream oss_png, oss_data;
    oss_png << "step_" << std::setw(4) << std::setfill('0') << step_index << ".png";
    oss_data << "step_" << std::setw(4) << std::setfill('0') << step_index << "_data";
    fs::path data_dir = snap_dir / oss_data.str();
    fs::create_directories(data_dir, ec);

    int cap = std::max(0, std::min(k_cap, (int)g2o_.nNodes.size() - 1));

    // init_nodes up to cap
    {
        std::ofstream fp((data_dir / "init_nodes.txt").string());
        int n = std::min((int)g2o_.nNodes.size(), cap + 1);
        for (int i = 0; i < n; ++i) {
            double* p = g2o_.nNodes[i]->p;
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
    }
    // optimized poses up to cap. Method 6 has single global solution
    auto write_opt = [&](const fs::path& path){
        std::ofstream fp(path.string());
        int n = std::min((int)g2o_.nNodes.size(), cap + 1);
        for (int i = 0; i < n; ++i) {
            double* p = g2o_.nNodes[i]->p;
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
    };
    write_opt(data_dir / "opt_nodes.txt");
    write_opt(data_dir / "opt_nodes_most_visited.txt");
    write_opt(data_dir / "opt_nodes_most_edges.txt");

    // Minimal method4-compatible stats
    {
        std::ofstream fp((data_dir / "method4_stats.txt").string());
        fp << "# layer_id visits total_reward avg_reward normalized_reward total_edges inherited_edges added_edges\n";
        int total_edges = 0;
        auto count_if_le = [&](const std::vector<Edge*>& edges){
            for (auto* e : edges) {
                int ia = e->a->index, ib = e->b->index;
                if (std::max(ia, ib) <= cap) total_edges++;
            }
        };
        count_if_le(g2o_.nEdgesOdometry);
        count_if_le(g2o_.nEdgesClosure);
        count_if_le(g2o_.nEdgesBogus);
        fp << "L1 0 0 0 0 " << total_edges << " 0 " << total_edges << "\n";
    }

    // Also refresh live files in save root for plotting
    write_opt(fs::path(save_path_) / "opt_nodes.txt");
    write_opt(fs::path(save_path_) / "opt_nodes_most_visited.txt");
    write_opt(fs::path(save_path_) / "opt_nodes_most_edges.txt");
    {
        std::ofstream fp((fs::path(save_path_) / "init_nodes.txt").string());
        int n = std::min((int)g2o_.nNodes.size(), cap + 1);
        for (int i = 0; i < n; ++i) {
            double* p = g2o_.nNodes[i]->p;
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
    }

    // Render png with method4 plotter
    std::string out_png = (snap_dir / oss_png.str()).string();
    std::string cmd = std::string("MPLBACKEND=Agg python ../drawer/plot_method4_results.py ") +
                      "--save_path " + data_dir.string() + " --output " + out_png +
                      " --no-show > /dev/null 2>&1";
    std::system(cmd.c_str());

    log_line(std::string("[method6-snapshot] saved ") + out_png + " (cap k=" + std::to_string(cap) + ")");
}
