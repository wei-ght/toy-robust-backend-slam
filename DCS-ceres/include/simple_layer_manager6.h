#ifndef SIMPLE_LAYER_MANAGER6_H
#define SIMPLE_LAYER_MANAGER6_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "graph.h"
#include "g2o_util.h"
#include "ceres_error.h"

// METHOD 6: Online-only, no SC/DCS. All edges use PoseGraph2dErrorTerm (1D blocks).

struct SimpleLayer6Config {
    int iters_per_step = 5;      // ceres iterations per online step
    double huber_delta = 0.05;   // robust kernel delta
    int snapshot_every = 20;      // 0=off, >0 means every N steps
};
class SimpleLayerManager6
{
public:
    SimpleLayerManager6(ReadG2O& g, const std::string& save_path, const SimpleLayer6Config& cfg = {});
    ~SimpleLayerManager6() = default;

    void run_online();

private:
    void log_line(const std::string& s);
    void save_nodes(const std::string& filepath);
    void save_snapshot(int step_index, int k_cap);

private:
    ReadG2O& g2o_;
    std::string save_path_;
    ceres::Problem problem_;
    bool anchor_added_ = false;
    int step_counter_ = 0;
    SimpleLayer6Config config_{};
    bool b_add_loop_edges_ = false;

};

#endif // SIMPLE_LAYER_MANAGER6_H
