#ifndef LAYER_MANAGER_H
#define LAYER_MANAGER_H

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <ceres/ceres.h>

#include "graph.h"
#include "g2o_util.h"
#include "ceres_error.h"

struct LayerConfig
{
    double new_layer_prob = 0.3;
    int max_layers = 50;
    int local_iters = 2;
    int commit_local_iters = 1;        // 새 엣지 커밋 시 로컬 최적화 이터 수 (2→1 감소)
    int commit_window_radius = 30;     // 커밋용 로컬 최적화 윈도우 크기 (80→30 감소)
    int window_radius = 20;            // 윈도우 최적화: 엣지 양끝 기준 노드 인덱스 범위 (50→20 감소)
    double huber_delta = 0.01;
    double ema_alpha = 0.1;
    double epsilon = 1e-3;
    double p_min = 0.05;
    double theta_weight = 1.0; // for residual L2 weighting
    // Conflict decision threshold (Delta = L_ij - min(L_i, L_e))
    double conflict_tau = 0.5;
    // UCT configs
    int uct_top_k = 3;   // evaluate conflict only for top-k layers
    double uct_C = 1.0;  // exploration constant
};

struct LayerStats
{
    double ema_residual = 0.0;
    int num_edges = 0;
};

struct Layer
{
    std::vector<double*> poses; // per-node copy of SE2 params [x,y,theta]
    std::vector<Edge*> edges;   // loop/bogus edges assigned to this layer
    LayerStats stats;
};

class SimpleLayerManager
{
public:
    SimpleLayerManager(ReadG2O& g, const std::string& save_path, const LayerConfig& cfg);

    // Runs the sequential probabilistic layering loop
    void run();

private:
    // Build and solve a small ceres problem for a given layer with limited iterations
    void optimize_layer(int layer_idx);
    void optimize_layer_local(int layer_idx, const Edge* ref_edge); // 추가된 엣지 중심 로컬 최적화

    // Compute residual L2 for an edge using the poses of a specific layer
    double compute_edge_residual_L2(const Edge* ed, int layer_idx) const;
    // Compute Mahalanobis distance r^T Omega r for an edge at current layer poses
    double compute_edge_mahalanobis(const Edge* ed, int layer_idx) const;
    // Approximate information gain (D-opt proxy) from an edge (no Jacobians): 0.5*logdet(I + Omega)
    double compute_info_gain_edge(const Edge* ed) const;
    // Count active loop-closure edges in a layer (edge_type == CLOSURE_EDGE)
    int count_closure_edges(int layer_idx) const;

    // Evaluate cost for a temporary problem with odometry + optional layer edges + extra edges
    double evaluate_cost(int base_layer_idx,
                         bool include_layer_edges,
                         const std::vector<Edge*>& extra_edges,
                         int iters) const;

    // MCTS/UCT helpers
    void ensure_stats_size();
    double compute_reward(double Li, double Le, double Lij) const; // negative relative delta
    double uct_score(int layer_idx) const;
    std::vector<int> pick_topk_layers(int k) const; // among existing layers (>=1)
    void update_stats(int layer_idx, double reward, bool success);
    
    // Li cache (per-layer evaluation cost without new edge)
    void ensure_Li_cache_size();
    double get_Li(int layer_idx) const;    // compute if invalid, then return
    void invalidate_Li(int layer_idx);     // call when layer is optimized/changed
    mutable std::vector<double> Li_cache;  // cached L_i per layer
    mutable std::vector<char> Li_valid;    // 0/1 flags per layer

    // Update selection probabilities based on layer EMA residuals
    void update_probabilities();

    // Sample a layer index according to categorical distribution in probs
    int sample_layer_index();

    // Create a new layer (if under limit); returns its index
    int create_new_layer();
    // Create a new layer by copying poses and edges from a base layer; returns new index
    int create_new_layer_from(int base_layer_idx);

    // Dump results: layer mapping and best layer poses
    void save_results();
    void log_line(const std::string& s);
    void print_summary();

private:
    ReadG2O& g2o;
    std::string save_path;
    LayerConfig config;

    std::vector<Layer> layers;
    std::vector<double> probs; // selection probabilities per layer (legacy; unused in UCT mode)
    struct LayerMctsStats { double visits=0.0; double total_reward=0.0; int success=0; int last_step=0; };
    std::vector<LayerMctsStats> mstats; // aligned with layers

    // Edges to assign, in order: closure edges, then bogus edges
    std::vector<Edge*> candidate_edges;
    // Records (global edge index in candidate_edges, assigned layer index)
    std::vector<std::pair<int,int>> assignments;

    // Logging
    std::ofstream logfile;
    int step_counter = 0;
};

#endif // LAYER_MANAGER_H
