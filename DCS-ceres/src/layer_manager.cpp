#include "layer_manager.h"

#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <future>
#include <Eigen/Dense>
#include <unordered_set>

using std::vector;
using std::pair;
using std::string;

SimpleLayerManager::SimpleLayerManager(ReadG2O& g, const std::string& save_path, const LayerConfig& cfg)
    : g2o(g), save_path(save_path), config(cfg)
{
    // Open log file
    logfile.open(save_path + "/method3.log", std::ios::out);
    if (logfile.is_open()) {
        log_line("[init] new_layer_prob=" + std::to_string(config.new_layer_prob) +
                 ", max_layers=" + std::to_string(config.max_layers) +
                 ", local_iters=" + std::to_string(config.local_iters));
    }

    // Candidate edges: first closures, then bogus
    candidate_edges.reserve(g2o.nEdgesClosure.size() + g2o.nEdgesBogus.size());
    for (auto* e : g2o.nEdgesClosure) candidate_edges.push_back(e);
    for (auto* e : g2o.nEdgesBogus) candidate_edges.push_back(e);

    // Layer 0: odometry-only baseline poses
    layers.emplace_back();
    layers[0].poses.resize(g2o.nNodes.size());
    for (size_t i = 0; i < g2o.nNodes.size(); ++i) {
        layers[0].poses[i] = new double[3]{ g2o.nNodes[i]->p[0], g2o.nNodes[i]->p[1], g2o.nNodes[i]->p[2] };
    }

    // One initial working layer for closures
    create_new_layer();

    // Initialize equal probabilities across non-odometry layers
    probs.assign(layers.size()-1, 1.0 / std::max<size_t>(1, layers.size()-1));

    // Seed log with initial state
    log_line("[init] layers=" + std::to_string(layers.size()) +
             " (including odo layer 0), candidates=" + std::to_string(candidate_edges.size()));

    // Init UCT stats (aligned with layers)
    mstats.resize(layers.size());
    // Init Li cache (aligned with layers)
    Li_cache.assign(layers.size(), 0.0);
    Li_valid.assign(layers.size(), 0);
}

int SimpleLayerManager::create_new_layer()
{
    if ((int)layers.size() >= config.max_layers) return (int)layers.size()-1;
    Layer newLayer;
    newLayer.poses.resize(g2o.nNodes.size());
    for (size_t i = 0; i < g2o.nNodes.size(); ++i) {
        newLayer.poses[i] = new double[3]{ layers[0].poses[i][0], layers[0].poses[i][1], layers[0].poses[i][2] };
    }
    layers.push_back(std::move(newLayer));
    // Resize probabilities for non-odo layers and renormalize
    probs.assign(layers.size()-1, 1.0 / std::max<size_t>(1, layers.size()-1));
    log_line("[layer] created new layer index=" + std::to_string(layers.size()-1) +
             ", total_layers=" + std::to_string(layers.size()));
    // UCT stats expand
    mstats.resize(layers.size());
    // Li cache expand
    Li_cache.resize(layers.size());
    Li_valid.resize(layers.size());
    Li_valid.back() = 0;
    return (int)layers.size()-1;
}

int SimpleLayerManager::create_new_layer_from(int base_layer_idx)
{
    if ((int)layers.size() >= config.max_layers) return (int)layers.size()-1;
    if (base_layer_idx < 0 || base_layer_idx >= (int)layers.size()) return (int)layers.size()-1;

    Layer newLayer;
    // copy poses from base layer
    newLayer.poses.resize(g2o.nNodes.size());
    for (size_t i = 0; i < g2o.nNodes.size(); ++i) {
        newLayer.poses[i] = new double[3]{ layers[base_layer_idx].poses[i][0], layers[base_layer_idx].poses[i][1], layers[base_layer_idx].poses[i][2] };
    }
    // copy edges from base layer (shared pointers)
    newLayer.edges = layers[base_layer_idx].edges;

    layers.push_back(std::move(newLayer));
    // resize helpers
    probs.assign(layers.size()-1, 1.0 / std::max<size_t>(1, layers.size()-1));
    log_line("[layer] created child layer index=" + std::to_string(layers.size()-1) +
             " from parent=" + std::to_string(base_layer_idx));
    mstats.resize(layers.size());
    Li_cache.resize(layers.size());
    Li_valid.resize(layers.size());
    Li_valid.back() = 0;
    return (int)layers.size()-1;
}

void SimpleLayerManager::optimize_layer(int layer_idx)
{
    if (layer_idx <= 0) return; // layer 0 is odometry baseline; skip optimizing it here
    Layer& L = layers[layer_idx];

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config.huber_delta);

    // Add odometry constraints across this layer's pose copies
    for (auto* ed : g2o.nEdgesOdometry) {
        ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost, loss, L.poses[ed->a->index], L.poses[ed->b->index]);
    }
    // Add this layer's loop/bogus edges
    for (auto* ed : L.edges) {
        int ia = ed->a->index, ib = ed->b->index;
        if (ia == ib) continue; // avoid self-loop causing duplicate parameter blocks
        ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost, loss, L.poses[ia], L.poses[ib]);
    }

    // Fix the first pose (anchor) in this layer
    problem.SetParameterBlockConstant(L.poses[0]);

    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, config.local_iters);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void SimpleLayerManager::optimize_layer_local(int layer_idx, const Edge* ref_edge)
{
    if (layer_idx <= 0) return;
    const int N = (int)g2o.nNodes.size();
    const int a = ref_edge ? ref_edge->a->index : 0;
    const int b = ref_edge ? ref_edge->b->index : 0;
    const int lo = std::max(0, std::min(a, b) - config.commit_window_radius);
    const int hi = std::min(N - 1, std::max(a, b) + config.commit_window_radius);

    Layer& L = layers[layer_idx];

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config.huber_delta);

    // Odometry constraints within window
    for (auto* ed : g2o.nEdgesOdometry) {
        int ia = ed->a->index, ib = ed->b->index;
        if (ia < lo || ia > hi || ib < lo || ib > hi) continue;
        ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost, loss, L.poses[ia], L.poses[ib]);
    }
    // Layer edges within window (including the newly added one)
    for (auto* ed : L.edges) {
        int ia = ed->a->index, ib = ed->b->index;
        if (ia < lo || ia > hi || ib < lo || ib > hi) continue;
        if (ia == ib) continue; // guard against self-loop
        ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost, loss, L.poses[ia], L.poses[ib]);
    }

    // Anchor: fix the first pose inside window (or 0 as fallback)
    int anchor = std::max(0, lo);
    problem.SetParameterBlockConstant(L.poses[anchor]);

    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, config.commit_local_iters);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

double SimpleLayerManager::compute_edge_residual_L2(const Edge* ed, int layer_idx) const
{
    const double* P1 = layers[layer_idx].poses[ed->a->index];
    const double* P2 = layers[layer_idx].poses[ed->b->index];

    // Build transforms
    const double c1 = std::cos(P1[2]);
    const double s1 = std::sin(P1[2]);
    const double c2 = std::cos(P2[2]);
    const double s2 = std::sin(P2[2]);

    // w_T_a
    double Ta[3][3] = {{c1, -s1, P1[0]}, {s1, c1, P1[1]}, {0.0, 0.0, 1.0}};
    // w_T_b
    double Tb[3][3] = {{c2, -s2, P2[0]}, {s2, c2, P2[1]}, {0.0, 0.0, 1.0}};
    // a_Tcap_b
    const double ct = std::cos(ed->theta);
    const double st = std::sin(ed->theta);
    double Tcap[3][3] = {{ct, -st, ed->x}, {st, ct, ed->y}, {0.0, 0.0, 1.0}};

    // Compute diff = Tcap^{-1} * (Ta^{-1} * Tb)
    auto inv = [](const double T[3][3], double R[3][3]){
        double c = T[0][0];
        double s = T[1][0];
        R[0][0]= c;  R[0][1]= s;  R[0][2]= -(c*T[0][2] + s*T[1][2]);
        R[1][0]= -s; R[1][1]= c;  R[1][2]= -(-s*T[0][2] + c*T[1][2]);
        R[2][0]= 0;  R[2][1]= 0;  R[2][2]= 1;
    };
    auto mul = [](const double A[3][3], const double B[3][3], double C[3][3]){
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                C[i][j]=0; for(int k=0;k<3;++k) C[i][j]+=A[i][k]*B[k][j];
            }
        }
    };

    double Ta_inv[3][3], Tb_in_a[3][3], Tcap_inv[3][3], diff[3][3];
    inv(Ta, Ta_inv);
    mul(Ta_inv, Tb, Tb_in_a);
    inv(Tcap, Tcap_inv);
    mul(Tcap_inv, Tb_in_a, diff);

    // residual vector (ex, ey, etheta)
    const double ex = diff[0][2];
    const double ey = diff[1][2];
    const double et = std::asin(std::min(1.0, std::max(-1.0, diff[1][0])));
    return std::sqrt(ex*ex + ey*ey + config.theta_weight*et*et);
}

double SimpleLayerManager::compute_edge_mahalanobis(const Edge* ed, int layer_idx) const
{
    // Compute residual vector r = [ex, ey, etheta] at current poses
    const double* P1 = layers[layer_idx].poses[ed->a->index];
    const double* P2 = layers[layer_idx].poses[ed->b->index];

    const double c1 = std::cos(P1[2]);
    const double s1 = std::sin(P1[2]);
    const double c2 = std::cos(P2[2]);
    const double s2 = std::sin(P2[2]);

    double Ta[3][3] = {{c1, -s1, P1[0]}, {s1, c1, P1[1]}, {0.0, 0.0, 1.0}};
    double Tb[3][3] = {{c2, -s2, P2[0]}, {s2, c2, P2[1]}, {0.0, 0.0, 1.0}};
    const double ct = std::cos(ed->theta);
    const double st = std::sin(ed->theta);
    double Tcap[3][3] = {{ct, -st, ed->x}, {st, ct, ed->y}, {0.0, 0.0, 1.0}};

    auto inv = [](const double T[3][3], double R[3][3]){
        double c = T[0][0];
        double s = T[1][0];
        R[0][0]= c;  R[0][1]= s;  R[0][2]= -(c*T[0][2] + s*T[1][2]);
        R[1][0]= -s; R[1][1]= c;  R[1][2]= -(-s*T[0][2] + c*T[1][2]);
        R[2][0]= 0;  R[2][1]= 0;  R[2][2]= 1;
    };
    auto mul = [](const double A[3][3], const double B[3][3], double C[3][3]){
        for(int i=0;i<3;++i){
            for(int j=0;j<3;++j){
                C[i][j]=0; for(int k=0;k<3;++k) C[i][j]+=A[i][k]*B[k][j];
            }
        }
    };

    double Ta_inv[3][3], Tb_in_a[3][3], Tcap_inv[3][3], diff[3][3];
    inv(Ta, Ta_inv);
    mul(Ta_inv, Tb, Tb_in_a);
    inv(Tcap, Tcap_inv);
    mul(Tcap_inv, Tb_in_a, diff);

    const double ex = diff[0][2];
    const double ey = diff[1][2];
    const double et = std::asin(std::min(1.0, std::max(-1.0, diff[1][0])));

    // Information matrix Omega from edge
    Eigen::Matrix3d Omega;
    Omega << ed->I11, ed->I12, ed->I13,
             ed->I12, ed->I22, ed->I23,
             ed->I13, ed->I23, ed->I33;

    Eigen::Vector3d rvec(ex, ey, et);
    double m = rvec.transpose() * Omega * rvec;
    if (m < 0.0) m = 0.0; // numerical safety
    return m;
}

double SimpleLayerManager::compute_info_gain_edge(const Edge* ed) const
{
    // Use edge information matrix only as a cheap D-opt proxy: ΔH ≈ 0.5*logdet(I + Omega)
    Eigen::Matrix3d Omega;
    Omega << ed->I11, ed->I12, ed->I13,
             ed->I12, ed->I22, ed->I23,
             ed->I13, ed->I23, ed->I33;
    // Numerical safety: ensure symmetry and clamp tiny negatives
    Omega = 0.5*(Omega + Omega.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Omega);
    Eigen::Vector3d evals = es.eigenvalues().cwiseMax(1e-12);
    // logdet(I + Omega) = sum log(1 + lambda_i)
    double logdet = std::log(1.0 + evals[0]) + std::log(1.0 + evals[1]) + std::log(1.0 + evals[2]);
    return 0.5 * logdet;
}

int SimpleLayerManager::count_closure_edges(int layer_idx) const
{
    if (layer_idx <= 0 || layer_idx >= (int)layers.size()) return 0;
    int cnt = 0;
    for (auto* ed : layers[layer_idx].edges) {
        if (ed->edge_type == CLOSURE_EDGE) cnt++;
    }
    return cnt;
}

void SimpleLayerManager::update_probabilities()
{
    const int K = (int)layers.size() - 1; // excluding odo layer 0
    if (K <= 0) return;
    vector<double> scores(K, 0.0);
    for (int k = 0; k < K; ++k) {
        double ema = layers[k+1].stats.ema_residual;
        scores[k] = 1.0 / (config.epsilon + ema);
    }
    double floor_val = config.p_min;
    double sum = 0.0;
    for (int k = 0; k < K; ++k) { scores[k] = std::max(scores[k], floor_val); sum += scores[k]; }
    if (sum <= 0) {
        probs.assign(K, 1.0 / K);
        return;
    }
    probs.resize(K);
    for (int k = 0; k < K; ++k) probs[k] = scores[k] / sum;
}

int SimpleLayerManager::sample_layer_index()
{
    // Return actual layer index (>=1)
    if (probs.empty()) return 1;
    double u = (double)std::rand() / (double)RAND_MAX;
    double cum = 0.0;
    for (int k = 0; k < (int)probs.size(); ++k) {
        cum += probs[k];
        if (u <= cum) return k+1; // offset by 1 because layer 0 is odo
    }
    return (int)probs.size(); // last
}

void SimpleLayerManager::run()
{
    // Sequentially process candidate edges using UCT top-k + conflict-aware assignment
    assignments.reserve(candidate_edges.size());
    for (int i = 0; i < (int)candidate_edges.size(); ++i) {
        step_counter++;
        Edge* ed = candidate_edges[i];

        // Precompute Le once per edge (odometry + e only)
        double L_e = evaluate_cost(0, false, {ed}, std::max(1, config.local_iters));

        // Pick top-k candidate layers by UCT
        std::vector<int> topk = pick_topk_layers(config.uct_top_k);
        std::string uctlog = "[uct] topk=";
        for (size_t t=0;t<topk.size();++t){
            int k = topk[t];
            uctlog += "L" + std::to_string(k) + "(" + std::to_string(uct_score(k)) + ")";
            if (t+1<topk.size()) uctlog += ", ";
        }
        log_line(uctlog);

        // Precompute L_i for top-k sequentially (uses cache; avoids races)
        std::vector<double> Li_vals(topk.size(), 0.0);
        for (size_t t = 0; t < topk.size(); ++t) {
            Li_vals[t] = get_Li(topk[t]);
        }

        // Compute Le per-candidate layer on the same base poses (swap test)
        std::vector<double> Le_vals(topk.size(), 0.0);
        int eval_iters = std::max(1, config.local_iters);
        for (size_t t = 0; t < topk.size(); ++t) {
            int k = topk[t];
            Le_vals[t] = evaluate_cost(k, false, std::vector<Edge*>{ed}, eval_iters);
        }

        // Launch parallel evaluation for L_ij on top-k candidates
        std::vector<std::future<double>> futures;
        futures.reserve(topk.size());
        for (int k : topk) {
            futures.emplace_back(std::async(std::launch::async, [this, k, ed, eval_iters]() {
                return this->evaluate_cost(k, true, std::vector<Edge*>{const_cast<Edge*>(ed)}, eval_iters);
            }));
        }

        // Collect results and pick best
        double best_delta = 1e100; int best_layer = -1; double best_Li=0.0; double best_Lij=0.0;
        for (size_t t = 0; t < topk.size(); ++t) {
            int k = topk[t];
            double L_i = Li_vals[t];
            double L_e_k = Le_vals[t];
            double L_ij = futures[t].get();
            double delta = L_ij - std::min(L_i, L_e_k);
            log_line("[conflict] edge_idx=" + std::to_string(i) +
                     ", try_layer=" + std::to_string(k) +
                     ", L_i=" + std::to_string(L_i) +
                     ", L_e(k)=" + std::to_string(L_e_k) +
                     ", L_ij=" + std::to_string(L_ij) +
                     ", Delta=" + std::to_string(delta));
            if (delta < best_delta) { best_delta = delta; best_layer = k; best_Li=L_i; best_Lij=L_ij; }
        }

        int target_layer = best_layer;
        bool request_split = (best_layer < 0) || (best_delta > config.conflict_tau);
        bool did_split = false;
        if (request_split) {
            size_t prev_layers = layers.size();
            int created_idx = (best_layer >= 1) ? create_new_layer_from(best_layer) : create_new_layer();
            if (layers.size() > prev_layers) {
                // Successfully created a new child from parent; assign new edge to parent (best_layer)
                target_layer = (best_layer >= 1) ? best_layer : created_idx;
                did_split = true;
                log_line("[split] edge_idx=" + std::to_string(i) +
                         ", Delta=" + std::to_string(best_delta) +
                         ", child_layer=" + std::to_string(created_idx) +
                         ", parent_assigned_layer=" + std::to_string(target_layer));
            } else {
                // Reached layer limit; fallback to best existing layer
                target_layer = (best_layer >= 1) ? best_layer : 1;
                log_line("[split-fallback] edge_idx=" + std::to_string(i) +
                         ", Delta=" + std::to_string(best_delta) +
                         ", fallback_layer=" + std::to_string(target_layer));
            }
        }

        log_line("[assign] edge_idx=" + std::to_string(i) +
                 ", a=" + std::to_string(ed->a->index) +
                 ", b=" + std::to_string(ed->b->index) +
                 ", type=" + std::to_string(ed->edge_type) +
                 ", to_layer=" + std::to_string(target_layer));
        layers[target_layer].edges.push_back(ed);
        assignments.emplace_back(i, target_layer);

        // Local optimization (new-edge-centered) and residual feedback
        double ema_prev = layers[target_layer].stats.ema_residual;
        optimize_layer_local(target_layer, ed);
        // Invalidate Li for modified layer
        invalidate_Li(target_layer);
        double r = compute_edge_residual_L2(ed, target_layer);
        LayerStats& st = layers[target_layer].stats;
        st.ema_residual = (1.0 - config.ema_alpha) * st.ema_residual + config.ema_alpha * r;
        st.num_edges += 1;
        log_line("[residual] layer=" + std::to_string(target_layer) +
                 ", r_new=" + std::to_string(r) +
                 ", ema_prev=" + std::to_string(ema_prev) +
                 ", ema_now=" + std::to_string(st.ema_residual));

        // Backprop: update UCT stats for selected layer
        // New reward: rollout negative log-posterior decrease + info gain proxy - sparsity penalty by active closures
        double delta_cost_rel = (best_Lij - best_Li) / (config.epsilon + best_Li);
        double info_gain = compute_info_gain_edge(ed);
        int n_lc = count_closure_edges(target_layer) + ((ed->edge_type == CLOSURE_EDGE) ? 1 : 0);
        const double ALPHA_INFO = 0.1; // weight for info gain
        const double BETA_SPARSE = 0.05; // sparsity penalty weight
        double reward = -delta_cost_rel + ALPHA_INFO * info_gain - BETA_SPARSE * static_cast<double>(n_lc);
        // clip to [-1, 1]
        if (reward > 1.0) reward = 1.0; else if (reward < -1.0) reward = -1.0;
        // Success if we didn't split and delta within threshold
        bool success = (!did_split) && (best_delta <= config.conflict_tau);
        update_stats(target_layer, reward, success);
    }

    // Save outputs
    save_results();
    // Print summary stats
    print_summary();
}

void SimpleLayerManager::ensure_stats_size()
{
    if (mstats.size() != layers.size()) mstats.resize(layers.size());
}

void SimpleLayerManager::ensure_Li_cache_size()
{
    if (Li_cache.size() != layers.size()) Li_cache.resize(layers.size(), 0.0);
    if (Li_valid.size() != layers.size()) Li_valid.resize(layers.size(), 0);
}

double SimpleLayerManager::get_Li(int layer_idx) const
{
    if (layer_idx <= 0) return 0.0; // layer 0 is odometry-only baseline
    if (Li_cache.size() != layers.size() || Li_valid.size() != layers.size()) {
        const_cast<SimpleLayerManager*>(this)->ensure_Li_cache_size();
    }
    if (!Li_valid[layer_idx]) {
        double Li = evaluate_cost(layer_idx, true, {}, std::max(1, config.local_iters));
        Li_cache[layer_idx] = Li;
        Li_valid[layer_idx] = 1;
    }
    return Li_cache[layer_idx];
}

void SimpleLayerManager::invalidate_Li(int layer_idx)
{
    if (Li_valid.size() != layers.size()) ensure_Li_cache_size();
    if (layer_idx >= 0 && layer_idx < (int)Li_valid.size()) Li_valid[layer_idx] = 0;
}

double SimpleLayerManager::compute_reward(double Li, double Le, double Lij) const
{
    double base = std::min(Li, Le);
    double delta = Lij - base;
    double drel = delta / (config.epsilon + base);
    // Clip reward to [-1, 1]
    double r = -drel;
    if (r > 1.0) r = 1.0; if (r < -1.0) r = -1.0;
    return r;
}

double SimpleLayerManager::uct_score(int layer_idx) const
{
    // Skip odo layer 0; caller supplies k>=1. Compute UCT with average reward.
    const auto& st = mstats[layer_idx];
    double q = st.total_reward / (1.0 + st.visits);
    double N = 1.0; // total visits across layers (approx)
    for (size_t i=1;i<mstats.size();++i) N += mstats[i].visits;
    double u = config.uct_C * std::sqrt(std::log(N) / (1.0 + st.visits));
    return q + u;
}

std::vector<int> SimpleLayerManager::pick_topk_layers(int k) const
{
    std::vector<int> idx;
    for (int i=1;i<(int)layers.size();++i) idx.push_back(i);
    if (idx.empty()) return idx;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){ return uct_score(a) > uct_score(b); });
    if ((int)idx.size() > k) idx.resize(k);
    return idx;
}

void SimpleLayerManager::update_stats(int layer_idx, double reward, bool success)
{
    ensure_stats_size();
    auto& st = mstats[layer_idx];
    st.visits += 1.0;
    st.total_reward += reward;
    if (success) st.success += 1;
    st.last_step = step_counter;
    log_line("[uct_update] layer=" + std::to_string(layer_idx) +
             ", visits=" + std::to_string(st.visits) +
             ", reward=" + std::to_string(reward));
}

void SimpleLayerManager::save_results()
{
    // Save assignments
    {
        std::ofstream fp(save_path + "/layers.txt");
        for (auto& pr : assignments) {
            fp << pr.first << " " << pr.second << "\n";
        }
    }

    // Pick best non-odo layer by minimal ema residual
    int best = -1; double best_val = 1e100;
    for (int k = 1; k < (int)layers.size(); ++k) {
        double v = layers[k].stats.ema_residual;
        if (k==1 || v < best_val) { best = k; best_val = v; }
    }
    if (best < 0) best = 1;

    // Save best layer poses
    {
        std::ofstream fp(save_path + "/opt_nodes_method3.txt");
        for (size_t i = 0; i < layers[best].poses.size(); ++i) {
            double* p = layers[best].poses[i];
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
    }
    // Also save to opt_nodes.txt for existing plotting pipeline compatibility
    {
        std::ofstream fp(save_path + "/opt_nodes.txt");
        for (size_t i = 0; i < layers[best].poses.size(); ++i) {
            double* p = layers[best].poses[i];
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
    }

    // Most selected layer by edges count (exclude odo layer 0)
    int most = -1; size_t most_edges = 0;
    for (int k = 1; k < (int)layers.size(); ++k) {
        size_t ecount = layers[k].edges.size();
        if (k == 1 || ecount > most_edges) { most = k; most_edges = ecount; }
    }
    if (most < 0) most = best;

    // Save most selected layer poses
    {
        std::ofstream fp(save_path + "/opt_nodes_most_selected.txt");
        for (size_t i = 0; i < layers[most].poses.size(); ++i) {
            double* p = layers[most].poses[i];
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
    }

    log_line("[finish] best_layer=" + std::to_string(best) +
             ", ema=" + std::to_string(best_val));
}

double SimpleLayerManager::evaluate_cost(int base_layer_idx,
                         bool include_layer_edges,
                         const std::vector<Edge*>& extra_edges,
                         int iters) const
{
    // Build temp poses copied from the chosen base layer (0 = odometry baseline)
    std::vector<double*> poses(g2o.nNodes.size(), nullptr);
    for (size_t i = 0; i < g2o.nNodes.size(); ++i) {
        poses[i] = new double[3]{ layers[base_layer_idx].poses[i][0], layers[base_layer_idx].poses[i][1], layers[base_layer_idx].poses[i][2] };
    }

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config.huber_delta);

    // Odometry constraints
    for (auto* ed : g2o.nEdgesOdometry) {
        ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost, loss, poses[ed->a->index], poses[ed->b->index]);
    }
    // Existing layer edges
    if (include_layer_edges) {
        for (auto* ed : layers[base_layer_idx].edges) {
            int ia = ed->a->index, ib = ed->b->index;
            if (ia == ib) continue; // skip self-loops
            ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
            problem.AddResidualBlock(cost, loss, poses[ia], poses[ib]);
        }
    }
    // Extra edges
    for (auto* ed : extra_edges) {
        int ia = ed->a->index, ib = ed->b->index;
        if (ia == ib) continue; // skip self-loops
        ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
        problem.AddResidualBlock(cost, loss, poses[ia], poses[ib]);
    }

    // Anchor: fix the first pose
    problem.SetParameterBlockConstant(poses[0]);

    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, iters);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // When running multiple evaluate_cost in parallel, keep per-problem threading minimal
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    double cost = summary.final_cost;

    for (auto* p : poses) delete [] p;
    return cost;
}

// (Approximation, batch processing, deferred optimization removed)

void SimpleLayerManager::log_line(const std::string& s)
{
    // console
    std::cout << s << std::endl;
    // file
    if (logfile.is_open()) { logfile << s << '\n'; logfile.flush(); }
}

void SimpleLayerManager::print_summary()
{
    // Determine most selected layer by number of assigned edges (exclude odo layer 0)
    int best_layer = -1; size_t best_edges = 0;
    for (int l = 1; l < (int)layers.size(); ++l) {
        size_t ecount = layers[l].edges.size();
        if (l == 1 || ecount > best_edges) { best_layer = l; best_edges = ecount; }
    }

    log_line("==== Method3 Summary ====");
    log_line("total_layers=" + std::to_string((int)layers.size()-1));
    if (best_layer >= 1)
        log_line("most_selected_layer=L" + std::to_string(best_layer) + " (edges=" + std::to_string(best_edges) + ")");

    for (int l = 1; l < (int)layers.size(); ++l) {
        // unique node count participating in layer's loop edges
        std::unordered_set<int> nodes;
        for (auto* ed : layers[l].edges) {
            nodes.insert(ed->a->index);
            nodes.insert(ed->b->index);
        }
        const auto& st = mstats[l];
        double avg_reward = st.total_reward / (1.0 + st.visits);
        std::string line = "layer=L" + std::to_string(l)
                         + ", edges=" + std::to_string(layers[l].edges.size())
                         + ", nodes=" + std::to_string(nodes.size())
                         + ", visits=" + std::to_string((long long)st.visits)
                         + ", success=" + std::to_string(st.success)
                         + ", total_reward=" + std::to_string(st.total_reward)
                         + ", avg_reward=" + std::to_string(avg_reward)
                         + ", ema_residual=" + std::to_string(layers[l].stats.ema_residual);
        log_line(line);
    }

    // Try to export images for best/most-selected layers using Python (headless)
    // Paths are relative to build/ when running main via do_build.sh
    std::string cmd_best = "MPLBACKEND=Agg python ../drawer/plot_results.py --initial_poses ../save/init_nodes.txt --optimized_poses ../save/opt_nodes.txt --output ../save/plot_best.png > /dev/null 2>&1";
    std::string cmd_most = "MPLBACKEND=Agg python ../drawer/plot_results.py --initial_poses ../save/init_nodes.txt --optimized_poses ../save/opt_nodes_most_selected.txt --output ../save/plot_most_selected.png > /dev/null 2>&1";
    std::system(cmd_best.c_str());
    std::system(cmd_most.c_str());
    log_line("saved images: ../save/plot_best.png, ../save/plot_most_selected.png");
}
