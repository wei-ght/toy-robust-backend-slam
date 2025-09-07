#include "simple_layer_manager.h"

#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>

using std::vector;
using std::pair;
using std::string;

SimpleLayerManagerV2::SimpleLayerManagerV2(ReadG2O& g, const std::string& save_path, const SimpleLayerConfig& cfg)
    : g2o_(g), save_path_(save_path), config_(cfg)
{
    // 로그 파일 생성
    logfile_.open(save_path + "/method4.log", std::ios::out);
    if (logfile_.is_open()) {
        log_line("[init] Simple Layer Manager for METHOD 4");
        log_line("[init] expansion_prob=" + std::to_string(config_.expansion_prob) +
                 ", max_layers=" + std::to_string(config_.max_layers));
    }
    
    // Candidate edges 수집: closure edges + bogus edges
    candidate_edges_.reserve(g2o_.nEdgesClosure.size() + g2o_.nEdgesBogus.size());
    for (auto* e : g2o_.nEdgesClosure) candidate_edges_.push_back(e);
    for (auto* e : g2o_.nEdgesBogus) candidate_edges_.push_back(e);
    
    // Root 레이어 생성 (모든 노드 포함, 엣지 없음)
    auto root_layer = std::make_unique<SimpleLayer>();
    root_layer_id_ = generate_layer_id();
    root_layer->id = root_layer_id_;
    root_layer->parent_id = "";
    
    // 모든 노드의 pose 복사본 생성
    root_layer->poses.resize(g2o_.nNodes.size());
    for (size_t i = 0; i < g2o_.nNodes.size(); ++i) {
        root_layer->poses[i] = new double[3]{
            g2o_.nNodes[i]->p[0], 
            g2o_.nNodes[i]->p[1], 
            g2o_.nNodes[i]->p[2]
        };
    }
    
    layers_[root_layer_id_] = std::move(root_layer);
    
    log_line("[init] root layer " + root_layer_id_ + " created with " + 
             std::to_string(g2o_.nNodes.size()) + " nodes");
    log_line("[init] candidate edges: " + std::to_string(candidate_edges_.size()));
}

SimpleLayerManagerV2::~SimpleLayerManagerV2()
{
    // 메모리 정리
    for (auto& pair : layers_) {
        for (auto* pose : pair.second->poses) {
            delete[] pose;
        }
    }
    
    if (logfile_.is_open()) {
        logfile_.close();
    }
}

void SimpleLayerManagerV2::run()
{
    log_line("[run] Starting METHOD 4 with " + std::to_string(candidate_edges_.size()) + " edges");
    
    assignments_.reserve(candidate_edges_.size());
    
    for (int i = 0; i < static_cast<int>(candidate_edges_.size()); ++i) {
        step_counter_++;
        Edge* edge = candidate_edges_[i];
        
        log_line("[step " + std::to_string(step_counter_) + "] Processing edge (" + 
                 std::to_string(edge->a->index) + "," + std::to_string(edge->b->index) + 
                 ") type=" + std::to_string(edge->edge_type));
        
        // MCTS로 최적 레이어 선택
        std::string selected_layer = select_layer_by_uct();
        
        // Residual 기반 엣지 필터링
        double residual = calculate_edge_residual(selected_layer, edge);
        log_line("[residual] edge residual=" + std::to_string(residual) + 
                 ", low=" + std::to_string(config_.residual_low) + 
                 ", high=" + std::to_string(config_.residual_high));
        
        // R_high 이상이면 스킵
        if (residual >= config_.residual_high) {
            log_line("[skip] edge residual too high, skipping");
            continue;
        }
        
        // R_low 이하이거나 확률적 선택으로 엣지 추가 결정
        bool should_add = should_add_edge(selected_layer, edge);
        if (!should_add) {
            log_line("[skip] edge not selected by probabilistic filtering");
            continue;
        }
        
        // 엣지 추가 후 레이어 분할 여부 결정
        if (layers_.size() < config_.max_layers && should_split_layer(selected_layer, edge)) {
            expand_layer(selected_layer, edge);
        } else {
            // 기존 레이어에 엣지 추가
            auto* layer = get_layer(selected_layer);
            if (layer) {
                layer->added_edges.push_back(edge);
                assignments_.emplace_back(edge, selected_layer);
                
                // 레이어 최적화: 국소 최적화 사용
                // optimize_local_window(selected_layer, 40);
                optimize_layer(selected_layer);
                
                // 보상 계산 및 backpropagation
                double reward = calculate_reward(selected_layer, edge);
                backpropagate(selected_layer, reward);
                
                log_line("[assign] edge to existing layer " + selected_layer + 
                         ", reward=" + std::to_string(reward));
            }
        }
    }
    
    save_results();
    log_line("[run] METHOD 4 completed");
}

std::string SimpleLayerManagerV2::select_layer_by_uct()
{
    if (layers_.size() == 1) {
        return root_layer_id_; // 루트 레이어만 있는 경우
    }
    
    std::string best_layer_id = root_layer_id_;
    double best_uct_value = -1e9;
    
    // 전체 방문 횟수 계산
    int total_visits = 0;
    for (auto& pair : layers_) {
        const auto& id = pair.first;
        auto& layer = pair.second;
        total_visits += layer->visits;
    }
    total_visits = std::max(total_visits, 1);
    
    // UCT 값 계산하여 최적 레이어 선택
    for (auto& pair : layers_) {
        const auto& id = pair.first;
        auto& layer = pair.second;
        if (layer->visits == 0) {
            // 방문하지 않은 레이어는 최우선 선택
            return id;
        }
        
        double avg_reward = layer->total_reward / layer->visits;
        double exploration = config_.mcts_exploration_c * 
                            std::sqrt(std::log(total_visits) / layer->visits);
        double uct_value = avg_reward + exploration;
        
        if (uct_value > best_uct_value) {
            best_uct_value = uct_value;
            best_layer_id = id;
        }
    }
    
    return best_layer_id;
}

bool SimpleLayerManagerV2::should_split_layer(const std::string& layer_id, Edge* new_edge)
{
    auto* layer = get_layer(layer_id);
    if (!layer || layer->added_edges.empty()) {
        return false; // 추가된 엣지가 없으면 분할할 필요 없음
    }
    
    // 3가지 경우의 비용 계산
    // Case 1: 상속받은 엣지들 + 현재 추가된 엣지들 (현재 상태)
    double cost_current = evaluate_layer_cost(layer_id);
    
    // Case 2: 상속받은 엣지들 + 새 엣지만 (기존 추가 엣지들 대신 새 엣지)
    std::vector<Edge*> backup_added_edges = layer->added_edges;
    layer->added_edges.clear();
    layer->added_edges.push_back(new_edge);
    double cost_new_only = evaluate_layer_cost(layer_id);
    
    // Case 3: 상속받은 엣지들 + 기존 추가 엣지들 + 새 엣지 (모든 엣지)
    layer->added_edges = backup_added_edges;
    layer->added_edges.push_back(new_edge);
    double cost_combined = evaluate_layer_cost(layer_id);
    
    // 원래 상태로 복원
    layer->added_edges = backup_added_edges;
    
    // 분할 조건: combined가 다른 두 경우보다 나쁜 경우
    // bool should_split = (cost_combined > cost_current) && (cost_combined > cost_new_only);
    double split_value = cost_combined - std::min(cost_current, cost_new_only);
    bool should_split = split_value > config_.conflict_tau;

    log_line("[split_check] layer=" + layer_id + 
             ", cost_current=" + std::to_string(cost_current) +
             ", cost_new_only=" + std::to_string(cost_new_only) +
             ", cost_combined=" + std::to_string(cost_combined) +
             ", should_split=" + (should_split ? "true" : "false") +
             ", split_value=" + std::to_string(split_value));
    
    return should_split;
}

void SimpleLayerManagerV2::expand_layer(const std::string& parent_id, Edge* new_edge)
{
    if (layers_.size() >= config_.max_layers) {
        log_line("[expand] max layers reached, adding to parent instead");
        auto* parent_layer = get_layer(parent_id);
        if (parent_layer) {
            parent_layer->added_edges.push_back(new_edge);
            assignments_.emplace_back(new_edge, parent_id);
            // 국소 최적화로 대체
            optimize_local_window(parent_id, 20);
            double reward = calculate_reward(parent_id, new_edge);
            backpropagate(parent_id, reward);
        }
        return;
    }
    
    // 두 개의 자식 레이어 생성: 하나는 엣지 포함, 하나는 엣지 미포함
    std::string child_include_id = create_child_layer(parent_id, new_edge, true);
    // std::string child_exclude_id = create_child_layer(parent_id, new_edge, false);
    
    auto* parent = get_layer(parent_id);
    if (parent) {
        parent->children.push_back(child_include_id);
        // parent->children.push_back(child_exclude_id);
    }
    
    // 포함하는 레이어에 엣지 할당
    assignments_.emplace_back(new_edge, child_include_id);
    
    // 두 자식 레이어 모두 최적화: 국소 최적화로 대체
    optimize_local_window(child_include_id, 20);
    // optimize_layer(child_exclude_id);
    
    // 보상 계산 및 backpropagation
    double reward_include = calculate_reward(child_include_id, new_edge);
    // double reward_exclude = calculate_reward(child_exclude_id, nullptr);
    
    backpropagate(child_include_id, reward_include);
    // backpropagate(child_exclude_id, reward_exclude);
    
    // log_line("[expand] created children: " + child_include_id + " (include), " + 
    //          child_exclude_id + " (exclude)");
    log_line("[expand] created children: " + child_include_id + " (include),");
    log_line("[rewards] include=" + std::to_string(reward_include));
}

std::string SimpleLayerManagerV2::create_child_layer(const std::string& parent_id, 
                                                   Edge* new_edge, bool include_edge)
{
    auto* parent = get_layer(parent_id);
    if (!parent) return "";
    
    auto child_layer = std::make_unique<SimpleLayer>();
    child_layer->id = generate_layer_id();
    child_layer->parent_id = parent_id;
    
    // 부모의 poses 복사
    child_layer->poses.resize(g2o_.nNodes.size());
    for (size_t i = 0; i < g2o_.nNodes.size(); ++i) {
        child_layer->poses[i] = new double[3]{
            parent->poses[i][0],
            parent->poses[i][1], 
            parent->poses[i][2]
        };
    }
    
    // 부모의 모든 엣지들을 상속받은 엣지로 설정
    child_layer->inherited_edges = parent->get_all_edges();
    
    // 새 엣지 추가 여부 결정
    if (include_edge && new_edge) {
        child_layer->added_edges.push_back(new_edge);
    }
    
    std::string child_id = child_layer->id;
    layers_[child_id] = std::move(child_layer);
    
    return child_id;
}

double SimpleLayerManagerV2::calculate_reward(const std::string& layer_id, Edge* added_edge)
{
    // r = −Δcost_rel + α·ΔH − β·n_lc(k)
    
    double delta_cost_rel = calculate_cost_delta_rel(layer_id, added_edge);
    double info_gain = added_edge ? calculate_info_gain(added_edge) : 0.0;
    int n_closure = count_closure_edges(layer_id, added_edge);
    
    double reward = -delta_cost_rel + 
                   config_.alpha_info * info_gain - 
                   config_.beta_sparse * static_cast<double>(n_closure);
    
    // [-1, 1] 범위로 클리핑
    reward = std::max(-1.0, std::min(1.0, reward));
    
    log_line("[reward] layer=" + layer_id + 
             ", delta_cost_rel=" + std::to_string(delta_cost_rel) +
             ", info_gain=" + std::to_string(info_gain) +
             ", n_closure=" + std::to_string(n_closure) +
             ", final_reward=" + std::to_string(reward));
    
    return reward;
}

double SimpleLayerManagerV2::calculate_cost_delta_rel(const std::string& layer_id, Edge* edge)
{
    if (!edge) return 0.0;
    
    auto* layer = get_layer(layer_id);
    if (!layer) return 0.0;
    
    // Li: 엣지 추가 전 비용
    double Li = evaluate_layer_cost(layer_id);
    
    // 임시로 엣지 제거하여 이전 상태 계산
    auto it = std::find(layer->added_edges.begin(), layer->added_edges.end(), edge);
    if (it != layer->added_edges.end()) {
        layer->added_edges.erase(it);
        double Li_prev = evaluate_layer_cost(layer_id);
        layer->added_edges.push_back(edge); // 다시 추가
        
        // Δcost_rel = (Lij - Li) / (ε + Li)
        return (Li - Li_prev) / (config_.epsilon + Li_prev);
    }
    
    return 0.0;
}

double SimpleLayerManagerV2::calculate_info_gain(Edge* edge)
{
    if (!edge) return 0.0;
    
    // ΔH ≈ 0.5·logdet(I + Ω_e)
    Eigen::Matrix3d Omega;
    Omega << edge->I11, edge->I12, edge->I13,
             edge->I12, edge->I22, edge->I23,
             edge->I13, edge->I23, edge->I33;
    
    // 수치적 안정성을 위한 대칭화
    Omega = 0.5 * (Omega + Omega.transpose());
    
    // 고유값 분해
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Omega);
    Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(1e-12);
    
    // logdet(I + Omega) = sum(log(1 + lambda_i))
    double logdet = 0.0;
    for (int i = 0; i < 3; ++i) {
        logdet += std::log(1.0 + eigenvalues[i]);
    }
    
    return 0.5 * logdet;
}

int SimpleLayerManagerV2::count_closure_edges(const std::string& layer_id, Edge* additional_edge)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return 0;
    
    int count = 0;
    auto all_edges = layer->get_all_edges();
    for (auto* edge : all_edges) {
        if (edge->edge_type == CLOSURE_EDGE) {
            count++;
        }
    }
    
    // 추가되는 엣지가 closure edge인 경우 +1
    if (additional_edge && additional_edge->edge_type == CLOSURE_EDGE) {
        count++;
    }
    
    return count;
}

double SimpleLayerManagerV2::calculate_edge_residual(const std::string& layer_id, Edge* edge)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return 1e6;
    
    // 엣지의 예상 측정값과 현재 pose로부터 계산된 값의 차이
    const Node* a = edge->a;
    const Node* b = edge->b;
    
    // layer의 pose 인덱스 찾기
    int a_idx = a->index;
    int b_idx = b->index;
    
    if (a_idx >= layer->poses.size() || b_idx >= layer->poses.size()) {
        return 1e6; // 유효하지 않은 인덱스면 매우 큰 residual 반환
    }
    
    // 현재 레이어의 pose 사용
    double* pose_a = layer->poses[a_idx];
    double* pose_b = layer->poses[b_idx];
    
    // SE2 relative pose 계산: T_ab = T_a^-1 * T_b
    double dx = pose_b[0] - pose_a[0];
    double dy = pose_b[1] - pose_a[1];
    double dtheta = pose_b[2] - pose_a[2];
    
    // 각도 정규화
    while (dtheta > M_PI) dtheta -= 2 * M_PI;
    while (dtheta < -M_PI) dtheta += 2 * M_PI;
    
    // 회전을 고려한 상대 변위 계산
    double cos_a = cos(pose_a[2]);
    double sin_a = sin(pose_a[2]);
    double rel_x = cos_a * dx + sin_a * dy;
    double rel_y = -sin_a * dx + cos_a * dy;
    
    // 측정값과의 차이 (residual)
    double rx = rel_x - edge->x;
    double ry = rel_y - edge->y;
    double rtheta = dtheta - edge->theta;
    
    // 각도 residual 정규화
    while (rtheta > M_PI) rtheta -= 2 * M_PI;
    while (rtheta < -M_PI) rtheta += 2 * M_PI;
    
    // Mahalanobis distance 계산: sqrt(r^T * Info * r)
    Eigen::Vector3d residual(rx, ry, rtheta);
    Eigen::Matrix3d info_matrix;
    info_matrix << edge->I11, edge->I12, edge->I13,
                   edge->I12, edge->I22, edge->I23,
                   edge->I13, edge->I23, edge->I33;
    
    double mahalanobis_dist = sqrt(residual.transpose() * info_matrix * residual);
    return mahalanobis_dist;
}

bool SimpleLayerManagerV2::should_add_edge(const std::string& layer_id, Edge* edge)
{
    double residual = calculate_edge_residual(layer_id, edge);
    
    // R_high 이상: skip
    if (residual >= config_.residual_high) {
        return false;
    }
    
    // R_high 미만: 추가
    return true;
}

void SimpleLayerManagerV2::optimize_layer(const std::string& layer_id)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return;
    
    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config_.huber_delta);
    
    // Odometry constraints
    for (auto* edge : g2o_.nEdgesOdometry) {
        ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, 
                               layer->poses[edge->a->index], 
                               layer->poses[edge->b->index]);
    }
    
    // Layer의 loop/bogus edges (inherited + added)
    auto all_edges = layer->get_all_edges();
    for (auto* edge : all_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue; // self-loop 방지
        
        ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, 
                               layer->poses[ia], 
                               layer->poses[ib]);
    }
    
    // 첫 번째 pose 고정 (앵커)
    if (!layer->poses.empty()) {
        problem.SetParameterBlockConstant(layer->poses[0]);
    }
    
    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, config_.local_iters);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void SimpleLayerManagerV2::optimize_local_window(const std::string& layer_id, int window_size)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return;
    if (window_size <= 0) window_size = 10;
    if (layer->added_edges.empty()) return; // nothing new to focus on

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config_.huber_delta);

    // Collect active nodes around newly added edges
    std::set<int> active_nodes;
    int radius = std::max(1, window_size / 2);
    for (auto* edge : layer->added_edges) {
        int ia = edge->a->index;
        int ib = edge->b->index;
        active_nodes.insert(ia);
        active_nodes.insert(ib);
        // expand around endpoints by index radius
        int a0 = std::max(0, ia - radius);
        int a1 = std::min((int)layer->poses.size() - 1, ia + radius);
        int b0 = std::max(0, ib - radius);
        int b1 = std::min((int)layer->poses.size() - 1, ib + radius);
        for (int i = a0; i <= a1; ++i) active_nodes.insert(i);
        for (int i = b0; i <= b1; ++i) active_nodes.insert(i);
    }

    // Add odometry constraints only within active window
    std::set<int> used_nodes; // nodes that appear in any residual
    for (auto* edge : g2o_.nEdgesOdometry) {
        int ia = edge->a->index;
        int ib = edge->b->index;
        if (active_nodes.count(ia) && active_nodes.count(ib)) {
            ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
            problem.AddResidualBlock(cost, loss, layer->poses[ia], layer->poses[ib]);
            used_nodes.insert(ia);
            used_nodes.insert(ib);
        }
    }

    // Add the newly added edges (focus constraints)
    for (auto* edge : layer->added_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue;
        ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, layer->poses[ia], layer->poses[ib]);
        used_nodes.insert(ia);
        used_nodes.insert(ib);
    }
    
    // Fix a single anchor among used nodes to remove gauge.
    // Prefer node 0 if it is used; otherwise pick the smallest used node id.
    if (!used_nodes.empty()) {
        int anchor = (used_nodes.count(0) ? 0 : *used_nodes.begin());
        problem.SetParameterBlockConstant(layer->poses[anchor]);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, config_.local_iters);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

double SimpleLayerManagerV2::evaluate_layer_cost(const std::string& layer_id)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return 1e9;
    
    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config_.huber_delta);
    
    // 현재 poses로 임시 복사본 생성
    std::vector<double*> temp_poses(g2o_.nNodes.size());
    for (size_t i = 0; i < g2o_.nNodes.size(); ++i) {
        temp_poses[i] = new double[3]{
            layer->poses[i][0],
            layer->poses[i][1],
            layer->poses[i][2]
        };
    }
    
    // Odometry constraints
    for (auto* edge : g2o_.nEdgesOdometry) {
        ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, 
                               temp_poses[edge->a->index], 
                               temp_poses[edge->b->index]);
    }
    
    // Layer edges (inherited + added)
    auto all_edges = layer->get_all_edges();
    for (auto* edge : all_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue;
        
        ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, temp_poses[ia], temp_poses[ib]);
    }
    
    problem.SetParameterBlockConstant(temp_poses[0]);
    
    ceres::Solver::Options options;
    options.max_num_iterations = 1;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    double cost = summary.final_cost;
    
    // 임시 poses 메모리 해제
    for (auto* pose : temp_poses) {
        delete[] pose;
    }
    
    return cost;
}

void SimpleLayerManagerV2::backpropagate(const std::string& layer_id, double reward)
{
    std::string current_id = layer_id;
    
    while (!current_id.empty()) {
        auto* layer = get_layer(current_id);
        if (!layer) break;
        
        layer->visits++;
        layer->total_reward += reward;
        
        log_line("[backprop] layer=" + current_id + 
                 ", visits=" + std::to_string(layer->visits) +
                 ", total_reward=" + std::to_string(layer->total_reward));
        
        current_id = layer->parent_id;
    }
}

double SimpleLayerManagerV2::normalize_reward_by_edge_count(double total_reward, int edge_count)
{
    // sqrt 정규화: reward / sqrt(1 + edge_count)
    return total_reward / sqrt(1.0 + edge_count);
}

std::string SimpleLayerManagerV2::get_best_layer()
{
    std::string best_layer_id = root_layer_id_;
    double best_normalized_reward = -1e9;
    
    for (auto& pair : layers_) {
        const auto& id = pair.first;
        auto& layer = pair.second;
        if (layer->visits > 0) {
            auto all_edges = layer->get_all_edges();
            double normalized_reward = normalize_reward_by_edge_count(layer->total_reward, all_edges.size());
            if (normalized_reward > best_normalized_reward) {
                best_normalized_reward = normalized_reward;
                best_layer_id = id;
            }
        }
    }
    
    return best_layer_id;
}

std::string SimpleLayerManagerV2::get_most_visited_layer()
{
    std::string most_visited_id = root_layer_id_;
    int max_visits = 0;
    
    for (auto& pair : layers_) {
        const auto& id = pair.first;
        auto& layer = pair.second;
        if (layer->visits > max_visits) {
            max_visits = layer->visits;
            most_visited_id = id;
        }
    }
    
    return most_visited_id;
}

std::string SimpleLayerManagerV2::get_most_edges_layer()
{
    std::string most_edges_id = root_layer_id_;
    int max_edges = 0;
    
    for (auto& pair : layers_) {
        const auto& id = pair.first;
        auto& layer = pair.second;
        auto all_edges = layer->get_all_edges();
        if (all_edges.size() > max_edges) {
            max_edges = all_edges.size();
            most_edges_id = id;
        }
    }
    
    return most_edges_id;
}

void SimpleLayerManagerV2::save_results()
{
    // 3가지 기준으로 레이어 선택
    std::string best_layer_id = get_best_layer();
    std::string most_visited_id = get_most_visited_layer();
    std::string most_edges_id = get_most_edges_layer();
    
    auto* best_layer = get_layer(best_layer_id);
    auto* most_visited_layer = get_layer(most_visited_id);
    auto* most_edges_layer = get_layer(most_edges_id);
    
    // 1. Best layer (정규화된 보상 기준) - 메인 결과
    if (best_layer) {
        std::ofstream fp(save_path_ + "/opt_nodes.txt");
        for (size_t i = 0; i < best_layer->poses.size(); ++i) {
            double* p = best_layer->poses[i];
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
        fp.close();
        
        auto best_edges = best_layer->get_all_edges();
        double normalized_reward = normalize_reward_by_edge_count(best_layer->total_reward, best_edges.size());
        
        log_line("[save] best layer (normalized reward): " + best_layer_id);
        log_line("[save] best layer visits: " + std::to_string(best_layer->visits));
        log_line("[save] best layer edges: " + std::to_string(best_edges.size()));
        log_line("[save] best layer normalized reward: " + std::to_string(normalized_reward));
    }
    
    // 2. Most visited layer
    if (most_visited_layer) {
        std::ofstream fp(save_path_ + "/opt_nodes_most_visited.txt");
        for (size_t i = 0; i < most_visited_layer->poses.size(); ++i) {
            double* p = most_visited_layer->poses[i];
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
        fp.close();
        
        auto visited_edges = most_visited_layer->get_all_edges();
        log_line("[save] most visited layer: " + most_visited_id);
        log_line("[save] most visited layer visits: " + std::to_string(most_visited_layer->visits));
        log_line("[save] most visited layer edges: " + std::to_string(visited_edges.size()));
    }
    
    // 3. Most edges layer
    if (most_edges_layer) {
        std::ofstream fp(save_path_ + "/opt_nodes_most_edges.txt");
        for (size_t i = 0; i < most_edges_layer->poses.size(); ++i) {
            double* p = most_edges_layer->poses[i];
            fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }
        fp.close();
        
        auto max_edges = most_edges_layer->get_all_edges();
        log_line("[save] most edges layer: " + most_edges_id);
        log_line("[save] most edges layer visits: " + std::to_string(most_edges_layer->visits));
        log_line("[save] most edges layer edges: " + std::to_string(max_edges.size()));
    }
    
    // 레이어별 통계 저장 (헤더 포함)
    std::ofstream stats_fp(save_path_ + "/method4_stats.txt");
    stats_fp << "# layer_id visits total_reward avg_reward normalized_reward total_edges inherited_edges added_edges\n";
    for (auto& pair : layers_) {
        const auto& id = pair.first;
        auto& layer = pair.second;
        double avg_reward = layer->visits > 0 ? layer->total_reward / layer->visits : 0.0;
        auto all_edges = layer->get_all_edges();
        double normalized_reward = normalize_reward_by_edge_count(layer->total_reward, all_edges.size());
        
        stats_fp << id << " " << layer->visits << " " << layer->total_reward << " " 
                 << avg_reward << " " << normalized_reward << " " << all_edges.size() << " " 
                 << layer->inherited_edges.size() << " " << layer->added_edges.size() << "\n";
    }
    stats_fp.close();
    
    // 요약 통계 로그
    log_line("[summary] ============ METHOD 4 SUMMARY ============");
    log_line("[summary] Total layers created: " + std::to_string(layers_.size()));
    log_line("[summary] Best layer (normalized): " + best_layer_id);
    log_line("[summary] Most visited layer: " + most_visited_id);  
    log_line("[summary] Most edges layer: " + most_edges_id);
    log_line("[summary] Results saved to " + save_path_);
}

SimpleLayer* SimpleLayerManagerV2::get_layer(const std::string& layer_id)
{
    auto it = layers_.find(layer_id);
    return it != layers_.end() ? it->second.get() : nullptr;
}

std::string SimpleLayerManagerV2::generate_layer_id()
{
    return "L" + std::to_string(++layer_id_counter_);
}

void SimpleLayerManagerV2::log_line(const std::string& s)
{
    std::cout << s << std::endl;
    if (logfile_.is_open()) {
        logfile_ << s << '\n';
        logfile_.flush();
    }
}
