#include "simple_layer_manager.h"

#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <limits>
#include <memory>
#include <iterator>

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
    // 설정값 로그 (헤더/호출자가 전달한 config 사용)
    log_line("[init] snapshot_every=" + std::to_string(config_.snapshot_every));
    log_line("[init] residual_mode=" + std::to_string(config_.residual_mode));
    log_line("[init] impact_iters=" + std::to_string(config_.impact_iters) +
             ", impact_window=" + std::to_string(config_.impact_window) +
             ", impact_theta_w=" + std::to_string(config_.impact_theta_weight) +
             ", impact_pose_w=" + std::to_string(config_.impact_pose_weight));
    
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
    base_layer_id_ = root_layer_id_;
    online_active_k_ = -1;
    
    log_line("[init] root layer " + root_layer_id_ + " created with " + 
             std::to_string(g2o_.nNodes.size()) + " nodes");
    log_line("[init] candidate edges: " + std::to_string(candidate_edges_.size()));
}

SimpleLayerManagerV2::~SimpleLayerManagerV2()
{
    // 메모리 정리
    for (auto& pair : layers_) {
        for (auto* pose : pair.second->poses) delete[] pose;
        for (auto& kv : pair.second->switch_vars) delete kv.second;
    }
    
    if (logfile_.is_open()) {
        logfile_.close();
    }
}

void SimpleLayerManagerV2::run()
{
    log_line("[run] Starting METHOD 4 with " + std::to_string(candidate_edges_.size()) + " edges");
    
    // Initialize method statistics tracking
    method_start_time_ = std::chrono::high_resolution_clock::now();
    
    assignments_.reserve(candidate_edges_.size());
    
    for (int i = 0; i < static_cast<int>(candidate_edges_.size()); ++i) {
        step_counter_++;
        Edge* edge = candidate_edges_[i];
        
        auto step_start_time = std::chrono::high_resolution_clock::now();
        
        log_line("[step " + std::to_string(step_counter_) + "] Processing edge (" + 
                 std::to_string(edge->a->index) + "," + std::to_string(edge->b->index) + 
                 ") type=" + std::to_string(edge->edge_type));
        
        // MCTS로 최적 레이어 선택
        std::string selected_layer = select_layer_by_uct();
        
        // Residual 기반 엣지 필터링 (+ 고잔차 1회 재최적화 후 재평가)
        double residual = calculate_edge_residual(selected_layer, edge);
        // log_line("[residual] edge residual=" + std::to_string(residual) +
        //          ", low=" + std::to_string(config_.residual_low) +
        //          ", high=" + std::to_string(config_.residual_high));

        // if (residual >= config_.residual_high) {
        //     // 빠른 국소 최적화 후 한 번 더 평가
        //     log_line("[gate] high residual; running quick local reopt then re-evaluate");
        //     optimize_local_window(selected_layer, 20);
        //     residual = calculate_edge_residual(selected_layer, edge);
        //     log_line("[residual-recheck] edge residual=" + std::to_string(residual));
        //     if (residual >= config_.residual_high) {
        //         log_line("[skip] edge residual still high after reopt; skipping");
        //         continue;
        //     }
        // }
        
        // R_low 이하이거나 확률적 선택으로 엣지 추가 결정
        // bool should_add = should_add_edge(selected_layer, edge, residual);
        // if (!should_add) {
        //     log_line("[skip] edge not selected by probabilistic filtering");
        //     continue;
        // }
        
        // 임시 레이어 개선 기반 파이프라인 적용 (Top-K는 병합 시 수행)
        double optimization_time = process_edge_with_temp_layer(selected_layer, edge);
        
        // Current maximum node index processed (based on edge nodes)
        int max_node_idx = std::max(edge->a->index, edge->b->index);
        double cum_distance = calculate_cumulative_distance_up_to_node(max_node_idx);
        int current_layer_count = static_cast<int>(layers_.size());
        
        // Adjust node count for closure/bogus edges (they don't add new nodes)
        int adjusted_node_count = max_node_idx + 1;
        if (edge->edge_type == CLOSURE_EDGE || edge->edge_type == BOGUS_EDGE) {
            // For closure/bogus edges, the node count should reflect actual unique nodes
            // Count closure/bogus edges processed so far in this step
            int closure_bogus_count = 0;
            for (int j = 0; j <= i; ++j) {
                Edge* prev_edge = candidate_edges_[j];
                if (prev_edge->edge_type == CLOSURE_EDGE || prev_edge->edge_type == BOGUS_EDGE) {
                    if (std::max(prev_edge->a->index, prev_edge->b->index) <= max_node_idx) {
                        closure_bogus_count++;
                    }
                }
            }
            // Subtract closure/bogus edges as they don't contribute new nodes
            adjusted_node_count = std::max(1, max_node_idx + 1 - closure_bogus_count);
            
            log_line("[stats] closure/bogus edge detected, adjusted node count: " + 
                     std::to_string(max_node_idx + 1) + " -> " + std::to_string(adjusted_node_count));
        }
        
        // Track statistics for DCS method (METHOD 4 is DCS-based)
        track_method_statistics("DCS", step_counter_, adjusted_node_count, cum_distance, 
                               current_layer_count, optimization_time);
        
        // 스냅샷 저장 (요청 시)
        if (config_.snapshot_every > 0 && (step_counter_ % config_.snapshot_every == 0)) {
            save_snapshot(step_counter_);
        }
    }
    
    save_results();
    if (config_.snapshot_every > 0) {
        save_snapshot(step_counter_);
    }
    
    // Output and save statistics
    output_method_statistics();
    save_statistics_to_file(save_path_ + "/method_statistics.txt");
    
    log_line("[run] METHOD 4 completed");
}

void SimpleLayerManagerV2::run_online()
{
    log_line("[run-online] Starting METHOD 4 ONLINE with sequential nodes");

    assignments_.clear();
    step_counter_ = 0;
    
    // Initialize method statistics tracking for online mode
    method_start_time_ = std::chrono::high_resolution_clock::now();
    
    // Optimization timing log file setup
    std::string timing_log_path = save_path_ + "/optimization_timing.txt";
    std::ofstream timing_log(timing_log_path);
    timing_log << "# step_counter k edge_a edge_b edge_type selected_layer optimization_type duration_ms\n";
    timing_log.flush();

    // 누적 처리 카운터 (검증용)
    int total_processed_edges = 0;
    int total_processed_closure = 0;
    int total_processed_bogus = 0;

    int N = static_cast<int>(g2o_.nNodes.size());
    for (int k = 1; k < N; ++k) {
        online_active_k_ = k;
        step_counter_++;

        // 수집: 새 노드 k와 관련된 루프/보거스 에지들만 처리
        std::vector<Edge*> new_edges;
        new_edges.reserve(32);
        int k_closure = 0, k_bogus = 0;
        for (auto* e : g2o_.nEdgesClosure) {
            int ia = e->a->index, ib = e->b->index;
            if (std::max(ia, ib) == k) { new_edges.push_back(e); k_closure++; }
        }
        for (auto* e : g2o_.nEdgesBogus) {
            int ia = e->a->index, ib = e->b->index;
            if (std::max(ia, ib) == k) { new_edges.push_back(e); k_bogus++; }
        }

        // 누적 갱신
        total_processed_edges += (int)new_edges.size();
        total_processed_closure += k_closure;
        total_processed_bogus += k_bogus;

        log_line("[run-online] activate node k=" + std::to_string(k) +
                 ", new_edges=" + std::to_string(new_edges.size()) +
                 ", closures_k=" + std::to_string(k_closure) +
                 ", bogus_k=" + std::to_string(k_bogus));

        if (new_edges.empty()) continue;

        for (auto* edge : new_edges) {
            
            log_line("[online step " + std::to_string(step_counter_) + "] edge (" +
                     std::to_string(edge->a->index) + "," + std::to_string(edge->b->index) + ") type=" + std::to_string(edge->edge_type));

            // MCTS로 최적 레이어 선택
            std::string selected_layer = select_layer_by_uct();

            // 방어적 가드: 레이어/인덱스 유효성 체크
            auto* sel_layer_ptr = get_layer(selected_layer);
            if (!sel_layer_ptr) {
                log_line("[warn] selected layer not found: " + selected_layer);
                continue;
            }
            int a_idx = edge->a->index;
            int b_idx = edge->b->index;
            if (a_idx < 0 || b_idx < 0 ||
                a_idx >= (int)sel_layer_ptr->poses.size() ||
                b_idx >= (int)sel_layer_ptr->poses.size()) {
                log_line("[warn] edge node index OOB for layer poses: a=" + std::to_string(a_idx) +
                         ", b=" + std::to_string(b_idx) +
                         ", poses_size=" + std::to_string(sel_layer_ptr->poses.size()));
                continue;
            }

            // Residual 기반 엣지 필터링 (고잔차 1회 재최적화 후 재평가 포함)
            double residual = calculate_edge_residual(selected_layer, edge);
            log_line("[residual] edge residual=" + std::to_string(residual));

            // if (residual >= config_.residual_high) {
            //     log_line("[gate] online high residual; quick reopt up to k and recheck: " + std::to_string(residual));
            //     optimize_layer_upto_k(selected_layer, online_active_k_ >= 0 ? online_active_k_ : k);
            //     residual = calculate_edge_residual(selected_layer, edge);
            //     log_line("[residual-recheck] online edge residual=" + std::to_string(residual));
            //     if (residual >= config_.residual_high) {
            //         log_line("[skip] online edge residual still high after reopt; skipping");
            //         continue;
            //     }
            // }

            // 확률적/임계 기반 엣지 추가 결정 (오프라인과 동일)
            bool should_add = should_add_edge(selected_layer, edge, residual);
            if (!should_add) {
                log_line("[skip] edge not selected by probabilistic filtering");
                continue;
            }
            
            // 임시 레이어 개선 기반 파이프라인 적용 (Top-K는 병합 시 수행)
            double optimization_time = process_edge_with_temp_layer(selected_layer, edge);
            
            // In online mode, track statistics based on current active node k
            double cum_distance = calculate_cumulative_distance_up_to_node(k);
            int current_layer_count = static_cast<int>(layers_.size());
            
            // Adjust node count for closure/bogus edges in online mode
            int adjusted_node_count = k + 1;
            if (edge->edge_type == CLOSURE_EDGE || edge->edge_type == BOGUS_EDGE) {
                // In online mode, closure/bogus edges don't add new nodes
                // The node count should reflect the actual maximum node index processed
                adjusted_node_count = k;  // Don't count the closure edge as adding a new node
                
                log_line("[stats] online closure/bogus edge detected, adjusted node count: " + 
                         std::to_string(k + 1) + " -> " + std::to_string(adjusted_node_count));
            }
            
            // Track statistics for DCS method in online mode
            track_method_statistics("DCS_Online", step_counter_, adjusted_node_count, cum_distance, 
                                   current_layer_count, optimization_time);
            
            // 스냅샷 저장 (요청 시)
            if (config_.snapshot_every > 0 && (step_counter_ % config_.snapshot_every == 0)) {
                save_snapshot(step_counter_);
            }
        }
    }

    // 처리 총계 검증 로그
    int expected_closure = (int)g2o_.nEdgesClosure.size();
    int expected_bogus = (int)g2o_.nEdgesBogus.size();
    int expected_total = expected_closure + expected_bogus;
    int missing_total = expected_total - total_processed_edges;
    int missing_closure = expected_closure - total_processed_closure;
    int missing_bogus = expected_bogus - total_processed_bogus;

    log_line("[run-online][verify] processed_total=" + std::to_string(total_processed_edges) +
             ", processed_closure=" + std::to_string(total_processed_closure) +
             ", processed_bogus=" + std::to_string(total_processed_bogus));
    log_line("[run-online][verify] expected_total=" + std::to_string(expected_total) +
             ", expected_closure=" + std::to_string(expected_closure) +
             ", expected_bogus=" + std::to_string(expected_bogus));
    if (missing_total != 0 || missing_closure != 0 || missing_bogus != 0) {
        log_line("[run-online][verify][warn] missing_total=" + std::to_string(missing_total) +
                 ", missing_closure=" + std::to_string(missing_closure) +
                 ", missing_bogus=" + std::to_string(missing_bogus));
    } else {
        log_line("[run-online][verify] all closure/bogus edges were processed exactly once");
    }

    save_results();
    if (config_.snapshot_every > 0) {
        save_snapshot(step_counter_);
    }
    
    // Output and save statistics for online mode
    output_method_statistics();
    save_statistics_to_file(save_path_ + "/method_statistics_online.txt");
    
    log_line("[run-online] METHOD 4 ONLINE completed");
}

void SimpleLayerManagerV2::save_snapshot(int step_index)
{
    // 스냅샷 디렉토리 생성
    std::filesystem::path snap_dir = std::filesystem::path(save_path_) / "snapshots";
    std::error_code ec;
    std::filesystem::create_directories(snap_dir, ec);

    // 출력 파일명
    std::ostringstream oss;
    oss << "step_" << std::setw(4) << std::setfill('0') << step_index << ".png";
    std::string out_png = (snap_dir / oss.str()).string();

    // 온라인이면 현재 k 이하만 포함한 임시 save_path를 구성 (명시적으로 생성)
    std::ostringstream oss2; 
    oss2 << "step_" << std::setw(4) << std::setfill('0') << step_index << "_data";
    std::filesystem::path snap_data_dir = snap_dir / oss2.str();
    std::filesystem::create_directories(snap_data_dir, ec);

    // 스냅샷 데이터 작성: init_nodes, best/visited/edges poses (k 이하로 자름)
    int cap = (online_active_k_ >= 0 ? online_active_k_ : (int)g2o_.nNodes.size()-1);
    // init
    write_initial_poses_capped((snap_data_dir / "init_nodes.txt").string(), cap);
    // layers
    std::string best_id = get_best_layer();
    std::string most_vis_id = get_most_visited_layer();
    std::string most_edges_id = get_most_edges_layer();
    write_layer_poses_capped(best_id, (snap_data_dir / "opt_nodes.txt").string(), cap);
    write_layer_poses_capped(most_vis_id, (snap_data_dir / "opt_nodes_most_visited.txt").string(), cap);
    write_layer_poses_capped(most_edges_id, (snap_data_dir / "opt_nodes_most_edges.txt").string(), cap);
    // stats 복사 (그대로)
    std::filesystem::path stats_src = std::filesystem::path(save_path_) / "method4_stats.txt";
    std::filesystem::path stats_dst = snap_data_dir / "method4_stats.txt";
    if (std::filesystem::exists(stats_src)) {
        std::filesystem::copy_file(stats_src, stats_dst, std::filesystem::copy_options::overwrite_existing, ec);
    }

    // Python 스크립트 호출 (headless backend), 스냅샷 데이터 디렉토리를 save_path로 사용
    std::string cmd = std::string("MPLBACKEND=Agg python ../drawer/plot_method4_results.py ") +
                      "--save_path " + snap_data_dir.string() + " --output " + out_png +
                      " --no-show > /dev/null 2>&1";
    std::system(cmd.c_str());

    log_line("[snapshot] saved " + out_png + " (cap k=" + std::to_string(cap) + ")");
}

void SimpleLayerManagerV2::write_layer_poses_capped(const std::string& layer_id, const std::string& filepath, int max_index)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return;
    std::ofstream fp(filepath);
    int n = std::min((int)layer->poses.size(), max_index + 1);
    for (int i = 0; i < n; ++i) {
        double* p = layer->poses[i];
        fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    fp.close();
}

void SimpleLayerManagerV2::write_initial_poses_capped(const std::string& filepath, int max_index)
{
    std::ofstream fp(filepath);
    int n = std::min((int)g2o_.nNodes.size(), max_index + 1);
    for (int i = 0; i < n; ++i) {
        double* p = g2o_.nNodes[i]->p;
        fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    fp.close();
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
        
        // Visit count bonus: 많이 방문된(성공적인) 레이어에 보너스
        double visit_bonus = 0.0;
        if (total_visits > 1) {
            double visit_ratio = (double)layer->visits / (double)total_visits;
            visit_bonus = 0.3 * visit_ratio; // TODO: expose via config (visit_bonus_weight)
        }
        
        // double uct_value = avg_reward + exploration + visit_bonus;
        double uct_value = avg_reward + visit_bonus;
        
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
            // best 레이어만 최적화
            if (config_.topk_layers > 0) {
                auto topk = get_topk_layers_by_reward(config_.topk_layers);
                std::string best_id = topk.empty() ? parent_id : topk[0];
                if (online_active_k_ >= 0) optimize_layer_upto_k(best_id, online_active_k_);
                else optimize_layer(best_id);
            } else {
                // K=0이면 부모 레이어만 간단 최적화
                if (online_active_k_ >= 0) optimize_layer_upto_k(parent_id, online_active_k_);
                else optimize_layer(parent_id);
            }
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
    
    // 자식 레이어는 여기서 최적화하지 않음 (best 레이어에서만 최적화 수행)
    
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

    // 스위치 변수 생성/복사 (상속 엣지)
    for (auto* e : child_layer->inherited_edges) {
        double init_s = 1.0;
        auto it = parent->switch_vars.find(e);
        if (it != parent->switch_vars.end() && it->second) init_s = *(it->second);
        child_layer->switch_vars[e] = new double(init_s);
    }
    
    // 새 엣지 추가 여부 결정
    if (include_edge && new_edge) {
        child_layer->added_edges.push_back(new_edge);
        if (child_layer->switch_vars.find(new_edge) == child_layer->switch_vars.end()) {
            child_layer->switch_vars[new_edge] = new double(1.0);
        }
    }
    
    std::string child_id = child_layer->id;
    layers_[child_id] = std::move(child_layer);
    
    return child_id;
}

double SimpleLayerManagerV2::calculate_reward(const std::string& layer_id, Edge* added_edge)
{
    // r = −Δcost_rel + α·ΔH − β·n_active_closure(k) − γ·outlier_penalty
    
    double delta_cost_rel = calculate_cost_delta_rel(layer_id, added_edge);
    double info_gain = added_edge ? calculate_info_gain(added_edge) : 0.0;
    // 현재 활성(스위치 s > threshold)인 closure 개수
    int n_closure = count_active_closure_edges(layer_id, online_active_k_);
    
    // Outlier penalty based on chi2 value
    double outlier_penalty = 0.0;
    if (added_edge) {
        auto* layer = get_layer(layer_id);
        if (layer) {
            int ia = added_edge->a->index, ib = added_edge->b->index;
            if (ia >= 0 && ib >= 0 && ia < (int)layer->poses.size() && ib < (int)layer->poses.size()) {
                // Chi2 계산을 위한 lambda (기존 코드에서 재사용)
                auto wrap = [](double a){ while(a > M_PI) a -= 2*M_PI; while(a < -M_PI) a += 2*M_PI; return a; };
                auto edge_chi2 = [&](double* pa, double* pb, const Edge* e){
                    double dx = pb[0] - pa[0];
                    double dy = pb[1] - pa[1];
                    double dth = wrap(pb[2] - pa[2]);
                    double ca = std::cos(pa[2]), sa = std::sin(pa[2]);
                    double rel_x = ca*dx + sa*dy;
                    double rel_y = -sa*dx + ca*dy;
                    double rx = rel_x - e->x;
                    double ry = rel_y - e->y;
                    double rth = wrap(dth - e->theta);
                    Eigen::Matrix<double,3,1> r; r << rx, ry, rth;
                    Eigen::Matrix3d I;
                    I << e->I11, e->I12, e->I13,
                         e->I12, e->I22, e->I23,
                         e->I13, e->I23, e->I33;
                    I = 0.5 * (I + I.transpose());
                    return (r.transpose() * I * r)(0,0);
                };
                
                double edge_chi2_val = edge_chi2(layer->poses[ia], layer->poses[ib], added_edge);
                
                // Progressive penalty based on chi2 thresholds
                const double chi2_threshold_95 = 7.815;   // 95% confidence
                const double chi2_threshold_99 = 11.345;  // 99% confidence
                const double gamma_outlier = 0.5; // TODO: expose via config
                
                if (edge_chi2_val > chi2_threshold_99) {
                    // Very likely outlier: strong penalty
                    outlier_penalty = gamma_outlier * 1.0;
                } else if (edge_chi2_val > chi2_threshold_95) {
                    // Suspicious: moderate penalty (linear interpolation)
                    double ratio = (edge_chi2_val - chi2_threshold_95) / (chi2_threshold_99 - chi2_threshold_95);
                    outlier_penalty = gamma_outlier * ratio;
                }
            }
        }
    }
    
    double reward = -delta_cost_rel + 
                   config_.alpha_info * info_gain - 
                   config_.beta_sparse * static_cast<double>(n_closure) -
                   outlier_penalty;
    
    // [-1, 1] 범위로 클리핑
    reward = std::max(-1.0, std::min(1.0, reward));
    
    log_line("[reward] layer=" + layer_id + 
             ", delta_cost_rel=" + std::to_string(delta_cost_rel) +
             ", info_gain=" + std::to_string(info_gain) +
             ", n_closure=" + std::to_string(n_closure) +
             ", outlier_penalty=" + std::to_string(outlier_penalty) +
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
    //eigenvalue normalization
    
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

int SimpleLayerManagerV2::count_active_closure_edges(const std::string& layer_id, int k_lim) const
{
    auto it = layers_.find(layer_id);
    if (it == layers_.end()) return 0;
    const auto* layer = it->second.get();
    int count = 0;
    auto consider = [&](Edge* e){
        if (!e) return;
        if (e->edge_type != CLOSURE_EDGE) return;
        int ia = e->a->index, ib = e->b->index;
        if (k_lim >= 0 && std::max(ia, ib) > k_lim) return;
        auto sit = layer->switch_vars.find(e);
        double s = 1.0;
        if (sit != layer->switch_vars.end() && sit->second) s = *(sit->second);
        if (s > config_.sc_active_threshold) count++;
    };
    for (auto* e : layer->inherited_edges) consider(e);
    for (auto* e : layer->added_edges) consider(e);
    return count;
}

double SimpleLayerManagerV2::calculate_edge_residual(const std::string& layer_id, Edge* edge)
{
    // 모드 1: 포함/미포함 최적화 영향 기반 평가
    if (config_.residual_mode == 1) {
        // return calculate_edge_residual_impact(layer_id, edge);
        return calculate_neighbor_residual_improvement(layer_id, edge);
    }
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
    // 수치 안정화를 위해 대칭화
    info_matrix = 0.5 * (info_matrix + info_matrix.transpose());
    
    // double mahalanobis_dist = sqrt(residual.transpose() * info_matrix * residual);
    double mahalanobis_dist = sqrt(residual.transpose() * residual);
    return mahalanobis_dist;
}

double SimpleLayerManagerV2::calculate_edge_residual_impact(const std::string& layer_id, Edge* edge)
{
    auto* layer = get_layer(layer_id);
    if (!layer || !edge) return 1e6;

    // 활성 노드 집합 구성: 온라인이면 [0..k], 아니면 impact_window 기반 또는 전체
    std::set<int> active_nodes;
    int a_idx = edge->a->index;
    int b_idx = edge->b->index;
    if (online_active_k_ >= 0 && config_.impact_window == 0) {
        for (int i = 0; i <= online_active_k_ && i < (int)layer->poses.size(); ++i) active_nodes.insert(i);
    } else if (config_.impact_window > 0) {
        int radius = config_.impact_window;
        int a0 = std::max(0, a_idx - radius);
        int a1 = std::min((int)layer->poses.size() - 1, a_idx + radius);
        int b0 = std::max(0, b_idx - radius);
        int b1 = std::min((int)layer->poses.size() - 1, b_idx + radius);
        for (int i = a0; i <= a1; ++i) active_nodes.insert(i);
        for (int i = b0; i <= b1; ++i) active_nodes.insert(i);
    } else {
        for (int i = 0; i < (int)layer->poses.size(); ++i) active_nodes.insert(i);
    }

    // 임시 포즈 복사본 준비
    int N = (int)layer->poses.size();
    std::vector<double*> poses_base(N), poses_with(N);
    for (int i = 0; i < N; ++i) {
        poses_base[i] = new double[3]{layer->poses[i][0], layer->poses[i][1], layer->poses[i][2]};
        poses_with[i] = new double[3]{layer->poses[i][0], layer->poses[i][1], layer->poses[i][2]};
    }

    // CERES 문제 구성
    ceres::LossFunction* loss_base = new ceres::HuberLoss(config_.huber_delta);
    ceres::LossFunction* loss_with = new ceres::HuberLoss(config_.huber_delta);
    ceres::Problem prob_base, prob_with;
    std::set<int> used_nodes_base, used_nodes_with;
    int odom_base = 0, odom_with = 0, loop_base = 0, loop_with = 0;

    auto add_edge_if_active = [&](ceres::Problem& prob, std::set<int>& used, Edge* e, std::vector<double*>& poses, ceres::LossFunction* loss, int& odc, int& lpc){
        int ia = e->a->index, ib = e->b->index;
        if (active_nodes.count(ia) && active_nodes.count(ib) && ia != ib) {
            ceres::CostFunction* cost = OdometryResidue::Create(e->x, e->y, e->theta);
            prob.AddResidualBlock(cost, loss, poses[ia], poses[ib]);
            used.insert(ia);
            used.insert(ib);
            if (e->edge_type == ODOMETRY_EDGE) odc++; else lpc++;
        }
    };

    // 오도메트리 엣지
    for (auto* e : g2o_.nEdgesOdometry) {
        add_edge_if_active(prob_base, used_nodes_base, e, poses_base, loss_base, odom_base, loop_base);
        add_edge_if_active(prob_with, used_nodes_with, e, poses_with, loss_with, odom_with, loop_with);
    }
    // 레이어 엣지(상속+추가). 후보 엣지는 base에는 포함하지 않고 with에는 포함
    auto layer_edges = layer->get_all_edges();
    for (auto* e : layer_edges) {
        if (e == edge) continue; // 후보 엣지는 분리 처리
        add_edge_if_active(prob_base, used_nodes_base, e, poses_base, loss_base, odom_base, loop_base);
        add_edge_if_active(prob_with, used_nodes_with, e, poses_with, loss_with, odom_with, loop_with);
    }
    // 후보 엣지는 with에만 추가
    add_edge_if_active(prob_with, used_nodes_with, edge, poses_with, loss_with, odom_with, loop_with);

    // 앵커 설정
    if (!used_nodes_base.empty()) {
        int anchor = (used_nodes_base.count(0) ? 0 : *used_nodes_base.begin());
        prob_base.AddParameterBlock(poses_base[anchor], 3);
        prob_base.SetParameterBlockConstant(poses_base[anchor]);
    }
    if (!used_nodes_with.empty()) {
        int anchor = (used_nodes_with.count(0) ? 0 : *used_nodes_with.begin());
        prob_with.AddParameterBlock(poses_with[anchor], 3);
        prob_with.SetParameterBlockConstant(poses_with[anchor]);
    }

    // 제약이 전혀 없는 경우 방어적 처리
    if (used_nodes_base.empty() || (odom_base + loop_base) == 0) {
        // base 문제가 비정상이면 영향 계산 불가 → 큰 값 반환
        for (int i = 0; i < N; ++i) { delete[] poses_base[i]; delete[] poses_with[i]; }
        log_line("[impact] base problem empty; returning large residual");
        return 1e6;
    }
    if (used_nodes_with.empty() || (odom_with + loop_with) == 0) {
        // with 문제가 비정상이면 영향 없음으로 간주
        for (int i = 0; i < N; ++i) { delete[] poses_base[i]; delete[] poses_with[i]; }
        log_line("[impact] with problem empty; returning small residual");
        return 0.0;
    }

    // Solve
    ceres::Solver::Options opts;
    opts.max_num_iterations = std::max(1, config_.impact_iters);
    opts.minimizer_progress_to_stdout = false;
    opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    opts.num_threads = 1;
    ceres::Solver::Summary sum_base, sum_with;
    ceres::Solve(opts, &prob_base, &sum_base);
    ceres::Solve(opts, &prob_with, &sum_with);

    // 포즈 델타 측정 (활성 노드 교집합 기준)
    std::set<int> used;
    used.insert(used_nodes_base.begin(), used_nodes_base.end());
    used.insert(used_nodes_with.begin(), used_nodes_with.end());
    double accum = 0.0; int cnt = 0; double maxv = 0.0;
    for (int i : used) {
        double dx = poses_with[i][0] - poses_base[i][0];
        double dy = poses_with[i][1] - poses_base[i][1];
        double dth = poses_with[i][2] - poses_base[i][2];
        while (dth > M_PI) dth -= 2 * M_PI;
        while (dth < -M_PI) dth += 2 * M_PI;
        double val = std::sqrt(dx*dx + dy*dy + config_.impact_theta_weight * dth*dth);
        accum += val; cnt++;
        if (val > maxv) maxv = val;
    }
    double pose_shift = (cnt > 0 ? accum / cnt : 0.0);

    // 비용 증가(나쁨만 반영)
    double bad_cost = std::max(0.0, sum_with.final_cost - sum_base.final_cost);
    double impact = bad_cost + config_.impact_pose_weight * pose_shift;

    // 메모리 정리
    for (int i = 0; i < N; ++i) { delete[] poses_base[i]; delete[] poses_with[i]; }
    // 디버그 로그
    log_line("[impact] a=" + std::to_string(a_idx) + ", b=" + std::to_string(b_idx) +
             ", used_base=" + std::to_string(used_nodes_base.size()) +
             ", used_with=" + std::to_string(used_nodes_with.size()) +
             ", odom_base=" + std::to_string(odom_base) + ", loop_base=" + std::to_string(loop_base) +
             ", odom_with=" + std::to_string(odom_with) + ", loop_with=" + std::to_string(loop_with) +
             ", cost_base=" + std::to_string(sum_base.final_cost) +
             ", cost_with=" + std::to_string(sum_with.final_cost) +
             ", pose_shift=" + std::to_string(pose_shift) +
             ", impact=" + std::to_string(impact));

    return impact;
}

// 새 엣지(edge)를 포함하기 전/후로, edge의 양 끝 노드(a,b)와 노드를 공유하는
// 인접 엣지들의 residual 합이 얼마나 감소했는지 계산한다.
// 반환값 = sum_residual_before - sum_residual_after (양수면 개선, 클수록 좋음)
double SimpleLayerManagerV2::calculate_neighbor_residual_improvement(const std::string& layer_id, Edge* edge)
{
    auto* layer = get_layer(layer_id);
    if (!layer || !edge) return 0.0;
    // 로컬 윈도우 제거: 전체 노드/엣지를 대상으로 평가
    int a_idx = edge->a->index;
    int b_idx = edge->b->index;

    // 포즈 복사본 준비
    int N = (int)layer->poses.size();
    std::vector<double*> poses_base(N), poses_with(N);
    for (int i = 0; i < N; ++i) {
        poses_base[i] = new double[3]{layer->poses[i][0], layer->poses[i][1], layer->poses[i][2]};
        poses_with[i] = new double[3]{layer->poses[i][0], layer->poses[i][1], layer->poses[i][2]};
    }

    ceres::LossFunction* loss_base = new ceres::HuberLoss(config_.huber_delta);
    ceres::LossFunction* loss_with = new ceres::HuberLoss(config_.huber_delta);
    ceres::Problem prob_base, prob_with;
    std::set<int> used_nodes_base, used_nodes_with;

    auto add_edge_if_active = [&](ceres::Problem& prob, std::set<int>& used, Edge* e, std::vector<double*>& poses, ceres::LossFunction* loss){
        int ia = e->a->index, ib = e->b->index;
        if (ia == ib) return;
        ceres::CostFunction* cost = OdometryResidue::Create(e->x, e->y, e->theta);
        prob.AddResidualBlock(cost, loss, poses[ia], poses[ib]);
        used.insert(ia);
        used.insert(ib);
    };

    // 오도메트리 엣지
    for (auto* e : g2o_.nEdgesOdometry) {
        add_edge_if_active(prob_base, used_nodes_base, e, poses_base, loss_base);
        add_edge_if_active(prob_with, used_nodes_with, e, poses_with, loss_with);
    }
    // 레이어 엣지(상속+추가), 후보 엣지는 base에선 제외, with에선 포함
    auto layer_edges = layer->get_all_edges();
    for (auto* e : layer_edges) {
        if (e == edge) continue;
        add_edge_if_active(prob_base, used_nodes_base, e, poses_base, loss_base);
        add_edge_if_active(prob_with, used_nodes_with, e, poses_with, loss_with);
    }
    add_edge_if_active(prob_with, used_nodes_with, edge, poses_with, loss_with);

    // 앵커 설정
    if (!used_nodes_base.empty()) {
        int anchor = (used_nodes_base.count(0) ? 0 : *used_nodes_base.begin());
        prob_base.AddParameterBlock(poses_base[anchor], 3);
        prob_base.SetParameterBlockConstant(poses_base[anchor]);
    }
    if (!used_nodes_with.empty()) {
        int anchor = (used_nodes_with.count(0) ? 0 : *used_nodes_with.begin());
        prob_with.AddParameterBlock(poses_with[anchor], 3);
        prob_with.SetParameterBlockConstant(poses_with[anchor]);
    }

    // 풀기 (짧게)
    ceres::Solver::Options opts;
    opts.max_num_iterations = std::max(1, config_.impact_iters);
    opts.minimizer_progress_to_stdout = false;
    opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    opts.num_threads = 1;
    ceres::Solver::Summary s0, s1;
    ceres::Solve(opts, &prob_base, &s0);
    ceres::Solve(opts, &prob_with, &s1);

    // 인접 엣지 선택: edge와 노드를 공유하는 모든 엣지(오도메트리 + 레이어엣지), 후보 제외
    std::vector<Edge*> neighbor_edges;
    auto consider_neighbor = [&](Edge* e){
        if (!e || e == edge) return;
        int ia = e->a->index, ib = e->b->index;
        if (ia == a_idx || ib == a_idx || ia == b_idx || ib == b_idx) {
            neighbor_edges.push_back(e);
        }
    };
    for (auto* e : g2o_.nEdgesOdometry) consider_neighbor(e);
    for (auto* e : layer_edges) consider_neighbor(e);

    // 잔차 계산 도우미 (L2, 각도 가중은 impact_theta_weight 재사용)
    auto residual_norm_with_poses = [&](std::vector<double*>& poses, Edge* e){
        int ia = e->a->index, ib = e->b->index;
        if (ia < 0 || ib < 0 || ia >= N || ib >= N) return 0.0;
        double* pa = poses[ia];
        double* pb = poses[ib];
        double dx = pb[0] - pa[0];
        double dy = pb[1] - pa[1];
        double dth = pb[2] - pa[2];
        while (dth > M_PI) dth -= 2 * M_PI;
        while (dth < -M_PI) dth += 2 * M_PI;
        double ca = std::cos(pa[2]);
        double sa = std::sin(pa[2]);
        double rel_x = ca * dx + sa * dy;
        double rel_y = -sa * dx + ca * dy;
        double rx = rel_x - e->x;
        double ry = rel_y - e->y;
        double rth = dth - e->theta;
        while (rth > M_PI) rth -= 2 * M_PI;
        while (rth < -M_PI) rth += 2 * M_PI;
        double wth = config_.neighbor_theta_weight;
        return std::sqrt(rx*rx + ry*ry + wth * rth*rth);
    };

    double sum_before = 0.0, sum_after = 0.0;
    for (auto* ne : neighbor_edges) {
        sum_before += residual_norm_with_poses(poses_base, ne);
        sum_after  += residual_norm_with_poses(poses_with, ne);
    }

    // 메모리 정리
    for (int i = 0; i < N; ++i) { delete[] poses_base[i]; delete[] poses_with[i]; }

    double improvement = sum_before - sum_after;
    double rel_improvement = improvement / (1e-9 + sum_before);
    log_line("[neighbor-improve] a=" + std::to_string(a_idx) + ", b=" + std::to_string(b_idx) +
             ", neighbors=" + std::to_string(neighbor_edges.size()) +
             ", sum_before=" + std::to_string(sum_before) +
             ", sum_after=" + std::to_string(sum_after) +
             ", improvement=" + std::to_string(improvement) +
             ", rel_improve=" + std::to_string(rel_improvement));
    return improvement;
}

// 부모/자식 레이어의 포즈를 각각 사용해 이웃 엣지 잔차 합의 개선량을 계산
// (disabled) 부모/자식 레이어의 포즈를 각각 사용해 이웃 엣지 잔차 합의 개선량을 계산
#if 0
double SimpleLayerManagerV2::compute_neighbor_improvement_between_layers(const std::string& parent_id,
                                                                         const std::string& child_id,
                                                                         Edge* edge)
{
    return 0.0;
}
#endif

void SimpleLayerManagerV2::merge_child_into_parent_and_delete(const std::string& parent_id,
                                                              const std::string& child_id,
                                                              Edge* edge)
{
    auto* parent = get_layer(parent_id);
    auto* child = get_layer(child_id);
    if (!parent || !child) return;

    // 부모 포즈를 자식(최적화된) 포즈로 복사
    copy_poses(child_id, parent_id);
    // 부모에 엣지 반영 및 스위치 초기화
    if (edge) {
        bool exists = false;
        for (auto* e : parent->added_edges) if (e == edge) { exists = true; break; }
        if (!exists) parent->added_edges.push_back(edge);
        if (parent->switch_vars.find(edge) == parent->switch_vars.end()) parent->switch_vars[edge] = new double(1.0);
    }

    // 부모 children 목록에서 child 제거
    auto it = std::find(parent->children.begin(), parent->children.end(), child_id);
    if (it != parent->children.end()) parent->children.erase(it);

    // 메모리 해제 및 레이어 삭제
    free_layer_poses(child);
    free_layer_switches(child);
    layers_.erase(child_id);

    log_line("[merge] merged child " + child_id + " into parent " + parent_id + " and deleted child");
}

double SimpleLayerManagerV2::process_edge_with_temp_layer(const std::string& parent_id, Edge* edge)
{
    auto* parent = get_layer(parent_id);
    if (!parent || !edge) return 0.0;

    // 1) 임시 레이어 생성 (엣지 포함)
    std::string child_id = create_child_layer(parent_id, edge, true);
    auto* parent_ptr = get_layer(parent_id);
    if (parent_ptr) parent_ptr->children.push_back(child_id);

    // 2) 임시 레이어 전체 최적화 (시간 측정)
    auto start_time = std::chrono::high_resolution_clock::now();
    optimize_layer(child_id);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_optimization_time = duration.count();
    
    // Timing log (append mode)
    std::string timing_log_path = save_path_ + "/optimization_timing.txt";
    std::ofstream timing_log(timing_log_path, std::ios::app);
    timing_log << step_counter_ << " " << online_active_k_ << " " 
               << edge->a->index << " " << edge->b->index << " " << edge->edge_type << " "
               << parent_id << " temp_layer " << duration.count() << "\n";
    timing_log.close();

    // 3) Chi-squared 기반 엣지 필터링
    auto wrap = [](double a){ while(a > M_PI) a -= 2*M_PI; while(a < -M_PI) a += 2*M_PI; return a; };
    auto edge_chi2 = [&](double* pa, double* pb, const Edge* e){
        double dx = pb[0] - pa[0];
        double dy = pb[1] - pa[1];
        double dth = wrap(pb[2] - pa[2]);
        double ca = std::cos(pa[2]), sa = std::sin(pa[2]);
        double rel_x = ca*dx + sa*dy;
        double rel_y = -sa*dx + ca*dy;
        double rx = rel_x - e->x;
        double ry = rel_y - e->y;
        double rth = wrap(dth - e->theta);
        Eigen::Matrix<double,3,1> r; r << rx, ry, rth;
        Eigen::Matrix3d I;
        I << e->I11, e->I12, e->I13,
             e->I12, e->I22, e->I23,
             e->I13, e->I23, e->I33;
        I = 0.5 * (I + I.transpose());
        return (r.transpose() * I * r)(0,0);
    };

    // 새 엣지의 chi2 값만 확인 (최적화된 자식 레이어에서)
    auto* child = get_layer(child_id);
    if (!child) return total_optimization_time;
    
    int ia = edge->a->index, ib = edge->b->index;
    if (ia < 0 || ib < 0 || ia >= (int)child->poses.size() || ib >= (int)child->poses.size()) return total_optimization_time;
    
    double new_edge_chi2 = edge_chi2(child->poses[ia], child->poses[ib], edge);
    
    // Calculate Euclidean distance between nodes (after optimization)
    double* pose_a = child->poses[ia];
    double* pose_b = child->poses[ib];
    double euclidean_dist = std::sqrt(
        (pose_a[0] - pose_b[0]) * (pose_a[0] - pose_b[0]) + 
        (pose_a[1] - pose_b[1]) * (pose_a[1] - pose_b[1])
    );
    
    // Angular distance (absolute difference in orientation)
    auto wrap_angle = [](double a) { while(a > M_PI) a -= 2*M_PI; while(a < -M_PI) a += 2*M_PI; return a; };
    double angular_dist = std::abs(wrap_angle(pose_a[2] - pose_b[2]));
    
    // Chi-squared threshold for 3 DOF (95% confidence: 7.815, 99% confidence: 11.345)
    const double chi2_threshold_95 = 7.815;  // TODO: expose via config
    const double chi2_threshold_99 = 11.345; // TODO: expose via config
    const double chi2_threshold = chi2_threshold_95; // Use 95% confidence by default
    
        
    // Statistical thresholds for node proximity (based on typical SLAM uncertainty)
    const double close_euclidean_threshold = 1.5;    // 1.5m - typical GPS/odometry accuracy
    const double close_angular_threshold = 5.0 * M_PI / 180.0; // 5 degrees - typical orientation uncertainty
    const double very_close_euclidean_threshold = 0.1; // 0.1m - very close nodes
    const double very_close_angular_threshold = 2.0 * M_PI / 180.0; // 2 degrees - very close orientation
    
    // Determine proximity level
    std::string proximity_level = "far";
    if (euclidean_dist <= very_close_euclidean_threshold && angular_dist <= very_close_angular_threshold) {
        proximity_level = "very_close";
    } else if (euclidean_dist <= close_euclidean_threshold && angular_dist <= close_angular_threshold) {
        proximity_level = "close";
    }
    


    log_line(std::string("[chi2-filter] new_edge_chi2=") + std::to_string(new_edge_chi2) +
             ", threshold=" + std::to_string(chi2_threshold) +
             ", edge=(" + std::to_string(ia) + "," + std::to_string(ib) + ")" +
             ", euclidean_dist=" + std::to_string(euclidean_dist) + "m" +
             ", angular_dist=" + std::to_string(angular_dist * 180.0 / M_PI) + "deg" +
             ", proximity=" + proximity_level);

    bool accept = (new_edge_chi2 < chi2_threshold);

    if (accept) {
        // 병합: 부모 포즈 갱신 + 엣지 추가 + 자식 삭제
        merge_child_into_parent_and_delete(parent_id, child_id, edge);
        // Top-K 전파 및 best 최적화 (시간 측정)
        if (config_.topk_layers > 0) {
            auto topk = get_topk_layers_by_reward(config_.topk_layers);
            std::string best_id = topk.empty() ? parent_id : topk[0];
            propagate_edge_to_layers(edge, topk, best_id, parent_id);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            optimize_layer(best_id);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto additional_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            total_optimization_time += additional_duration.count();
            
            // Timing log
            std::string timing_log_path = save_path_ + "/optimization_timing.txt";
            std::ofstream timing_log(timing_log_path, std::ios::app);
            timing_log << step_counter_ << " " << online_active_k_ << " " 
                       << edge->a->index << " " << edge->b->index << " " << edge->edge_type << " "
                       << best_id << " topk_best " << additional_duration.count() << "\n";
            timing_log.close();
        } else {
            auto start_time = std::chrono::high_resolution_clock::now();
            optimize_layer(parent_id);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto additional_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            total_optimization_time += additional_duration.count();
            
            // Timing log
            std::string timing_log_path = save_path_ + "/optimization_timing.txt";
            std::ofstream timing_log(timing_log_path, std::ios::app);
            timing_log << step_counter_ << " " << online_active_k_ << " " 
                       << edge->a->index << " " << edge->b->index << " " << edge->edge_type << " "
                       << parent_id << " parent " << additional_duration.count() << "\n";
            timing_log.close();
        }
        // 보상/백프로파게이션(부모 기준)
        double reward = calculate_reward(parent_id, edge);
        backpropagate(parent_id, reward);
    } else {
        log_line("[chi2-decision] reject; keep child layer " + child_id);
        // 기록용
        assignments_.emplace_back(edge, child_id);
        // 선택적으로 자식 보상 기록
        double reward = calculate_reward(child_id, edge);
        backpropagate(child_id, reward);
    }
    
    return total_optimization_time;
}

bool SimpleLayerManagerV2::should_add_edge(const std::string& layer_id, Edge* edge, double precomputed_residual)
{
    double residual = std::isnan(precomputed_residual) ? calculate_edge_residual(layer_id, edge)
                                                      : precomputed_residual;
    
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
    
    // Odometry constraints (add all, like method 5 after full accumulation)
    for (auto* edge : g2o_.nEdgesOdometry) {
        ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, 
                               layer->poses[edge->a->index], 
                               layer->poses[edge->b->index]);
    }

    // Layer의 loop/bogus edges (inherited + added) with switching constraints
    auto all_edges = layer->get_all_edges();
    for (auto* edge : all_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue; // self-loop 방지
        // Switchable constraint for closures/bogus
        if (layer->switch_vars.find(edge) == layer->switch_vars.end()) {
            layer->switch_vars[edge] = new double(1.0);
        }
        double* s = layer->switch_vars[edge];
        ceres::CostFunction* cost = SwitchableClosureResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, layer->poses[ia], layer->poses[ib], s);
        ceres::CostFunction* prior = SwitchPriorResidue::Create(config_.sc_prior_lambda);
        problem.AddResidualBlock(prior, nullptr, s);
    }
    
    // Fix first pose (anchor). Ensure it's registered.
    if (!layer->poses.empty()) {
        problem.AddParameterBlock(layer->poses[0], 3);
        problem.SetParameterBlockConstant(layer->poses[0]);
    }
    

    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, config_.local_iters);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = std::thread::hardware_concurrency() != 0
        ? std::thread::hardware_concurrency()
        : 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

void SimpleLayerManagerV2::optimize_layer_upto_k(const std::string& layer_id, int k)
{
    auto* layer = get_layer(layer_id);
    if (!layer) return;

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(config_.huber_delta);
    
    std::set<int> used_nodes;
    
    // Odometry constraints only up to node index k (like method 5 accumulated up to k)
    int odom_added = 0;
    for (auto* edge : g2o_.nEdgesOdometry) {
        int ia = edge->a->index;
        int ib = edge->b->index;
        if (std::max(ia, ib) <= k) {
            ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
            problem.AddResidualBlock(cost, loss, layer->poses[ia], layer->poses[ib]);
            used_nodes.insert(ia);
            used_nodes.insert(ib);
            odom_added++;
        }
    }

    // Layer loop/bogus edges (inherited + added) up to node index k, with switches
    auto all_edges = layer->get_all_edges();
    int loop_added = 0;
    for (auto* edge : all_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue;
        if (std::max(ia, ib) <= k) {
            if (layer->switch_vars.find(edge) == layer->switch_vars.end()) {
                layer->switch_vars[edge] = new double(1.0);
            }
            double* s = layer->switch_vars[edge];
            ceres::CostFunction* cost = SwitchableClosureResidue::Create(edge->x, edge->y, edge->theta);
            problem.AddResidualBlock(cost, loss, layer->poses[ia], layer->poses[ib], s);
            ceres::CostFunction* prior = SwitchPriorResidue::Create(config_.sc_prior_lambda);
            problem.AddResidualBlock(prior, nullptr, s);
            used_nodes.insert(ia);
            used_nodes.insert(ib);
            loop_added++;
        }
    }

    // Anchor among used nodes to remove gauge
    if (!used_nodes.empty()) {
        int anchor = (used_nodes.count(0) ? 0 : *used_nodes.begin());
        problem.AddParameterBlock(layer->poses[anchor], 3);
        problem.SetParameterBlockConstant(layer->poses[anchor]);
    } else {
        // 추가된 제약이 없다면 바로 반환
        log_line("[opt<=k] no active constraints; skip solve (layer=" + layer_id + ", k=" + std::to_string(k) + ")");
        return;
    }

    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, 100);
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = std::thread::hardware_concurrency() != 0
        ? std::thread::hardware_concurrency()
        : 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 간단한 요약 로그 (디버깅용)
    log_line("[opt<=k] layer=" + layer_id + 
             ", k=" + std::to_string(k) +
             ", odom_added=" + std::to_string(odom_added) +
             ", loop_added=" + std::to_string(loop_added) +
             ", used_nodes=" + std::to_string(used_nodes.size()) +
             ", final_cost=" + std::to_string(summary.final_cost));
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

    // Add the newly added edges (focus constraints) only if endpoints are active
    for (auto* edge : layer->added_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue;
        if (active_nodes.count(ia) && active_nodes.count(ib)) {
            ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
            problem.AddResidualBlock(cost, loss, layer->poses[ia], layer->poses[ib]);
            used_nodes.insert(ia);
            used_nodes.insert(ib);
        }
    }
    
    // Fix a single anchor among used nodes to remove gauge.
    // Prefer node 0 if it is used; otherwise pick the smallest used node id.
    if (!used_nodes.empty()) {
        int anchor = (used_nodes.count(0) ? 0 : *used_nodes.begin());
        problem.AddParameterBlock(layer->poses[anchor], 3);
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
    // 임시 스위치 변수 (레퍼런스 값 복사) — 주소 안정성을 위해 unique_ptr 사용
    std::vector<std::unique_ptr<double>> local_switches;
    local_switches.reserve(layer->inherited_edges.size() + layer->added_edges.size());

    // Determine active nodes from layer edges up to k_lim
    auto all_edges = layer->get_all_edges();
    int k_lim = (online_active_k_ >= 0 ? online_active_k_ : std::numeric_limits<int>::max());
    std::unordered_set<int> active_nodes;
    for (auto* edge : all_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue;
        if (std::max(ia, ib) > k_lim) continue;
        active_nodes.insert(ia);
        active_nodes.insert(ib);
    }

    // Odometry constraints only between active nodes
    for (auto* edge : g2o_.nEdgesOdometry) {
        int ia = edge->a->index, ib = edge->b->index;
        if (active_nodes.count(ia) && active_nodes.count(ib)) {
            ceres::CostFunction* cost = OdometryResidue::Create(edge->x, edge->y, edge->theta);
            problem.AddResidualBlock(cost, loss, temp_poses[ia], temp_poses[ib]);
        }
    }

    // Layer edges (inherited + added)
    for (auto* edge : all_edges) {
        int ia = edge->a->index, ib = edge->b->index;
        if (ia == ib) continue;
        if (std::max(ia, ib) > k_lim) continue; // online 모드에선 k 이후 엣지 배제
        // switchable constraint (임시 s)
        double s_val = 1.0;
        auto sit = layer->switch_vars.find(edge);
        if (sit != layer->switch_vars.end() && sit->second) s_val = *(sit->second);
        local_switches.emplace_back(new double(s_val));
        double* s_ptr = local_switches.back().get();
        ceres::CostFunction* cost = SwitchableClosureResidue::Create(edge->x, edge->y, edge->theta);
        problem.AddResidualBlock(cost, loss, temp_poses[ia], temp_poses[ib], s_ptr);
        ceres::CostFunction* prior = SwitchPriorResidue::Create(config_.sc_prior_lambda);
        problem.AddResidualBlock(prior, nullptr, s_ptr);
    }
    
    problem.AddParameterBlock(temp_poses[0], 3);
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

std::vector<std::string> SimpleLayerManagerV2::get_topk_layers_by_reward(int k)
{
    std::vector<std::pair<std::string, double>> vec;
    vec.reserve(layers_.size());
    for (auto& p : layers_) {
        auto* layer = p.second.get();
        auto all_edges = layer->get_all_edges();
        double score = normalize_reward_by_edge_count(layer->total_reward, (int)all_edges.size());
        vec.emplace_back(p.first, score);
    }
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
    std::vector<std::string> out;
    for (int i = 0; i < k && i < (int)vec.size(); ++i) out.push_back(vec[i].first);
    return out;
}

std::vector<std::string> SimpleLayerManagerV2::get_parent_and_sibling_layers(const std::string& layer_id,
                                                                             bool include_parent,
                                                                             bool include_siblings)
{
    std::vector<std::string> result;
    auto* layer = get_layer(layer_id);
    if (!layer) return result;

    // 부모가 없으면 전파 대상 없음 (루트)
    if (layer->parent_id.empty()) return result;

    std::unordered_set<std::string> uniq;

    // 부모 포함
    if (include_parent) {
        uniq.insert(layer->parent_id);
    }

    // 형제 포함: 부모의 children 전체 (best 포함)
    if (include_siblings) {
        auto* parent = get_layer(layer->parent_id);
        if (parent) {
            for (const auto& cid : parent->children) {
                uniq.insert(cid);
            }
        }
    }

    result.reserve(uniq.size());
    for (const auto& id : uniq) result.push_back(id);
    return result;
}

void SimpleLayerManagerV2::copy_poses(const std::string& src_layer, const std::string& dst_layer)
{
    auto* src = get_layer(src_layer);
    auto* dst = get_layer(dst_layer);
    if (!src || !dst) return;
    if (src->poses.size() != dst->poses.size()) return;
    for (size_t i = 0; i < src->poses.size(); ++i) {
        dst->poses[i][0] = src->poses[i][0];
        dst->poses[i][1] = src->poses[i][1];
        dst->poses[i][2] = src->poses[i][2];
    }
}

void SimpleLayerManagerV2::propagate_edge_to_layers(Edge* e, const std::vector<std::string>& layer_ids,
                                                    const std::string& best_id, const std::string& exclude_id)
{
    for (const auto& id : layer_ids) {
        if (id == exclude_id) continue;
        auto* layer = get_layer(id);
        if (!layer) continue;
        // dedup: skip if already present
        bool exists = false;
        for (auto* ee : layer->added_edges) { if (ee == e) { exists = true; break; } }
        if (!exists) {
            for (auto* ee : layer->inherited_edges) { if (ee == e) { exists = true; break; } }
        }
        if (!exists) {
            layer->added_edges.push_back(e);
            assignments_.emplace_back(e, id);
            // initialize a switch variable for new edge
            if (layer->switch_vars.find(e) == layer->switch_vars.end()) {
                layer->switch_vars[e] = new double(1.0);
            }
        }
        if (config_.propagate_poses_from_best && !best_id.empty() && id != best_id) {
            copy_poses(best_id, id);
        }
    }
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

bool SimpleLayerManagerV2::should_merge_layer(const std::string& layer_id)
{
    if (layer_id.empty() || layer_id == base_layer_id_) return false;
    auto* layer = get_layer(layer_id);
    if (!layer) return false;
    auto all_edges = layer->get_all_edges();
    int edge_count = static_cast<int>(all_edges.size());
    if (edge_count < config_.min_edges_to_merge) return false;
    if (layer->visits <= 0) return false;
    double normalized = normalize_reward_by_edge_count(layer->total_reward, edge_count);
    log_line("[merge-eval] layer=" + layer_id +
             ", visits=" + std::to_string(layer->visits) +
             ", edges=" + std::to_string(edge_count) +
             ", normalized=" + std::to_string(normalized) +
             ", threshold=" + std::to_string(config_.merge_threshold));
    return normalized >= config_.merge_threshold;
}

void SimpleLayerManagerV2::free_layer_poses(SimpleLayer* layer)
{
    if (!layer) return;
    for (auto* p : layer->poses) {
        delete[] p;
    }
    layer->poses.clear();
}

void SimpleLayerManagerV2::free_layer_switches(SimpleLayer* layer)
{
    if (!layer) return;
    for (auto& kv : layer->switch_vars) {
        delete kv.second;
    }
    layer->switch_vars.clear();
}

void SimpleLayerManagerV2::merge_layer_into_base(const std::string& layer_id)
{
    if (layer_id.empty() || layer_id == base_layer_id_) return;
    auto it = layers_.find(layer_id);
    if (it == layers_.end()) return;
    SimpleLayer* child = it->second.get();
    SimpleLayer* base = get_layer(base_layer_id_);
    if (!base || !child) return;

    // Base poses <- child poses (Option A)
    if (base->poses.size() != child->poses.size()) {
        log_line("[merge] pose size mismatch; abort merge");
        return;
    }
    for (size_t i = 0; i < base->poses.size(); ++i) {
        base->poses[i][0] = child->poses[i][0];
        base->poses[i][1] = child->poses[i][1];
        base->poses[i][2] = child->poses[i][2];
    }

    // Merge edges (dedup)
    std::unordered_set<Edge*> exists;
    exists.reserve(base->inherited_edges.size() + base->added_edges.size());
    for (auto* e : base->inherited_edges) exists.insert(e);
    for (auto* e : base->added_edges) exists.insert(e);
    for (auto* e : child->added_edges) {
        if (exists.insert(e).second) base->added_edges.push_back(e);
    }

    // Detach from parent children list
    if (!child->parent_id.empty()) {
        auto* parent = get_layer(child->parent_id);
        if (parent) {
            auto& vec = parent->children;
            vec.erase(std::remove(vec.begin(), vec.end(), layer_id), vec.end());
        }
    }

    // Remove child layer
    free_layer_poses(child);
    free_layer_switches(child);
    layers_.erase(it);

    log_line("[merge] merged layer " + layer_id + " into base " + base_layer_id_);
}

// =============================
// METHOD 5: SimpleLayerManager2
// =============================

SimpleLayerManager2::SimpleLayerManager2(ReadG2O& g, const std::string& save_path, const SimpleLayer2Config& cfg)
    : g2o_(g), save_path_(save_path), config_(cfg)
{
    node_added_.assign(g2o_.nNodes.size(), false);
    // Apply robust loss to both odometry and loop/bogus constraints
    loss_odom_ = new ceres::HuberLoss(config_.huber_delta);
    loss_loop_ = new ceres::HuberLoss(config_.huber_delta);
}

SimpleLayerManager2::~SimpleLayerManager2()
{
    // free switch variables created
    for (auto& kv : switch_vars_) {
        delete kv.second;
    }
    // loss_loop_ owned here
    delete loss_loop_;
    delete loss_odom_;
}

void SimpleLayerManager2::run_online()
{
    const int N = static_cast<int>(g2o_.nNodes.size());
    if (N == 0) return;
    
    // Initialize method statistics tracking for SC method
    method_start_time_ = std::chrono::high_resolution_clock::now();
    int step_counter = 0;

    // Explicitly add and fix anchor node 0
    if (!anchor_added_) {
        problem_.AddParameterBlock(g2o_.nNodes[0]->p, 3);
        problem_.SetParameterBlockConstant(g2o_.nNodes[0]->p);
        node_added_[0] = true;
        anchor_added_ = true;
    }

    auto add_node_if_needed = [&](int idx) {
        if (idx < 0 || idx >= N) return;
        if (!node_added_[idx]) {
            problem_.AddParameterBlock(g2o_.nNodes[idx]->p, 3);
            node_added_[idx] = true;
        }
    };

    auto add_odometry_edge = [&](Edge* e) {
        int ia = e->a->index, ib = e->b->index;
        if (ia == ib) return; // guard
        add_node_if_needed(ia);
        add_node_if_needed(ib);
        ceres::CostFunction* cost = OdometryResidue::Create(e->x, e->y, e->theta);
        problem_.AddResidualBlock(cost, loss_odom_, e->a->p, e->b->p);
    };

    auto add_sc_edge = [&](Edge* e) {
        int ia = e->a->index, ib = e->b->index;
        if (ia == ib) return; // guard
        add_node_if_needed(ia);
        add_node_if_needed(ib);
        double*& s = switch_vars_[e];
        if (!s) s = new double(1.0);
        ceres::CostFunction* cost = SwitchableClosureResidue::Create(e->x, e->y, e->theta);
        problem_.AddResidualBlock(cost, loss_loop_, e->a->p, e->b->p, s);
        ceres::CostFunction* prior = SwitchPriorResidue::Create(config_.sc_prior_lambda);
        problem_.AddResidualBlock(prior, nullptr, s);
        if (config_.bound_switch_01) {
            problem_.SetParameterLowerBound(s, 0, 0.0);
            problem_.SetParameterUpperBound(s, 0, 1.0);
        }
    };

    int step = 0;
    for (int k = 1; k < N; ++k) {
        step++;
        step_counter++;
        auto step_start_time = std::chrono::high_resolution_clock::now();
        
        int add_odo = 0, add_cl = 0, add_bg = 0;

        // Add odometry edges that become active at k
        for (auto* e : g2o_.nEdgesOdometry) {
            if (std::max(e->a->index, e->b->index) == k) {
                add_odometry_edge(e);
                add_odo++;
            }
        }

        // Add closure edges: only detect loops from current node k to past nodes
        for (auto* e : g2o_.nEdgesClosure) {
            // Realistic online SLAM: current node k can only detect loops to past nodes
            if ((e->a->index == k && e->b->index < k) || 
                (e->b->index == k && e->a->index < k)) {
                add_sc_edge(e);
                add_cl++;
                // METHOD 4-like reward (log only)
                double rew = calculate_reward(k, e);
                log_line("[m5-reward] k=" + std::to_string(k) +
                         " edge(CLOSURE)=" + std::to_string(e->a->index) + "," + std::to_string(e->b->index) +
                         " reward=" + std::to_string(rew));
                b_add_loop_edges_ = true;
            }
        }

        // Add bogus edges: only detect from current node k to past nodes
        for (auto* e : g2o_.nEdgesBogus) {
            // Realistic online SLAM: current node k can only detect bogus loops to past nodes
            if ((e->a->index == k && e->b->index < k) || 
                (e->b->index == k && e->a->index < k)) {
                add_sc_edge(e);
                add_bg++;
                // METHOD 4-like reward (log only)
                double rew = calculate_reward(k, e);
                log_line("[m5-reward] k=" + std::to_string(k) +
                         " edge(BOGUS)=" + std::to_string(e->a->index) + "," + std::to_string(e->b->index) +
                         " reward=" + std::to_string(rew));
                b_add_loop_edges_ = true;
            }
        }

        log_line("[m5-online] k=" + std::to_string(k) +
                 ", odom_added=" + std::to_string(add_odo) +
                 ", closure_added=" + std::to_string(add_cl) +
                 ", bogus_added=" + std::to_string(add_bg));

        double optimization_time = 0.0;
        if(b_add_loop_edges_) {
            // Short solve at each step (시간 측정)
            ceres::Solver::Options opts;
            opts.max_num_iterations = std::max(1, config_.iters_per_step);
            opts.minimizer_progress_to_stdout = false;
            opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            opts.num_threads = std::thread::hardware_concurrency() != 0
                ? std::thread::hardware_concurrency()
                : 4;
            ceres::Solver::Summary sum;
            
            auto opt_start_time = std::chrono::high_resolution_clock::now();
            ceres::Solve(opts, &problem_, &sum);
            auto opt_end_time = std::chrono::high_resolution_clock::now();
            auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end_time - opt_start_time);
            optimization_time = opt_duration.count();
            
            b_add_loop_edges_ = false;
        }

        if (config_.snapshot_every > 0 && (step % 20 == 0)) {
            // Create snapshot dirs (PNG + data) compatible with METHOD 4 plotter
            std::filesystem::path snap_root = std::filesystem::path(save_path_) / "snapshots5";
            std::error_code ec;
            std::filesystem::create_directories(snap_root, ec);

            // Names similar to method4
            std::ostringstream oss_png, oss_data;
            oss_png << "step_" << std::setw(4) << std::setfill('0') << k << ".png";
            oss_data << "step_" << std::setw(4) << std::setfill('0') << k << "_data";
            std::filesystem::path data_dir = snap_root / oss_data.str();
            std::filesystem::create_directories(data_dir, ec);

            // Write init poses (0..k)
            {
                std::ofstream fp((data_dir / "init_nodes.txt").string());
                for (int i = 0; i <= k && i < N; ++i) {
                    double* p = g2o_.nNodes[i]->p;
                    fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
                }
            }
            // Write optimized poses (0..k) — Method 5 uses single global optimization
            {
                std::ofstream fp((data_dir / "opt_nodes.txt").string());
                for (int i = 0; i <= k && i < N; ++i) {
                    double* p = g2o_.nNodes[i]->p;
                    fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
                }
            }

            // Minimal method4-compatible stats file
            {
                std::ofstream fp((data_dir / "method4_stats.txt").string());
                fp << "# layer_id visits total_reward avg_reward normalized_reward total_edges inherited_edges added_edges\n";
                // Single surrogate layer statistics
                int total_edges = 0;
                for (auto* e : g2o_.nEdgesOdometry) {
                    int ia = e->a->index, ib = e->b->index;
                    if (std::max(ia, ib) <= k) total_edges++;
                }
                for (auto* e : g2o_.nEdgesClosure) {
                    int ia = e->a->index, ib = e->b->index;
                    if (std::max(ia, ib) <= k) total_edges++;
                }
                for (auto* e : g2o_.nEdgesBogus) {
                    int ia = e->a->index, ib = e->b->index;
                    if (std::max(ia, ib) <= k) total_edges++;
                }
                fp << "L1 0 0 0 0 " << total_edges << " 0 " << total_edges << "\n";
            }

            // Render with METHOD 4 plotter for consistency


            std::string out_png = (snap_root / oss_png.str()).string();
            std::string cmd = std::string("MPLBACKEND=Agg python3 ../drawer/plot_method4_results.py ") +
                              "--save_path " + data_dir.string() + " --output " + out_png +
                              " --no-show > /dev/null 2>&1";
            std::system(cmd.c_str());
        }
        
        // Track statistics for SC (Switchable Constraints) method
        double cum_distance = calculate_cumulative_distance_up_to_node(k);
        int layer_count = 1; // METHOD 5 uses single global problem, so always 1 layer
        
        track_method_statistics("SC", step_counter, k + 1, cum_distance, 
                               layer_count, optimization_time);
    }
    ceres::Solver::Options opts;
    opts.max_num_iterations = std::max(1, 100);
    opts.minimizer_progress_to_stdout = false;
    opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    opts.num_threads = std::thread::hardware_concurrency() != 0
        ? std::thread::hardware_concurrency()
        : 4;
    ceres::Solver::Summary sum;
    ceres::Solve(opts, &problem_, &sum);

    // Save final results
    save_nodes(save_path_ + "/opt_nodes.txt");
    save_switches(save_path_ + "/switches.txt");
    
    // Output and save statistics for SC method
    output_method_statistics();
    save_statistics_to_file(save_path_ + "/method_statistics_sc.txt");
}

void SimpleLayerManager2::log_line(const std::string& s)
{
    std::cout << s << std::endl;
}

void SimpleLayerManager2::save_nodes(const std::string& filepath)
{
    std::ofstream fp(filepath);
    for (size_t i = 0; i < g2o_.nNodes.size(); ++i) {
        double* p = g2o_.nNodes[i]->p;
        fp << i << " " << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
}

void SimpleLayerManager2::save_switches(const std::string& filepath)
{
    // Build priors and optimized arrays in order: closures first, then bogus
    std::vector<double> priors;
    std::vector<double*> optimized;
    priors.reserve(g2o_.nEdgesClosure.size() + g2o_.nEdgesBogus.size());
    optimized.reserve(g2o_.nEdgesClosure.size() + g2o_.nEdgesBogus.size());

    auto push_for = [&](const std::vector<Edge*>& vec) {
        for (auto* e : vec) {
            priors.push_back(1.0);
            auto it = switch_vars_.find(e);
            if (it != switch_vars_.end()) optimized.push_back(it->second);
            else {
                // If not created (edge never added online), create a dummy 1.0 value
                double* s = new double(1.0);
                switch_vars_[e] = s;
                optimized.push_back(s);
            }
        }
    };

    push_for(g2o_.nEdgesClosure);
    push_for(g2o_.nEdgesBogus);

    // Reuse writer format
    std::ofstream fp(filepath);
    fp << "Odometry EDGES AHEAD\n";
    for (auto* ed : g2o_.nEdgesOdometry) {
        fp << ed->a->index << " " << ed->b->index << " " << ed->edge_type << " " << 1.0 << " " << 1.0 << "\n";
    }
    fp << "Closure EDGES AHEAD\n";
    for (size_t i = 0; i < g2o_.nEdgesClosure.size(); ++i) {
        auto* ed = g2o_.nEdgesClosure[i];
        fp << ed->a->index << " " << ed->b->index << " " << ed->edge_type << " "
           << priors[i] << " " << *(optimized[i]) << "\n";
    }
    fp << "BOGUS EDGES AHEAD\n";
    size_t of = g2o_.nEdgesClosure.size();
    for (size_t i = 0; i < g2o_.nEdgesBogus.size(); ++i) {
        auto* ed = g2o_.nEdgesBogus[i];
        fp << ed->a->index << " " << ed->b->index << " " << ed->edge_type << " "
           << priors[of + i] << " " << *(optimized[of + i]) << "\n";
    }
}

// ---- METHOD 4-like reward components for METHOD 5 ----

double SimpleLayerManager2::calculate_info_gain(Edge* edge)
{
    if (!edge) return 0.0;
    Eigen::Matrix3d Omega;
    Omega << edge->I11, edge->I12, edge->I13,
             edge->I12, edge->I22, edge->I23,
             edge->I13, edge->I23, edge->I33;
    Omega = 0.5 * (Omega + Omega.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(Omega);
    Eigen::Vector3d evals = solver.eigenvalues().cwiseMax(1e-12);
    double logdet = 0.0;
    for (int i = 0; i < 3; ++i) logdet += std::log(1.0 + evals[i]);
    return 0.5 * logdet;
}

int SimpleLayerManager2::count_active_closure_edges_upto_k(int k) const
{
    int cnt = 0;
    auto consider = [&](Edge* e){
        if (!e) return;
        if (e->edge_type != CLOSURE_EDGE) return;
        int ia = e->a->index, ib = e->b->index;
        if (std::max(ia, ib) > k) return;
        double s = 1.0;
        auto it = switch_vars_.find(e);
        if (it != switch_vars_.end() && it->second) s = *(it->second);
        if (s > config_.sc_active_threshold) cnt++;
    };
    for (auto* e : g2o_.nEdgesClosure) consider(e);
    // bogus edges are not counted as closures here
    return cnt;
}

double SimpleLayerManager2::evaluate_global_cost_upto_k(int k, Edge* exclude_edge, bool include_candidate)
{
    // Build a temporary problem with pose copies and local switch copies
    const int N = static_cast<int>(g2o_.nNodes.size());
    k = std::min(k, N - 1);
    if (k < 0) return 0.0;

    std::vector<double*> temp_poses(N);
    for (int i = 0; i < N; ++i) {
        temp_poses[i] = new double[3]{g2o_.nNodes[i]->p[0], g2o_.nNodes[i]->p[1], g2o_.nNodes[i]->p[2]};
    }

    std::vector<std::unique_ptr<double>> local_switches; // for closures/bogus
    ceres::Problem prob;
    ceres::LossFunction* loss = new ceres::HuberLoss(config_.huber_delta);

    // Odometry up to k
    for (auto* e : g2o_.nEdgesOdometry) {
        int ia = e->a->index, ib = e->b->index;
        if (std::max(ia, ib) <= k) {
            ceres::CostFunction* c = OdometryResidue::Create(e->x, e->y, e->theta);
            prob.AddResidualBlock(c, loss, temp_poses[ia], temp_poses[ib]);
        }
    }

    auto add_sc_local = [&](Edge* e){
        int ia = e->a->index, ib = e->b->index;
        if (ia == ib) return;
        if (std::max(ia, ib) > k) return;
        double s_val = 1.0;
        auto it = switch_vars_.find(e);
        if (it != switch_vars_.end() && it->second) s_val = *(it->second);
        local_switches.emplace_back(new double(s_val));
        double* s_ptr = local_switches.back().get();
        ceres::CostFunction* c = SwitchableClosureResidue::Create(e->x, e->y, e->theta);
        prob.AddResidualBlock(c, loss, temp_poses[ia], temp_poses[ib], s_ptr);
        ceres::CostFunction* prior = SwitchPriorResidue::Create(config_.sc_prior_lambda);
        prob.AddResidualBlock(prior, nullptr, s_ptr);
    };

    // Closures up to k (excluding candidate if requested)
    for (auto* e : g2o_.nEdgesClosure) {
        if (e == exclude_edge && !include_candidate) continue;
        add_sc_local(e);
    }
    // Bogus up to k (excluding candidate if requested)
    for (auto* e : g2o_.nEdgesBogus) {
        if (e == exclude_edge && !include_candidate) continue;
        add_sc_local(e);
    }

    // If include_candidate=false, it was excluded above; if true, it was included by the loops.

    // Anchor among used indices (prefer 0)
    prob.SetParameterBlockConstant(temp_poses[0]);

    ceres::Solver::Options opts;
    opts.max_num_iterations = 1;
    opts.minimizer_progress_to_stdout = false;
    opts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary sum;
    ceres::Solve(opts, &prob, &sum);

    double cost = sum.final_cost;
    for (int i = 0; i < N; ++i) delete[] temp_poses[i];
    return cost;
}

double SimpleLayerManager2::calculate_cost_delta_rel(int k, Edge* added_edge)
{
    if (!added_edge) return 0.0;
    double Li_prev = evaluate_global_cost_upto_k(k, added_edge, false);
    double Li = evaluate_global_cost_upto_k(k, added_edge, true);
    return (Li - Li_prev) / (config_.epsilon + Li_prev);
}

double SimpleLayerManager2::calculate_reward(int k, Edge* added_edge)
{
    double dcost_rel = calculate_cost_delta_rel(k, added_edge);
    double info_gain = calculate_info_gain(added_edge);
    int n_active = count_active_closure_edges_upto_k(k);
    double reward = -dcost_rel + config_.alpha_info * info_gain - config_.beta_sparse * (double)n_active;
    if (reward > 1.0) reward = 1.0; if (reward < -1.0) reward = -1.0;
    return reward;
}

// Statistics tracking implementation for SimpleLayerManager2 (METHOD 5)
void SimpleLayerManager2::track_method_statistics(const std::string& method_name, int time_step, int node_count, 
                                                 double cumulative_distance, int layer_count, double processing_time)
{
    if (method_statistics_.find(method_name) == method_statistics_.end()) {
        method_statistics_[method_name] = MethodStats{method_name, {}, {}, {}, {}, {}};
    }
    
    auto& stats = method_statistics_[method_name];
    stats.time_steps.push_back(time_step);
    stats.node_counts.push_back(node_count);
    stats.cumulative_distances.push_back(cumulative_distance);
    stats.layer_counts.push_back(layer_count);
    stats.processing_times.push_back(processing_time);
    
    log_line("[stats] " + method_name + " - step=" + std::to_string(time_step) + 
             ", nodes=" + std::to_string(node_count) + 
             ", cum_dist=" + std::to_string(cumulative_distance) + "m" +
             ", layers=" + std::to_string(layer_count) + 
             ", time=" + std::to_string(processing_time) + "ms");
}

double SimpleLayerManager2::calculate_cumulative_distance_up_to_node(int node_idx)
{
    double total_distance = 0.0;
    
    // Calculate cumulative distance from odometry edges up to node_idx
    for (auto* edge : g2o_.nEdgesOdometry) {
        int max_node = std::max(edge->a->index, edge->b->index);
        if (max_node <= node_idx) {
            // Calculate Euclidean distance from odometry measurement
            double dx = edge->x;
            double dy = edge->y;
            double distance = std::sqrt(dx*dx + dy*dy);
            total_distance += distance;
        }
    }
    
    return total_distance;
}

void SimpleLayerManager2::output_method_statistics()
{
    log_line("\n=== METHOD STATISTICS SUMMARY ===");
    
    for (const auto& pair : method_statistics_) {
        const auto& method_name = pair.first;
        const auto& stats = pair.second;
        
        if (stats.time_steps.empty()) continue;
        
        log_line("\n--- " + method_name + " STATISTICS ---");
        log_line("Total steps processed: " + std::to_string(stats.time_steps.size()));
        log_line("Final node count: " + std::to_string(stats.node_counts.back()));
        log_line("Final cumulative distance: " + std::to_string(stats.cumulative_distances.back()) + "m");
        log_line("Final layer count: " + std::to_string(stats.layer_counts.back()));
        
        // Calculate average processing time
        double total_time = std::accumulate(stats.processing_times.begin(), stats.processing_times.end(), 0.0);
        double avg_time = total_time / stats.processing_times.size();
        log_line("Average processing time per step: " + std::to_string(avg_time) + "ms");
        log_line("Total processing time: " + std::to_string(total_time) + "ms");
    }
}

void SimpleLayerManager2::save_statistics_to_file(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        log_line("[error] Failed to open statistics file: " + filename);
        return;
    }
    
    file << "# Method Statistics Data\n";
    file << "# Format: method_name,time_step,node_count,cumulative_distance,layer_count,processing_time_ms\n";
    
    for (const auto& pair : method_statistics_) {
        const auto& method_name = pair.first;
        const auto& stats = pair.second;
        
        for (size_t i = 0; i < stats.time_steps.size(); ++i) {
            file << method_name << ","
                 << stats.time_steps[i] << ","
                 << stats.node_counts[i] << ","
                 << std::fixed << std::setprecision(6) << stats.cumulative_distances[i] << ","
                 << stats.layer_counts[i] << ","
                 << std::fixed << std::setprecision(3) << stats.processing_times[i] << "\n";
        }
    }
    
    file.close();
    log_line("[stats] Statistics saved to: " + filename);
}

// Statistics tracking implementation
void SimpleLayerManagerV2::track_method_statistics(const std::string& method_name, int time_step, int node_count, 
                                                   double cumulative_distance, int layer_count, double processing_time)
{
    if (method_statistics_.find(method_name) == method_statistics_.end()) {
        method_statistics_[method_name] = MethodStats{method_name, {}, {}, {}, {}, {}};
    }
    
    auto& stats = method_statistics_[method_name];
    stats.time_steps.push_back(time_step);
    stats.node_counts.push_back(node_count);
    stats.cumulative_distances.push_back(cumulative_distance);
    stats.layer_counts.push_back(layer_count);
    stats.processing_times.push_back(processing_time);
    
    log_line("[stats] " + method_name + " - step=" + std::to_string(time_step) + 
             ", nodes=" + std::to_string(node_count) + 
             ", cum_dist=" + std::to_string(cumulative_distance) + "m" +
             ", layers=" + std::to_string(layer_count) + 
             ", time=" + std::to_string(processing_time) + "ms");
}

double SimpleLayerManagerV2::calculate_cumulative_distance_up_to_node(int node_idx)
{
    double total_distance = 0.0;
    
    // Calculate cumulative distance from odometry edges up to node_idx
    for (auto* edge : g2o_.nEdgesOdometry) {
        int max_node = std::max(edge->a->index, edge->b->index);
        if (max_node <= node_idx) {
            // Calculate Euclidean distance from odometry measurement
            double dx = edge->x;
            double dy = edge->y;
            double distance = std::sqrt(dx*dx + dy*dy);
            total_distance += distance;
        }
    }
    
    return total_distance;
}

void SimpleLayerManagerV2::output_method_statistics()
{
    log_line("\n=== METHOD STATISTICS SUMMARY ===");
    
    for (const auto& pair : method_statistics_) {
        const auto& method_name = pair.first;
        const auto& stats = pair.second;
        
        if (stats.time_steps.empty()) continue;
        
        log_line("\n--- " + method_name + " ---");
        log_line("Total time steps: " + std::to_string(stats.time_steps.size()));
        log_line("Final node count: " + std::to_string(stats.node_counts.back()));
        log_line("Final cumulative distance: " + std::to_string(stats.cumulative_distances.back()) + " m");
        log_line("Final layer count: " + std::to_string(stats.layer_counts.back()));
        
        // Calculate average processing time
        double avg_time = 0.0;
        for (double t : stats.processing_times) avg_time += t;
        avg_time /= stats.processing_times.size();
        log_line("Average processing time: " + std::to_string(avg_time) + " ms");
        
        // Show some detailed entries
        log_line("Sample entries:");
        int sample_interval = std::max(1, (int)stats.time_steps.size() / 10);
        for (size_t i = 0; i < stats.time_steps.size(); i += sample_interval) {
            log_line("  Step " + std::to_string(stats.time_steps[i]) + 
                     ": nodes=" + std::to_string(stats.node_counts[i]) + 
                     ", dist=" + std::to_string(stats.cumulative_distances[i]) + "m" +
                     ", layers=" + std::to_string(stats.layer_counts[i]));
        }
    }
}

void SimpleLayerManagerV2::save_statistics_to_file(const std::string& filename)
{
    std::ofstream file(save_path_ + "/" + filename);
    if (!file.is_open()) {
        log_line("[error] Could not open statistics file: " + filename);
        return;
    }
    
    // Write header
    file << "# Method Statistics\n";
    file << "# Format: method_name,time_step,node_count,cumulative_distance_m,layer_count,processing_time_ms\n";
    
    for (const auto& pair : method_statistics_) {
        const auto& method_name = pair.first;
        const auto& stats = pair.second;
        
        for (size_t i = 0; i < stats.time_steps.size(); ++i) {
            file << method_name << ","
                 << stats.time_steps[i] << ","
                 << stats.node_counts[i] << ","
                 << stats.cumulative_distances[i] << ","
                 << stats.layer_counts[i] << ","
                 << stats.processing_times[i] << "\n";
        }
    }
    
    file.close();
    log_line("[stats] Statistics saved to " + filename);
}
