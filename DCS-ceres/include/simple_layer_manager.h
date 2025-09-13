#ifndef SIMPLE_LAYER_MANAGER_H
#define SIMPLE_LAYER_MANAGER_H

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <ceres/ceres.h>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Dense>
#include <thread>

#include "graph.h"
#include "g2o_util.h"
#include "ceres_error.h"

struct SimpleLayerConfig
{
    double expansion_prob = 0.3;     // 30% 확률로 레이어 확장
    int max_layers = 20;
    int local_iters = 5;
    double huber_delta = 0.01;
    double ema_alpha = 0.1;
    double epsilon = 1e-3;
    double conflict_tau = 0.5; // 분할 결정 임계값 (Δ = L_ij - min(L_i, L_e))
    
    // Snapshot/Visualization
    int snapshot_every = 20; // 0이면 비활성화, N>0이면 N스텝마다 스냅샷 저장
    
    // MCTS reward 파라미터
    double alpha_info = 0.4;         // ΔH 정보 이득 가중치
    double beta_sparse = 0.2;       // n_lc 희소성 페널티 가중치
    double mcts_exploration_c = 1.414; // UCT 탐색 상수
    
    // Residual 기반 엣지 필터링
    double residual_low = 0.0;        // 모드 0: Mahalanobis 기준의 저역 게이트
    double residual_high = 0.8;     // 모드 0: Mahalanobis 기준의 고역 게이트
    int residual_mode = 1;            // 0: Mahalanobis, 1: 포함/미포함 최적화 영향 기반
    int impact_iters = 1;             // 모드 1: solve 반복 횟수
    int impact_window = 0;            // 모드 1: >0이면 a,b 주변 index window만 사용, 0이면 온라인 k 사용
    double impact_theta_weight = 1.0; // 모드 1: 포즈 델타 각도 가중치
    double impact_pose_weight = 1.0;  // 모드 1: 포즈 델타 항의 가중치

    // Neighbor residual improvement metric (for analysis; not enforced yet)
    double neighbor_improve_abs_tau = 0.05;   // 절대 개선 임계 제안값
    double neighbor_improve_rel_tau = 0.03;   // 상대 개선 임계 제안값
    double neighbor_neg_slack = 0.005;        // 소폭 악화 허용 슬랙
    double neighbor_theta_weight = 1.0;       // 이웃 잔차 각도 가중(기본=1)

    // Merge-to-base 정책
    double merge_threshold = 0.4;     // 정규화 보상이 이 값 이상이면 Base로 머지
    int min_edges_to_merge = 3;       // 머지 최소 엣지 수 조건

    // Top-K/관계 전파 정책
    int topk_layers = 5;              // 매 스텝 최상위 K개 레이어에도 엣지를 추가(최적화는 best만)
    bool propagate_poses_from_best = true; // best 레이어의 포즈를 다른 상위 레이어로 복사 갱신
    bool propagate_to_parent_siblings = false; // Top-K 대신 best의 부모/형제 레이어로 전파

    // Switching Constraints (SC)
    double sc_prior_lambda = 1.0;     // 스위치 prior 강도 (1에 가깝게 유지)
    double sc_active_threshold = 0.5; // 활성 판단 임계값 (s > this)
};

struct SimpleLayerStats
{
    double ema_residual = 0.0;
    int num_edges = 0;
};

struct SimpleLayer
{
    std::string id;
    std::string parent_id;
    std::vector<double*> poses;        // 각 레이어별 독립적인 pose 복사본
    std::vector<Edge*> inherited_edges; // 부모로부터 상속받은 엣지들
    std::vector<Edge*> added_edges;    // 이 레이어에서 추가된 엣지들
    SimpleLayerStats stats;

    // Switch variables per edge (loop/bogus)
    std::unordered_map<Edge*, double*> switch_vars; // new/delete 관리 필요
    
    // MCTS 통계
    int visits = 0;
    double total_reward = 0.0;
    std::vector<std::string> children; // 자식 레이어 ID들
    
    // 전체 엣지를 반환하는 헬퍼 함수
    std::vector<Edge*> get_all_edges() const {
        std::vector<Edge*> all_edges;
        all_edges.reserve(inherited_edges.size() + added_edges.size());
        all_edges.insert(all_edges.end(), inherited_edges.begin(), inherited_edges.end());
        all_edges.insert(all_edges.end(), added_edges.begin(), added_edges.end());
        return all_edges;
    }
};

class SimpleLayerManagerV2
{
public:
    SimpleLayerManagerV2(ReadG2O& g, const std::string& save_path, const SimpleLayerConfig& cfg);
    ~SimpleLayerManagerV2();
    
    // METHOD 4 실행
    void run();
    // 온라인 모드 실행 (노드 순차 처리)
    void run_online();

private:
    // 레이어 관리
    std::string create_child_layer(const std::string& parent_id, Edge* new_edge, bool include_edge);
    std::string generate_layer_id();
    SimpleLayer* get_layer(const std::string& layer_id);
    
    // MCTS 관련
    std::string select_layer_by_uct();
    void expand_layer(const std::string& layer_id, Edge* new_edge);
    bool should_split_layer(const std::string& layer_id, Edge* new_edge);
    double simulate_layer(const std::string& layer_id);
    void backpropagate(const std::string& layer_id, double reward);
    
    // 보상 함수 계산
    double calculate_reward(const std::string& layer_id, Edge* added_edge);
    double calculate_cost_delta_rel(const std::string& layer_id, Edge* edge);
    double calculate_info_gain(Edge* edge);
    int count_closure_edges(const std::string& layer_id, Edge* additional_edge = nullptr);
    int count_active_closure_edges(const std::string& layer_id, int k_lim = -1) const; // k_lim>=0이면 k 이하만
    
    // Residual 기반 필터링
    double calculate_edge_residual(const std::string& layer_id, Edge* edge);
    double calculate_edge_residual_impact(const std::string& layer_id, Edge* edge);
    // 새 엣지 추가 전/후로, 해당 엣지와 노드를 공유하는 인접 엣지들의
    // residual 합(또는 평균)이 얼마나 개선되는지(감소) 측정하는 척도
    // 반환: sum_residual_before - sum_residual_after (양수면 개선)
    double calculate_neighbor_residual_improvement(const std::string& layer_id, Edge* edge);
    // 임시 레이어 기반 엣지 평가/병합 파이프라인
    std::pair<double, std::string> process_edge_with_temp_layer(const std::string& parent_id, Edge* edge);
    double compute_neighbor_improvement_between_layers(const std::string& parent_id, const std::string& child_id, Edge* edge);
    void merge_child_into_parent_and_delete(const std::string& parent_id, const std::string& child_id, Edge* edge);
    bool should_add_edge(const std::string& layer_id, Edge* edge, double precomputed_residual = std::numeric_limits<double>::quiet_NaN());
    
    // 최적화
    void optimize_layer(const std::string& layer_id);
    void optimize_layer_upto_k(const std::string& layer_id, int k);
    void optimize_local_window(const std::string& layer_id, int window_size = 10);
    double evaluate_layer_cost(const std::string& layer_id);
    
    // Statistics tracking for different methods
    struct MethodStats {
        std::string method_name;
        std::vector<int> time_steps;
        std::vector<int> node_counts;
        std::vector<double> cumulative_distances;
        std::vector<int> layer_counts;
        std::vector<double> processing_times;
    };
    
    void track_method_statistics(const std::string& method_name, int time_step, int node_count, 
                               double cumulative_distance, int layer_count, double processing_time);
    void output_method_statistics();
    void save_statistics_to_file(const std::string& filename);
    double calculate_cumulative_distance_up_to_node(int node_idx);
    
    // 유틸리티
    void log_line(const std::string& s);
    void save_results();
    std::string get_best_layer();
    std::string get_most_visited_layer();
    std::string get_most_edges_layer();
    double normalize_reward_by_edge_count(double total_reward, int edge_count);
    std::vector<std::string> get_topk_layers_by_reward(int k);
    std::vector<std::string> get_parent_and_sibling_layers(const std::string& layer_id,
                                                           bool include_parent = true,
                                                           bool include_siblings = true);
    void propagate_edge_to_layers(Edge* e, const std::vector<std::string>& layer_ids,
                                  const std::string& best_id, const std::string& exclude_id);
    void copy_poses(const std::string& src_layer, const std::string& dst_layer);
    
    // Merge-to-base
    bool should_merge_layer(const std::string& layer_id);
    void merge_layer_into_base(const std::string& layer_id);
    void free_layer_poses(SimpleLayer* layer);
    void free_layer_switches(SimpleLayer* layer);
    void save_snapshot(int step_index);
    void write_layer_poses_capped(const std::string& layer_id, const std::string& filepath, int max_index);
    void write_initial_poses_capped(const std::string& filepath, int max_index);

private:
    ReadG2O& g2o_;
    std::string save_path_;
    SimpleLayerConfig config_;
    
    std::unordered_map<std::string, std::unique_ptr<SimpleLayer>> layers_;
    std::string root_layer_id_;
    std::string base_layer_id_;
    
    std::vector<Edge*> candidate_edges_;
    std::vector<std::pair<Edge*, std::string>> assignments_; // (edge, layer_id)
    
    std::ofstream logfile_;
    int iters_per_step = 100;      // ceres iterations per online step
    int step_counter_ = 0;
    int layer_id_counter_ = 0;
    int online_active_k_ = -1; // 현재 온라인 단계의 활성 최대 노드 인덱스 (오프라인: -1)
    bool b_add_loop_edges_ = false; // for testing
    
    // Method statistics tracking
    std::map<std::string, MethodStats> method_statistics_;
    std::chrono::high_resolution_clock::time_point method_start_time_;

    // 가까운(중복 처리 대상) 노드 인덱스 집합
    // - 중복 삽입 방지: std::set으로 유일성 및 정렬 유지
    // - 카운트는 집합 크기이거나, 구간별로 upper_bound로 빠르게 계산 가능
    std::set<int> excluded_nodes_;
    int count_excluded_up_to(int idx) const;
};

//
// METHOD 5: SimpleLayerManager2 (no branching, online, dynamic parameter blocks)
// - Maintains a single global problem
// - At step k, explicitly adds parameter blocks for nodes first seen and adds
//   odometry/closure/bogus edges that touch node k
// - Uses Switchable Constraints for closure/bogus edges
//
struct SimpleLayer2Config
{
    int iters_per_step = 400;      // ceres iterations per online step
    double huber_delta = 0.05;   // robust kernel delta
    double sc_prior_lambda = 1.0; // prior weight for switch variables
    int snapshot_every = 20;      // 0=off, >0 means save every N steps
    bool bound_switch_01 = true; // clamp s in [0,1]

    // Reward (borrowed from METHOD 4)
    double alpha_info = 1.1;           // information gain weight
    double beta_sparse = 0.1;          // active closures sparsity penalty
    double epsilon = 1e-3;             // small for relative cost
    double sc_active_threshold = 0.5;  // consider s>threshold as active
};

class SimpleLayerManager2
{
public:
    SimpleLayerManager2(ReadG2O& g, const std::string& save_path, const SimpleLayer2Config& cfg);
    ~SimpleLayerManager2();

    // Online-only execution
    void run_online();

private:
    void log_line(const std::string& s);
    void save_nodes(const std::string& filepath);
    void save_switches(const std::string& filepath);

    // METHOD 4-like reward pieces
    double calculate_info_gain(Edge* edge);
    int count_active_closure_edges_upto_k(int k) const;
    double evaluate_global_cost_upto_k(int k, Edge* exclude_edge, bool include_candidate);
    double calculate_cost_delta_rel(int k, Edge* added_edge);
    double calculate_reward(int k, Edge* added_edge);
    
    // Statistics tracking for different methods
    struct MethodStats {
        std::string method_name;
        std::vector<int> time_steps;
        std::vector<int> node_counts;
        std::vector<double> cumulative_distances;
        std::vector<int> layer_counts;
        std::vector<double> processing_times;
    };
    
    void track_method_statistics(const std::string& method_name, int time_step, int node_count, 
                               double cumulative_distance, int layer_count, double processing_time);
    void output_method_statistics();
    void save_statistics_to_file(const std::string& filename);
    double calculate_cumulative_distance_up_to_node(int node_idx);

private:
    ReadG2O& g2o_;
    std::string save_path_;
    SimpleLayer2Config config_;

    ceres::Problem problem_;
    ceres::LossFunction* loss_odom_ = nullptr;
    ceres::LossFunction* loss_loop_ = nullptr;

    std::vector<bool> node_added_;
    bool anchor_added_ = false;
    bool b_add_loop_edges_ = false; // for testing

    // map Edge* -> switch variable pointer for SC
    std::unordered_map<Edge*, double*> switch_vars_;
    
    // Method statistics tracking
    std::map<std::string, MethodStats> method_statistics_;
    std::chrono::high_resolution_clock::time_point method_start_time_;
};

#endif // SIMPLE_LAYER_MANAGER_H
