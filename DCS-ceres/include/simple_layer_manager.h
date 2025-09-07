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
#include <Eigen/Dense>

#include "graph.h"
#include "g2o_util.h"
#include "ceres_error.h"

struct SimpleLayerConfig
{
    double expansion_prob = 0.3;     // 30% 확률로 레이어 확장
    int max_layers = 20;
    int local_iters = 2;
    double huber_delta = 0.01;
    double ema_alpha = 0.1;
    double epsilon = 1e-3;
    double conflict_tau = 0.3; // 분할 결정 임계값 (Δ = L_ij - min(L_i, L_e))
    
    // MCTS reward 파라미터
    double alpha_info = 1.1;         // ΔH 정보 이득 가중치
    double beta_sparse = 0.1;       // n_lc 희소성 페널티 가중치
    double mcts_exploration_c = 1.414; // UCT 탐색 상수
    
    // Residual 기반 엣지 필터링 (Mahalanobis distance 기준)
    double residual_low = 3.0;       // R_low: 이 값 이하면 반드시 추가 (chi2 분포 99% 임계값)
    double residual_high = 50.0;     // R_high: 이 값 이상이면 skip (명백한 outlier)
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
    
    // Residual 기반 필터링
    double calculate_edge_residual(const std::string& layer_id, Edge* edge);
    bool should_add_edge(const std::string& layer_id, Edge* edge);
    
    // 최적화
    void optimize_layer(const std::string& layer_id);
    void optimize_local_window(const std::string& layer_id, int window_size = 10);
    double evaluate_layer_cost(const std::string& layer_id);
    
    // 유틸리티
    void log_line(const std::string& s);
    void save_results();
    std::string get_best_layer();
    std::string get_most_visited_layer();
    std::string get_most_edges_layer();
    double normalize_reward_by_edge_count(double total_reward, int edge_count);

private:
    ReadG2O& g2o_;
    std::string save_path_;
    SimpleLayerConfig config_;
    
    std::unordered_map<std::string, std::unique_ptr<SimpleLayer>> layers_;
    std::string root_layer_id_;
    
    std::vector<Edge*> candidate_edges_;
    std::vector<std::pair<Edge*, std::string>> assignments_; // (edge, layer_id)
    
    std::ofstream logfile_;
    int step_counter_ = 0;
    int layer_id_counter_ = 0;
};

#endif // SIMPLE_LAYER_MANAGER_H
