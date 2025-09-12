#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>       /* timing measurement */
#include <iomanip>      /* setprecision, fixed */

#include "ceres_error.h"
#include "g2o_util.h"
#include "graph.h"
#include "simple_layer_manager.h" // METHOD 4 + METHOD 5
#include "simple_layer_manager6.h" // METHOD 6

using std::cout;
using std::endl;
using std::string;

string BASE_PATH = std::string( "../data");
string SAVE_PATH = std::string( "../save");

/*
DATASET_NAME_WITHOUGH_DOTG2O: {INTEL, M3500, M3500b, M3500c, CSAIL, FR079, FRH, M10000}

 * How to use
 * $ ./main DATASET_NAME_WITHOUGH_DOTG2O NUM_OUTLIER_LOOPS METHOD
 * - METHOD: 0=baseline, 1=DCS, 2=Switchable Constraints, 4=Simple Layer MCTS, 5=Simple Layer Online
 * - e.g., $ ./main INTEL 50 1 # USING DCS
 * - e.g., $ ./main INTEL 50 0 # NOT USING DCS
 * - e.g., $ ./main INTEL 50 2 # USING Switchable Constraints
*/
static bool has_flag(int argc, char* argv[], const std::string& flag)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == flag) return true;
        // allow short variant
        if (flag == "--online" && (a == "-online")) return true;
    }
    return false;
}

static bool env_on(const char* name)
{
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    for (auto& c : s) c = std::tolower(c);
    return (s == "1" || s == "true" || s == "on");
}

static int run_online_baseline_dcs_sc(ReadG2O& g2o_manager, int METHOD)
{
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.01);

    // Anchor first pose: add parameter block explicitly before fixing it
    if (!g2o_manager.nNodes.empty()) {
        problem.AddParameterBlock(g2o_manager.nNodes[0]->p, 3);
        problem.SetParameterBlockConstant(g2o_manager.nNodes[0]->p);
    }
    
    // Statistics tracking structure
    struct MethodStats {
        std::string method_name;
        std::vector<int> time_steps;
        std::vector<int> node_counts;
        std::vector<double> cumulative_distances;
        std::vector<int> layer_counts;
        std::vector<double> processing_times;
    };
    
    // Optimization timing log file setup for method1/2
    std::string method_name = (METHOD == 1) ? "method1_dcs" : (METHOD == 2) ? "method2_sc" : "method0_baseline";
    std::string timing_log_path = SAVE_PATH + "/" + method_name + "_optimization_timing.txt";
    std::ofstream timing_log(timing_log_path);
    timing_log << "# k edge_type num_edges optimization_type duration_ms\n";
    timing_log.flush();
    
    // Statistics tracking
    MethodStats stats;
    stats.method_name = method_name;
    
    // Function to calculate cumulative distance
    auto calculate_cumulative_distance = [&](int node_idx) -> double {
        double total = 0.0;
        for (auto* edge : g2o_manager.nEdgesOdometry) {
            int ia = edge->a->index, ib = edge->b->index;
            if (std::max(ia, ib) <= node_idx) {
                double dx = edge->x, dy = edge->y;
                total += std::sqrt(dx*dx + dy*dy);
            }
        }
        return total;
    };

    std::vector<double*> switch_variables; // for SC
    std::vector<double> switch_priors;
    const double sc_prior_lambda = 1.0;

    int N = (int)g2o_manager.nNodes.size();
    for (int k = 1; k < N; ++k) {
        int added_odo = 0, added_closure = 0, added_bogus = 0;
        // add odometry with max(a,b) == k
        for (auto* ed : g2o_manager.nEdgesOdometry) {
            int ia = ed->a->index, ib = ed->b->index;
            if (std::max(ia, ib) != k) continue;
            ceres::CostFunction* cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
            problem.AddResidualBlock(cost, loss_function, ed->a->p, ed->b->p);
            added_odo++;
        }
        // add closures (and bogus) touching the new node k
        auto add_closure_like = [&](Edge* ed){
            ceres::CostFunction* cost = nullptr;
            if (METHOD==1) {
                cost = DCSClosureResidue::Create(ed->x, ed->y, ed->theta);
                problem.AddResidualBlock(cost, loss_function, ed->a->p, ed->b->p);
                added_closure++;
            } else if (METHOD==2) {
                double* s = new double(1.0);
                switch_variables.push_back(s);
                switch_priors.push_back(1.0);
                cost = SwitchableClosureResidue::Create(ed->x, ed->y, ed->theta);
                problem.AddResidualBlock(cost, loss_function, ed->a->p, ed->b->p, s);
                ceres::CostFunction* prior = SwitchPriorResidue::Create(sc_prior_lambda);
                problem.AddResidualBlock(prior, nullptr, s);
                // treat as closure-type for logging
                added_closure++;
            } else {
                cost = OdometryResidue::Create(ed->x, ed->y, ed->theta);
                problem.AddResidualBlock(cost, loss_function, ed->a->p, ed->b->p);
                added_closure++;
            }
        };
        for (auto* ed : g2o_manager.nEdgesClosure) {
            int ia = ed->a->index, ib = ed->b->index;
            if (std::max(ia, ib) == k) add_closure_like(ed);
        }
        for (auto* ed : g2o_manager.nEdgesBogus) {
            int ia = ed->a->index, ib = ed->b->index;
            if (std::max(ia, ib) == k) {
                add_closure_like(ed);
                added_bogus++;
            }
        }

        // Track statistics for this step
        auto step_start_time = std::chrono::high_resolution_clock::now();
        double cum_distance = calculate_cumulative_distance(k);
        stats.time_steps.push_back(k);
        stats.node_counts.push_back(k + 1);  // node count includes node 0 to k
        stats.cumulative_distances.push_back(cum_distance);
        stats.layer_counts.push_back(1);  // baseline methods use single layer
        
        // log per-node activation summary
        std::cout << "[online] activate node k=" << k
                  << ", odom_added=" << added_odo
                  << ", closure_added=" << added_closure
                  << ", bogus_added=" << added_bogus
                  << ", cum_dist=" << std::fixed << std::setprecision(3) << cum_distance << "m"
                  << std::endl;

        // Only optimize when loop closure or bogus edges are added
        bool has_loop_edges = (added_closure > 0) || (added_bogus > 0);
        if (has_loop_edges) {
            std::cout << "[optimize] Loop edges detected, running optimization..." << std::endl;
            
            ceres::Solver::Options options;
            options.max_num_iterations = std::max(1, 100);
            options.minimizer_progress_to_stdout = false;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.num_threads = std::thread::hardware_concurrency() != 0
                ? std::thread::hardware_concurrency()
                : 4;
            ceres::Solver::Summary summary;
            
            // Measure optimization time
            auto start_time = std::chrono::high_resolution_clock::now();
            ceres::Solve(options, &problem, &summary);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Log timing data
            std::ofstream timing_log(timing_log_path, std::ios::app);
            timing_log << k << " loop_edges " << (added_closure + added_bogus) 
                       << " loop_solve " << duration.count() << "\n";
            timing_log.close();
            
            // Record processing time for statistics
            stats.processing_times.push_back(duration.count());
        } else {
            std::cout << "[skip] Only odometry edges added, skipping optimization" << std::endl;
            // Record 0 processing time when no optimization is performed
            stats.processing_times.push_back(0.0);
        }
            
    }

    // Final optimization after all nodes are processed
    std::cout << "[final] Running final optimization..." << std::endl;
    ceres::Solver::Options final_options;
    final_options.max_num_iterations = std::max(1, 200);
    final_options.minimizer_progress_to_stdout = false;
    final_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    final_options.num_threads = std::thread::hardware_concurrency() != 0
        ? std::thread::hardware_concurrency()
        : 4;
    ceres::Solver::Summary final_summary;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    ceres::Solve(final_options, &problem, &final_summary);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Log final optimization timing
    std::ofstream final_timing_log(timing_log_path, std::ios::app);
    final_timing_log << N-1 << " all " << (g2o_manager.nEdgesClosure.size() + g2o_manager.nEdgesBogus.size()) 
                     << " final_solve " << duration.count() << "\n";
    final_timing_log.close();

    // Output statistics summary
    std::cout << "\n=== " << method_name << " Statistics ===" << std::endl;
    if (!stats.time_steps.empty()) {
        std::cout << "Total steps processed: " << stats.time_steps.size() << std::endl;
        std::cout << "Final node count: " << stats.node_counts.back() << std::endl;
        std::cout << "Final cumulative distance: " << std::fixed << std::setprecision(3) 
                  << stats.cumulative_distances.back() << "m" << std::endl;
        
        double total_processing_time = 0.0;
        for (double t : stats.processing_times) {
            total_processing_time += t;
        }
        std::cout << "Total processing time: " << total_processing_time << "ms" << std::endl;
    }
    
    // Save statistics to file
    std::string stats_filename = SAVE_PATH + "/" + method_name + "_statistics.txt";
    std::ofstream stats_file(stats_filename);
    if (stats_file.is_open()) {
        stats_file << "# Method Statistics Data\n";
        stats_file << "# Format: method_name,time_step,node_count,cumulative_distance,layer_count,processing_time_ms\n";
        
        for (size_t i = 0; i < stats.time_steps.size(); ++i) {
            stats_file << stats.method_name << ","
                      << stats.time_steps[i] << ","
                      << stats.node_counts[i] << ","
                      << std::fixed << std::setprecision(6) << stats.cumulative_distances[i] << ","
                      << stats.layer_counts[i] << ","
                      << std::fixed << std::setprecision(3) << stats.processing_times[i] << "\n";
        }
        stats_file.close();
        std::cout << "Statistics saved to: " << stats_filename << std::endl;
    }

    // write outputs (poses and edges)
    g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/opt_nodes.txt");
    g2o_manager.writePoseGraph_edges(SAVE_PATH+"/opt_edges.txt");
    if (METHOD==2) {
        g2o_manager.writePoseGraph_switches(SAVE_PATH+"/switches.txt", switch_priors, switch_variables);
    }
    return 0;
}

auto main(int argc, char *argv[]) ->int
{
    // 인자 체크
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " DATASET NUM_OUTLIER_LOOPS METHOD [--online]\n";
        cout << "METHOD: 0=baseline, 1=DCS, 2=Switchable, 3=Layer, 4=Simple Layer MCTS\n";
        cout << "Example: " << argv[0] << " INTEL 50 4\n";
        return -1;
    }

    // @ random seed change for tests
    std::srand((unsigned int) time(0)); // for random bogus add 

    // @ Read g2o file and add noise edges
    string fname{argv[1]};
    string fpath = BASE_PATH + "/" + fname + ".g2o";
    cout << "Start Reading PoseGraph\n";
    ReadG2O g2o_manager( fpath );

    int num_bogus_loops{atoi(argv[2])};
    g2o_manager.add_random_C(num_bogus_loops); // adding bogus (false) loops

    int METHOD{atoi(argv[3])};
    bool DCS_ON = (METHOD==1);
    bool SC_ON  = (METHOD==2);

    g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/init_nodes.txt");
    g2o_manager.writePoseGraph_edges(SAVE_PATH+"/init_edges.txt");
    cout << "total nodes : "<< g2o_manager.nNodes.size() << endl;
    cout << "total nEdgesOdometry : "<< g2o_manager.nEdgesOdometry.size() << endl;
    cout << "total nEdgesClosure : "<< g2o_manager.nEdgesClosure.size() << endl;
    cout << "total nEdgesBogus : "<< g2o_manager.nEdgesBogus.size() << endl;

    // @ Make a cost function 
    ceres::Problem problem;
    ceres::LossFunction * loss_function = NULL;
    loss_function = new ceres::HuberLoss(0.01); // robust kernel

    // METHOD 3: lightweight probabilistic layering with local optimization per layer
    bool ONLINE = has_flag(argc, argv, "--online") || env_on("ONLINE");
    cout << "[mode] ONLINE=" << (ONLINE?"1":"0")
         << ", METHOD=" << METHOD
         << ", DATASET='" << fname << "'" << endl;

    // METHOD 3 removed

    // METHOD 4: Layer Manager (MCTS-based layering)
    if (!ONLINE && METHOD==4) {
        SimpleLayerConfig cfg; // defaults with 30% expansion, MCTS parameters
        SimpleLayerManagerV2 manager(g2o_manager, SAVE_PATH, cfg);
        manager.run();

        // Also dump init/edges for consistency with plotting
        g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/init_nodes.txt");
        g2o_manager.writePoseGraph_edges(SAVE_PATH+"/init_edges.txt");
        return 0;
    }

    // ONLINE modes
    if (ONLINE) {
        if (METHOD==0 || METHOD==1 || METHOD==2) {
            return run_online_baseline_dcs_sc(g2o_manager, METHOD);
        }
        if (METHOD==4) {
            SimpleLayerConfig cfg;
            SimpleLayerManagerV2 manager(g2o_manager, SAVE_PATH, cfg);
            manager.run_online();
            // Also dump init/edges for consistency with plotting
            g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/init_nodes.txt");
            g2o_manager.writePoseGraph_edges(SAVE_PATH+"/init_edges.txt");
            return 0;
        }
        if (METHOD==6) {
            SimpleLayer6Config cfg6; SimpleLayerManager6 manager(g2o_manager, SAVE_PATH, cfg6);
            manager.run_online();
            g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/init_nodes.txt");
            g2o_manager.writePoseGraph_edges(SAVE_PATH+"/init_edges.txt");
            return 0;
        }
        if (METHOD==5) {
            // METHOD 5: SimpleLayerManager2 (no branching, online only, dynamic parameter blocks)
            SimpleLayer2Config cfg;
            // Enable periodic snapshots similar to method 4
            cfg.snapshot_every = 5;
            SimpleLayerManager2 manager(g2o_manager, SAVE_PATH, cfg);
            manager.run_online();
            // Also dump init/edges for consistency with plotting
            g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/init_nodes.txt");
            g2o_manager.writePoseGraph_edges(SAVE_PATH+"/init_edges.txt");
            return 0;
        }
    }

    // @ 1. Odometry Constraints
    for( int i=0 ; i<g2o_manager.nEdgesOdometry.size() ; i++ )
    {
        Edge* ed = g2o_manager.nEdgesOdometry[i];
        ceres::CostFunction * cost_function = OdometryResidue::Create( ed->x, ed->y, ed->theta );
        problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p );
    }

    // @ 2. Loop Closure Constaints 
    // - DCS: see DCSClosureResidue
    // - Switchable Constraints: add a scalar switch with prior per loop
    std::vector<double> switch_priors;     // for SC
    std::vector<double*> switch_variables; // for SC
    const double sc_prior_lambda = 1.0;    // weighting for prior term
    for( int i=0 ; i<g2o_manager.nEdgesClosure.size() ; i++ )
    {   // for clean edges
        Edge* ed = g2o_manager.nEdgesClosure[i];
        ceres::CostFunction * cost_function;
        if (DCS_ON) {
            cost_function = DCSClosureResidue::Create( ed->x, ed->y, ed->theta );
            problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p );
        } else if (SC_ON) {
            // create switch var s initialized to 1.0
            double* s = new double(1.0);
            switch_variables.push_back(s);
            switch_priors.push_back(1.0);
            // closure residual scaled by switch
            cost_function = SwitchableClosureResidue::Create( ed->x, ed->y, ed->theta );
            problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p, s );
            // prior on switch s ~ 1.0
            ceres::CostFunction* prior = SwitchPriorResidue::Create(sc_prior_lambda);
            problem.AddResidualBlock( prior, nullptr, s );
        } else {
            cost_function = OdometryResidue::Create( ed->x, ed->y, ed->theta );
            problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p );
        }
    } 
    for( int i=0 ; i<g2o_manager.nEdgesBogus.size() ; i++ )
    {   // for bogus edges
        Edge* ed = g2o_manager.nEdgesBogus[i];
        ceres::CostFunction * cost_function;
        if (DCS_ON) {
            cost_function = DCSClosureResidue::Create( ed->x, ed->y, ed->theta );
            problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p );
        } else if (SC_ON) {
            double* s = new double(1.0);
            switch_variables.push_back(s);
            switch_priors.push_back(1.0);
            cost_function = SwitchableClosureResidue::Create( ed->x, ed->y, ed->theta );
            problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p, s );
            ceres::CostFunction* prior = SwitchPriorResidue::Create(sc_prior_lambda);
            problem.AddResidualBlock( prior, nullptr, s );
        } else {
            cost_function = OdometryResidue::Create( ed->x, ed->y, ed->theta );
            problem.AddResidualBlock( cost_function, loss_function, ed->a->p, ed->b->p );
        }
    }

    // @ Solve 
    problem.SetParameterBlockConstant(g2o_manager.nNodes[0]->p); // i.e., 1st pose be origin
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    // options.preconditioner_type = ceres::SCHUR_JACOBI;
    ceres::Solver::Summary summary;
    
    // Measure optimization time for offline mode
    std::string method_name = (METHOD == 1) ? "method1_dcs" : (METHOD == 2) ? "method2_sc" : "method0_baseline";
    std::string timing_log_path = SAVE_PATH + "/" + method_name + "_optimization_timing.txt";
    std::ofstream timing_log(timing_log_path);
    timing_log << "# k edge_type num_edges optimization_type duration_ms\n";
    
    int total_edges = g2o_manager.nEdgesOdometry.size() + g2o_manager.nEdgesClosure.size() + g2o_manager.nEdgesBogus.size();
    auto start_time = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, &problem, &summary);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    timing_log << g2o_manager.nNodes.size()-1 << " all " << total_edges 
               << " batch_solve " << duration.count() << "\n";
    timing_log.close();
    
    cout << summary.FullReport() << endl;

    // @ Write Pose Graph file after Optimization
    g2o_manager.writePoseGraph_nodes(SAVE_PATH+"/opt_nodes.txt");
    g2o_manager.writePoseGraph_edges(SAVE_PATH+"/opt_edges.txt");
    if (SC_ON) {
        g2o_manager.writePoseGraph_switches(SAVE_PATH+"/switches.txt", switch_priors, switch_variables);
    }

} // End main 
