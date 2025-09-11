#ifndef CERES_ERROR_H
#define CERES_ERROR_H

#include <algorithm>  
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

struct OdometryResidue
{
public:
    OdometryResidue(double dx, double dy, double dtheta);
    template <typename T> bool operator()(const T* const P1, const T* const P2, T* e) const;
    static ceres::CostFunction* Create(const double dx, const double dy, const double dtheta);

public:
    double dx;
    double dy;
    double dtheta;
    Eigen::Matrix<double,3,3> a_Tcap_b;
};


struct DCSClosureResidue
{
public:
    DCSClosureResidue (double dx, double dy, double dtheta );
    template <typename T> bool operator() (const T* const P1, const T* const P2, T* e) const;
    static ceres::CostFunction* Create(const double dx, const double dy, const double dtheta);

public:
    double dx;
    double dy;
    double dtheta;
    Eigen::Matrix<double,3,3> a_Tcap_b;
};

// Switchable constraints for loop closures
// Reference: "Switchable Constraints for Robust Pose Graph SLAM",
// Sunderhauf & Protzel, IROS 2012
struct SwitchableClosureResidue
{
public:
    SwitchableClosureResidue(double dx, double dy, double dtheta);
    template <typename T> bool operator()(const T* const P1, const T* const P2, const T* const S, T* e) const;
    static ceres::CostFunction* Create(const double dx, const double dy, const double dtheta);

public:
    double dx;
    double dy;
    double dtheta;
    Eigen::Matrix<double,3,3> a_Tcap_b;
};

// Simple prior term to keep switch close to 1.0
// Residual: sqrt_lambda * (1 - s)
struct SwitchPriorResidue
{
public:
    explicit SwitchPriorResidue(double lambda);
    template <typename T> bool operator()(const T* const S, T* e) const;
    static ceres::CostFunction* Create(const double lambda);

private:
    double lambda; // weighting factor
};
// PoseGraph2dErrorTerm with 1D blocks per parameter (x,y,yaw for A and B)
class PoseGraph2dErrorTerm {
 public:
  PoseGraph2dErrorTerm(double x_ab,
                       double y_ab,
                       double yaw_ab_radians,
                       const Eigen::Matrix3d& sqrt_information)
      : p_ab_(x_ab, y_ab),
        yaw_ab_radians_(yaw_ab_radians),
        sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const x_a,
                  const T* const y_a,
                  const T* const yaw_a,
                  const T* const x_b,
                  const T* const y_b,
                  const T* const yaw_b,
                  T* residuals_ptr) const {
    const Eigen::Matrix<T, 2, 1> p_a(*x_a, *y_a);
    const Eigen::Matrix<T, 2, 1> p_b(*x_b, *y_b);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals_ptr);

    const T ca = ceres::cos(*yaw_a);
    const T sa = ceres::sin(*yaw_a);
    Eigen::Matrix<T,2,2> R_a;
    R_a << ca, -sa,
           sa,  ca;

    residuals_map.template head<2>() =
        R_a.transpose() * (p_b - p_a) - p_ab_.template cast<T>();

    T dyaw = (*yaw_b - *yaw_a) - static_cast<T>(yaw_ab_radians_);
    dyaw = ceres::atan2(ceres::sin(dyaw), ceres::cos(dyaw));
    residuals_map(2) = dyaw;

    residuals_map = sqrt_information_.template cast<T>() * residuals_map;
    return true;
  }

  static ceres::CostFunction* Create(double x_ab,
                                     double y_ab,
                                     double yaw_ab_radians,
                                     const Eigen::Matrix3d& sqrt_information) {
    return (new ceres::AutoDiffCostFunction<PoseGraph2dErrorTerm, 3, 1, 1, 1, 1, 1, 1>(
        new PoseGraph2dErrorTerm(x_ab, y_ab, yaw_ab_radians, sqrt_information)));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Eigen::Vector2d p_ab_;
  const double yaw_ab_radians_;
  const Eigen::Matrix3d sqrt_information_;
};

#endif // CERES_ERROR_H
