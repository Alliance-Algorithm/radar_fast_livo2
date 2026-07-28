#pragma once
// esikf_state.hpp - SO3 state math + pointWithVar + ESIKF 19维状态（跳过 inv_expo_time = 18 维）
// 原始来源: hku-mars/FAST-LIVO2/include/common_lib.h (StatesGroup, pointWithVar)
//           hku-mars/FAST-LIVO2/include/utils/so3_math.h (SO3 exponential, logarithm, skew)
//
// 移植: radar_fast_livo2 项目（纯 Eigen 实现，不依赖 Sophus）
//
// 状态布局 (DIM_STATE=18):
//   [0-2]   rot  (SO(3) tangent space)
//   [3-5]   pos  (world frame)
//   [6-8]   vel  (world frame)
//   [9-11]  bias_g (gyroscope bias)
//   [12-14] bias_a (accelerometer bias)
//   [15-17] gravity (world frame)
//
// 跳过: inv_expo_time (仅用于 VIO 曝光时间估计，纯 LIO 不需要)

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "radar_fast_livo2/lie.hpp"

namespace radar::fast_livo2 {

// ── 维度常量 ────────────────────────────────────────────────────────
constexpr int DIM_STATE   = 18; // 状态维度（不含 inv_expo_time）
constexpr double INIT_COV = 0.01;

// ── 类型别名 ────────────────────────────────────────────────────────
using M3D                        = Eigen::Matrix3d;
using V3D                        = Eigen::Vector3d;
template <int R, int C> using MD = Eigen::Matrix<double, R, C>;
template <int N> using VD        = Eigen::Matrix<double, N, 1>;

/// 旋转矩阵 → 欧拉角 (ZYX)
inline Eigen::Vector3d RotMtoEuler(const Eigen::Matrix3d& rot) {
    double sy     = std::sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(rot(2, 1), rot(2, 2));
        y = std::atan2(-rot(2, 0), sy);
        z = std::atan2(rot(1, 0), rot(0, 0));
    } else {
        x = std::atan2(-rot(1, 2), rot(1, 1));
        y = std::atan2(-rot(2, 0), sy);
        z = 0;
    }
    return Eigen::Vector3d(x, y, z);
}

// ── pointWithVar: 带方差的点（用于 voxel_map 体素插入和 ESIKF 残差构建）──
// 等价于 FAST-LIVO2 common_lib.h: pointWithVar
struct alignas(16) pointWithVar {
    Eigen::Vector3d point_b;        // LiDAR body frame
    Eigen::Vector3d point_i;        // IMU body frame
    Eigen::Vector3d point_w;        // world frame
    Eigen::Matrix3d var_nostate;    // covariance without state uncertainty
    Eigen::Matrix3d body_var;       // body-frame covariance (from sensor model)
    Eigen::Matrix3d var;            // full world-frame covariance
    Eigen::Matrix3d point_crossmat; // skew-sym of point (for Jacobian propagation)
    Eigen::Vector3d normal;         // matched plane normal

    pointWithVar() {
        var_nostate    = Eigen::Matrix3d::Zero();
        var            = Eigen::Matrix3d::Zero();
        body_var       = Eigen::Matrix3d::Zero();
        point_crossmat = Eigen::Matrix3d::Zero();
        point_b        = Eigen::Vector3d::Zero();
        point_i        = Eigen::Vector3d::Zero();
        point_w        = Eigen::Vector3d::Zero();
        normal         = Eigen::Vector3d::Zero();
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ── StatesGroup: ESIKF 状态组 ─────────────────────────────────────
//
// 等价于 FAST-LIVO2 common_lib.h: StatesGroup
// 差异:
//   - DIM_STATE = 18（原 19，去掉了 inv_expo_time）
//   - 不使用 Sophus，纯 Eigen 实现 SO(3) 运算
//   - operator+ 使用 SO3 exponential 更新旋转（so(3) → SO(3)）
//   - operator- 使用 SO3 logarithm 计算旋转差（SO(3) → so(3)）
//   - 协方差初始化与原版一致（仅索引平移一位）
struct alignas(16) StatesGroup {
    M3D rot_end;                  // 帧末姿态旋转矩阵 (world frame)
    V3D pos_end;                  // 帧末位置 (world frame)
    V3D vel_end;                  // 帧末速度 (world frame)
    V3D bias_g;                   // 陀螺仪零偏
    V3D bias_a;                   // 加速度计零偏
    V3D gravity;                  // 重力加速度 (world frame)
    MD<DIM_STATE, DIM_STATE> cov; // 状态协方差矩阵

    StatesGroup() {
        rot_end = M3D::Identity();
        pos_end = V3D::Zero();
        vel_end = V3D::Zero();
        bias_g  = V3D::Zero();
        bias_a  = V3D::Zero();
        gravity = V3D::Zero();
        cov     = MD<DIM_STATE, DIM_STATE>::Identity() * INIT_COV;
        // bias_g(3) + bias_a(3) + gravity(3) = 9 dims, indices 9-17
        cov.block<9, 9>(9, 9) = MD<9, 9>::Identity() * 0.00001;
    }

    StatesGroup(const StatesGroup& b) {
        rot_end = b.rot_end;
        pos_end = b.pos_end;
        vel_end = b.vel_end;
        bias_g  = b.bias_g;
        bias_a  = b.bias_a;
        gravity = b.gravity;
        cov     = b.cov;
    }

    StatesGroup& operator=(const StatesGroup& b) {
        rot_end = b.rot_end;
        pos_end = b.pos_end;
        vel_end = b.vel_end;
        bias_g  = b.bias_g;
        bias_a  = b.bias_a;
        gravity = b.gravity;
        cov     = b.cov;
        return *this;
    }

    /// state += delta: 将 DIM_STATE 维增量应用到状态
    /// 布局: [0-2]=rot, [3-5]=pos, [6-8]=vel, [9-11]=bg, [12-14]=ba, [15-17]=gravity
    StatesGroup operator+(const VD<DIM_STATE>& state_add) const {
        StatesGroup a;
        const V3D rotation_delta = state_add.block<3, 1>(0, 0);
        a.rot_end                = this->rot_end * lie::SO3d::exp(rotation_delta);
        a.pos_end                = this->pos_end + state_add.block<3, 1>(3, 0);
        a.vel_end                = this->vel_end + state_add.block<3, 1>(6, 0);
        a.bias_g                 = this->bias_g + state_add.block<3, 1>(9, 0);
        a.bias_a                 = this->bias_a + state_add.block<3, 1>(12, 0);
        a.gravity                = this->gravity + state_add.block<3, 1>(15, 0);
        a.cov                    = this->cov;
        return a;
    }

    StatesGroup& operator+=(const VD<DIM_STATE>& state_add) {
        const V3D rotation_delta = state_add.block<3, 1>(0, 0);
        this->rot_end            = this->rot_end * lie::SO3d::exp(rotation_delta);
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->vel_end += state_add.block<3, 1>(6, 0);
        this->bias_g += state_add.block<3, 1>(9, 0);
        this->bias_a += state_add.block<3, 1>(12, 0);
        this->gravity += state_add.block<3, 1>(15, 0);
        return *this;
    }

    /// state_a - state_b: 返回 DIM_STATE 维差分
    VD<DIM_STATE> operator-(const StatesGroup& b) const {
        VD<DIM_STATE> a;
        a.block<3, 1>(0, 0)  = lie::SO3d::log(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(3, 0)  = this->pos_end - b.pos_end;
        a.block<3, 1>(6, 0)  = this->vel_end - b.vel_end;
        a.block<3, 1>(9, 0)  = this->bias_g - b.bias_g;
        a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(15, 0) = this->gravity - b.gravity;
        return a;
    }

    void reset_pose() {
        rot_end = M3D::Identity();
        pos_end = V3D::Zero();
        vel_end = V3D::Zero();
    }
};

} // namespace radar::fast_livo2
