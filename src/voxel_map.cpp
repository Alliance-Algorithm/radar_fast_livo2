// voxel_map.cpp - 哈希体素八叉树 + 平面拟合 + ESIKF 地图管理
// 原始来源: hku-mars/FAST-LIVO2/src/voxel_map.cpp (全部算法逻辑，逐行移植)
// 移植: radar_fast_livo2 项目（ROS2 + 纯 Eigen）
// 移植范围（完整）:
//   - calcBodyCov()            传感器点协方差模型
//   - VoxelOctoTree::init_plane()     PCA + 特征值分析 → 平面性判断
//   - VoxelOctoTree::init_octo_tree()  平面拟合 + 递归分裂入口
//   - VoxelOctoTree::cut_octo_tree()    8 分裂 + 递归
//   - VoxelOctoTree::UpdateOctoTree()   增量更新
//   - VoxelOctoTree::find_correspond()  体素查找
//   - VoxelOctoTree::Insert()           插入（建图用）
//   - VoxelMapManager::StateEstimation()  ESIKF 主循环（核心）
//   - VoxelMapManager::BuildVoxelMap()    首次建图
//   - VoxelMapManager::UpdateVoxelMap()   增量地图更新
//   - VoxelMapManager::BuildResidualListOMP() OMP 并行残差构建
//   - VoxelMapManager::build_single_residual() 单点残差
//   - VoxelMapManager::mapSliding()      局部地图滑动
//   - VoxelMapManager::clearMemOutOfMap() 清理越界体素
// TODO: 跳过:
//   - pubVoxelMap / pubSinglePlane / mapJet / CalcVectQuation / GetUpdatePlane
//     (ROS1 visualization_msgs 可视化，ROS2 阶段暂不实现)

#include "radar_fast_livo2/voxel_map.hpp"
#include "radar_fast_livo2/esikf_state.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace radar::fast_livo2 {

namespace {
    // ── 辅助函数 ────────────────────────────────────────────────────────
    // deg_to_rad (等价于原版 DEG2RAD，避免与 PCL 宏冲突)
    inline constexpr double deg_to_rad(double deg) { return deg * M_PI / 180.0; }

    // 静态全局平面 ID（等价于原版 static int voxel_plane_id）
    static int g_voxel_plane_id = 0;
} // namespace

// ══════════════════════════════════════════════════════════════════
// calcBodyCov — 传感器点协方差模型
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:15-34 (逐行等价)
// ══════════════════════════════════════════════════════════════════

void calcBodyCov(const Eigen::Vector3d& pb, const float range_inc, const float degree_inc,
    Eigen::Matrix3d& cov) {
    double range = pb.norm();
    if (range < 1e-6) range = 0.0001;

    double range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << std::pow(std::sin(deg_to_rad(degree_inc)), 2), 0, 0,
        std::pow(std::sin(deg_to_rad(degree_inc)), 2);

    Eigen::Vector3d direction(pb);
    direction.normalize();

    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1),
        direction(0), 0;

    // 注意: 原版使用 pb[2]（引用，允许修改为 0.0001），
    // 这里 range 已经做了 clamp，保持与原版一致
    Eigen::Vector3d base_vector1(
        1, 1, -(direction(0) + direction(1)) / (direction(2) == 0 ? 0.0001 : direction(2)));
    base_vector1.normalize();

    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();

    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2),
        base_vector2(2);

    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

// ══════════════════════════════════════════════════════════════════
// VoxelOctoTree::init_plane — PCA 平面拟合
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:55-135 (逐行等价)
//
// 流程:
//   1. 计算点集均值 center_ 和协方差 covariance_
//   2. 特征值分解: 最小特征值 < planer_threshold_ → 平面
//   3. 计算平面不确定性 plane_var_ (6×6: 中心+法向量)
// ══════════════════════════════════════════════════════════════════

void VoxelOctoTree::init_plane(const std::vector<pointWithVar>& points, VoxelPlane* plane) {
    plane->plane_var_   = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance_  = Eigen::Matrix3d::Zero();
    plane->center_      = Eigen::Vector3d::Zero();
    plane->normal_      = Eigen::Vector3d::Zero();
    plane->points_size_ = static_cast<int>(points.size());
    plane->radius_      = 0.0f;

    for (const auto& pv : points) {
        plane->covariance_ += pv.point_w * pv.point_w.transpose();
        plane->center_ += pv.point_w;
    }
    plane->center_ = plane->center_ / plane->points_size_;
    plane->covariance_ =
        plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();

    // 特征值分解
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
    Eigen::Matrix3cd evecs    = es.eigenvectors();
    Eigen::Vector3cd evals    = es.eigenvalues();
    Eigen::Vector3d evalsReal = evals.real();

    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - static_cast<int>(evalsMin) - static_cast<int>(evalsMax);

    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

    // ── 平面性判断 ──
    if (evalsReal(evalsMin) < planer_threshold_) {
        Eigen::Matrix3d J_Q;
        J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0,
            1.0 / plane->points_size_;

        for (size_t i = 0; i < points.size(); i++) {
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++) {
                if (m != static_cast<int>(evalsMin)) {
                    Eigen::Matrix<double, 1, 3> F_m =
                        (points[i].point_w - plane->center_).transpose()
                        / (plane->points_size_ * (evalsReal[evalsMin] - evalsReal[m]))
                        * (evecs.real().col(m) * evecs.real().col(evalsMin).transpose()
                            + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
                    F.row(m) = F_m;
                } else {
                    F.row(m) << 0, 0, 0;
                }
            }
            J.block<3, 3>(0, 0) = evecs.real() * F;
            J.block<3, 3>(3, 0) = J_Q;
            plane->plane_var_ += J * points[i].var * J.transpose();
        }

        plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
            evecs.real()(2, evalsMin);
        plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
            evecs.real()(2, evalsMid);
        plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
            evecs.real()(2, evalsMax);
        plane->min_eigen_value_ = static_cast<float>(evalsReal(evalsMin));
        plane->mid_eigen_value_ = static_cast<float>(evalsReal(evalsMid));
        plane->max_eigen_value_ = static_cast<float>(evalsReal(evalsMax));
        plane->radius_          = static_cast<float>(std::sqrt(evalsReal(evalsMax)));
        plane->d_               = static_cast<float>(-(plane->normal_(0) * plane->center_(0)
            + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2)));
        plane->is_plane_        = true;
        plane->is_update_       = true;

        if (!plane->is_init_) {
            plane->id_ = g_voxel_plane_id;
            g_voxel_plane_id++;
            plane->is_init_ = true;
        }
    } else {
        plane->is_update_ = true;
        plane->is_plane_  = false;
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelOctoTree::init_octo_tree — 初始化八叉树
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:137-161
// ══════════════════════════════════════════════════════════════════

void VoxelOctoTree::init_octo_tree() {
    if (temp_points_.size() > static_cast<size_t>(points_size_threshold_)) {
        init_plane(temp_points_, plane_ptr_.get());
        if (plane_ptr_->is_plane_) {
            octo_state_ = 0;
            // 点数过多则停止更新，释放内存
            if (temp_points_.size() > static_cast<size_t>(max_points_num_)) {
                update_enable_ = false;
                std::vector<pointWithVar>().swap(temp_points_);
                new_points_ = 0;
            }
        } else {
            octo_state_ = 1;
            cut_octo_tree();
        }
        init_octo_  = true;
        new_points_ = 0;
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelOctoTree::cut_octo_tree — 递归 8 分裂
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:163-217
// ══════════════════════════════════════════════════════════════════

void VoxelOctoTree::cut_octo_tree() {
    if (layer_ >= max_layer_) {
        octo_state_ = 0;
        return;
    }
    for (size_t i = 0; i < temp_points_.size(); i++) {
        int xyz[3] = { 0, 0, 0 };
        if (temp_points_[i].point_w[0] > voxel_center_[0]) xyz[0] = 1;
        if (temp_points_[i].point_w[1] > voxel_center_[1]) xyz[1] = 1;
        if (temp_points_[i].point_w[2] > voxel_center_[2]) xyz[2] = 1;
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

        if (!leaves_[leafnum]) {
            leaves_[leafnum] = std::make_unique<VoxelOctoTree>(max_layer_, layer_ + 1,
                layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
            leaves_[leafnum]->layer_init_num_ = layer_init_num_;
            leaves_[leafnum]->voxel_center_[0] =
                voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[1] =
                voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[2] =
                voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
            leaves_[leafnum]->quater_length_ = quater_length_ / 2;
        }
        leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
        leaves_[leafnum]->new_points_++;
    }
    for (int i = 0; i < 8; i++) {
        if (leaves_[i]) {
            if (static_cast<int>(leaves_[i]->temp_points_.size())
                > leaves_[i]->points_size_threshold_) {
                init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_.get());
                if (leaves_[i]->plane_ptr_->is_plane_) {
                    leaves_[i]->octo_state_ = 0;
                    if (static_cast<int>(leaves_[i]->temp_points_.size())
                        > leaves_[i]->max_points_num_) {
                        leaves_[i]->update_enable_ = false;
                        std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
                        leaves_[i]->new_points_ = 0;
                    }
                } else {
                    leaves_[i]->octo_state_ = 1;
                    leaves_[i]->cut_octo_tree();
                }
                leaves_[i]->init_octo_  = true;
                leaves_[i]->new_points_ = 0;
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelOctoTree::UpdateOctoTree — 增量更新
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:219-290
//
// 行为:
//   - 未初始化 → 累积点直到触发 init_octo_tree()
//   - 平面节点 + update_enable → 累积新点，定期重新拟合平面
//   - 非平面节点 + 未达最大层 → 递归到子节点
//   - 非平面节点 + 已达最大层 → 直接累积点并定期拟合
// ══════════════════════════════════════════════════════════════════

void VoxelOctoTree::UpdateOctoTree(const pointWithVar& pv) {
    if (!init_octo_) {
        new_points_++;
        temp_points_.push_back(pv);
        if (static_cast<int>(temp_points_.size()) > points_size_threshold_) {
            init_octo_tree();
        }
    } else {
        if (plane_ptr_->is_plane_) {
            if (update_enable_) {
                new_points_++;
                temp_points_.push_back(pv);
                if (new_points_ > update_size_threshold_) {
                    init_plane(temp_points_, plane_ptr_.get());
                    new_points_ = 0;
                }
                if (static_cast<int>(temp_points_.size()) >= max_points_num_) {
                    update_enable_ = false;
                    std::vector<pointWithVar>().swap(temp_points_);
                    new_points_ = 0;
                }
            }
        } else {
            if (layer_ < max_layer_) {
                int xyz[3] = { 0, 0, 0 };
                if (pv.point_w[0] > voxel_center_[0]) xyz[0] = 1;
                if (pv.point_w[1] > voxel_center_[1]) xyz[1] = 1;
                if (pv.point_w[2] > voxel_center_[2]) xyz[2] = 1;
                int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
                if (leaves_[leafnum]) {
                    leaves_[leafnum]->UpdateOctoTree(pv);
                } else {
                    leaves_[leafnum] = std::make_unique<VoxelOctoTree>(max_layer_, layer_ + 1,
                        layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
                    leaves_[leafnum]->layer_init_num_ = layer_init_num_;
                    leaves_[leafnum]->voxel_center_[0] =
                        voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
                    leaves_[leafnum]->voxel_center_[1] =
                        voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
                    leaves_[leafnum]->voxel_center_[2] =
                        voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
                    leaves_[leafnum]->quater_length_ = quater_length_ / 2;
                    leaves_[leafnum]->UpdateOctoTree(pv);
                }
            } else {
                if (update_enable_) {
                    new_points_++;
                    temp_points_.push_back(pv);
                    if (new_points_ > update_size_threshold_) {
                        init_plane(temp_points_, plane_ptr_.get());
                        new_points_ = 0;
                    }
                    if (static_cast<int>(temp_points_.size()) > max_points_num_) {
                        update_enable_ = false;
                        std::vector<pointWithVar>().swap(temp_points_);
                        new_points_ = 0;
                    }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelOctoTree::find_correspond — 递归查找包含点 pw 的叶节点
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:292-305
// ══════════════════════════════════════════════════════════════════

VoxelOctoTree* VoxelOctoTree::find_correspond(Eigen::Vector3d pw) {
    if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) {
        return this;
    }
    int xyz[3]  = { 0, 0, 0 };
    xyz[0]      = pw[0] > voxel_center_[0] ? 1 : 0;
    xyz[1]      = pw[1] > voxel_center_[1] ? 1 : 0;
    xyz[2]      = pw[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    return leaves_[leafnum] ? leaves_[leafnum]->find_correspond(pw) : this;
}

// ══════════════════════════════════════════════════════════════════
// VoxelOctoTree::Insert — 插入新点（首次建图用）
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:307-336
// ══════════════════════════════════════════════════════════════════

VoxelOctoTree* VoxelOctoTree::Insert(const pointWithVar& pv) {
    if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_)
        || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_))) {
        new_points_++;
        temp_points_.push_back(pv);
        return this;
    }

    int xyz[3]  = { 0, 0, 0 };
    xyz[0]      = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1]      = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2]      = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

    if (leaves_[leafnum]) {
        return leaves_[leafnum]->Insert(pv);
    } else {
        leaves_[leafnum]                   = std::make_unique<VoxelOctoTree>(max_layer_, layer_ + 1,
            layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
        leaves_[leafnum]->layer_init_num_  = layer_init_num_;
        leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_   = quater_length_ / 2;
        return leaves_[leafnum]->Insert(pv);
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::StateEstimation — ESIKF 主循环（核心算法）
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:338-511 (逐行等价)
//
// 流程:
//   1. 预计算每个降采样点的 body_cov 和 cross_mat
//   2. ESIKF 迭代 (max_iterations_ 次):
//      a. 用当前 state_ 将点云变换到世界系
//      b. 更新每个点的 world-frame 协方差
//      c. BuildResidualListOMP 并行构建点到平面残差
//      d. 构建雅可比 H(6×N) 和信息矩阵
//      e. 卡尔曼增益 K = (H^T R^-1 H + P^-1)^-1 H^T R^-1
//      f. 状态更新: state += K * z + (I-KH)*(state_propagat - state_)
//      g. 收敛判断: Δrot < 0.01°, Δpos < 0.15mm
//      h. rematch 机制: 收敛或倒数第2次迭代 → 多一次残差重匹配
//      i. 协方差更新: P = (I - KH) * P
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::StateEstimation(StatesGroup& state_propagat) {
    cross_mat_list_.clear();
    cross_mat_list_.reserve(feats_down_size_);
    body_cov_list_.clear();
    body_cov_list_.reserve(feats_down_size_);

    // ── 预计算每个点的 body_cov 和 cross_mat ──
    for (size_t i = 0; i < feats_down_body_->size(); i++) {
        V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y,
            feats_down_body_->points[i].z);
        if (std::abs(point_this[2]) < 1e-6) point_this[2] = 0.001;

        M3D var;
        calcBodyCov(point_this, static_cast<float>(config_setting_.dept_err_),
            static_cast<float>(config_setting_.beam_err_), var);
        body_cov_list_.push_back(var);

        point_this         = extR_ * point_this + extT_;
        M3D point_crossmat = skewSym(point_this);
        cross_mat_list_.push_back(point_crossmat);
    }

    pv_list_.clear();
    pv_list_.resize(feats_down_size_);

    int rematch_num = 0;
    MD<DIM_STATE, DIM_STATE> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    bool EKF_stop_flg = false;

    for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++) {
        double total_residual = 0.0;

        // a. 变换点云到世界系
        PointCloudT::Ptr world_lidar(new PointCloudT());
        TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);

        M3D rot_var = state_.cov.block<3, 3>(0, 0);
        M3D t_var   = state_.cov.block<3, 3>(3, 3);

        for (size_t i = 0; i < feats_down_body_->size(); i++) {
            pointWithVar& pv = pv_list_[i];
            pv.point_b       = V3D(feats_down_body_->points[i].x, feats_down_body_->points[i].y,
                feats_down_body_->points[i].z);
            pv.point_w =
                V3D(world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z);

            M3D cov            = body_cov_list_[i];
            M3D point_crossmat = cross_mat_list_[i];
            // world-frame 协方差: sensor + rotation uncertainty + translation uncertainty
            cov = state_.rot_end * cov * state_.rot_end.transpose()
                + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
            pv.var      = cov;
            pv.body_var = body_cov_list_[i];
        }

        ptpl_list_.clear();

        // c. 并行构建残差列表
        BuildResidualListOMP(pv_list_, ptpl_list_);

        for (size_t i = 0; i < ptpl_list_.size(); i++) {
            total_residual += std::fabs(ptpl_list_[i].dis_to_plane_);
        }
        effct_feat_num_ = static_cast<int>(ptpl_list_.size());

        std::cout << "[ LIO ] Raw feature num: " << feats_undistort_->size()
                  << ", downsampled feature num: " << feats_down_size_
                  << ", effective feature num: " << effct_feat_num_ << ", average residual: "
                  << (effct_feat_num_ > 0 ? total_residual / effct_feat_num_ : 0.0) << std::endl;

        // FIXME: 有效特征数过低（<min_effct_feat_for_correction_）时跳过本轮
        // ESIKF 修正，state_ 保持为 IMU 传播先验（state_propagat）不变，但
        // 循环上面的 BuildResidualListOMP 已经跑过，UpdateVoxelMap 仍会在
        // 调用方无条件执行（地图继续吸收看到的点，不受此处影响）。原因：
        // 极少数点（个位数到几百个）大多来自单一小平面，只能约束1-2个自由
        // 度，却被当全局6维观测去修正，容易在欠约束方向注入错误运动——
        // 实测复现：dToF 面阵传感器转弯/遇到低反射率场景时 raw 点数瞬间
        // 骤降(如47k→16k持续2-3帧)，几十到几百个点的"病态修正"把 state
        // 带偏，后续即使 raw 点恢复也无法重新匹配（voxel_map 相关注释里
        // 记录的 frame858+ 发散链路）。阈值给 500：低于此值默认信息量
        // 不足以支撑 6 维状态观测，宁可信任 IMU 短时传播的平滑性。
        constexpr int kMinEffectFeatForCorrection = 500;
        if (effct_feat_num_ < kMinEffectFeatForCorrection) {
            EKF_stop_flg = true;
            break;
        }

        // d. 构建雅可比 H 和信息矩阵
        Eigen::MatrixXd Hsub(effct_feat_num_, 6);
        Eigen::MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
        Eigen::VectorXd R_inv(effct_feat_num_);
        Eigen::VectorXd meas_vec(effct_feat_num_);
        meas_vec.setZero();

        for (int i = 0; i < effct_feat_num_; i++) {
            auto& ptpl = ptpl_list_[i];
            V3D point_this(ptpl.point_b_);
            point_this = extR_ * point_this + extT_;

            M3D point_crossmat = skewSym(point_this);

            V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;

            // 残差雅可比: J_nq = [p_w - center, -normal]
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = point_world - ptpl.center_;
            J_nq.block<1, 3>(0, 3) = -ptpl.normal_;

            // point_body cov (原版使用 body frame 协方差)
            M3D var = state_propagat.rot_end * extR_ * ptpl.body_cov_
                * (state_propagat.rot_end * extR_).transpose();

            double sigma_l = J_nq * ptpl.plane_var_ * J_nq.transpose();
            R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl.normal_.transpose() * var * ptpl.normal_);

            // 测量雅可比（仅对 rot+pos 可观测）
            V3D A(point_crossmat * state_.rot_end.transpose() * ptpl.normal_);
            Hsub.row(i) << A[0], A[1], A[2], ptpl.normal_[0], ptpl.normal_[1], ptpl.normal_[2];
            Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i),
                ptpl.normal_[0] * R_inv(i), ptpl.normal_[1] * R_inv(i), ptpl.normal_[2] * R_inv(i);
            meas_vec(i) = -ptpl.dis_to_plane_;
        }

        // ── ESIKF 更新 ──
        auto HTz                = Hsub_T_R_inv * meas_vec; // 6×1
        H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;

        // K_1 = (H^T R^-1 H + P^-1)^-1
        MD<DIM_STATE, DIM_STATE> K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0)
            + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse())
                                           .inverse();

        // G = K_1 * H^T R^-1 H  (Kalman gain matrix for cov update)
        G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);

        // vec = state_propagat - state_ (DIM_STATE 维差分)
        VD<DIM_STATE> vec = state_propagat - state_;

        // solution = K * z + (I - G) * vec (MAP 估计)
        VD<DIM_STATE> solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0)
            - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

        state_ += solution;

        // 收敛判断
        auto rot_add           = solution.block<3, 1>(0, 0);
        auto t_add             = solution.block<3, 1>(3, 0);
        bool flg_EKF_converged = (rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015);

        V3D euler_cur = RotMtoEuler(state_.rot_end);

        // ── Rematch 判断 ──
        if (flg_EKF_converged
            || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) {
            rematch_num++;
        }

        // ── 收敛 → 协方差更新 + 停止 ──
        if (!EKF_stop_flg
            && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1))) {
            // P = (I - G) * P
            state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
                (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0))
                * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);

            // FIXME: 数值保护：有效特征数很大时 G 可能接近单位阵，(I-G) 接近零矩阵，
            // 连续多帧乘积 + 浮点误差会让 cov 出现负/近零特征值甚至丢失
            // 对称性，导致后续 .inverse() 求 K_1 时数值爆炸（对应实测 frame
            // 234+ 静止发散链路里 ESIKF 数值失稳的最后一环）。这里强制对称化
            // 并给对角线加最小方差地板，防止协方差矩阵退化为不可逆或病态。
            state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
                0.5 * (state_.cov.block<DIM_STATE, DIM_STATE>(0, 0)
                     + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).transpose());
            constexpr double kMinCovDiag = 1e-12;
            for (int d = 0; d < DIM_STATE; ++d) {
                if (state_.cov(d, d) < kMinCovDiag) state_.cov(d, d) = kMinCovDiag;
            }

            position_last_ = state_.pos_end;
            EKF_stop_flg   = true;
        }
        if (EKF_stop_flg) break;
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::TransformLidar — 坐标变换
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:513-530
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::TransformLidar(const M3D& rot, const V3D& t,
    const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& trans_cloud) {
    trans_cloud->clear();
    trans_cloud->reserve(input_cloud->size());
    for (size_t i = 0; i < input_cloud->size(); i++) {
        const auto& p_c = input_cloud->points[i];
        V3D p(p_c.x, p_c.y, p_c.z);
        // LiDAR body → IMU body → World
        p = rot * (extR_ * p + extT_) + t;
        pcl::PointXYZINormal pi;
        pi.x         = static_cast<float>(p(0));
        pi.y         = static_cast<float>(p(1));
        pi.z         = static_cast<float>(p(2));
        pi.intensity = p_c.intensity;
        pi.curvature = p_c.curvature;
        trans_cloud->points.push_back(pi);
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::BuildVoxelMap — 首次建图
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:532-591
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::BuildVoxelMap() {
    float voxel_size                = static_cast<float>(config_setting_.max_voxel_size_);
    float planer_threshold          = static_cast<float>(config_setting_.planner_threshold_);
    int max_layer                   = config_setting_.max_layer_;
    int max_points_num              = config_setting_.max_points_num_;
    std::vector<int> layer_init_num = config_setting_.layer_init_num_;

    std::vector<pointWithVar> input_points;

    // 构建带方差的点列表
    for (size_t i = 0; i < feats_down_world_->size(); i++) {
        pointWithVar pv;
        pv.point_w = V3D(feats_down_world_->points[i].x, feats_down_world_->points[i].y,
            feats_down_world_->points[i].z);

        V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y,
            feats_down_body_->points[i].z);
        M3D var;
        calcBodyCov(point_this, static_cast<float>(config_setting_.dept_err_),
            static_cast<float>(config_setting_.beam_err_), var);

        M3D point_crossmat = skewSym(point_this);

        // world-frame 协方差: sensor noise + rotation uncertainty + translation uncertainty
        var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose()
            + (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose()
            + state_.cov.block<3, 3>(3, 3);
        pv.var = var;
        input_points.push_back(pv);
    }

    // ── 按体素哈希分组 ──
    for (size_t i = 0; i < input_points.size(); i++) {
        const pointWithVar& p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = static_cast<float>(p_v.point_w[j] / voxel_size);
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0f;
        }
        VOXEL_LOCATION position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
            static_cast<int64_t>(loc_xyz[2]));
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            iter->second->temp_points_.push_back(p_v);
            iter->second->new_points_++;
        } else {
            auto octo_tree = std::make_unique<VoxelOctoTree>(
                max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
            voxel_map_[position]                   = std::move(octo_tree);
            voxel_map_[position]->quater_length_   = voxel_size / 4.0f;
            voxel_map_[position]->voxel_center_[0] = (0.5f + position.x) * voxel_size;
            voxel_map_[position]->voxel_center_[1] = (0.5f + position.y) * voxel_size;
            voxel_map_[position]->voxel_center_[2] = (0.5f + position.z) * voxel_size;
            voxel_map_[position]->temp_points_.push_back(p_v);
            voxel_map_[position]->new_points_++;
            voxel_map_[position]->layer_init_num_ = layer_init_num;
        }
    }

    // ── 初始化所有体素的八叉树 ──
    for (auto& pair : voxel_map_) {
        pair.second->init_octo_tree();
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::RGBFromVoxel — 体素着色（非核心，保留兼容性）
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:593-607
// ══════════════════════════════════════════════════════════════════

Eigen::Vector3f VoxelMapManager::RGBFromVoxel(const V3D& input_point) {
    int64_t loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] =
            static_cast<int64_t>(std::floor(input_point[j] / config_setting_.max_voxel_size_));
    }
    int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
    int k       = static_cast<int>((ind + 100000) % 3);
    return Eigen::Vector3f((k == 0) * 255.0f, (k == 1) * 255.0f, (k == 2) * 255.0f);
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::UpdateVoxelMap — 增量地图更新
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:609-641
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar>& input_points) {
    float voxel_size                = static_cast<float>(config_setting_.max_voxel_size_);
    float planer_threshold          = static_cast<float>(config_setting_.planner_threshold_);
    int max_layer                   = config_setting_.max_layer_;
    int max_points_num              = config_setting_.max_points_num_;
    std::vector<int> layer_init_num = config_setting_.layer_init_num_;

    for (size_t i = 0; i < input_points.size(); i++) {
        const pointWithVar& p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = static_cast<float>(p_v.point_w[j] / voxel_size);
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0f;
        }
        VOXEL_LOCATION position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
            static_cast<int64_t>(loc_xyz[2]));
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            iter->second->UpdateOctoTree(p_v);
        } else {
            auto octo_tree = std::make_unique<VoxelOctoTree>(
                max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
            octo_tree->layer_init_num_  = layer_init_num;
            octo_tree->quater_length_   = voxel_size / 4.0f;
            octo_tree->voxel_center_[0] = (0.5f + position.x) * voxel_size;
            octo_tree->voxel_center_[1] = (0.5f + position.y) * voxel_size;
            octo_tree->voxel_center_[2] = (0.5f + position.z) * voxel_size;
            octo_tree->UpdateOctoTree(p_v);
            voxel_map_[position] = std::move(octo_tree);
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::BuildResidualListOMP — 并行残差构建
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:643-711
//
// 对每个点:
//   1. 查哈希表找到对应体素
//   2. 调用 build_single_residual 递归搜索最佳匹配平面
//   3. 若体素内未匹配，尝试相邻体素
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::BuildResidualListOMP(
    std::vector<pointWithVar>& pv_list, std::vector<PointToPlane>& ptpl_list) {
    double voxel_size = config_setting_.max_voxel_size_;
    std::mutex mylock;
    ptpl_list.clear();

    std::vector<PointToPlane> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size(), false);
    std::vector<size_t> index(pv_list.size());
    for (size_t i = 0; i < index.size(); ++i)
        index[i] = i;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(index.size()); i++) {
        pointWithVar& pv = pv_list[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = static_cast<float>(pv.point_w[j] / voxel_size);
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0f;
        }
        VOXEL_LOCATION position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
            static_cast<int64_t>(loc_xyz[2]));

        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            VoxelOctoTree* current_octo = iter->second.get();
            PointToPlane single_ptpl;
            bool is_sucess = false;
            double prob    = 0.0;
            build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);

            // 若当前体素未匹配成功，尝试相邻体素
            if (!is_sucess) {
                VOXEL_LOCATION near_position = position;
                if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
                    near_position.x = near_position.x + 1;
                } else if (loc_xyz[0]
                    < (current_octo->voxel_center_[0] - current_octo->quater_length_)) {
                    near_position.x = near_position.x - 1;
                }
                if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
                    near_position.y = near_position.y + 1;
                } else if (loc_xyz[1]
                    < (current_octo->voxel_center_[1] - current_octo->quater_length_)) {
                    near_position.y = near_position.y - 1;
                }
                if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
                    near_position.z = near_position.z + 1;
                } else if (loc_xyz[2]
                    < (current_octo->voxel_center_[2] - current_octo->quater_length_)) {
                    near_position.z = near_position.z - 1;
                }
                auto iter_near = voxel_map_.find(near_position);
                if (iter_near != voxel_map_.end()) {
                    build_single_residual(
                        pv, iter_near->second.get(), 0, is_sucess, prob, single_ptpl);
                }
            }

            if (is_sucess) {
                std::lock_guard<std::mutex> lock(mylock);
                useful_ptpl[i]   = true;
                all_ptpl_list[i] = single_ptpl;
            }
        }
    }

    for (size_t i = 0; i < useful_ptpl.size(); i++) {
        if (useful_ptpl[i]) ptpl_list.push_back(all_ptpl_list[i]);
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::build_single_residual — 单点残差构建
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:713-786
//
// 递归搜索八叉树找到与点 pw 最匹配的平面:
//   - 平面节点 → 检查点是否在平面半径内 + 残差 < sigma * σ_l
//   - 非平面节点 → 递归搜索 8 个子节点
//   - 选择概率最高（1/σ * exp(-0.5 * r²/σ²)）的匹配
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::build_single_residual(pointWithVar& pv, const VoxelOctoTree* current_octo,
    int current_layer, bool& is_sucess, double& prob, PointToPlane& single_ptpl) {
    const double sigma_num = config_setting_.sigma_num_;
    const double radius_k  = 3.0;
    Eigen::Vector3d p_w    = pv.point_w;

    if (current_octo->plane_ptr_->is_plane_) {
        const VoxelPlane& plane = *current_octo->plane_ptr_;

        float dis_to_plane = std::fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1)
            + plane.normal_(2) * p_w(2) + plane.d_);
        float dis_to_center =
            static_cast<float>((plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0))
                + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1))
                + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2)));
        float range_dis = std::sqrt(dis_to_center - dis_to_plane * dis_to_plane);

        if (range_dis <= radius_k * plane.radius_) {
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
            J_nq.block<1, 3>(0, 3) = -plane.normal_;
            double sigma_l         = J_nq * plane.plane_var_ * J_nq.transpose();
            sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;

            if (dis_to_plane < sigma_num * std::sqrt(sigma_l)) {
                is_sucess        = true;
                double this_prob = 1.0 / std::sqrt(sigma_l)
                    * std::exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
                if (this_prob > prob) {
                    prob                      = this_prob;
                    pv.normal                 = plane.normal_;
                    single_ptpl.body_cov_     = pv.body_var;
                    single_ptpl.point_b_      = pv.point_b;
                    single_ptpl.point_w_      = pv.point_w;
                    single_ptpl.plane_var_    = plane.plane_var_;
                    single_ptpl.normal_       = plane.normal_;
                    single_ptpl.center_       = plane.center_;
                    single_ptpl.d_            = plane.d_;
                    single_ptpl.layer_        = current_layer;
                    single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0)
                        + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
                }
                return;
            }
            return; // 超出 sigma 倍数，不匹配
        }
        return; // 超出平面半径，不匹配
    } else {
        // 非平面节点：递归搜索子节点
        if (current_layer < config_setting_.max_layer_) {
            for (int leafnum = 0; leafnum < 8; leafnum++) {
                if (current_octo->leaves_[leafnum]) {
                    build_single_residual(pv, current_octo->leaves_[leafnum].get(),
                        current_layer + 1, is_sucess, prob, single_ptpl);
                }
            }
            return;
        }
        return; // 达到最大层数，无法匹配
    }
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::mapSliding — 局部地图滑动
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:924-948
//
// 当机器人移动超过 sliding_thresh 时，清理距离当前位置
// 超过 half_map_size 个体素的旧地图数据。
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::mapSliding() {
    if ((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh) {
        std::cout << "\033[31m[DEBUG]: Last sliding length "
                  << (position_last_ - last_slide_position).norm() << "\033[0m\n";
        return;
    }

    last_slide_position = position_last_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = static_cast<float>(position_last_[j] / config_setting_.max_voxel_size_);
        if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0f;
    }
    clearMemOutOfMap(static_cast<int>(loc_xyz[0]) + config_setting_.half_map_size,
        static_cast<int>(loc_xyz[0]) - config_setting_.half_map_size,
        static_cast<int>(loc_xyz[1]) + config_setting_.half_map_size,
        static_cast<int>(loc_xyz[1]) - config_setting_.half_map_size,
        static_cast<int>(loc_xyz[2]) + config_setting_.half_map_size,
        static_cast<int>(loc_xyz[2]) - config_setting_.half_map_size);
}

// ══════════════════════════════════════════════════════════════════
// VoxelMapManager::clearMemOutOfMap — 清理越界体素
// 原始来源: FAST-LIVO2/src/voxel_map.cpp:950-971
// ══════════════════════════════════════════════════════════════════

void VoxelMapManager::clearMemOutOfMap(const int& x_max, const int& x_min, const int& y_max,
    const int& y_min, const int& z_max, const int& z_min) {
    int delete_voxel_cout = 0;
    for (auto it = voxel_map_.begin(); it != voxel_map_.end();) {
        const VOXEL_LOCATION& loc = it->first;
        bool should_remove        = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min
            || loc.z > z_max || loc.z < z_min;
        if (should_remove) {
            it = voxel_map_.erase(it);
            delete_voxel_cout++;
        } else {
            ++it;
        }
    }
    std::cout << "\033[31m[DEBUG]: Delete " << delete_voxel_cout << " root voxels\033[0m\n";
}

} // namespace radar::fast_livo2
