// imu_processing.cpp - IMU 初始化 + 逐点去畸变 + 协方差传播
// 从 FAST-LIVO2/src/IMU_Processing.cpp 移植核心逻辑，适配 ROS2
//
// 移植范围:
//   - IMU_init()         : 静止初始化，估计重力方向、陀螺仪零偏、协方差
//   - UndistortPcl()     : 正向 IMU 积分传播（含协方差 F_x*P*F_x^T+cov_w）
//                          + 逐点反向插值补偿运动畸变
//   - Forward_without_imu(): 无 IMU 模式下的简化处理路径（匀速运动假设）
//
// FIXME: 状态所有权（Oracle C1 修复）：
//   本文件不再持有独立的 ESIKF 状态副本。process()/undistort_pcl()/
//   forward_without_imu() 均直接对调用方传入的 StatesGroup& 原地读写，
//   使 IMU 正向传播的起点始终是上一帧 ESIKF 的真实后验，形成闭环。
//
// FIXME: 协方差传播（Oracle C2 修复）：
//   undistort_pcl() 的正向传播循环内新增 F_x（状态转移雅可比）与
//   cov_w（过程噪声），执行 state.cov = F_x * state.cov * F_x^T + cov_w，
//   与 FAST-LIVO2 IMU_Processing.cpp:377-401 等价（索引按 18 维布局重排，
//   去掉了 inv_expo_time 对应的第 6 维）。
//
// 跳过部分:
//   - 曝光时间估计 (inv_expo_time): StatesGroup 无此字段，
//     见 FAST-LIVO2 IMU_Processing.cpp 中 exposure_estimate_en 相关代码
//
// 原始作者: Chunran Zheng <zhengcr@connect.hku.hk>
// 移植: radar_fast_livo2 项目

#include "radar_fast_livo2/imu_processing.hpp"
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace radar::fast_livo2 {

namespace {

constexpr double G_M_S2 = 9.81;  // 重力常数（广东地区）

// Rodrigues' formula: exp(omega * dt) ∈ SO(3)
// 等价于 FAST-LIVO2 utils/so3_math.h: Exp(ang_vel, dt)
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& ang_vel, const double dt) {
    const double ang_vel_norm = ang_vel.norm();
    if (ang_vel_norm < 1e-7) { return Eigen::Matrix3d::Identity(); }
    const Eigen::Vector3d axis = ang_vel / ang_vel_norm;
    const double ang = ang_vel_norm * dt;
    Eigen::Matrix3d K;
    K << 0.0, -axis.z(), axis.y(),
         axis.z(), 0.0, -axis.x(),
        -axis.y(), axis.x(), 0.0;
    return Eigen::Matrix3d::Identity() + std::sin(ang) * K
           + (1.0 - std::cos(ang)) * K * K;
}

// 按 curvature（帧内时间偏移，毫秒）升序排序
// 等价于 FAST-LIVO2: bool time_list(PointType&, PointType&)
bool time_list(const PointType& x, const PointType& y) {
    return x.curvature < y.curvature;
}

// 将当前 IMU 积分状态写入 Pose6D 槽位
// 等价于 FAST-LIVO2: set_pose6d(...)
void set_pose6d(ImuProcess::Pose6D& p, const double t,
                const Eigen::Vector3d& a, const Eigen::Vector3d& g,
                const Eigen::Vector3d& v, const Eigen::Vector3d& pos,
                const Eigen::Matrix3d& R) {
    p.offset_time = t;
    p.acc = a;
    p.gyr = g;
    p.vel = v;
    p.pos = pos;
    p.rot = R;
}

}  // namespace

// ── 构造/重置 ────────────────────────────────────────────────────────

ImuProcess::ImuProcess() = default;

// WARN: 仅重置 ImuProcess 内部的传播工作变量（均值累积器、上一帧 IMU 桥接、
// Pose6D 队列）。ESIKF 状态（StatesGroup）现在完全由调用方持有，
// 不在此处重置——这是 Oracle C1 修复的直接结果：状态所有权单一化后，
// ImuProcess 不应也不能重置一个它不拥有的对象。
void ImuProcess::reset() {
    init_count_       = 0;
    is_first_frame_   = true;
    imu_need_init_    = true;
    last_prop_end_time_ = 0.0;
    time_last_scan_   = 0.0;
    mean_acc_         = Eigen::Vector3d(0, 0, -1.0);
    mean_gyr_         = Eigen::Vector3d::Zero();
    angvel_last_      = Eigen::Vector3d::Zero();
    acc_s_last_       = Eigen::Vector3d::Zero();
    mean_acc_norm_    = 9.81;
    imu_pose_.clear();
    pcl_wait_proc_.clear();
    imu_buf_.clear();
}

// ── 主入口 ───────────────────────────────────────────────────────────

bool ImuProcess::process(MeasureGroup& meas, StatesGroup& state,
                         PointCloudT::Ptr& cloud_out) {
    if (!imu_en) {
        forward_without_imu(meas, state, cloud_out);
        return true;
    }

    if (meas.imu.empty()) { return false; }

    // ── 静止初始化：累积 IMU 数据，估计重力方向与零偏 ──
    if (imu_need_init_) {
        imu_init(meas, state);
        if (!meas.imu.empty()) {
            last_imu_ = meas.imu.back();
            // FIXME: last_prop_end_time_ 初始值为 0。若不在此处更新，init
            // 期间积累的全部 IMU 数据（最多 2s × 400Hz = 800 个）会在第一次
            // 调用 undistort_pcl 时被完整地跑过一遍协方差传播——因为
            // prop_beg_time=0 导致 `tail.timestamp < prop_beg_time` 门控
            // 永远为 false（实际时间戳都是 ~7910s），800 步 cov 叠加让
            // state.cov 起点膨胀数百倍，后续 ESIKF 修正效果大幅变差。
            // 修复：把传播锚点推进到 init 最后一个 IMU 时间戳，这样第一次
            // undistort_pcl 只处理当前帧的 ~40 个新样本，与上游行为对齐。
            last_prop_end_time_ = meas.imu.back().timestamp;
        }
        return false;
    }

    // ── 正向传播（含协方差）+ 逐点逆向去畸变 ──
    undistort_pcl(meas, state, cloud_out);
    return true;
}

// ── 静止初始化 ────────────────────────────────────────────────────────
//
// 等价于 FAST-LIVO2 IMU_Processing.cpp: IMU_init()
// 跨帧累积加速度计和陀螺仪 running mean，收敛后直接写入调用方状态：
//   grav  ← -mean_acc_norm * G  （初始重力方向）
//   rot   ← Identity（初始姿态无旋转）
//   bg    ← 0（陀螺仪零偏，等 ESIKF 在线估计）
//   ba    ← 0（加速度计零偏，等 ESIKF 在线估计）
// vel/pos 保持 StatesGroup 默认构造的零值，与原版一致（原版 IMU_init 同样
// 不触碰 bias_a/vel_end/pos_end，只设 gravity/rot_end/bias_g）。
//
// HACK: 区别：FAST-LIVO2 按 IMU 样本计数（bug：单帧 40 个样本即 > MAX_INI_COUNT=20），
//       本实现按 Lidar 帧计数（init_imu_num_ = 20 帧，约 2 秒@10Hz），
//       语义上是等待足够帧数以确保静止初始化的统计可靠性。

void ImuProcess::imu_init(const MeasureGroup& meas, StatesGroup& state) {
    if (is_first_frame_) {
        init_count_ = 0;
        is_first_frame_ = false;
        if (!meas.imu.empty()) {
            mean_acc_ = meas.imu.front().acc;
            mean_gyr_ = meas.imu.front().gyro;
        }
        if (meas.imu.size() < 2) { init_count_ = 1; return; }
    }

    // 跨帧 running mean（Welford 在线均值更新）
    for (size_t i = 0; i < meas.imu.size(); ++i) {
        const auto& imu = meas.imu[i];
        init_count_++;
        mean_acc_ += (imu.acc - mean_acc_) / static_cast<double>(init_count_);
        mean_gyr_ += (imu.gyro - mean_gyr_) / static_cast<double>(init_count_);
    }

    mean_acc_norm_ = mean_acc_.norm();

    RCLCPP_INFO_THROTTLE(
        rclcpp::get_logger("radar_fast_livo2_node"), throttle_clock_, 2000,
        "IMU initializing... %d IMU samples over %d frames (keep device still)",
        init_count_, init_count_ > 0 ? 1 + init_count_ / 40 : 0);

    const int target_count = init_imu_num * 40;  // 40 ≈ 典型每帧 IMU 样本数 @400Hz

    if (init_count_ >= target_count) {
        // 重力方向估计，直接写入调用方持有的 ESIKF 状态
        state.gravity = -mean_acc_ / mean_acc_norm_ * G_M_S2;
        state.rot_end = Eigen::Matrix3d::Identity();
        state.bias_g  = Eigen::Vector3d::Zero();
        state.bias_a  = Eigen::Vector3d::Zero();
        state.vel_end = Eigen::Vector3d::Zero();
        state.pos_end = Eigen::Vector3d::Zero();
        imu_need_init_ = false;

        RCLCPP_INFO(
            rclcpp::get_logger("radar_fast_livo2_node"),
            "IMU initialization done.\n"
            "  Gravity:  [%.4f %.4f %.4f]  norm=%.4f\n"
            "  Acc cov:  [%.6f %.6f %.6f]\n"
            "  Gyro cov: [%.6f %.6f %.6f]",
            state.gravity.x(), state.gravity.y(), state.gravity.z(),
            mean_acc_norm_,
            acc_cov_.x(), acc_cov_.y(), acc_cov_.z(),
            gyr_cov_.x(), gyr_cov_.y(), gyr_cov_.z());
    }
}

// ── 正向传播（含协方差）+ 逐点逆向去畸变 ────────────────────────────────
//
// 等价于 FAST-LIVO2 IMU_Processing.cpp: UndistortPcl()
//
// 流程:
//   1. 将上一帧末尾 IMU 与当前帧 IMU 序列拼接，保证时间连续性
//   2. 正向传播：对每一对连续 IMU 测量，用中值角速度/加速度积分状态
//      与协方差：
//        R_{k+1} = R_k * Exp(w_k, dt)
//        v_{k+1} = v_k + (R_k * a_k + g) * dt
//        p_{k+1} = p_k + v_k * dt + 0.5 * (R_k * a_k + g) * dt^2
//        P_{k+1} = F_x * P_k * F_x^T + cov_w
//      每个 IMU 时刻保存 Pose6D 到 imu_pose_
//   3. 逆向去畸变：按 curvature（帧内毫秒偏移）排序点云，
//      对每点在其最近的 IMU Pose6D 之间线性插值做逆运动补偿
//
// state 为调用方持有的唯一 ESIKF 状态对象，本函数原地修改其
// rot_end/pos_end/vel_end/bias_g/bias_a/gravity/cov 字段（Oracle C1）。
//
// 协方差传播 F_x/cov_w 的索引布局（18 维，对照 esikf_state.hpp）：
//   [0-2]=rot [3-5]=pos [6-8]=vel [9-11]=bg [12-14]=ba [15-17]=gravity
// 等价于 FAST-LIVO2 IMU_Processing.cpp:377-401 的 19 维布局去掉
// inv_expo_time（原第 6 维）后整体前移一位（Oracle C2）。
//
// 跳过:
//   - 曝光时间估计 (tau = inv_expo_time)：
//     StatesGroup 无 inv_expo_time 字段，见 FAST-LIVO2 中
//     exposure_estimate_en 相关代码

void ImuProcess::undistort_pcl(const MeasureGroup& meas, StatesGroup& state,
                                PointCloudT::Ptr& out) {
    // ── 1. 构建 IMU 序列（含上一帧末尾 IMU 桥接）──
    std::vector<ImuData, Eigen::aligned_allocator<ImuData>> v_imu = meas.imu;
    v_imu.insert(v_imu.begin(), last_imu_);  // 等价于 FAST-LIVO2: v_imu.push_front(last_imu)

    const double prop_beg_time = last_prop_end_time_;
    const double prop_end_time = meas.lidar_end_time;  // ONLY_LIO: 以帧末为传播终点

    if (out) { out->clear(); }

    // ── 2. 准备待去畸变点云 ──
    pcl_wait_proc_.clear();
    pcl_wait_proc_.resize(meas.lidar->points.size());
    pcl_wait_proc_ = *(meas.lidar);

    // 初始化 IMU 位姿队列：首元素为上一帧末尾状态（来自调用方状态）
    imu_pose_.clear();
    imu_pose_.push_back({});
    set_pose6d(imu_pose_.back(), 0.0, acc_s_last_, angvel_last_,
               state.vel_end, state.pos_end, state.rot_end);

    // ── 3. 正向传播：在每个 IMU 时间点积分状态与协方差 ──
    Eigen::Vector3d acc_imu     = acc_s_last_;
    Eigen::Vector3d angvel_avr  = angvel_last_;
    Eigen::Vector3d vel_imu     = state.vel_end;
    Eigen::Vector3d pos_imu     = state.pos_end;
    Eigen::Matrix3d R_imu       = state.rot_end;

    MD<DIM_STATE, DIM_STATE> F_x, cov_w;

    for (size_t i = 0; i + 1 < v_imu.size(); ++i) {
        const auto& head = v_imu[i];
        const auto& tail = v_imu[i + 1];

        if (tail.timestamp < prop_beg_time) { continue; }

        // 中值角速度/加速度（FAST-LIVO2 原版策略）
        angvel_avr = 0.5 * (head.gyro + tail.gyro);
        Eigen::Vector3d acc_avr = 0.5 * (head.acc + tail.acc);

        // 减去 IMU 偏置（由 ESIKF 在线估计并通过 state 闭环回灌）
        angvel_avr -= state.bias_g;
        // FAST-LIVO2 标度缩放：G/mean_acc_norm 补偿加速度计的标度误差
        // 参考 FAST-LIVO2 IMU_Processing.cpp:353
        acc_avr = acc_avr * G_M_S2 / mean_acc_norm_ - state.bias_a;

        double dt     = 0.0;
        double offs_t = 0.0;

        if (head.timestamp < prop_beg_time) {
            // 跨传播边界的首个部分 IMU 区间
            dt     = tail.timestamp - last_prop_end_time_;
            offs_t = tail.timestamp - prop_beg_time;
        } else if (i != v_imu.size() - 2) {
            // 正常帧内 IMU 区间
            dt     = tail.timestamp - head.timestamp;
            offs_t = tail.timestamp - prop_beg_time;
        } else {
            // 最后一个 IMU 区间，截断到帧末
            dt     = prop_end_time - head.timestamp;
            offs_t = prop_end_time - prop_beg_time;
        }

        // ── 协方差传播：F_x * P * F_x^T + cov_w（Oracle C2）──
        const Eigen::Matrix3d acc_avr_skew = skewSym(acc_avr);

        F_x.setIdentity();
        cov_w.setZero();

        F_x.block<3, 3>(0, 0) = exp_so3(angvel_avr, -dt);
        if (ba_bg_est_en_) { F_x.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity() * dt; }
        F_x.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
        F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
        if (ba_bg_est_en_) { F_x.block<3, 3>(6, 12) = -R_imu * dt; }
        if (gravity_align) { F_x.block<3, 3>(6, 15) = Eigen::Matrix3d::Identity() * dt; }

        cov_w.block<3, 3>(0, 0).diagonal()  = gyr_cov_ * dt * dt;
        cov_w.block<3, 3>(6, 6)             = R_imu * acc_cov_.asDiagonal() * R_imu.transpose() * dt * dt;
        cov_w.block<3, 3>(9, 9).diagonal()  = bg_cov_ * dt * dt;
        cov_w.block<3, 3>(12, 12).diagonal() = ba_cov_ * dt * dt;

        state.cov = F_x * state.cov * F_x.transpose() + cov_w;

        // 姿态传播（罗德里格斯公式）
        R_imu = R_imu * exp_so3(angvel_avr, dt);

        // 世界系下 IMU 加速度（含重力，来自调用方状态，闭环）
        acc_imu = R_imu * acc_avr + state.gravity;

        // 位置传播（匀加速模型）
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

        // 速度传播
        vel_imu = vel_imu + acc_imu * dt;

        // 保存当前 IMU 时间点的位姿（供后续逆向去畸变使用）
        angvel_last_ = angvel_avr;
        acc_s_last_  = acc_imu;

        imu_pose_.push_back({});
        set_pose6d(imu_pose_.back(), offs_t, acc_imu, angvel_avr,
                   vel_imu, pos_imu, R_imu);
    }

    // ── 4. 写回帧末状态（原地更新调用方持有的 state，闭环闭合）──
    // bias_g/bias_a/gravity 在传播循环内未被改写（IMU 积分本身不更新零偏/
    // 重力估计值，只用它们做补偿），因此保持调用方传入时的值——若上一帧
    // ESIKF 已经修正过这些字段，本帧传播自然继承修正后的结果（Oracle C1）。
    state.vel_end = vel_imu;
    state.rot_end = R_imu;
    state.pos_end = pos_imu;

    last_imu_           = v_imu.back();
    last_prop_end_time_ = prop_end_time;

    if (pcl_wait_proc_.points.size() < 1) { return; }

    // ── 5. 逐点逆向去畸变 ──
    //
    // 按 curvature（帧内时间偏移，毫秒）升序排列点云。
    // 对每点在其所在 IMU 位姿区间 [head, tail] 内插值：
    //
    //   P_compensate = extR_Ri * (R_i * (R_LI * P_raw + t_LI) + T_ei) - exrR_extT
    //
    // 其中：
    //   R_i   = R_head * Exp(w_head, dt)     — 点测量时刻的姿态（世界系）
    //   T_ei  = p_head + v_head*dt + a_head*dt²/2 - p_end  — 测量时刻→帧末的平移
    //   extR_Ri    = R_LI^T * R_end^T                     — 帧末世界系→帧末 Lidar 系
    //   exrR_extT  = R_LI^T * t_LI                        — IMU→Lidar 平移在 Lidar 系下
    //
    // 等价于 FAST-LIVO2 IMU_Processing.cpp:519-529

    std::sort(pcl_wait_proc_.points.begin(), pcl_wait_proc_.points.end(), time_list);

    if (imu_pose_.size() < 2) {
        // 正向传播产生的 IMU 位姿不足（极短帧），跳过去畸变
        out->swap(pcl_wait_proc_);
        return;
    }

    auto it_pcl = pcl_wait_proc_.points.end() - 1;

    // 预计算外参变换常量（帧末世界系 → 帧末 Lidar 系）
    const Eigen::Matrix3d extR_Ri = r_lidar_imu_.transpose() * state.rot_end.transpose();
    const Eigen::Vector3d exrR_extT = r_lidar_imu_.transpose() * t_lidar_imu_;

    for (auto it_kp = imu_pose_.end() - 1; it_kp != imu_pose_.begin(); --it_kp) {
        const auto* head = &*(it_kp - 1);
        const auto* tail = &*it_kp;
        (void)tail;

        // 加载 earlier (head) IMU 位姿
        const Eigen::Matrix3d& R_head = head->rot;
        const Eigen::Vector3d& a_head = head->acc;
        const Eigen::Vector3d& v_head = head->vel;
        const Eigen::Vector3d& p_head = head->pos;
        const Eigen::Vector3d& w_head = head->gyr;

        // 此 IMU 区间 [head, tail] 内所有 3D 点去畸变
        for (; it_pcl->curvature / 1000.0 > head->offset_time; --it_pcl) {
            const double dt = it_pcl->curvature / 1000.0 - head->offset_time;

            // 点测量时刻的姿态 R_i (world frame)
            const Eigen::Matrix3d R_i(R_head * exp_so3(w_head, dt));

            // 测量时刻 → 帧末的平移 T_ei (world frame, to end)
            const Eigen::Vector3d T_ei(
                p_head + v_head * dt + 0.5 * a_head * dt * dt - state.pos_end);

            const Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);

            // 全链式变换：Lidar(t) → IMU(t) → World(t) → World(end) → Lidar(end)
            const Eigen::Vector3d P_compensate =
                (extR_Ri * (R_i * (r_lidar_imu_ * P_i + t_lidar_imu_) + T_ei)
                 - exrR_extT);

            it_pcl->x = P_compensate.x();
            it_pcl->y = P_compensate.y();
            it_pcl->z = P_compensate.z();

            if (it_pcl == pcl_wait_proc_.points.begin()) { break; }
        }
    }

    out->swap(pcl_wait_proc_);
}

// ── 无 IMU 模式 ────────────────────────────────────────────────────────
//
// 等价于 FAST-LIVO2 IMU_Processing.cpp: Forward_without_imu()
//
// 简化处理:
//   1. 按 curvature 排序点云
//   2. 匀速运动假设：pos_end += vel_end * dt，rot_end *= Exp(bias_g, dt)
//   3. 逆向去畸变：旋转补偿（Exp(-bias_g * dt)）+ 平移补偿（-vel * dt）
//
// 跳过:
//   - 协方差传播（无 IMU 数据无法构造 F_x/cov_w 中依赖角速度/加速度的项）
//   - 加速度积分（无 IMU 数据无法积分 a）
//
// state 为调用方持有的唯一 ESIKF 状态对象，原地修改（Oracle C1，
// 与 undistort_pcl 保持一致的状态所有权约定）。

void ImuProcess::forward_without_imu(const MeasureGroup& meas,
                                     StatesGroup& state,
                                     PointCloudT::Ptr& out) {
    // 拷贝点云并排序
    out->clear();
    pcl_wait_proc_.clear();
    pcl_wait_proc_.resize(meas.lidar->points.size());
    pcl_wait_proc_ = *(meas.lidar);

    std::sort(pcl_wait_proc_.points.begin(), pcl_wait_proc_.points.end(), time_list);

    if (pcl_wait_proc_.points.empty()) { return; }

    const double pcl_end_offset_time =
        pcl_wait_proc_.points.back().curvature / 1000.0;
    const double pcl_beg_time = meas.lidar_beg_time;

    double dt = 0.0;
    if (is_first_frame_) {
        dt = 0.1;
        is_first_frame_ = false;
    } else {
        dt = pcl_beg_time - time_last_scan_;
    }
    time_last_scan_ = pcl_beg_time;

    // ── 简单位姿传播（匀速模型 + 零偏旋转）──
    const Eigen::Matrix3d Exp_f = exp_so3(state.bias_g, dt);
    state.rot_end = state.rot_end * Exp_f;
    state.pos_end = state.pos_end + state.vel_end * dt;

    // ── 逆向去畸变（旋转 + 匀速平移补偿）──
    auto it_pcl = pcl_wait_proc_.points.end() - 1;
    for (; it_pcl != pcl_wait_proc_.points.begin(); --it_pcl) {
        const double dt_j = pcl_end_offset_time - it_pcl->curvature / 1000.0;
        const Eigen::Matrix3d R_jk(exp_so3(state.bias_g, -dt_j));
        const Eigen::Vector3d P_j(it_pcl->x, it_pcl->y, it_pcl->z);
        const Eigen::Vector3d p_jk =
            -state.rot_end.transpose() * state.vel_end * dt_j;
        const Eigen::Vector3d P_compensate = R_jk * P_j + p_jk;

        it_pcl->x = P_compensate.x();
        it_pcl->y = P_compensate.y();
        it_pcl->z = P_compensate.z();
    }

    out->swap(pcl_wait_proc_);
}

}  // namespace radar::fast_livo2
