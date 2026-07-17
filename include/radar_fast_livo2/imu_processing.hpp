#pragma once
// imu_processing.hpp - IMU 初始化 + 逐点去畸变
// 原始来源: hku-mars/FAST-LIVO2/include/IMU_Processing.h，适配 ROS2

#include "radar_fast_livo2/common_lib.hpp"
#include "radar_fast_livo2/esikf_state.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <rclcpp/clock.hpp>
#include <vector>

namespace radar::fast_livo2 {

class alignas(16) ImuProcess {
public:
    ImuProcess();
    ~ImuProcess() = default;

    // 主入口：IMU 初始化 + 逐点去畸变
    //
    // state 是调用方持有的唯一 ESIKF 状态对象（in/out）：
    //   - 输入：上一帧 ESIKF 后验（rot_end/pos_end/vel_end/bias_g/bias_a/gravity/cov）
    //   - 输出：本帧 IMU 正向传播后的先验（同一对象原地修改，
    //     供 process_frame() 直接作为 state_propagat 使用）
    // 状态所有权只有这一份，不再有独立的 last_state_ 副本，
    // 避免 ESIKF 后验无法回灌传播起点（Oracle C1）。
    //
    // 若初始化未完成返回 false，否则填充 cloud_undistorted 的每个点坐标（去畸变后）
    bool process(MeasureGroup& meas, StatesGroup& state,
                 PointCloudT::Ptr& cloud_undistorted);

    void set_extrinsic(const Eigen::Vector3d& t, const Eigen::Matrix3d& r) {
        t_lidar_imu_ = t;
        r_lidar_imu_ = r;
    }
    void set_gyr_cov(const Eigen::Vector3d& cov) { gyr_cov_ = cov; }
    void set_acc_cov(const Eigen::Vector3d& cov) { acc_cov_ = cov; }
    void set_gyr_bias_cov(const Eigen::Vector3d& cov) { bg_cov_ = cov; }
    void set_acc_bias_cov(const Eigen::Vector3d& cov) { ba_cov_ = cov; }
    void disable_imu()        { imu_en = false; imu_need_init_ = false; }
    void disable_gravity_est(){ gravity_align = false; }
    void disable_bias_est()   { ba_bg_est_en_ = false; }
    void set_imu_init_frame_num(int n) { init_imu_num = n; }
    void reset();

    // 已消费到的传播锚点（上一次 undistort_pcl/imu_init 结束时刻）。
    // 调用方用它来裁剪 imu_buf_：早于此时刻的样本已被消费，不会再被
    // undistort_pcl 用到（其内部 prop_beg_time 门控会直接 continue 跳过），
    // 可安全丢弃。比固定 wall-clock 窗口更贴近上游"处理完就 pop"的语义,
    // 也避免处理卡顿超过固定窗口时误删尚未消费的 IMU 数据。
    double last_prop_end_time() const { return last_prop_end_time_; }

    bool   imu_en         = true;
    int    init_imu_num   = 20;      // 静止初始化需要的 Lidar 帧数（非 IMU 样本数）
    bool   gravity_align  = true;    // 是否对齐重力方向

private:
    void imu_init(const MeasureGroup& meas, StatesGroup& state);
    void undistort_pcl(const MeasureGroup& meas, StatesGroup& state,
                       PointCloudT::Ptr& out);
    void forward_without_imu(const MeasureGroup& meas, StatesGroup& state,
                             PointCloudT::Ptr& out);

    // ── 正向传播状态 ──
    int    init_count_      = 0;
    bool   is_first_frame_  = true;
    bool   imu_need_init_   = true;
    bool   ba_bg_est_en_    = true;
    double last_prop_end_time_ = 0.0;
    double time_last_scan_  = 0.0;

    ImuData last_imu_;

    Eigen::Vector3d mean_acc_    = Eigen::Vector3d(0, 0, -1.0);
    Eigen::Vector3d mean_gyr_    = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel_last_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_s_last_  = Eigen::Vector3d::Zero();
    double mean_acc_norm_ = 9.81;

    // ── LiDAR-IMU 外参 ──
    Eigen::Vector3d t_lidar_imu_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d r_lidar_imu_ = Eigen::Matrix3d::Identity();

    // ── 协方差参数 ──
    Eigen::Vector3d gyr_cov_ = Eigen::Vector3d(0.1, 0.1, 0.1);
    Eigen::Vector3d acc_cov_ = Eigen::Vector3d(0.1, 0.1, 0.1);
    Eigen::Vector3d bg_cov_  = Eigen::Vector3d(0.0001, 0.0001, 0.0001);
    Eigen::Vector3d ba_cov_  = Eigen::Vector3d(0.0001, 0.0001, 0.0001);

    // ── 内部位姿队列（等价于 FAST-LIVO2 Pose6D）──
public:
    // 等价于 FAST-LIVO2 Pose6D，用 Eigen 类型替代 C 式数组
    struct Pose6D {
        double          offset_time = 0.0;
        Eigen::Vector3d acc         = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyr         = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel         = Eigen::Vector3d::Zero();
        Eigen::Vector3d pos         = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rot         = Eigen::Matrix3d::Identity();
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
private:
    std::vector<Pose6D, Eigen::aligned_allocator<Pose6D>> imu_pose_;
    PointCloudT pcl_wait_proc_;
    std::vector<ImuData, Eigen::aligned_allocator<ImuData>> imu_buf_;

    // FIXME: RCLCPP_INFO_THROTTLE 需要跨调用存活的 Clock（用于记录上次打印时间）；
    // 每次用临时 Clock 会在语句结束时销毁，导致下次调用时出现 use-after-free。
    rclcpp::Clock throttle_clock_{RCL_STEADY_TIME};
};

}  // namespace radar::fast_livo2
