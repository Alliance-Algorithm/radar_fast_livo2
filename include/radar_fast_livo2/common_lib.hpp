#pragma once
// common_lib.h - 公共类型定义（从 FAST-LIVO2 移植，适配 ROS2）
// 原始来源: hku-mars/FAST-LIVO2/include/common_lib.h

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <deque>
#include <string>
#include <vector>

namespace radar::fast_livo2 {

// ── 点类型 ────────────────────────────────────────────────────────
// curvature 字段被复用为帧内时间偏移（毫秒），与 FAST-LIO2 约定一致
using PointType    = pcl::PointXYZINormal;
using PointCloudT  = pcl::PointCloud<PointType>;

// ── 传感器类型 ────────────────────────────────────────────────────
enum LidarType {
    AVIA    = 1,  // Livox AVIA（原始支持）
    VELO16  = 2,  // Velodyne VLP-16
    OUST64  = 3,  // Ouster OS-64
    L515    = 4,  // Intel RealSense L515（固态 LiDAR，面阵 dToF）
    ODIN1   = 5,  // Manifold Tech Odin1 dToF（新增）
};

enum SlamMode {
    ONLY_LO  = 0,
    ONLY_LIO = 1,  // 纯 LiDAR-IMU，不使用相机
    LIVO     = 2,  // 完整 LiDAR-IMU-Visual 融合
};

// ── IMU 数据 ──────────────────────────────────────────────────────
struct alignas(16) ImuData {
    double timestamp = 0.0;  // 秒
    Eigen::Vector3d acc;     // m/s^2, IMU 本体系
    Eigen::Vector3d gyro;    // rad/s, IMU 本体系
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ── 同步数据包 ────────────────────────────────────────────────────
// 一帧 LiDAR 点云 + 其时间窗口内的所有 IMU 数据
struct MeasureGroup {
    double lidar_beg_time = 0.0;  // 帧起始时间（秒）
    double lidar_end_time = 0.0;  // 帧结束时间（秒）
    PointCloudT::Ptr lidar;       // 去畸变前的原始帧
    std::vector<ImuData, Eigen::aligned_allocator<ImuData>> imu;  // 时间区间内的 IMU 数据
    bool has_image = false;
    double img_time = 0.0;
};

// ── 常量 ─────────────────────────────────────────────────────────
constexpr double LASER_POINT_COV = 0.001;
constexpr double NUM_MATCH_POINTS = 5;

}  // namespace radar::fast_livo2
