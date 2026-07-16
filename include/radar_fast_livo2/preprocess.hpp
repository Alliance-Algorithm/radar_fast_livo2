#pragma once
// preprocess.hpp - 点云预处理：各传感器 → 统一 PointType
// 原始来源: hku-mars/FAST-LIVO2/src/preprocess.h，适配 Odin1 dToF

#include "radar_fast_livo2/common_lib.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

namespace radar::fast_livo2 {

class Preprocess {
public:
    Preprocess();
    ~Preprocess() = default;

    // 主入口：将 ROS2 PointCloud2 转换为统一的 PointCloudT
    // 内部根据 lidar_type_ 分发到对应 handler
    void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                 PointCloudT::Ptr& out);

    // ── 配置 ────────────────────────────────────────────────────
    int    lidar_type      = LidarType::ODIN1;
    int    point_filter_num = 4;     // 每 N 个点取 1 个（降采样）
    float  blind            = 0.1f;  // 最近有效距离 (m)
    float  max_range        = 30.0f; // 最远有效距离 (m)
    uint16_t confidence_threshold = 35;  // Odin1 专用：置信度阈值

private:
    // ── 各传感器处理函数 ─────────────────────────────────────────
    void odin1_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                       PointCloudT::Ptr& out);
    void l515_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                      PointCloudT::Ptr& out);
    void velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                          PointCloudT::Ptr& out);
    void oust64_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                        PointCloudT::Ptr& out);

    int point_count_ = 0;  // 用于 point_filter_num 的计数器
};

}  // namespace radar::fast_livo2

// FIXME: 注：Odin1 cloud_raw 字段偏移曾用 #pragma pack(1) 结构体
// + pcl::fromROSMsg 解析，因 PCL 内部 SIMD 对齐指令访问非对齐 float
// 触发 SIGSEGV（详见 preprocess.cpp odin1_handler 注释），已改为
// 直接按 msg->fields 偏移用 memcpy 读取，不再需要该结构体和 PCL 类型注册。
