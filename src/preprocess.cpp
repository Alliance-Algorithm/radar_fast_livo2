// preprocess.cpp - Odin1 dToF 点云预处理实现
#include "radar_fast_livo2/preprocess.hpp"
#include <algorithm>
#include <pcl/common/transforms.h>
#include <rclcpp/rclcpp.hpp>

namespace radar::fast_livo2 {

Preprocess::Preprocess() = default;

void Preprocess::process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                         PointCloudT::Ptr& out) {
    out->clear();
    switch (lidar_type) {
        case LidarType::ODIN1: odin1_handler(msg, out); break;
        case LidarType::L515:  l515_handler(msg, out);  break;
        case LidarType::VELO16:
        case LidarType::OUST64:
        default:
            RCLCPP_WARN_ONCE(rclcpp::get_logger("radar_fast_livo2"),
                "Unsupported lidar_type %d, using odin1_handler fallback", lidar_type);
            odin1_handler(msg, out);
            break;
    }
}

// FIXME: 不用 pcl::fromROSMsg + pcl::PointCloud<odin::Point>（曾用 #pragma pack(1)
// 让结构体偏移匹配设备紧凑打包格式）：PCL 内部对点云数据做 SIMD 操作时会用
// 对齐指令（如 movaps）访问点字段，而 pack(1) 结构体里 offset_time 落在非
// 4 字节对齐地址，实测触发 SIGSEGV（节点收到第一帧后立即崩溃，exit -11）。
// 改成直接按 msg->fields 里的偏移量用 memcpy 读原始字节，memcpy 对任意对齐
// 都安全，且不依赖 PCL 点类型注册，同时更贴近协议本身（字段偏移来自消息
// 元数据而非编译期假设，设备端字段顺序变化也不会导致静默错位）。
void Preprocess::odin1_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                                PointCloudT::Ptr& out) {
    uint32_t off_x = 0, off_y = 4, off_z = 8;
    uint32_t off_intensity = 12, off_confidence = 13, off_offset_time = 15;
    for (const auto& f : msg->fields) {
        if (f.name == "x") off_x = f.offset;
        else if (f.name == "y") off_y = f.offset;
        else if (f.name == "z") off_z = f.offset;
        else if (f.name == "intensity") off_intensity = f.offset;
        else if (f.name == "confidence") off_confidence = f.offset;
        else if (f.name == "offset_time") off_offset_time = f.offset;
    }

    const uint32_t point_step = msg->point_step;
    const uint32_t n_points = msg->width * msg->height;
    if (point_step == 0 || n_points == 0) { return; }

    // Bounds check: ensure field offsets don't exceed point_step
    const uint32_t max_needed = std::max({off_x + 4, off_y + 4, off_z + 4,
                                          off_intensity + 1, off_confidence + 2,
                                          off_offset_time + 4});
    if (point_step < max_needed) {
        RCLCPP_ERROR_ONCE(rclcpp::get_logger("radar_fast_livo2"),
            "Point step %u too small for fields (need >= %u). "
            "x=%u y=%u z=%u intensity=%u confidence=%u offset_time=%u",
            point_step, max_needed,
            off_x, off_y, off_z,
            off_intensity, off_confidence, off_offset_time);
        return;
    }

    out->reserve(n_points / std::max(1, point_filter_num) + 1);

    for (uint32_t i = 0; i < n_points; ++i) {
        const uint8_t* base = msg->data.data() + static_cast<size_t>(i) * point_step;

        uint16_t confidence = 0;
        std::memcpy(&confidence, base + off_confidence, sizeof(confidence));

        // 1. 置信度过滤（dToF 噪声主要来自低置信度点）
        if (confidence < confidence_threshold) continue;

        float x = 0.0f, y = 0.0f, z = 0.0f;
        std::memcpy(&x, base + off_x, sizeof(x));
        std::memcpy(&y, base + off_y, sizeof(y));
        std::memcpy(&z, base + off_z, sizeof(z));

        // 2. 距离范围过滤
        const float r = std::sqrt(x * x + y * y + z * z);
        if (r < blind || r > max_range) continue;

        // 3. 均匀下采样
        if ((point_count_++ % point_filter_num) != 0) continue;

        uint8_t intensity = 0;
        float offset_time = 0.0f;
        std::memcpy(&intensity, base + off_intensity, sizeof(intensity));
        std::memcpy(&offset_time, base + off_offset_time, sizeof(offset_time));

        // 4. 转换到统一 PointType
        // FIXME: Odin1 dToF 是全局曝光面阵传感器（同 L515），所有像素同时曝光，
        // 不存在逐点扫描时间差。驱动层虚构的 offset_time（group/ODR，量纲是
        // rate 不是时间）不能用于运动去畸变——用了等价于把旋转向量乘了一个随机
        // 幅值，运动中点云被"拧成麻花"，导致 zero feature。
        // 正确做法与 L515 路径（preprocess.cpp:129）一致：curvature = 0。
        PointType pt;
        pt.x         = x;
        pt.y         = y;
        pt.z         = z;
        pt.intensity = static_cast<float>(intensity);
        pt.curvature = 0.0f;  // 全局曝光面阵，无帧内时间偏移（同 L515 路径）
        out->push_back(pt);
    }

    out->header = pcl_conversions::toPCL(msg->header);
    out->is_dense = false;
}

void Preprocess::l515_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg,
                               PointCloudT::Ptr& out) {
    // Intel L515：标准 PointXYZRGB，无 offset_time，无曝光去畸变
    pcl::PointCloud<pcl::PointXYZRGB> raw;
    pcl::fromROSMsg(*msg, raw);

    out->reserve(raw.size() / std::max(1, point_filter_num) + 1);

    for (std::size_t i = 0; i < raw.size(); ++i) {
        const auto& p = raw[i];

        if ((point_count_++ % point_filter_num) != 0) continue;

        const float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (r < blind || r > max_range) continue;

        PointType pt;
        pt.x         = p.x;
        pt.y         = p.y;
        pt.z         = p.z;
        pt.intensity = 0.0f;
        pt.curvature = 0.0f;  // 面阵同时曝光，无帧内时间偏移
        out->push_back(pt);
    }

    out->header = pcl_conversions::toPCL(msg->header);
    out->is_dense = false;
}

}  // namespace radar::fast_livo2
