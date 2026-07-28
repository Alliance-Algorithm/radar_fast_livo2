#pragma once
// ros_image_camera.hpp — ROS Image replay-safe camera adapter for radar_fast_livo2
//
// Provides a stateless BGR8 → grayscale converter that reads timestamps
// directly from the ROS Image header (no offset). Odin's official ROS driver
// publishes the decoded camera image as bgr8; this is the single conversion
// needed by the grayscale FAST-LIVO2 frontend. Intended for MCAP replay
// where image stamps are already in the LiDAR/ROS clock domain.
//
// Usage:
//   auto frame = RosImageCamera::convert(image_msg, target_w, target_h, local_seq);
//   if (frame) { queue.push(std::move(*frame)); }

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <sensor_msgs/msg/image.hpp>

#include "radar_fast_livo2/shm_camera.hpp" // CameraFrame

namespace radar::fast_livo2 {

/// Stateless BGR8 replay converter.
///
/// Takes a sensor_msgs/Image with "bgr8" encoding, validates it,
/// converts to owned CV_8UC1 grayscale at the target resolution,
/// and returns a CameraFrame with:
///   - timestamp_seconds = rclcpp::Time(image.header.stamp).seconds()
///     (NO img_time_offset — ROS stamps are already in LiDAR time domain)
///   - sequence = locally-assigned monotonically-increasing counter
///   - host_monotonic_ns = 0 (not applicable)
///   - frame_id = 0 (not applicable)
struct RosImageCamera {
    RosImageCamera()  = delete;
    ~RosImageCamera() = delete;

    /// @param image         sensor_msgs/Image with "bgr8" encoding
    /// @param target_width  desired grayscale output width
    /// @param target_height desired grayscale output height
    /// @param local_seq     monotonically-increasing local sequence number
    /// @return CameraFrame if conversion succeeded, nullopt otherwise
    [[nodiscard]] static auto convert(const sensor_msgs::msg::Image& image, int target_width,
        int target_height, uint64_t local_seq) -> std::optional<CameraFrame>;

    /// Validate camera input mode and ROS topic for ros_image mode.
    /// Throws std::invalid_argument if mode is "ros_image" and topic is empty.
    static void validate_topic_not_empty(const std::string& mode, const std::string& topic);
};

} // namespace radar::fast_livo2
