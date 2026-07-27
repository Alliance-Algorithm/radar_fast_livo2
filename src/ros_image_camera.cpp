// ros_image_camera.cpp — RosImageCamera implementation
//
// Converts sensor_msgs::msg::Image (BGR8) → CameraFrame (CV_8UC1 grayscale).
// Timestamp is computed directly from image.header.stamp (sec + nanosec*1e-9)
// with NO img_time_offset — ROS stamps are already in LiDAR time domain for replay.

#include "radar_fast_livo2/ros_image_camera.hpp"

#include <opencv2/imgproc.hpp>

namespace radar::fast_livo2 {

auto RosImageCamera::convert(const sensor_msgs::msg::Image& image, int target_width,
    int target_height, uint64_t local_seq) -> std::optional<CameraFrame> {
    // The Odin ROS2 driver labels this stream as bgr8. Some firmware revisions
    // leave encoding empty even though the payload is still the decoded
    // 3-channel BGR image produced by publishRgb(). Empty encoding is accepted
    // only as this driver compatibility case; no RGB/BGR swap is performed.
    if (!image.encoding.empty() && image.encoding != "bgr8") return std::nullopt;

    // ── 2. Validate dimensions ────────────────────────────────────
    if (image.width <= 0 || image.height <= 0 || target_width <= 0 || target_height <= 0)
        return std::nullopt;

    // 3. Validate data size matches declared dimensions
    const size_t row_bytes = static_cast<size_t>(image.width) * 3U;
    if (image.step < row_bytes) return std::nullopt;
    const size_t expected_bytes = static_cast<size_t>(image.height) * image.step;
    if (expected_bytes == 0 || image.data.size() < expected_bytes) return std::nullopt;

    // ── 4. Wrap raw BGR8 data as cv::Mat (no copy) ────────────────
    // const_cast is safe: cvtColor only reads from source.
    cv::Mat bgr(image.height, image.width, CV_8UC3,
        const_cast<uint8_t*>(image.data.data()), image.step);

    // ── 5. BGR → gray at source resolution ────────────────────────
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    CameraFrame frame;

    // ── 6. Resize to target resolution ────────────────────────────
    if (gray.cols == target_width && gray.rows == target_height) {
        frame.gray = std::move(gray);
    } else {
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
        frame.gray = std::move(resized);
    }

    // ── 7. Metadata ───────────────────────────────────────────────
    frame.sequence = local_seq;
    // host_monotonic_ns / frame_id: not applicable for ROS replay mode
    frame.host_monotonic_ns = 0;
    frame.frame_id          = 0;

    // ── 8. Timestamp: ROS header stamp, NO img_time_offset ────────
    // Image stamps are already in the ROS/LiDAR time domain for MCAP replay.
    // Compute directly from builtin_interfaces/Time fields (avoids rclcpp header chain).
    frame.timestamp_seconds = static_cast<double>(image.header.stamp.sec)
        + static_cast<double>(image.header.stamp.nanosec) * 1e-9;

    return frame;
}

void RosImageCamera::validate_topic_not_empty(const std::string& mode, const std::string& topic) {
    if (mode == "ros_image" && topic.empty()) {
        throw std::invalid_argument(
            "camera/image_topic must be set (non-empty) when camera_input_mode is 'ros_image'");
    }
}

} // namespace radar::fast_livo2
