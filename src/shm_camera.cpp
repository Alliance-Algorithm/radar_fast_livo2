// shm_camera.cpp — ShmCamera implementation

#include "radar_fast_livo2/shm_camera.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace radar::fast_livo2 {

// ══════════════════════════════════════════════════════════════════
// Constructor / destructor
// ══════════════════════════════════════════════════════════════════

ShmCamera::ShmCamera(std::string shm_name, int target_width,
                     int target_height, double img_time_offset)
    : shm_name_(std::move(shm_name)),
      target_width_(target_width),
      target_height_(target_height),
      img_time_offset_(img_time_offset) {}

ShmCamera::~ShmCamera() = default;  // reader_ RAII closes on destruction

// ══════════════════════════════════════════════════════════════════
// open
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::open() -> std::expected<void, std::string> {
    auto result = reader_.open(shm_name_.c_str());
    if (!result) return std::unexpected(result.error());
    is_open_ = true;
    return {};
}

auto ShmCamera::is_open() const noexcept -> bool { return is_open_; }

// ══════════════════════════════════════════════════════════════════
// wait_next
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::wait_next(std::chrono::milliseconds timeout)
    -> std::expected<CameraFrame, std::string> {
    auto sf_result = reader_.wait_next(timeout);
    if (!sf_result) return std::unexpected(sf_result.error());

    const auto& sf = *sf_result;
    CameraFrame frame = convert(sf.mat(), target_width_, target_height_,
                                sf.metadata(), img_time_offset_);
    if (frame.gray.empty()) {
        return std::unexpected(
            "ShmCamera: conversion produced empty frame (invalid SHM source)");
    }
    return frame;
}

// ══════════════════════════════════════════════════════════════════
// convert — pure function, testable without hardware
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::convert(const cv::Mat& bgr, int target_width,
                         int target_height,
                         const hikcamera::FrameMetadata& meta,
                         double img_time_offset) -> CameraFrame {
    CameraFrame frame;

    if (bgr.empty() || bgr.type() != CV_8UC3 || target_width <= 0 || target_height <= 0) {
        frame.sequence          = meta.committed_sequence;
        frame.host_monotonic_ns = meta.host_monotonic_ns;
        frame.frame_id          = meta.frame_id;
        frame.timestamp_seconds = static_cast<double>(meta.host_monotonic_ns) * 1e-9
                                  + img_time_offset;
        return frame;
    }

    // BGR8 → gray at source resolution, then resize to target.
    // No full BGR clone; small allocations: gray(source res) then target.
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // 2. Resize to target resolution (only if needed; uses INTER_AREA for downscale)
    if (gray.cols == target_width && gray.rows == target_height) {
        frame.gray = std::move(gray);
    } else {
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(target_width, target_height),
                   0.0, 0.0, cv::INTER_AREA);
        frame.gray = std::move(resized);
    }

    // 3. Metadata
    frame.sequence          = meta.committed_sequence;
    frame.host_monotonic_ns = meta.host_monotonic_ns;
    frame.frame_id          = meta.frame_id;

    // 4. Timestamp: steady-clock epoch seconds + offset
    frame.timestamp_seconds = static_cast<double>(meta.host_monotonic_ns) * 1e-9
                              + img_time_offset;

    return frame;
}

}  // namespace radar::fast_livo2
