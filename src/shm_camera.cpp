// shm_camera.cpp — ShmCamera implementation (hikcamera SHMRead adapter)

#include "radar_fast_livo2/shm_camera.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>

#include <opencv2/imgproc.hpp>

#include "hikcamera/shm.hpp"

namespace radar::fast_livo2 {

namespace {

auto map_shm_error(const std::string& msg) -> FrameReadErrorCode {
    // SHMRead reports string errors; classify common cases for typed branch.
    if (msg.find("not initialized") != std::string::npos
        || msg.find("Failed to map") != std::string::npos
        || msg.find("Failed to create") != std::string::npos) {
        return FrameReadErrorCode::ShmError;
    }
    // sem_timedwait expiry still returns success path with stale data in some
    // versions; empty mat / invalid layout → InvalidFrame at convert.
    return FrameReadErrorCode::ShmError;
}

auto steady_ns(std::chrono::steady_clock::time_point tp) -> uint64_t {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch());
    return static_cast<uint64_t>(ns.count());
}

} // namespace

// ══════════════════════════════════════════════════════════════════
// Constructor / destructor
// ══════════════════════════════════════════════════════════════════

ShmCamera::ShmCamera(std::string shm_name, int target_width, int target_height,
    double img_time_offset, int source_width, int source_height)
    : shm_name_(std::move(shm_name))
    , target_width_(target_width)
    , target_height_(target_height)
    , source_width_(source_width)
    , source_height_(source_height)
    , img_time_offset_(img_time_offset) { }

ShmCamera::~ShmCamera() {
    if (shm_fd_ >= 0) {
        // Prefer SDK close helper; fall back to close(2).
        std::ignore = hikcamera::SHMClose(shm_fd_);
        shm_fd_ = -1;
    }
    is_open_ = false;
}

// ══════════════════════════════════════════════════════════════════
// open
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::open() -> std::expected<void, ShmCameraError> {
    if (target_width_ <= 0 || target_height_ <= 0) {
        return std::unexpected { ShmCameraError { FrameReadErrorCode::InvalidFrame,
            "ShmCamera: target dimensions must be positive" } };
    }
    if (source_width_ <= 0 || source_height_ <= 0) {
        return std::unexpected { ShmCameraError { FrameReadErrorCode::InvalidFrame,
            "ShmCamera: source dimensions must be positive" } };
    }
    if (is_open_) {
        return { };
    }

    // Reader path: open existing segment created by hikcamera_ros_driver writer.
    const int fd = ::shm_open(shm_name_.c_str(), O_RDWR, 0666);
    if (fd == -1) {
        return std::unexpected { ShmCameraError { FrameReadErrorCode::ShmError,
            std::string("ShmCamera: shm_open failed for '") + shm_name_ + "': "
                + std::strerror(errno) } };
    }

    shm_fd_  = fd;
    is_open_ = true;
    return { };
}

auto ShmCamera::is_open() const noexcept -> bool { return is_open_; }

// ══════════════════════════════════════════════════════════════════
// wait_next
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::wait_next(std::chrono::milliseconds /*timeout*/)
    -> std::expected<CameraFrame, ShmCameraError> {
    if (!is_open_ || shm_fd_ < 0) {
        return std::unexpected {
            ShmCameraError { FrameReadErrorCode::NotOpen, "ShmCamera: not open" } };
    }

    // hikcamera::SHMRead owns the wait (sem_timedwait ~1s) and returns a
    // BGR8 clone (or resized BGR).  We pass dst size so the SDK can resize
    // in-place to target before we convert to gray.
    cv::Mat bgr;
    std::chrono::steady_clock::time_point ts {};
    auto ret = hikcamera::SHMRead(
        shm_fd_, bgr, ts, source_width_, source_height_, target_width_, target_height_);
    if (!ret) {
        return std::unexpected {
            ShmCameraError { map_shm_error(ret.error()), ret.error() } };
    }

    FrameMetadata meta {};
    meta.host_monotonic_ns  = steady_ns(ts);
    meta.committed_sequence = next_sequence_;
    meta.frame_id           = next_sequence_;
    ++next_sequence_;

    CameraFrame frame = convert(bgr, target_width_, target_height_, meta, img_time_offset_);
    if (frame.gray.empty()) {
        return std::unexpected { ShmCameraError { FrameReadErrorCode::InvalidFrame,
            "ShmCamera: conversion produced empty frame (invalid SHM source)" } };
    }
    return frame;
}

// ══════════════════════════════════════════════════════════════════
// convert — pure function, testable without hardware
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::convert(const cv::Mat& bgr, int target_width, int target_height,
    const FrameMetadata& meta, double img_time_offset) -> CameraFrame {
    CameraFrame frame;

    if (bgr.empty() || bgr.type() != CV_8UC3 || target_width <= 0 || target_height <= 0) {
        frame.sequence          = meta.committed_sequence;
        frame.host_monotonic_ns = meta.host_monotonic_ns;
        frame.frame_id          = meta.frame_id;
        frame.timestamp_seconds =
            static_cast<double>(meta.host_monotonic_ns) * 1e-9 + img_time_offset;
        return frame;
    }

    // BGR8 → gray at current resolution, then resize to target if needed.
    // When SHMRead already resized to target, this is a single cvtColor.
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    if (gray.cols == target_width && gray.rows == target_height) {
        frame.gray = std::move(gray);
    } else {
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
        frame.gray = std::move(resized);
    }

    frame.sequence          = meta.committed_sequence;
    frame.host_monotonic_ns = meta.host_monotonic_ns;
    frame.frame_id          = meta.frame_id;
    frame.timestamp_seconds = static_cast<double>(meta.host_monotonic_ns) * 1e-9 + img_time_offset;

    return frame;
}

} // namespace radar::fast_livo2
