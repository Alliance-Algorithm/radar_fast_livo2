// shm_camera.cpp — ShmCamera implementation using hikcamera::SharedFrameReader
//
// Image-dimension contract: source_width/source_height describe the raw
// SHM RGB resolution (hikcamera.yaml → 5472×3648).  target_width/target_height
// describe the VIO working resolution (odin_livo2.yaml → cam_width/cam_height).
// wait_next() copies FULL source-resolution RGB with correct stride, then
// convert() does cvtColor → resize to target.  This prevents row-stride
// corruption: the SHM has source_width×3 bytes per row, NOT target_width×3.
//
// New SDK: hikcamera::SharedFrameReader wraps the SHM ring (latest-frame
// broadcast).  open() = reader_.open(name); wait_next() = reader_.wait_next()
// returning a SharedFrame lease whose mat() is a BGR8 non-owning view.
// Gray conversion uses COLOR_BGR2GRAY (new SDK outputs BGR8; legacy RGB8
// protocol and completed-slot polling are gone).

#include "radar_fast_livo2/shm_camera.hpp"

#include <chrono>
#include <cmath>
#include <thread>

#include <opencv2/imgproc.hpp>

namespace radar::fast_livo2 {

// ══════════════════════════════════════════════════════════════════
// Constructor / destructor
// ══════════════════════════════════════════════════════════════════

ShmCamera::ShmCamera(std::string shm_name, int source_width, int source_height, int target_width,
    int target_height, double img_time_offset)
    : shm_name_(std::move(shm_name))
    , source_width_(source_width)
    , source_height_(source_height)
    , target_width_(target_width)
    , target_height_(target_height)
    , img_time_offset_(img_time_offset) { }

ShmCamera::~ShmCamera() = default;

// ══════════════════════════════════════════════════════════════════
// open — open existing SHM ring via SharedFrameReader
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::open() -> std::expected<void, ShmCameraError> {
    if (source_width_ <= 0 || source_height_ <= 0) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: source dimensions must be positive" } };
    }
    if (target_width_ <= 0 || target_height_ <= 0) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: target dimensions must be positive" } };
    }

    // Validate source RGB image fits within MAX_IMAGE_SIZE
    const auto w               = static_cast<size_t>(source_width_);
    const auto h               = static_cast<size_t>(source_height_);
    constexpr size_t kChannels = 3;
    const size_t pixels        = w * h;
    if (h != 0 && pixels / h != w) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: source width * height overflow" } };
    }
    image_bytes_ = pixels * kChannels;
    if (image_bytes_ / kChannels != pixels) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: source width * height * 3 overflow" } };
    }
    if (image_bytes_ > MAX_IMAGE_SIZE) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: source RGB image size " + std::to_string(image_bytes_)
                + " exceeds MAX_IMAGE_SIZE " + std::to_string(MAX_IMAGE_SIZE) } };
    }

    auto open_ret = reader_.open(shm_name_.c_str());
    if (!open_ret.has_value()) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::ShmOpenFailed,
            "ShmCamera: open('" + shm_name_ + "') failed: " + open_ret.error() } };
    }
    is_open_ = true;
    return { };
}

auto ShmCamera::is_open() const noexcept -> bool { return is_open_; }

// ══════════════════════════════════════════════════════════════════
// wait_next — latest-frame read via SharedFrameReader
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::wait_next(std::chrono::milliseconds timeout)
    -> std::expected<CameraFrame, ShmCameraError> {
    if (!is_open_) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: not open — call open() first" } };
    }

    auto frame_ret = reader_.wait_next(timeout);
    if (!frame_ret.has_value()) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::Timeout, "ShmCamera: wait_next timeout" } };
    }
    const auto& frame = *frame_ret;
    if (!frame.valid()) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: invalid frame lease" } };
    }

    // mat() is a BGR8 non-owning view over SHM pixels; clone once so the
    // conversion below is safe after the lease is dropped.
    cv::Mat bgr = frame.mat().clone();
    if (bgr.empty()) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::InvalidFrame, "ShmCamera: empty BGR frame" } };
    }

    // convert() expects RGB — SharedFrameReader delivers BGR8.  Convert to
    // RGB first so convert()'s COLOR_RGB2GRAY and source-stride contract hold.
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    const uint64_t host_ns = frame.metadata().host_monotonic_ns;

    CameraFrame out = convert(rgb, source_width_, source_height_, target_width_, target_height_,
        host_ns, /*frame_id=*/0, frame.sequence(), img_time_offset_);
    if (out.gray.empty()) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: conversion produced empty frame (invalid SHM source)" } };
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════
// convert — pure function, testable without hardware
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::convert(const cv::Mat& rgb, int source_width, int source_height, int target_width,
    int target_height, uint64_t host_monotonic_ns, uint64_t frame_id, uint64_t committed_sequence,
    double img_time_offset) -> CameraFrame {
    CameraFrame frame;

    if (rgb.empty() || rgb.type() != CV_8UC3 || source_width <= 0 || source_height <= 0
        || target_width <= 0 || target_height <= 0) {
        frame.sequence          = committed_sequence;
        frame.host_monotonic_ns = host_monotonic_ns;
        frame.frame_id          = frame_id;
        frame.timestamp_seconds = static_cast<double>(host_monotonic_ns) * 1e-9 + img_time_offset;
        return frame;
    }

    // Verify the input matches the declared source dimensions.
    if (rgb.cols != source_width || rgb.rows != source_height) {
        frame.sequence          = committed_sequence;
        frame.host_monotonic_ns = host_monotonic_ns;
        frame.frame_id          = frame_id;
        frame.timestamp_seconds = static_cast<double>(host_monotonic_ns) * 1e-9 + img_time_offset;
        return frame;
    }

    // RGB → gray at source resolution, then resize to target.
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);

    // Resize to target resolution (only if needed)
    if (gray.cols == target_width && gray.rows == target_height) {
        frame.gray = std::move(gray);
    } else {
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
        frame.gray = std::move(resized);
    }

    // Metadata
    frame.sequence          = committed_sequence;
    frame.host_monotonic_ns = host_monotonic_ns;
    frame.frame_id          = frame_id;

    // Timestamp: steady-clock epoch seconds + offset
    frame.timestamp_seconds = static_cast<double>(host_monotonic_ns) * 1e-9 + img_time_offset;

    return frame;
}

} // namespace radar::fast_livo2
