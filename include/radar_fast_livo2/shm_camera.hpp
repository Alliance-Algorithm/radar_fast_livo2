#pragma once
// shm_camera.hpp — Hik SHM camera adapter for radar_fast_livo2
//
// Replaces ROS Image subscription with a SharedFrameReader-backed camera
// thread.  CameraFrame holds an owned grayscale cv::Mat at the target
// resolution.
//
// Timestamp: host_monotonic_ns is std::chrono::steady_clock epoch, NOT
// the Odin device clock or ROS clock.  img_time_offset must be measured
// to map the two clock domains when running LIVO mode.  Default 0.0 is
// a placeholder; use calibration to determine the actual offset.
//
// Usage (production):
//   ShmCamera cam("/hikcamera_shm", 2736, 1824, 0.0);
//   if (auto ok = cam.open(); !ok) { /* fatal */ }
//   while (true) {
//       auto frame = cam.wait_next(100ms);
//       if (frame) { queue.push(*frame); }
//   }
//
// Usage (test):
//   CameraFrame f = ShmCamera::convert(bgr_mat, tw, th, meta, offset);

#include <chrono>
#include <cstdint>
#include <expected>
#include <string>

#include <opencv2/core/mat.hpp>

#include "hikcamera/shared_frame.hpp"
#include "hikcamera/shared_frame_reader.hpp"
#include "hikcamera/shm_types.hpp"

namespace radar::fast_livo2 {

// ── ShmCameraError: typed error for ShmCamera operations ────────────
// Wraps hikcamera::FrameReadErrorCode so consumers can branch on code
// instead of fragile string matching.  Reader errors propagate with
// their original code; camera-specific failures use InvalidFrame.
struct ShmCameraError {
    hikcamera::FrameReadErrorCode code;
    std::string message;
};

// ── CameraFrame: one converted grayscale frame with metadata ──────────
struct CameraFrame {
    cv::Mat gray;                     // owned CV_8UC1 at target resolution
    uint64_t sequence { 0 };          // SHM committed_sequence (strictly increasing)
    uint64_t host_monotonic_ns { 0 }; // steady-clock epoch nanoseconds
    uint64_t frame_id { 0 };          // device frame counter

    double timestamp_seconds { 0.0 }; // host_monotonic_ns/1e9 + img_time_offset
};

// ── ShmCamera: RAII adapter over hikcamera::SharedFrameReader ─────────
//
// Opens an existing POSIX SHM ring (created by a hikcamera::SharedFrameWriter).
// Each wait_next() blocks until a new committed sequence is available, then
// converts the leased BGR8 pixels directly into an owned CV_8UC1 grayscale
// frame at the configured target resolution.
class ShmCamera {
public:
    /// @param shm_name      POSIX SHM segment name (e.g. "/hikcamera_shm")
    /// @param target_width  desired grayscale output width
    /// @param target_height desired grayscale output height
    /// @param img_time_offset additive offset applied to host_monotonic_ns/1e9
    ShmCamera(std::string shm_name, int target_width, int target_height, double img_time_offset);

    ShmCamera(const ShmCamera&)            = delete;
    ShmCamera& operator=(const ShmCamera&) = delete;
    ShmCamera(ShmCamera&&)                 = default;
    ShmCamera& operator=(ShmCamera&&)      = default;
    ~ShmCamera();

    /// Open the SHM ring.  Validates target dimensions are positive before
    /// attempting to open; fails with InvalidFrame if width or height <= 0.
    [[nodiscard]] auto open() -> std::expected<void, ShmCameraError>;

    /// Block until a new frame arrives, convert it, and return.
    /// `timeout` uses CLOCK_MONOTONIC.  Returns ShmCameraError with the
    /// underlying FrameReadErrorCode so callers can branch on code.
    [[nodiscard]] auto wait_next(std::chrono::milliseconds timeout = std::chrono::milliseconds {
                                     2000 }) -> std::expected<CameraFrame, ShmCameraError>;

    [[nodiscard]] auto is_open() const noexcept -> bool;

    /// Pure conversion: BGR8 cv::Mat → owned CV_8UC1 at target resolution.
    /// No hardware required — callable from tests.
    /// Allocates source-resolution gray via cvtColor then resize to target;
    /// no full BGR clone is made.
    [[nodiscard]] static auto convert(const cv::Mat& bgr, int target_width, int target_height,
        const hikcamera::FrameMetadata& meta, double img_time_offset) -> CameraFrame;

private:
    std::string shm_name_;
    int target_width_ { 0 };
    int target_height_ { 0 };
    double img_time_offset_ { 0.0 };
    hikcamera::SharedFrameReader reader_;
    bool is_open_ { false };
};

} // namespace radar::fast_livo2
