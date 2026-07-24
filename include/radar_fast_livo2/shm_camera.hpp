#pragma once
// shm_camera.hpp — Hik SHM camera adapter for radar_fast_livo2
//
// Replaces ROS Image subscription with hikcamera POSIX SHM (shm.hpp /
// SHMRead). CameraFrame holds an owned grayscale cv::Mat at the target
// resolution.
//
// Timestamp: host_monotonic_ns is std::chrono::steady_clock epoch, NOT
// the Odin device clock or ROS clock.  img_time_offset must be measured
// to map the two clock domains when running LIVO mode.  Default 0.0 is
// a placeholder; use calibration to determine the actual offset.
//
// Usage (production):
//   ShmCamera cam("/hikcamera_shm", 2736, 1824, 0.0, 5472, 3648);
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

namespace radar::fast_livo2 {

// Local error codes — SDK exposes SHMRead as expected<void,string> only.
// Keep a typed surface so callers (and tests) can branch without string match.
enum class FrameReadErrorCode {
    Timeout,
    InvalidFrame,
    NotOpen,
    ShmError,
};

// Metadata carried from SHM into CameraFrame / convert().
// committed_sequence is a strictly increasing local counter (SDK SHMRead
// does not return frame_counter to the caller).
struct FrameMetadata {
    uint64_t host_monotonic_ns { 0 };
    uint64_t frame_id { 0 };
    uint64_t committed_sequence { 0 };
};

// ── ShmCameraError: typed error for ShmCamera operations ────────────
struct ShmCameraError {
    FrameReadErrorCode code;
    std::string message;
};

// ── CameraFrame: one converted grayscale frame with metadata ──────────
struct CameraFrame {
    cv::Mat gray;                     // owned CV_8UC1 at target resolution
    uint64_t sequence { 0 };          // strictly increasing accept counter
    uint64_t host_monotonic_ns { 0 }; // steady-clock epoch nanoseconds
    uint64_t frame_id { 0 };          // same as sequence for this adapter

    double timestamp_seconds { 0.0 }; // host_monotonic_ns/1e9 + img_time_offset
};

// ── ShmCamera: RAII adapter over hikcamera::SHMRead ───────────────────
//
// Opens an existing POSIX SHM segment (created by hikcamera_ros_driver /
// SHMInit+SHMWrite). Each wait_next() blocks on the SHM semaphore (SDK
// uses a ~1s timed wait internally), then converts BGR8 pixels into an
// owned CV_8UC1 grayscale frame at the configured target resolution.
class ShmCamera {
public:
    /// @param shm_name        POSIX SHM segment name (e.g. "/hikcamera_shm")
    /// @param target_width    desired grayscale output width
    /// @param target_height   desired grayscale output height
    /// @param img_time_offset additive offset applied to host_monotonic_ns/1e9
    /// @param source_width    SHM slot image width (writer layout; default CS200)
    /// @param source_height   SHM slot image height (writer layout; default CS200)
    ShmCamera(std::string shm_name, int target_width, int target_height, double img_time_offset,
        int source_width = 5472, int source_height = 3648);

    ShmCamera(const ShmCamera&)            = delete;
    ShmCamera& operator=(const ShmCamera&) = delete;
    ShmCamera(ShmCamera&&)                 = delete;
    ShmCamera& operator=(ShmCamera&&)      = delete;
    ~ShmCamera();

    /// Open the SHM segment (reader). Validates target dimensions are positive
    /// before attempting to open; fails with InvalidFrame if width or height <= 0.
    [[nodiscard]] auto open() -> std::expected<void, ShmCameraError>;

    /// Block until a new frame arrives (via hikcamera::SHMRead), convert it,
    /// and return.  `timeout` is best-effort: the SDK SHMRead uses a fixed
    /// ~1s semaphore wait; on failure we map the string error to a typed code.
    [[nodiscard]] auto wait_next(std::chrono::milliseconds timeout = std::chrono::milliseconds {
                                     2000 }) -> std::expected<CameraFrame, ShmCameraError>;

    [[nodiscard]] auto is_open() const noexcept -> bool;

    /// Pure conversion: BGR8 cv::Mat → owned CV_8UC1 at target resolution.
    /// No hardware required — callable from tests.
    /// Allocates source-resolution gray via cvtColor then resize to target;
    /// no full BGR clone is made.
    [[nodiscard]] static auto convert(const cv::Mat& bgr, int target_width, int target_height,
        const FrameMetadata& meta, double img_time_offset) -> CameraFrame;

private:
    std::string shm_name_;
    int target_width_ { 0 };
    int target_height_ { 0 };
    int source_width_ { 5472 };
    int source_height_ { 3648 };
    double img_time_offset_ { 0.0 };
    int shm_fd_ { -1 };
    bool is_open_ { false };
    uint64_t next_sequence_ { 1 };
};

} // namespace radar::fast_livo2
