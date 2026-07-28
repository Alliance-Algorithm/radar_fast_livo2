#pragma once
// shm_camera.hpp — Hik SHM camera adapter for radar_fast_livo2
//
<<<<<<< HEAD
// Replaces ROS Image subscription with hikcamera POSIX SHM (shm.hpp /
// SHMRead). CameraFrame holds an owned grayscale cv::Mat at the target
// resolution.
=======
// Replaces ROS Image subscription with a raw hikcamera::imageSHM polling
// reader.  CameraFrame holds an owned grayscale cv::Mat at the VIO target
// resolution (target_width × target_height).
>>>>>>> feat/lio-tuning-safety-gate
//
// Image-dimension contract (single-source):
//   source_width / source_height — raw SHM RGB resolution, must match the
//     HIK driver output (hikcamera.yaml → 5472×3648).
//   target_width / target_height — VIO working resolution after downsample
//     (odin_livo2.yaml → cam_width/cam_height, e.g. 2736×1824).
//
<<<<<<< HEAD
// Usage (production):
//   ShmCamera cam("/hikcamera_shm", 2736, 1824, 0.0, 5472, 3648);
=======
//   wait_next() copies the FULL source-resolution RGB with correct stride,
//   then convert() produces grayscale at source resolution and resizes to
//   the VIO target.  This prevents row-stride corruption that would occur
//   if consumers treated source stride as target width × 3.
//
// Protocol: completed-frame-counter (same as radar_fast_livo2_rgb).
//   Slot = (frame_counter - 1) % SLOT_NUM  after acquire-load.
//   One full RGB copy from SHM → owned local storage; gray conversion
//   happens outside the SHM critical section.  Post-copy frame_counter
//   stability check with bounded retry.  Does NOT consume the shared
//   semaphore or lock the SHM mutex.
//
// The HIK capturer converts to PixelType_Gvsp_RGB8_Packed, so raw SHM
// bytes are R,G,B per pixel (NOT B,G,R).  All consumers must treat the
// source as RGB.
//
// Timestamp: host_monotonic_ns derived from shm_ptr_->timestamp[slot]
// (std::chrono::steady_clock epoch).  img_time_offset maps to ROS domain.
//
// Usage (production — live camera_input_mode: shm):
//   ShmCamera cam("/hikcamera_shm", /*source_w=*/5472, /*source_h=*/3648,
//                 /*target_w=*/2736, /*target_h=*/1824, 0.0);
>>>>>>> feat/lio-tuning-safety-gate
//   if (auto ok = cam.open(); !ok) { /* fatal */ }
//   while (true) {
//       auto frame = cam.wait_next(100ms);
//       if (frame) { queue.push(*frame); }
//   }
//
// Usage (test):
//   CameraFrame f = ShmCamera::convert(rgb_mat, sw, sh, tw, th,
//       host_monotonic_ns, frame_id, committed_sequence, offset);

#include <chrono>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>

#include <opencv2/core/mat.hpp>

<<<<<<< HEAD
=======
#include "hikcamera/shm.hpp"

    >>>>>>> feat/lio-tuning-safety-gate namespace radar::fast_livo2 {

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
<<<<<<< HEAD
    struct ShmCameraError {
        FrameReadErrorCode code;
=======
    enum class ShmCameraErrorCode {
        Timeout,
        InvalidFrame,
        ShmOpenFailed,
    };

    struct ShmCameraError {
        ShmCameraErrorCode code;
>>>>>>> feat/lio-tuning-safety-gate
        std::string message;
    };

    // ── CameraFrame: one converted grayscale frame with metadata ──────────
    struct CameraFrame {
        cv::Mat gray;                     // owned CV_8UC1 at target resolution
<<<<<<< HEAD
        uint64_t sequence { 0 };          // strictly increasing accept counter
        uint64_t host_monotonic_ns { 0 }; // steady-clock epoch nanoseconds
        uint64_t frame_id { 0 };          // same as sequence for this adapter
=======
        uint64_t sequence { 0 };          // SHM frame_counter value (strictly increasing)
        uint64_t host_monotonic_ns { 0 }; // steady-clock epoch nanoseconds
        uint64_t frame_id { 0 };          // currently always 0 (HIK SHM has no per-device frame ID)
>>>>>>> feat/lio-tuning-safety-gate

        double timestamp_seconds { 0.0 }; // host_monotonic_ns/1e9 + img_time_offset
    };

<<<<<<< HEAD
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
=======
    // ── ShmCamera: RAII adapter over raw hikcamera::imageSHM ─────────────
    //
    // Opens an existing POSIX SHM segment (created by hikcamera writer).
    // Persistently maps the segment once at open() — no per-frame mmap.
    // Each wait_next() polls frame_counter without consuming the semaphore.
    // Completed slot is (frame_counter - 1) % SLOT_NUM; RGB is memcpy'd
    // to owned storage, then gray-converted outside the SHM critical section.
    class ShmCamera {
    public:
        /// @param shm_name        POSIX SHM segment name (e.g. "/hikcamera_shm")
        /// @param source_width     raw SHM RGB width  (hikcamera.yaml, e.g. 5472)
        /// @param source_height    raw SHM RGB height (hikcamera.yaml, e.g. 3648)
        /// @param target_width     VIO grayscale output width  (odin_livo2.yaml, e.g. 2736)
        /// @param target_height    VIO grayscale output height (odin_livo2.yaml, e.g. 1824)
        /// @param img_time_offset  additive offset applied to host_monotonic_ns/1e9
        ShmCamera(std::string shm_name, int source_width, int source_height, int target_width,
            int target_height, double img_time_offset);
>>>>>>> feat/lio-tuning-safety-gate

        ShmCamera(const ShmCamera&)            = delete;
        ShmCamera& operator=(const ShmCamera&) = delete;
        ShmCamera(ShmCamera&&)                 = delete;
        ShmCamera& operator=(ShmCamera&&)      = delete;
        ~ShmCamera();

<<<<<<< HEAD
        /// Open the SHM segment (reader). Validates target dimensions are positive
        /// before attempting to open; fails with InvalidFrame if width or height <= 0.
        [[nodiscard]] auto open() -> std::expected<void, ShmCameraError>;

        /// Block until a new frame arrives (via hikcamera::SHMRead), convert it,
        /// and return.  `timeout` is best-effort: the SDK SHMRead uses a fixed
        /// ~1s semaphore wait; on failure we map the string error to a typed code.
=======
        /// Open the SHM segment (SHMInit + SHMGetPtr).  Validates target
        /// dimensions are positive and BGR image size ≤ MAX_IMAGE_SIZE.
        [[nodiscard]] auto open() -> std::expected<void, ShmCameraError>;

        /// Poll frame_counter until a new completed frame arrives, copy RGB
        /// from the completed slot, stability-check, convert to grayscale.
        /// `timeout` uses steady_clock (CLOCK_MONOTONIC).
>>>>>>> feat/lio-tuning-safety-gate
        [[nodiscard]] auto wait_next(std::chrono::milliseconds timeout = std::chrono::milliseconds {
                                         2000 }) -> std::expected<CameraFrame, ShmCameraError>;

        [[nodiscard]] auto is_open() const noexcept -> bool;

<<<<<<< HEAD
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
=======
        /// Pure conversion: RGB (CV_8UC3) cv::Mat at source resolution → owned
        /// CV_8UC1 grayscale at target resolution.  No hardware required —
        /// callable from tests.  Does cvtColor at source resolution then resize.
        [[nodiscard]] static auto convert(const cv::Mat& rgb, int source_width, int source_height,
            int target_width, int target_height, uint64_t host_monotonic_ns, uint64_t frame_id,
            uint64_t committed_sequence, double img_time_offset) -> CameraFrame;

    private:
        std::string shm_name_;
        int source_width_ { 0 };  // raw SHM RGB width  (hikcamera.yaml)
        int source_height_ { 0 }; // raw SHM RGB height (hikcamera.yaml)
        int target_width_ { 0 };  // VIO grayscale output width
        int target_height_ { 0 }; // VIO grayscale output height
        double img_time_offset_ { 0.0 };
        int shm_fd_ { -1 };                        // SHM file descriptor from SHMInit
        hikcamera::imageSHM* shm_ptr_ { nullptr }; // persistent mapping from SHMGetPtr
        size_t image_bytes_ { 0 };                 // expected RGB byte count (source_w * source_h * 3)
>>>>>>> feat/lio-tuning-safety-gate
        bool is_open_ { false };
        uint64_t next_sequence_ { 1 };
    };

} // namespace radar::fast_livo2
