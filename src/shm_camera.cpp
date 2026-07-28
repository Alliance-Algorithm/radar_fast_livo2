// shm_camera.cpp — ShmCamera implementation using raw hikcamera::imageSHM
//
// Image-dimension contract: source_width/source_height describe the raw
// SHM RGB resolution (hikcamera.yaml → 5472×3648).  target_width/target_height
// describe the VIO working resolution (odin_livo2.yaml → cam_width/cam_height).
// wait_next() copies FULL source-resolution RGB with correct stride, then
// convert() does cvtColor → resize to target.  This prevents row-stride
// corruption: the SHM has source_width×3 bytes per row, NOT target_width×3.
//
// Protocol (same as reviewed radar_fast_livo2_rgb completed-slot pattern):
//   1. Poll frame_counter (acquire) without consuming semaphore.
//   2. Completed slot = (counter - 1) % SLOT_NUM.
//   3. memcpy RGB from SHM → owned local cv::Mat (one full copy required
//      before gray conversion outside SHM critical section).
//   4. Post-copy frame_counter stability check.
//   5. Bounded retry (3 attempts) on instability; drop frame on exhaustion.
//   6. Convert the already-RGB SHM bytes to gray outside the SHM section
//      (cvtColor + resize). No RGB/BGR format conversion is performed here.
//
// The HIK capturer writes PixelType_Gvsp_RGB8_Packed, so raw SHM bytes
// are R,G,B per pixel. Gray conversion uses COLOR_RGB2GRAY. Replay ROS
// Image subscribers receive standard bgr8 and use COLOR_BGR2GRAY.
//
// Semaphore & mutex: NOT consumed/locked.  The hikcamera writer
// (SHMWrite) does not use the mutex for write-side protection, and
// other readers (e.g. radar_bridge) need the semaphore intact.

#include "radar_fast_livo2/shm_camera.hpp"

#include <chrono>
#include <cmath>
#include <thread>

#include <opencv2/imgproc.hpp>

namespace radar::fast_livo2 {

// ══════════════════════════════════════════════════════════════════
// Internal helpers — zero SHM dependency, testable
// ══════════════════════════════════════════════════════════════════

namespace {

/// Completed slot protocol: the writer increments write_index BEFORE
/// writing data but release-stores frame_counter AFTER data+timestamp
/// are complete.  After observing counter N (N>0), slot (N-1)%SLOT_NUM
/// is guaranteed to hold a fully-written frame.
[[nodiscard]] inline auto completed_slot(uint64_t counter, unsigned int slot_num) -> unsigned int {
    return static_cast<unsigned int>((counter - 1) % static_cast<uint64_t>(slot_num));
}

[[nodiscard]] inline auto is_valid_counter(uint64_t counter) -> bool { return counter > 0; }

[[nodiscard]] inline auto is_stable(uint64_t before, uint64_t after) -> bool {
    return before == after;
}

[[nodiscard]] inline auto has_advanced(uint64_t current, uint64_t last_seen) -> bool {
    return current > 0 && current != last_seen;
}

/// Convert steady_clock::time_point to nanoseconds since epoch.
[[nodiscard]] inline auto to_nanos(const std::chrono::steady_clock::time_point& tp) -> uint64_t {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
}

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════
// Constructor / destructor
// ══════════════════════════════════════════════════════════════════

ShmCamera::ShmCamera(
    std::string shm_name, int source_width, int source_height,
    int target_width, int target_height, double img_time_offset)
    : shm_name_(std::move(shm_name))
    , source_width_(source_width)
    , source_height_(source_height)
    , target_width_(target_width)
    , target_height_(target_height)
    , img_time_offset_(img_time_offset) { }

ShmCamera::~ShmCamera() {
    if (shm_ptr_ != nullptr) {
        std::ignore = hikcamera::SHMReleasePtr(shm_ptr_);
        shm_ptr_ = nullptr;
    }
    if (shm_fd_ != -1) {
        std::ignore = hikcamera::SHMClose(shm_fd_);
        shm_fd_ = -1;
    }
    is_open_ = false;
}

// ══════════════════════════════════════════════════════════════════
// open — persistent mapping, dimension + size validation
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::open() -> std::expected<void, ShmCameraError> {
    if (source_width_ <= 0 || source_height_ <= 0) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: source dimensions must be positive" } };
    }
    if (target_width_ <= 0 || target_height_ <= 0) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: target dimensions must be positive" } };
    }

    // Validate source RGB image fits within MAX_IMAGE_SIZE
    const auto w = static_cast<size_t>(source_width_);
    const auto h = static_cast<size_t>(source_height_);
    constexpr size_t kChannels = 3;
    const size_t pixels = w * h;
    if (h != 0 && pixels / h != w) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: source width * height overflow" } };
    }
    image_bytes_ = pixels * kChannels;
    if (image_bytes_ / kChannels != pixels) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: source width * height * 3 overflow" } };
    }
    if (image_bytes_ > MAX_IMAGE_SIZE) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: source RGB image size " + std::to_string(image_bytes_)
                + " exceeds MAX_IMAGE_SIZE " + std::to_string(MAX_IMAGE_SIZE) } };
    }

    auto fd_result = hikcamera::SHMInit(shm_name_, sizeof(hikcamera::imageSHM));
    if (!fd_result.has_value()) {
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::ShmOpenFailed,
            "ShmCamera: SHMInit('" + shm_name_ + "') failed: " + fd_result.error() } };
    }
    shm_fd_ = fd_result.value();

    auto ptr_result = hikcamera::SHMGetPtr(shm_fd_);
    if (!ptr_result.has_value()) {
        std::ignore = hikcamera::SHMClose(shm_fd_);
        shm_fd_ = -1;
        return std::unexpected { ShmCameraError {
            ShmCameraErrorCode::ShmOpenFailed,
            "ShmCamera: SHMGetPtr failed: " + ptr_result.error() } };
    }
    shm_ptr_ = ptr_result.value();

    is_open_ = true;
    return { };
}

auto ShmCamera::is_open() const noexcept -> bool { return is_open_; }

// ══════════════════════════════════════════════════════════════════
// wait_next — polling + completed-slot protocol
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::wait_next(std::chrono::milliseconds timeout)
    -> std::expected<CameraFrame, ShmCameraError> {
    if (!is_open_ || shm_ptr_ == nullptr) {
        return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
            "ShmCamera: not open — call open() first" } };
    }

    constexpr int kPollIntervalMs = 1;
    constexpr int kMaxCopyRetries = 3;
    using clock = std::chrono::steady_clock;

    const auto deadline = clock::now() + timeout;

    // ── Phase 1: poll until frame_counter advances ─────────────────
    uint64_t last_seen = shm_ptr_->frame_counter.load(std::memory_order_acquire);
    uint64_t poll_start = last_seen; // snapshot before poll — used for timeout predicate

    while (clock::now() < deadline) {
        uint64_t counter = shm_ptr_->frame_counter.load(std::memory_order_acquire);
        if (is_valid_counter(counter) && has_advanced(counter, last_seen)) {
            last_seen = counter;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
    }

    // Check if we timed out (no new frame since poll_start)
    {
        uint64_t counter_now = shm_ptr_->frame_counter.load(std::memory_order_acquire);
        if (!is_valid_counter(counter_now) || !has_advanced(counter_now, poll_start)) {
            return std::unexpected { ShmCameraError { ShmCameraErrorCode::Timeout,
                "ShmCamera: no new frame within timeout" } };
        }
        // Update last_seen to latest even if we didn't advance during poll
        if (counter_now != last_seen) {
            last_seen = counter_now;
        }
    }

    // ── Phase 2: copy from completed slot with stability check ─────
    uint64_t latest_seen = last_seen;

    for (int retry = 0; retry < kMaxCopyRetries; ++retry) {
        uint64_t counter_before = shm_ptr_->frame_counter.load(std::memory_order_acquire);
        if (!is_valid_counter(counter_before)) {
            // Counter wrapped to 0? Unlikely but handle gracefully.
            if (retry + 1 < kMaxCopyRetries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
                continue;
            }
            return std::unexpected { ShmCameraError { ShmCameraErrorCode::InvalidFrame,
                "ShmCamera: frame_counter is zero (uninitialized SHM?)" } };
        }

        latest_seen   = counter_before;
        unsigned int slot = completed_slot(counter_before, SLOT_NUM);

        // ── Copy RGB from SHM slot → owned local cv::Mat ─────────
        // Copy FULL source-resolution RGB (source_width × source_height)
        // so the cv::Mat stride matches the SHM stride.  The resize to
        // VIO target resolution happens later in convert().
        // Must be done inside the stability window (between
        // counter_before and counter_after reads).
        cv::Mat rgb(source_height_, source_width_, CV_8UC3);
        std::memcpy(rgb.data, shm_ptr_->imagedata[slot], image_bytes_);

        // ── Capture timestamp from completed slot ─────────────────
        uint64_t host_ns = to_nanos(shm_ptr_->timestamp[slot]);

        // ── Post-copy stability check ─────────────────────────────
        uint64_t counter_after = shm_ptr_->frame_counter.load(std::memory_order_acquire);
        latest_seen            = counter_after;

        if (is_stable(counter_before, counter_after)) {
            // ── Stable: convert outside SHM critical section ──────
            CameraFrame frame = convert(rgb, source_width_, source_height_,
                target_width_, target_height_,
                host_ns, /*frame_id=*/0, counter_before, img_time_offset_);
            if (frame.gray.empty()) {
                return std::unexpected { ShmCameraError {
                    ShmCameraErrorCode::InvalidFrame,
                    "ShmCamera: conversion produced empty frame (invalid SHM source)" } };
            }
            return frame;
        }

        // Writer moved: retry or drop
        if (retry + 1 < kMaxCopyRetries) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
        }
    }

    // Exhausted retries: drop this frame, log would go here
    return std::unexpected { ShmCameraError { ShmCameraErrorCode::Timeout,
        "ShmCamera: frame_counter unstable after " + std::to_string(kMaxCopyRetries)
            + " retries; frame dropped" } };
}

// ══════════════════════════════════════════════════════════════════
// convert — pure function, testable without hardware
// ══════════════════════════════════════════════════════════════════

auto ShmCamera::convert(const cv::Mat& rgb,
    int source_width, int source_height,
    int target_width, int target_height,
    uint64_t host_monotonic_ns, uint64_t frame_id, uint64_t committed_sequence,
    double img_time_offset) -> CameraFrame {
    CameraFrame frame;

    if (rgb.empty() || rgb.type() != CV_8UC3
        || source_width <= 0 || source_height <= 0
        || target_width <= 0 || target_height <= 0) {
        frame.sequence          = committed_sequence;
        frame.host_monotonic_ns = host_monotonic_ns;
        frame.frame_id          = frame_id;
        frame.timestamp_seconds =
            static_cast<double>(host_monotonic_ns) * 1e-9 + img_time_offset;
        return frame;
    }

    // Verify the input matches the declared source dimensions.
    if (rgb.cols != source_width || rgb.rows != source_height) {
        frame.sequence          = committed_sequence;
        frame.host_monotonic_ns = host_monotonic_ns;
        frame.frame_id          = frame_id;
        frame.timestamp_seconds =
            static_cast<double>(host_monotonic_ns) * 1e-9 + img_time_offset;
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
