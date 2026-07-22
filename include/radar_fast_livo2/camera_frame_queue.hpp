#pragma once
// camera_frame_queue.hpp — Production CameraFrame queue with at-most-once semantics
//
// Thread-safe bounded queue for CameraFrame.  Each SHM sequence enters
// at most once — even after bounded-capacity eviction or consumption via
// take_nearest().  Sequences must be strictly increasing per `last_accepted_sequence_`.
// Sequence 0 (uncommitted SHM slot) is rejected.

#include "radar_fast_livo2/shm_camera.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace radar::fast_livo2 {

class CameraFrameQueue {
public:
    explicit CameraFrameQueue(size_t max_size = 5);

    CameraFrameQueue(const CameraFrameQueue&)            = delete;
    CameraFrameQueue& operator=(const CameraFrameQueue&) = delete;

    /// Push a frame.  Returns false if the sequence is zero, not strictly
    /// greater than `last_accepted_sequence_`, or max_size_ is zero.
    /// Returns true and emplaces the frame on success; oldest frame is
    /// evicted if size exceeds max_size_.  `last_accepted_sequence_` is
    /// updated only on successful push, so an evicted (or consumed) sequence
    /// can never re-enter.
    [[nodiscard]] auto push(CameraFrame frame) -> bool;

    /// Find the frame nearest to `target_time` within `tolerance_sec`
    /// seconds, move its gray mat out, and erase it plus all older frames.
    /// Returns nullopt if no frame is within tolerance.
    [[nodiscard]] auto take_nearest(double target_time,
                                     double tolerance_sec) -> std::optional<CameraFrame>;

    [[nodiscard]] auto empty() const -> bool;
    [[nodiscard]] auto size() const -> size_t;
    [[nodiscard]] auto max_size() const noexcept -> size_t { return max_size_; }

private:
    size_t                         max_size_;
    uint64_t                       last_accepted_sequence_{0};
    mutable std::mutex             mutex_;
    std::deque<CameraFrame>        frames_;
};

}  // namespace radar::fast_livo2
