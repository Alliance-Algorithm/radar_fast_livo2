// camera_frame_queue.cpp — CameraFrameQueue implementation

#include "radar_fast_livo2/camera_frame_queue.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace radar::fast_livo2 {

CameraFrameQueue::CameraFrameQueue(size_t max_size)
    : max_size_(max_size) { }

auto CameraFrameQueue::push(CameraFrame frame) -> bool {
    std::lock_guard<std::mutex> lock(mutex_);

    if (max_size_ == 0) return false;
    if (frame.sequence == 0) return false;
    if (frame.sequence <= last_accepted_sequence_) return false;

    last_accepted_sequence_ = frame.sequence;
    frames_.push_back(std::move(frame));
    if (frames_.size() > max_size_) frames_.pop_front();
    return true;
}

auto CameraFrameQueue::take_nearest(double target_time, double tolerance_sec)
    -> std::optional<CameraFrame> {
    std::lock_guard<std::mutex> lock(mutex_);

    if (frames_.empty()) return std::nullopt;

    size_t best_idx = 0;
    double best_dt  = std::numeric_limits<double>::max();
    for (size_t i = 0; i < frames_.size(); ++i) {
        double dt = std::abs(frames_[i].timestamp_seconds - target_time);
        if (dt < best_dt) {
            best_dt  = dt;
            best_idx = i;
        }
    }

    if (best_dt > tolerance_sec) return std::nullopt;

    CameraFrame selected = std::move(frames_[best_idx]);
    frames_.erase(frames_.begin(), frames_.begin() + static_cast<long>(best_idx) + 1);
    return selected;
}

auto CameraFrameQueue::oldest_timestamp() const -> std::optional<double>
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (frames_.empty()) return std::nullopt;
    return frames_.front().timestamp_seconds;
}

auto CameraFrameQueue::nearest_timestamp(double target_time) const -> std::optional<double>
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (frames_.empty()) return std::nullopt;

    double best_time = frames_.front().timestamp_seconds;
    double best_dt   = std::abs(best_time - target_time);
    for (const auto& frame : frames_) {
        const double dt = std::abs(frame.timestamp_seconds - target_time);
        if (dt < best_dt) {
            best_dt   = dt;
            best_time = frame.timestamp_seconds;
        }
    }
    return best_time;
}

auto CameraFrameQueue::take_oldest() -> std::optional<CameraFrame>
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (frames_.empty()) return std::nullopt;

    CameraFrame selected = std::move(frames_.front());
    frames_.pop_front();
    return selected;
}

void CameraFrameQueue::set_max_size(size_t max_size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    max_size_ = max_size;
    while (frames_.size() > max_size_) frames_.pop_front();
}

auto CameraFrameQueue::empty() const -> bool {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_.empty();
}

auto CameraFrameQueue::size() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_.size();
}

} // namespace radar::fast_livo2
