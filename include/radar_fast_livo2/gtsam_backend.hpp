#pragma once
// gtsam_backend.hpp - Lightweight GTSAM ISAM2 loop closure backend
//
// Drop-in replacement for KISS-Matcher-SAM.  Uses GTSAM ISAM2 with
// identity loop edges (spatial proximity = same place).
//
// Designed for sub-50m paths with <1% LIO drift (RoboMaster field).

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <deque>
#include <vector>

namespace radar::fast_livo2 {

class GtsamBackend {
public:
    struct Config {
        double keyframe_dist = 0.5;   // meters between keyframes
        double keyframe_ang  = 15.0;   // degrees between keyframes
        double loop_radius   = 1.0;    // loop closure search radius (m)
        int    loop_min_skip = 20;     // minimum keyframe index gap
    };

    explicit GtsamBackend(const Config& cfg);

    /// Add an odometry pose. Returns true if this was a new keyframe.
    bool add_odometry(const gtsam::Pose3& pose, double timestamp);

    /// Try to add a loop closure at this pose. Returns number of loops found.
    int try_loop_closure(const gtsam::Pose3& pose, int kf_idx);

    /// Run ISAM2 and return the latest corrected pose.
    /// If no update needed, returns the input pose unchanged.
    gtsam::Pose3 optimize();

    /// Get corrected pose for a specific keyframe index.
    gtsam::Pose3 pose_at(int idx) const;

    /// Number of keyframes stored.
    size_t num_keyframes() const { return keyframes_.size(); }

    /// Number of loop edges added.
    int num_loops() const { return loop_count_; }

    /// Total optimized edges.
    int num_edges() const { return edge_count_; }

    /// Latest keyframe index.
    int latest_idx() const { return latest_idx_; }

    /// Access raw keyframe data: (idx, Pose3, timestamp)
    const std::vector<std::tuple<int, gtsam::Pose3, double>>& keyframes() const {
        return keyframes_;
    }

private:
    void update_isam();

    Config cfg_;
    gtsam::ISAM2 isam_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_;
    gtsam::Values result_;
    int latest_idx_ = 0;
    int edge_count_ = 0;
    int loop_count_ = 0;
    bool need_update_ = false;

    std::vector<std::tuple<int, gtsam::Pose3, double>> keyframes_;
    gtsam::Pose3 last_keyframe_pose_;
};

} // namespace radar::fast_livo2
