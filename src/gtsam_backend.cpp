// gtsam_backend.cpp - GTSAM ISAM2 loop closure backend implementation

#include "radar_fast_livo2/gtsam_backend.hpp"
#include <cmath>
#include <iostream>

namespace radar::fast_livo2 {

using gtsam::symbol_shorthand::X;

namespace {
    double point_distance(const gtsam::Point3& a, const gtsam::Point3& b) {
        return (a - b).norm();
    }
}

GtsamBackend::GtsamBackend(const Config& cfg) : cfg_(cfg) {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam_ = gtsam::ISAM2(params);
}

bool GtsamBackend::add_odometry(const gtsam::Pose3& pose, double timestamp) {
    bool is_keyframe = false;
    if (keyframes_.empty()) {
        is_keyframe = true;
    } else {
        double dist = point_distance(pose.translation(), last_keyframe_pose_.translation());
        double ang = pose.rotation().localCoordinates(
            last_keyframe_pose_.rotation()).norm();
        is_keyframe = (dist >= cfg_.keyframe_dist ||
                       ang >= cfg_.keyframe_ang * M_PI / 180.0);
    }

    if (!is_keyframe) return false;

    latest_idx_++;
    int idx = latest_idx_;
    keyframes_.emplace_back(idx, pose, timestamp);
    last_keyframe_pose_ = pose;

    // Prior on first keyframe
    if (idx == 1) {
        auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());
        graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(idx), pose, prior_noise));
        initial_.insert(X(idx), pose);
        need_update_ = true;
        return true;
    }

    // Odometry factor: 1% LIO drift → ~0.005m per 0.5m keyframe
    // Use 0.02m std to allow correction without over-trusting loop edges.
    auto& [prev_idx, prev_pose, _] = keyframes_[keyframes_.size() - 2];
    gtsam::Pose3 delta = prev_pose.between(pose);
    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 2e-2, 2e-2, 2e-2).finished());
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        X(prev_idx), X(idx), delta, odom_noise));
    initial_.insert(X(idx), pose);
    edge_count_++;
    need_update_ = true;

    return true;
}

int GtsamBackend::try_loop_closure(const gtsam::Pose3& pose, int kf_idx) {
    int found = 0;
    for (const auto& [candidate_idx, candidate_pose, _] : keyframes_) {
        if (kf_idx - candidate_idx < cfg_.loop_min_skip) continue;
        double dist = point_distance(pose.translation(), candidate_pose.translation());
        if (dist < cfg_.loop_radius) {
            // Validate: heading must agree within ~60° to avoid false positive
            // from crossing paths (same location, opposite direction)
            double ang = pose.rotation().localCoordinates(
                candidate_pose.rotation()).norm();
            if (ang > M_PI / 3.0) continue;  // >60° heading diff → skip

            auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 5e-2, 5e-2, 5e-2, 0.3, 0.3, 0.3).finished());
            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                X(kf_idx), X(candidate_idx), gtsam::Pose3(), loop_noise));
            loop_count_++;
            found++;
            need_update_ = true;
        }
    }
    return found;
}

gtsam::Pose3 GtsamBackend::optimize() {
    if (!need_update_ || graph_.empty()) {
        if (result_.exists(X(latest_idx_)))
            return result_.at<gtsam::Pose3>(X(latest_idx_));
        auto& [_, pose, _t] = keyframes_.back();
        return pose;
    }

    try {
        isam_.update(graph_, initial_);
        result_ = isam_.calculateEstimate();
        graph_ = gtsam::NonlinearFactorGraph();
        initial_ = gtsam::Values();
        need_update_ = false;
    } catch (const std::exception& e) {
        graph_ = gtsam::NonlinearFactorGraph();
        initial_ = gtsam::Values();
        need_update_ = false;
    }

    if (result_.exists(X(latest_idx_)))
        return result_.at<gtsam::Pose3>(X(latest_idx_));
    auto& [_, pose, _t] = keyframes_.back();
    return pose;
}

gtsam::Pose3 GtsamBackend::pose_at(int idx) const {
    if (result_.exists(X(idx)))
        return result_.at<gtsam::Pose3>(X(idx));
    for (const auto& [kf_idx, pose, _] : keyframes_)
        if (kf_idx == idx) return pose;
    return gtsam::Pose3();
}

} // namespace radar::fast_livo2
