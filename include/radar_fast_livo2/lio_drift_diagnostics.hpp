#pragma once
// lio_drift_diagnostics.hpp — Pure helpers for static LIO drift diagnostics
//
// Provides a single pure function compute_lio_drift_metrics() that computes
// all drift-related metrics from the ESIKF posterior and IMU prior states.
// No ROS, no side effects — trivially testable.
//
// Integration: call from livmapper_node.cpp after StateEstimation (posterior
// available) using the IMU-prior saved before StateEstimation. Capture the
// reference position once after lidar_map_inited_ transitions to true.

#include <cmath>
#include <numbers>

#include "radar_fast_livo2/esikf_state.hpp"

namespace radar::fast_livo2 {

/// All drift metrics for one frame, ready to log.
struct LioDriftMetrics {
    int frame_count;

    double displacement;     // [m] posterior position from first-post-init reference
    double total_path;       // [m] cumulative path length (sum of frame-to-frame deltas)
    double speed_norm;       // [m/s] posterior linear velocity norm
    double pos_correction;   // [m] IMU-prior → LiDAR-posterior position delta norm
    double ang_correction;   // [deg] IMU-prior → LiDAR-posterior rotation delta norm

    double gyro_bias_norm;   // [rad/s] gyroscope bias norm
    double accel_bias_norm;  // [m/s^2] accelerometer bias norm

    // Posterior covariance diagonal entries (representative subset)
    double cov_rot;  // rotation (index 0)
    double cov_pos;  // position (index 3)
    double cov_vel;  // velocity (index 6)
    double cov_bg;   // gyro bias (index 9)
    double cov_ba;   // accel bias (index 12)
};

/// Compute all LIO drift metrics from ESIKF state snapshots.
///
/// @param frame_count     Current frame number (1-indexed)
/// @param posterior       ESIKF posterior state (after StateEstimation)
/// @param prior           IMU-only prior state (saved before StateEstimation)
/// @param ref_position    Reference position captured after map init
/// @return                Populated LioDriftMetrics struct
[[nodiscard]] inline LioDriftMetrics compute_lio_drift_metrics(
    int frame_count,
    const StatesGroup& posterior,
    const StatesGroup& prior,
    const V3D& ref_position) noexcept
{
    LioDriftMetrics m{};
    m.frame_count = frame_count;

    // ── Displacement from first-post-init reference ──
    m.displacement = (posterior.pos_end - ref_position).norm();

    // ── Linear speed ──
    m.speed_norm = posterior.vel_end.norm();

    // ── IMU-prior → LiDAR-posterior corrections ──
    m.pos_correction = (posterior.pos_end - prior.pos_end).norm();

    // Rotation delta: Log(prior_R^T * posterior_R) → axis-angle vector
    const M3D rot_diff = prior.rot_end.transpose() * posterior.rot_end;
    m.ang_correction   = Log(rot_diff).norm() * 180.0 / std::numbers::pi;

    // ── Bias norms ──
    m.gyro_bias_norm  = posterior.bias_g.norm();
    m.accel_bias_norm = posterior.bias_a.norm();

    // ── Covariance diagonal subset ──
    m.cov_rot = posterior.cov(0, 0);
    m.cov_pos = posterior.cov(3, 3);
    m.cov_vel = posterior.cov(6, 6);
    m.cov_bg  = posterior.cov(9, 9);
    m.cov_ba  = posterior.cov(12, 12);

    return m;
}

} // namespace radar::fast_livo2
