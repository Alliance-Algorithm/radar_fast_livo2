// test_lio_drift_diagnostics.cpp — Unit tests for LIO drift metrics computation
//
// Tests the pure compute_lio_drift_metrics() helper against known geometries:
//   - Identity state: all zero displacements
//   - Known translation/rotation/bias: verify each metric's numerical value
//
// RED-GREEN contract:
//   Before the header exists, this file won't compile (RED).
//   After lio_drift_diagnostics.hpp is implemented, these tests pass (GREEN).

#include <cmath>
#include <gtest/gtest.h>

#include "radar_fast_livo2/esikf_state.hpp"
#include "radar_fast_livo2/lio_drift_diagnostics.hpp"

namespace radar::fast_livo2 {

// ══════════════════════════════════════════════════════════════════
// Zero-motion identity state → every metric should be zero
// ══════════════════════════════════════════════════════════════════

TEST(LioDriftMetricsTest, AllZeroForIdentityState) {
    StatesGroup posterior;
    StatesGroup prior;
    // Both at identity: same pos=0, rot=I, vel=0, bias=0
    V3D ref_pos = V3D::Zero();

    auto m = compute_lio_drift_metrics(42, posterior, prior, ref_pos);

    EXPECT_EQ(m.frame_count, 42);
    EXPECT_NEAR(m.displacement, 0.0, 1e-10);
    EXPECT_NEAR(m.speed_norm, 0.0, 1e-10);
    EXPECT_NEAR(m.pos_correction, 0.0, 1e-10);
    EXPECT_NEAR(m.ang_correction, 0.0, 1e-10);
    EXPECT_NEAR(m.gyro_bias_norm, 0.0, 1e-10);
    EXPECT_NEAR(m.accel_bias_norm, 0.0, 1e-10);
}

// ══════════════════════════════════════════════════════════════════
// Known offsets — verify each metric independently
// ══════════════════════════════════════════════════════════════════

TEST(LioDriftMetricsTest, KnownDisplacementAndSpeed) {
    StatesGroup posterior;
    StatesGroup prior;

    // Posterior moved to (1, 2, 3) with velocity (0.3, 0.4, 0.0)
    posterior.pos_end = V3D(1.0, 2.0, 3.0);
    posterior.vel_end = V3D(0.3, 0.4, 0.0);

    // Reference is origin → displacement = sqrt(1+4+9) = sqrt(14) ≈ 3.741657
    // Speed norm = sqrt(0.09+0.16) = 0.5
    V3D ref_pos = V3D::Zero();

    auto m = compute_lio_drift_metrics(1, posterior, prior, ref_pos);

    EXPECT_NEAR(m.displacement, std::sqrt(14.0), 1e-6);
    EXPECT_NEAR(m.speed_norm, 0.5, 1e-6);
}

TEST(LioDriftMetricsTest, KnownPositionCorrection) {
    StatesGroup posterior;
    StatesGroup prior;

    // Prior at (1, 0, 0), posterior at (3, 0, 0)
    // → correction = 2.0
    prior.pos_end     = V3D(1.0, 0.0, 0.0);
    posterior.pos_end = V3D(3.0, 0.0, 0.0);
    // Same rotation → no angular correction
    prior.rot_end     = M3D::Identity();
    posterior.rot_end = M3D::Identity();

    auto m = compute_lio_drift_metrics(2, posterior, prior, V3D::Zero());

    EXPECT_NEAR(m.pos_correction, 2.0, 1e-6);
    EXPECT_NEAR(m.ang_correction, 0.0, 1e-6);
}

TEST(LioDriftMetricsTest, KnownAngularCorrectionDegrees) {
    StatesGroup posterior;
    StatesGroup prior;

    // Prior has identity rotation, posterior has 2° around Z
    const double angle_rad = 2.0 * M_PI / 180.0;
    prior.rot_end          = M3D::Identity();
    posterior.rot_end      = Eigen::AngleAxisd(angle_rad, V3D::UnitZ()).toRotationMatrix();
    // Same position → no position correction
    prior.pos_end     = V3D(0, 0, 0);
    posterior.pos_end = V3D(0, 0, 0);

    auto m = compute_lio_drift_metrics(3, posterior, prior, V3D::Zero());

    EXPECT_NEAR(m.pos_correction, 0.0, 1e-6);
    EXPECT_NEAR(m.ang_correction, 2.0, 1e-6); // 2 degrees exactly
}

TEST(LioDriftMetricsTest, KnownBiasNorms) {
    StatesGroup posterior;
    StatesGroup prior;

    posterior.bias_g = V3D(0.01, 0.0, 0.0); // norm = 0.01
    posterior.bias_a = V3D(0.0, 0.02, 0.0); // norm = 0.02

    auto m = compute_lio_drift_metrics(4, posterior, prior, V3D::Zero());

    EXPECT_NEAR(m.gyro_bias_norm, 0.01, 1e-8);
    EXPECT_NEAR(m.accel_bias_norm, 0.02, 1e-8);
}

// ══════════════════════════════════════════════════════════════════
// Covariance diagonals: test that values flow through correctly
// ══════════════════════════════════════════════════════════════════

TEST(LioDriftMetricsTest, CovarianceDiagonalsSurface) {
    StatesGroup posterior;
    StatesGroup prior;

    // Set known diagonal values
    posterior.cov(0, 0)   = 1.0e-4; // rot x
    posterior.cov(3, 3)   = 2.0e-3; // pos x
    posterior.cov(6, 6)   = 3.0e-2; // vel x
    posterior.cov(9, 9)   = 4.0e-5; // bg x
    posterior.cov(12, 12) = 5.0e-6; // ba x

    auto m = compute_lio_drift_metrics(5, posterior, prior, V3D::Zero());

    EXPECT_NEAR(m.cov_rot, 1.0e-4, 1e-12);
    EXPECT_NEAR(m.cov_pos, 2.0e-3, 1e-12);
    EXPECT_NEAR(m.cov_vel, 3.0e-2, 1e-12);
    EXPECT_NEAR(m.cov_bg, 4.0e-5, 1e-12);
    EXPECT_NEAR(m.cov_ba, 5.0e-6, 1e-12);
}

// ══════════════════════════════════════════════════════════════════
// Prior-dominates scenario: posterior ≈ prior → small corrections
// ══════════════════════════════════════════════════════════════════

TEST(LioDriftMetricsTest, SmallCorrectionWhenPosteriorCloseToPrior) {
    StatesGroup posterior;
    StatesGroup prior;

    // Prior and posterior almost the same: tiny dP, tiny dR
    prior.pos_end         = V3D(10.0, 10.0, 10.0);
    prior.rot_end         = M3D::Identity();
    posterior.pos_end     = V3D(10.001, 10.000, 10.000); // 1 mm offset
    const double tiny_deg = 0.01 * M_PI / 180.0;         // 0.01°
    posterior.rot_end     = Eigen::AngleAxisd(tiny_deg, V3D::UnitZ()).toRotationMatrix();

    auto m = compute_lio_drift_metrics(10, posterior, prior, V3D::Zero());

    // Displacement is from reference (origin), not from prior
    EXPECT_NEAR(m.displacement, std::sqrt(3.0 * 100.0), 0.002); // ≈ 17.32
    EXPECT_NEAR(m.pos_correction, 0.001, 1e-6);
    EXPECT_NEAR(m.ang_correction, 0.01, 1e-4);
}

} // namespace radar::fast_livo2

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
