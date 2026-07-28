#include "radar_fast_livo2/lie.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using radar::fast_livo2::lie::SO3d;
using radar::fast_livo2::lie::SO3f;

TEST(SO3Math, HatMatchesCrossProduct) {
    const Eigen::Vector3d v(1.0, -2.0, 3.0);
    const Eigen::Vector3d w(-4.0, 5.0, 0.5);
    EXPECT_TRUE((SO3d::hat(v) * w).isApprox(v.cross(w), 1e-12));
}

TEST(SO3Math, IdentityAndRoundTrip) {
    EXPECT_TRUE(SO3d::exp(Eigen::Vector3d::Zero()).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE(SO3d::log(Eigen::Matrix3d::Identity()).isApprox(Eigen::Vector3d::Zero(), 1e-12));

    const Eigen::Vector3d phi(0.2, -0.1, 0.3);
    EXPECT_TRUE(SO3d::log(SO3d::exp(phi)).isApprox(phi, 1e-12));
}

TEST(SO3Math, SupportsFloat) {
    const Eigen::Vector3f phi(0.2F, -0.1F, 0.3F);
    EXPECT_TRUE(SO3f::log(SO3f::exp(phi)).isApprox(phi, 1e-5F));
}

TEST(SO3Math, LogNearPiIsFinite) {
    const double pi                        = std::acos(-1.0);
    const Eigen::Vector3d axis             = Eigen::Vector3d(1.0, 2.0, -3.0).normalized();
    const Eigen::Vector3d recovered        = SO3d::log(SO3d::exp(axis * (pi - 1e-10)));
    const Eigen::Matrix3d near_pi_rotation = SO3d::exp(axis * (pi - 1e-6));
    const Eigen::Vector3d near_pi_log      = SO3d::log(near_pi_rotation);

    EXPECT_TRUE(recovered.allFinite());
    EXPECT_NEAR(recovered.norm(), pi, 1e-8);
    EXPECT_NEAR(std::abs(recovered.normalized().dot(axis)), 1.0, 1e-8);
    EXPECT_TRUE(SO3d::exp(near_pi_log).isApprox(near_pi_rotation, 1e-10));
}

TEST(SO3Math, AlignsSameOrdinaryAndOppositeVectors) {
    const Eigen::Vector3d x = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d y = Eigen::Vector3d::UnitY();

    EXPECT_TRUE(SO3d::fromTwoVectors(x, x).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE((SO3d::fromTwoVectors(x, y) * x).isApprox(y, 1e-12));
    EXPECT_TRUE((SO3d::fromTwoVectors(x, -x) * x).isApprox(-x, 1e-12));

    const Eigen::Vector3f float_from(0.468309999F, 0.698848188F, 0.540644944F);
    const Eigen::Vector3f float_to(-0.468309939F, -0.698848248F, -0.540645063F);
    const Eigen::Vector3f float_mapped =
        SO3f::fromTwoVectors(float_from, float_to) * float_from.normalized();
    EXPECT_NEAR((float_mapped - float_to.normalized()).norm(), 0.0F, 1e-5F);

    const Eigen::Vector3d double_from(0.468309999, 0.698848188, 0.540644944);
    const Eigen::Vector3d double_to(-0.468309939, -0.698848248, -0.540645063);
    const Eigen::Vector3d double_mapped =
        SO3d::fromTwoVectors(double_from, double_to) * double_from.normalized();
    EXPECT_NEAR((double_mapped - double_to.normalized()).norm(), 0.0, 1e-10);
}

} // namespace
