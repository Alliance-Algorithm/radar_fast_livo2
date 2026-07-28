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
    EXPECT_TRUE(SO3d::exp(Eigen::Vector3d::Zero()).isApprox(
        Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE(SO3d::log(Eigen::Matrix3d::Identity()).isApprox(
        Eigen::Vector3d::Zero(), 1e-12));

    const Eigen::Vector3d phi(0.2, -0.1, 0.3);
    EXPECT_TRUE(SO3d::log(SO3d::exp(phi)).isApprox(phi, 1e-12));
}

TEST(SO3Math, SupportsFloat) {
    const Eigen::Vector3f phi(0.2F, -0.1F, 0.3F);
    EXPECT_TRUE(SO3f::log(SO3f::exp(phi)).isApprox(phi, 1e-5F));
}

TEST(SO3Math, LogNearPiIsFinite) {
    const double pi = std::acos(-1.0);
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, -3.0).normalized();
    const Eigen::Vector3d recovered =
        SO3d::log(SO3d::exp(axis * (pi - 1e-10)));

    EXPECT_TRUE(recovered.allFinite());
    EXPECT_NEAR(recovered.norm(), pi, 1e-8);
    EXPECT_NEAR(std::abs(recovered.normalized().dot(axis)), 1.0, 1e-8);
}

TEST(SO3Math, AlignsSameOrdinaryAndOppositeVectors) {
    const Eigen::Vector3d x = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d y = Eigen::Vector3d::UnitY();

    EXPECT_TRUE(SO3d::fromTwoVectors(x, x).isApprox(
        Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE((SO3d::fromTwoVectors(x, y) * x).isApprox(y, 1e-12));
    EXPECT_TRUE((SO3d::fromTwoVectors(x, -x) * x).isApprox(-x, 1e-12));
}

}  // namespace
