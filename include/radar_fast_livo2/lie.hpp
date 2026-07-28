#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>

namespace radar::fast_livo2::lie {

template <typename Scalar> struct SO3Math {
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

    SO3Math() = delete;

    static Mat3 hat(const Vec3& v) {
        Mat3 result;
        result << Scalar(0), -v.z(), v.y(), v.z(), Scalar(0), -v.x(), -v.y(), v.x(), Scalar(0);
        return result;
    }

    static Mat3 exp(const Vec3& phi) {
        const Scalar theta_squared = phi.squaredNorm();
        const Scalar small_angle   = std::sqrt(std::numeric_limits<Scalar>::epsilon());
        const Mat3 K               = hat(phi);

        Scalar a;
        Scalar b;
        if (theta_squared < small_angle * small_angle) {
            a = Scalar(1) - theta_squared / Scalar(6);
            b = Scalar(0.5) - theta_squared / Scalar(24);
        } else {
            const Scalar theta = std::sqrt(theta_squared);
            a                  = std::sin(theta) / theta;
            b                  = (Scalar(1) - std::cos(theta)) / theta_squared;
        }

        return Mat3::Identity() + a * K + b * K * K;
    }

    template <typename Time> static Mat3 exp(const Vec3& angular_velocity, Time dt) {
        return exp(angular_velocity * static_cast<Scalar>(dt));
    }

    static Vec3 log(const Mat3& rotation) {
        const Scalar epsilon     = std::numeric_limits<Scalar>::epsilon();
        const Scalar small_angle = std::sqrt(epsilon);
        const Scalar cos_theta =
            std::clamp((rotation.trace() - Scalar(1)) / Scalar(2), Scalar(-1), Scalar(1));
        const Vec3 vee(rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
            rotation(1, 0) - rotation(0, 1));
        const Scalar vee_norm  = vee.norm();
        const Scalar sin_theta = Scalar(0.5) * vee_norm;
        const Scalar theta     = std::atan2(sin_theta, cos_theta);

        if (theta < small_angle) {
            return Scalar(0.5) * vee;
        }

        const Scalar pi = std::acos(Scalar(-1));
        if (pi - theta < small_angle) {
            const Mat3 symmetric          = rotation + Mat3::Identity();
            Eigen::Index largest_diagonal = 0;
            if (symmetric(1, 1) > symmetric(largest_diagonal, largest_diagonal)) {
                largest_diagonal = 1;
            }
            if (symmetric(2, 2) > symmetric(largest_diagonal, largest_diagonal)) {
                largest_diagonal = 2;
            }

            Vec3 axis              = Vec3::Zero();
            axis(largest_diagonal) = std::sqrt(
                std::max(Scalar(0), symmetric(largest_diagonal, largest_diagonal) / Scalar(2)));
            const Scalar denominator = Scalar(4) * axis(largest_diagonal);
            if (std::abs(denominator) < small_angle) {
                axis.setZero();
                axis(largest_diagonal) = Scalar(1);
            } else {
                for (Eigen::Index index = 0; index < 3; ++index) {
                    if (index != largest_diagonal) {
                        axis(index) =
                            (rotation(largest_diagonal, index) + rotation(index, largest_diagonal))
                            / denominator;
                    }
                }
            }

            const Scalar axis_norm = axis.norm();
            if (axis_norm < small_angle || !axis.allFinite()) {
                axis.setZero();
                axis(0) = Scalar(1);
            } else {
                axis /= axis_norm;
            }
            if (vee_norm > Scalar(16) * epsilon && axis.dot(vee) < Scalar(0)) {
                axis = -axis;
            }
            return theta * axis;
        }

        return theta / vee_norm * vee;
    }

    static Mat3 fromTwoVectors(const Vec3& from, const Vec3& to) {
        const Scalar small_angle = std::sqrt(std::numeric_limits<Scalar>::epsilon());
        const Scalar from_norm   = from.norm();
        const Scalar to_norm     = to.norm();
        if (from_norm < small_angle || to_norm < small_angle) {
            return Mat3::Identity();
        }

        const Vec3 from_normalized = from / from_norm;
        const Vec3 to_normalized   = to / to_norm;
        const Vec3 cross           = from_normalized.cross(to_normalized);
        const Scalar cosine = std::clamp(from_normalized.dot(to_normalized), Scalar(-1), Scalar(1));
        const Scalar cross_squared_norm = cross.squaredNorm();
        const Scalar epsilon            = std::numeric_limits<Scalar>::epsilon();

        if (cross_squared_norm <= epsilon * epsilon) {
            if (cosine >= Scalar(0)) {
                return Mat3::Identity();
            }

            Eigen::Index least_aligned = 0;
            if (std::abs(from_normalized(1)) < std::abs(from_normalized(least_aligned))) {
                least_aligned = 1;
            }
            if (std::abs(from_normalized(2)) < std::abs(from_normalized(least_aligned))) {
                least_aligned = 2;
            }

            Vec3 coordinate_axis           = Vec3::Zero();
            coordinate_axis(least_aligned) = Scalar(1);
            Vec3 axis                      = from_normalized.cross(coordinate_axis);
            axis.normalize();
            const Scalar pi = std::acos(Scalar(-1));
            return exp(axis * pi);
        }

        const Scalar anti_parallel_cross_threshold = Scalar(16) * small_angle;
        if (cosine < Scalar(0)
            && cross_squared_norm
                <= anti_parallel_cross_threshold * anti_parallel_cross_threshold) {
            using HighVec3                      = Eigen::Matrix<long double, 3, 1>;
            const HighVec3 from_high            = from.template cast<long double>();
            const HighVec3 to_high              = to.template cast<long double>();
            const HighVec3 from_high_normalized = from_high / from_high.norm();
            const HighVec3 to_high_normalized   = to_high / to_high.norm();
            const HighVec3 high_cross           = from_high_normalized.cross(to_high_normalized);
            const long double high_cross_norm   = high_cross.norm();
            if (high_cross_norm > std::numeric_limits<long double>::epsilon()) {
                const long double high_cosine =
                    std::clamp(from_high_normalized.dot(to_high_normalized), -1.0L, 1.0L);
                const long double high_angle = std::atan2(high_cross_norm, high_cosine);
                const Vec3 rotation_vector =
                    (high_cross / high_cross_norm * high_angle).template cast<Scalar>();
                return exp(rotation_vector);
            }
        }

        const Mat3 K = hat(cross);
        return Mat3::Identity() + K + K * K * ((Scalar(1) - cosine) / cross_squared_norm);
    }
};

using SO3d = SO3Math<double>;
using SO3f = SO3Math<float>;

} // namespace radar::fast_livo2::lie
