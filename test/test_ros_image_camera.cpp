// test_ros_image_camera.cpp — Unit tests for RosImageCamera
//
// RosImageCamera tests (1-3): BGR8→gray conversion, header-stamp timestamp,
//   encoding rejection.
//
// Unlike ShmCamera tests, these tests verify ROS replay-mode semantics:
//   - Timestamp comes from image.header.stamp with NO offset applied
//   - Sequence is a locally-assigned monotonically-increasing counter
//   - Encoding must be exactly "bgr8"

#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "radar_fast_livo2/ros_image_camera.hpp"

using namespace radar::fast_livo2;

// ══════════════════════════════════════════════════════════════════
// RosImageCamera::convert tests
// ══════════════════════════════════════════════════════════════════

TEST(RosImageCamera, ConvertsBgr8AndUsesHeaderStampWithoutOffset) {
    auto image                 = sensor_msgs::msg::Image();
    image.header.stamp.sec     = 12;
    image.header.stamp.nanosec = 345000000;
    image.encoding             = "bgr8";
    image.width                = 2;
    image.height               = 2;
    image.step                 = 6;
    image.data                 = { 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255 };

    const auto frame = RosImageCamera::convert(image, 2, 2, 9);

    ASSERT_TRUE(frame.has_value());
    EXPECT_DOUBLE_EQ(frame->timestamp_seconds, 12.345);
    EXPECT_EQ(frame->sequence, 9U);
    EXPECT_EQ(frame->gray.type(), CV_8UC1);
}

TEST(RosImageCamera, RejectsNonBgr8Image) {
    sensor_msgs::msg::Image image;
    image.encoding = "mono8";
    EXPECT_FALSE(RosImageCamera::convert(image, 2, 2, 1).has_value());
}

TEST(RosImageCamera, GrayscaleValuesMatchCvtColor) {
    // 2×2 BGR8: Red=255, Green=255, Blue=255, White=255
    // BGR2GRAY = 0.114*B + 0.587*G + 0.299*R
    auto image                 = sensor_msgs::msg::Image();
    image.header.stamp.sec     = 1;
    image.header.stamp.nanosec = 0;
    image.encoding             = "bgr8";
    image.width                = 2;
    image.height               = 2;
    image.step                 = 6;
    // Pixel (0,0): B=0,   G=0,   R=255 → Red   → gray ≈ 76
    // Pixel (0,1): B=0,   G=255, R=0   → Green → gray ≈ 150
    // Pixel (1,0): B=255, G=0,   R=0   → Blue  → gray ≈ 29
    // Pixel (1,1): B=255, G=255, R=255 → White → gray = 255
    image.data = { 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255 };

    const auto frame = RosImageCamera::convert(image, 2, 2, 1);

    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->gray.at<uint8_t>(0, 0),
        static_cast<uint8_t>(0.299 * 255 + 0.5)); // Red → gray
    EXPECT_EQ(frame->gray.at<uint8_t>(0, 1),
        static_cast<uint8_t>(0.587 * 255 + 0.5)); // Green → gray
    EXPECT_EQ(frame->gray.at<uint8_t>(1, 0),
        static_cast<uint8_t>(0.114 * 255 + 0.5));  // Blue → gray
    EXPECT_EQ(frame->gray.at<uint8_t>(1, 1), 255); // White
}

TEST(RosImageCamera, SequenceIsMonotonicallyAssigned) {
    auto image     = sensor_msgs::msg::Image();
    image.encoding = "bgr8";
    image.width    = 1;
    image.height   = 1;
    image.step     = 3;
    image.data     = { 128, 128, 128 };

    const auto f1 = RosImageCamera::convert(image, 1, 1, 1);
    const auto f2 = RosImageCamera::convert(image, 1, 1, 42);
    const auto f3 = RosImageCamera::convert(image, 1, 1, 100);

    ASSERT_TRUE(f1.has_value());
    ASSERT_TRUE(f2.has_value());
    ASSERT_TRUE(f3.has_value());
    EXPECT_EQ(f1->sequence, 1U);
    EXPECT_EQ(f2->sequence, 42U);
    EXPECT_EQ(f3->sequence, 100U);
}

TEST(RosImageCamera, ResizeToTargetResolution) {
    auto image     = sensor_msgs::msg::Image();
    image.encoding = "bgr8";
    image.width    = 4;
    image.height   = 4;
    image.step     = 12; // 4 pixels * 3 channels
    image.data.resize(4 * 4 * 3, 128);

    const auto frame = RosImageCamera::convert(image, 2, 2, 1);

    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->gray.cols, 2);
    EXPECT_EQ(frame->gray.rows, 2);
    EXPECT_EQ(frame->gray.type(), CV_8UC1);
}

TEST(RosImageCamera, EmptyImageDataReturnsNullopt) {
    sensor_msgs::msg::Image image;
    image.encoding = "bgr8";
    image.width    = 10;
    image.height   = 10;
    image.step     = 30;
    // data left empty

    EXPECT_FALSE(RosImageCamera::convert(image, 10, 10, 1).has_value());
}

TEST(RosImageCamera, ZeroTargetDimensionsReturnNullopt) {
    auto image     = sensor_msgs::msg::Image();
    image.encoding = "bgr8";
    image.width    = 10;
    image.height   = 10;
    image.step     = 30;
    image.data.resize(10 * 10 * 3, 128);

    EXPECT_FALSE(RosImageCamera::convert(image, 0, 10, 1).has_value());
    EXPECT_FALSE(RosImageCamera::convert(image, 10, 0, 1).has_value());
    EXPECT_FALSE(RosImageCamera::convert(image, -1, 10, 1).has_value());
}

// ══════════════════════════════════════════════════════════════════
// RosImageCamera::validate_topic_not_empty tests (M3-1 regression)
// ══════════════════════════════════════════════════════════════════

TEST(RosImageCameraValidateTopic, ThrowsForRosImageModeWithEmptyTopic) {
    EXPECT_THROW(
        { RosImageCamera::validate_topic_not_empty("ros_image", ""); }, std::invalid_argument);
}

TEST(RosImageCameraValidateTopic, NoThrowForRosImageModeWithNonEmptyTopic) {
    EXPECT_NO_THROW(RosImageCamera::validate_topic_not_empty("ros_image", "/camera/bgr8"));
}

TEST(RosImageCameraValidateTopic, NoThrowForShmModeWithEmptyTopic) {
    EXPECT_NO_THROW(RosImageCamera::validate_topic_not_empty("shm", ""));
}

TEST(RosImageCameraValidateTopic, NoThrowForShmModeWithNonEmptyTopic) {
    EXPECT_NO_THROW(RosImageCamera::validate_topic_not_empty("shm", "/camera/bgr8"));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
