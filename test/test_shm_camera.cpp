// test_shm_camera.cpp — Unit tests for ShmCamera + CameraFrameQueue
//
// ShmCamera tests (1-4): geometry, grayscale conversion, timestamp, metadata
// ShmCamera boundary tests (5-6): invalid input rejection
// CameraFrameQueue tests (7-14): push, duplicate, eviction, nearest selection,
//   tolerance, consume-through, post-consume rejection

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

#include "radar_fast_livo2/shm_camera.hpp"
#include "radar_fast_livo2/camera_frame_queue.hpp"

using namespace radar::fast_livo2;

// ══════════════════════════════════════════════════════════════════
// ShmCamera::convert tests
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraConvert, ConvertsBGR8ToGrayAtTargetResolution) {
    const int src_w = 5472;
    const int src_h = 3648;
    const int tgt_w = 2736;
    const int tgt_h = 1824;

    cv::Mat bgr(src_h, src_w, CV_8UC3);
    for (int r = 0; r < src_h; ++r) {
        auto* row = bgr.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src_w; ++c) {
            uint8_t v = static_cast<uint8_t>((r + c) % 256);
            row[c] = cv::Vec3b(v, v, v);
        }
    }

    hikcamera::FrameMetadata meta{};
    meta.host_monotonic_ns = 1'700'000'000'000ULL;
    meta.frame_id = 42;
    meta.committed_sequence = 7;

    CameraFrame result = ShmCamera::convert(bgr, tgt_w, tgt_h, meta, 0.0);

    EXPECT_EQ(result.gray.cols, tgt_w);
    EXPECT_EQ(result.gray.rows, tgt_h);
    EXPECT_EQ(result.gray.type(), CV_8UC1);
}

TEST(ShmCameraConvert, GrayscaleValuesAreCorrect) {
    cv::Mat bgr(4, 4, CV_8UC3, cv::Scalar(0, 0, 255));
    hikcamera::FrameMetadata meta{};
    meta.host_monotonic_ns = 1'000'000'000ULL;
    meta.frame_id = 1;
    meta.committed_sequence = 1;

    CameraFrame result = ShmCamera::convert(bgr, 4, 4, meta, 0.0);

    const int expected = static_cast<int>(0.299 * 255 + 0.5);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_EQ(result.gray.at<uint8_t>(r, c), expected);
        }
    }
}

TEST(ShmCameraConvert, TimestampFromHostMonotonicNsPlusOffset) {
    cv::Mat bgr(2, 2, CV_8UC3, cv::Scalar(128, 128, 128));
    hikcamera::FrameMetadata meta{};
    meta.host_monotonic_ns = 1'700'000'000'000ULL;
    meta.frame_id = 100;
    meta.committed_sequence = 42;

    CameraFrame result = ShmCamera::convert(bgr, 2, 2, meta, 0.05);

    EXPECT_DOUBLE_EQ(result.timestamp_seconds, 1700.0 + 0.05);
    EXPECT_EQ(result.host_monotonic_ns, meta.host_monotonic_ns);
}

TEST(ShmCameraConvert, MetadataPreserved) {
    cv::Mat bgr(10, 10, CV_8UC3, cv::Scalar(0, 128, 255));
    hikcamera::FrameMetadata meta{};
    meta.host_monotonic_ns = 9'876'543'210'000'000ULL;
    meta.frame_id = 999;
    meta.committed_sequence = 12345;

    CameraFrame result = ShmCamera::convert(bgr, 5, 5, meta, 0.01);

    EXPECT_EQ(result.sequence, meta.committed_sequence);
    EXPECT_EQ(result.host_monotonic_ns, meta.host_monotonic_ns);
    EXPECT_EQ(result.frame_id, meta.frame_id);
}

// ══════════════════════════════════════════════════════════════════
// ShmCamera::convert boundary tests
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraConvert, EmptyInputReturnsEmptyGray) {
    cv::Mat tiny(0, 0, CV_8UC3);
    hikcamera::FrameMetadata meta{};
    CameraFrame result = ShmCamera::convert(tiny, 10, 10, meta, 0.0);
    EXPECT_TRUE(result.gray.empty());
}

TEST(ShmCameraConvert, WrongTypeInputReturnsEmptyGray) {
    cv::Mat gray(10, 10, CV_8UC1, cv::Scalar(128));
    hikcamera::FrameMetadata meta{};
    CameraFrame result = ShmCamera::convert(gray, 10, 10, meta, 0.0);
    EXPECT_TRUE(result.gray.empty());
}

// ══════════════════════════════════════════════════════════════════
// CameraFrameQueue tests
// ══════════════════════════════════════════════════════════════════

static auto make_frame(uint64_t seq, double ts) -> CameraFrame {
    CameraFrame f;
    f.gray = cv::Mat(2, 2, CV_8UC1, cv::Scalar(static_cast<uint8_t>(seq)));
    f.sequence = seq;
    f.timestamp_seconds = ts;
    return f;
}

// ── Acceptance: strictly increasing ─────────────────────────────────

TEST(CameraFrameQueue, PushSucceedsForNewSequence) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(1, 1.0)));
    EXPECT_EQ(q.size(), 1u);
}

TEST(CameraFrameQueue, RejectsSequenceZero) {
    CameraFrameQueue q(5);
    EXPECT_FALSE(q.push(make_frame(0, 1.0)));
    EXPECT_EQ(q.size(), 0u);
}

TEST(CameraFrameQueue, RejectsOutOfOrderSequence) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(5, 1.0)));
    EXPECT_FALSE(q.push(make_frame(3, 1.1)));
    EXPECT_EQ(q.size(), 1u);
}

TEST(CameraFrameQueue, RejectsDuplicateWhileQueued) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(7, 1.0)));
    EXPECT_FALSE(q.push(make_frame(7, 1.01)));
    EXPECT_EQ(q.size(), 1u);
}

// ── Eviction: evicted sequences cannot re-enter ────────────────────

TEST(CameraFrameQueue, RejectsEvictedOldSequence) {
    CameraFrameQueue q(2);
    EXPECT_TRUE(q.push(make_frame(1, 1.0)));
    EXPECT_TRUE(q.push(make_frame(2, 2.0)));
    EXPECT_TRUE(q.push(make_frame(3, 3.0)));  // evicts seq 1
    EXPECT_EQ(q.size(), 2u);
    EXPECT_FALSE(q.push(make_frame(1, 1.5)));
    EXPECT_EQ(q.size(), 2u);
}

// ── Eviction + consumption: consumed (erased-through) can't re-enter ─

TEST(CameraFrameQueue, RejectsConsumedSequence) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(1, 0.90)));
    EXPECT_TRUE(q.push(make_frame(3, 1.10)));
    EXPECT_TRUE(q.push(make_frame(5, 1.30)));

    auto result = q.take_nearest(1.11, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sequence, 3u);

    EXPECT_FALSE(q.push(make_frame(1, 0.95)));
    EXPECT_FALSE(q.push(make_frame(3, 1.05)));
    EXPECT_EQ(q.size(), 1u);
}

// ── take_nearest semantics ─────────────────────────────────────────

TEST(CameraFrameQueue, NearestSelection) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(1, 1.00)));
    EXPECT_TRUE(q.push(make_frame(2, 1.03)));
    EXPECT_TRUE(q.push(make_frame(3, 1.07)));
    EXPECT_TRUE(q.push(make_frame(4, 1.12)));

    auto result = q.take_nearest(1.04, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sequence, 2u);
    EXPECT_DOUBLE_EQ(result->timestamp_seconds, 1.03);
}

TEST(CameraFrameQueue, OutsideTolerance) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(1, 1.00)));
    EXPECT_TRUE(q.push(make_frame(2, 1.20)));

    auto result = q.take_nearest(1.06, 0.05);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(q.size(), 2u);
}

TEST(CameraFrameQueue, ErasesThroughSelected) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(1, 1.00)));
    EXPECT_TRUE(q.push(make_frame(3, 1.03)));
    EXPECT_TRUE(q.push(make_frame(5, 1.07)));
    EXPECT_TRUE(q.push(make_frame(7, 1.12)));

    auto result = q.take_nearest(1.08, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sequence, 5u);

    EXPECT_EQ(q.size(), 1u);
    auto result2 = q.take_nearest(1.12, 0.01);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2->sequence, 7u);
}

TEST(CameraFrameQueue, PostConsumeOlderSequencesBlocked) {
    CameraFrameQueue q(5);
    EXPECT_TRUE(q.push(make_frame(10, 1.00)));
    EXPECT_TRUE(q.push(make_frame(20, 1.10)));

    auto result = q.take_nearest(1.11, 0.05);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sequence, 20u);

    EXPECT_FALSE(q.push(make_frame(10, 0.90)));
    EXPECT_FALSE(q.push(make_frame(20, 1.10)));
    EXPECT_TRUE(q.push(make_frame(30, 1.20)));
    EXPECT_EQ(q.size(), 1u);
}

TEST(CameraFrameQueue, BoundedEvictionKeepsNewest) {
    CameraFrameQueue q(3);
    for (uint64_t s = 1; s <= 6; ++s) {
        EXPECT_TRUE(q.push(make_frame(s, static_cast<double>(s))));
    }
    EXPECT_EQ(q.size(), 3u);
    auto result = q.take_nearest(4.0, 0.01);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sequence, 4u);
}

// ── max_size=0 boundary ────────────────────────────────────────────

TEST(CameraFrameQueue, MaxSizeZeroAlwaysRejects) {
    CameraFrameQueue q(0);
    EXPECT_FALSE(q.push(make_frame(1, 1.0)));
    EXPECT_EQ(q.size(), 0u);
    EXPECT_FALSE(q.take_nearest(1.0, 0.1).has_value());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
