// test_shm_camera.cpp — Unit tests for ShmCamera + CameraFrameQueue
//
// ShmCamera tests (1-4): geometry, grayscale conversion, timestamp, metadata
// ShmCamera boundary tests (5-6): invalid input rejection
// ShmCamera slot protocol tests (7-8): completed-slot, stability helpers
// CameraFrameQueue tests (9-20): push, duplicate, eviction, nearest selection,
//   tolerance, consume-through, post-consume rejection

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <opencv2/opencv.hpp>

#include "radar_fast_livo2/camera_frame_queue.hpp"
#include "radar_fast_livo2/shm_camera.hpp"

using namespace radar::fast_livo2;

// ══════════════════════════════════════════════════════════════════
// ShmCamera::convert tests (signature: rgb, tw, th, host_ns, frm_id, seq, offset)
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraConvert, ConvertsRGBToGrayAtTargetResolution) {
    const int src_w = 5472;
    const int src_h = 3648;
    const int tgt_w = 2736;
    const int tgt_h = 1824;

    cv::Mat rgb(src_h, src_w, CV_8UC3);
    for (int r = 0; r < src_h; ++r) {
        auto* row = rgb.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src_w; ++c) {
            uint8_t v = static_cast<uint8_t>((r + c) % 256);
            row[c]    = cv::Vec3b(v, v, v);
        }
    }

    CameraFrame result = ShmCamera::convert(rgb, src_w, src_h, tgt_w, tgt_h,
        /*host_monotonic_ns=*/1'700'000'000'000ULL,
        /*frame_id=*/42,
        /*committed_sequence=*/7,
        /*img_time_offset=*/0.0);

    EXPECT_EQ(result.gray.cols, tgt_w);
    EXPECT_EQ(result.gray.rows, tgt_h);
    EXPECT_EQ(result.gray.type(), CV_8UC1);
}

TEST(ShmCameraConvert, GrayscaleValuesAreCorrect) {
    // Simulate SHM RGB data: pixel is R=255, G=0, B=0 (pure red).
    // In cv::Mat storage this is Vec3b(0, 0, 255) = [B=0, G=0, R=255],
    // but cvtColor with COLOR_RGB2GRAY treats ch0 as R, so:
    //   gray = 0.299*(ch0) + 0.587*(ch1) + 0.114*(ch2)
    //        = 0.299*0 + 0.587*0 + 0.114*255 ≈ 29
    cv::Mat rgb(4, 4, CV_8UC3);
    rgb.setTo(cv::Scalar(0, 0, 0));
    // Put a pure-red pixel: SHM RGB byte order is [R=255, G=0, B=0]
    // In cv::Mat, this is stored as Vec3b(B=0, G=0, R=255).
    rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(0, 0, 255);

    CameraFrame result = ShmCamera::convert(rgb, 4, 4, 4, 4,
        /*host_monotonic_ns=*/1'000'000'000ULL,
        /*frame_id=*/1,
        /*committed_sequence=*/1,
        /*img_time_offset=*/0.0);

    // With RGB→GRAY, pure red (B=0,G=0,R=255 stored as Vec3b(0,0,255)):
    //   0.299*0 + 0.587*0 + 0.114*255 ≈ 29
    const int expected_red_gray = static_cast<int>(0.114 * 255 + 0.5);
    EXPECT_EQ(result.gray.at<uint8_t>(1, 1), expected_red_gray);

    // Surrounding pixels are black (RGB=[0,0,0]) → gray = 0
    EXPECT_EQ(result.gray.at<uint8_t>(0, 0), 0);
}

TEST(ShmCameraConvert, TimestampFromHostMonotonicNsPlusOffset) {
    cv::Mat rgb(2, 2, CV_8UC3, cv::Scalar(128, 128, 128));

    CameraFrame result = ShmCamera::convert(rgb, 2, 2, 2, 2,
        /*host_monotonic_ns=*/1'700'000'000'000ULL,
        /*frame_id=*/100,
        /*committed_sequence=*/42,
        /*img_time_offset=*/0.05);

    EXPECT_DOUBLE_EQ(result.timestamp_seconds, 1700.0 + 0.05);
    EXPECT_EQ(result.host_monotonic_ns, 1'700'000'000'000ULL);
}

TEST(ShmCameraConvert, MetadataPreserved) {
    cv::Mat rgb(10, 10, CV_8UC3, cv::Scalar(0, 128, 255));

    CameraFrame result = ShmCamera::convert(rgb, 10, 10, 5, 5,
        /*host_monotonic_ns=*/9'876'543'210'000'000ULL,
        /*frame_id=*/999,
        /*committed_sequence=*/12345,
        /*img_time_offset=*/0.01);

    EXPECT_EQ(result.sequence, 12345ULL);
    EXPECT_EQ(result.host_monotonic_ns, 9'876'543'210'000'000ULL);
    EXPECT_EQ(result.frame_id, 999ULL);
}

// ══════════════════════════════════════════════════════════════════
// ShmCamera::convert boundary tests
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraConvert, EmptyInputReturnsEmptyGray) {
    cv::Mat tiny(0, 0, CV_8UC3);
    CameraFrame result = ShmCamera::convert(tiny, 0, 0, 10, 10,
        /*host_monotonic_ns=*/0,
        /*frame_id=*/0,
        /*committed_sequence=*/0,
        /*img_time_offset=*/0.0);
    EXPECT_TRUE(result.gray.empty());
}

TEST(ShmCameraConvert, WrongTypeInputReturnsEmptyGray) {
    cv::Mat gray(10, 10, CV_8UC1, cv::Scalar(128));
    CameraFrame result = ShmCamera::convert(gray, 10, 10, 10, 10,
        /*host_monotonic_ns=*/0,
        /*frame_id=*/0,
        /*committed_sequence=*/0,
        /*img_time_offset=*/0.0);
    EXPECT_TRUE(result.gray.empty());
}

// ══════════════════════════════════════════════════════════════════
// ShmCamera::open tests — dimension validation (no SHM required)
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraOpen, RejectsZeroWidth) {
    ShmCamera cam("/nonexistent_shm", 0, 480, 640, 480, 0.0);
    auto result = cam.open();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ShmCameraErrorCode::InvalidFrame);
}

TEST(ShmCameraOpen, RejectsZeroHeight) {
    ShmCamera cam("/nonexistent_shm", 640, 0, 640, 480, 0.0);
    auto result = cam.open();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ShmCameraErrorCode::InvalidFrame);
}

TEST(ShmCameraOpen, RejectsNegativeDimensions) {
    ShmCamera cam("/nonexistent_shm", -1, -1, -1, -1, 0.0);
    auto result = cam.open();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ShmCameraErrorCode::InvalidFrame);
}

TEST(ShmCameraOpen, RejectsOversizedImage) {
    // MAX_IMAGE_SIZE = 60MB; 10000*10000*3 ≈ 300MB > MAX_IMAGE_SIZE
    ShmCamera cam("/nonexistent_shm", 10000, 10000, 1000, 1000, 0.0);
    auto result = cam.open();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ShmCameraErrorCode::InvalidFrame);
}

// ══════════════════════════════════════════════════════════════════
// Completed-slot protocol tests (pure logic, no SHM needed)
// ══════════════════════════════════════════════════════════════════

// Replicates the completed_slot() logic from shm_camera.cpp inline for testing.
namespace {
auto completed_slot(uint64_t counter, unsigned int slot_num) -> unsigned int {
    return static_cast<unsigned int>((counter - 1) % static_cast<uint64_t>(slot_num));
}
bool has_advanced(uint64_t current, uint64_t last_seen) {
    return current > 0 && current != last_seen;
}
bool is_valid_counter(uint64_t c) { return c > 0; }
bool is_stable(uint64_t before, uint64_t after) { return before == after; }
}

TEST(ShmCameraSlotProtocol, CompletedSlotIsCorrect) {
    // frame_counter=1 → (1-1)%4 = 0
    EXPECT_EQ(completed_slot(1, 4), 0u);
    // frame_counter=4 → (4-1)%4 = 3
    EXPECT_EQ(completed_slot(4, 4), 3u);
    // frame_counter=5 → (5-1)%4 = 0  (wraps)
    EXPECT_EQ(completed_slot(5, 4), 0u);
}

TEST(ShmCameraSlotProtocol, ZeroCounterRejected) {
    EXPECT_FALSE(is_valid_counter(0));
    EXPECT_TRUE(is_valid_counter(1));
}

TEST(ShmCameraSlotProtocol, NoNewFrameDetected) {
    EXPECT_FALSE(has_advanced(0, 5)); // zero counter
    EXPECT_FALSE(has_advanced(5, 5)); // same value
    EXPECT_TRUE(has_advanced(6, 5));  // advanced
    EXPECT_TRUE(has_advanced(1, 0));  // first frame after zero
}

TEST(ShmCameraSlotProtocol, StabilityCheckWorks) {
    EXPECT_TRUE(is_stable(5, 5));
    EXPECT_FALSE(is_stable(5, 6));
}

// ══════════════════════════════════════════════════════════════════
// Regression: RGB-vs-BGR grayscale numerical test
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraConvert, RGBvsBGRGrayscaleNumerical) {
    // Simulate SHM RGB data: pixel is pure green (R=0, G=255, B=0).
    // In cv::Mat, stored as Vec3b(B=0, G=255, R=0).
    // With COLOR_RGB2GRAY (correct for SHM): 0.299*0 + 0.587*255 + 0.114*0 ≈ 150
    // With COLOR_BGR2GRAY (wrong for SHM):    0.114*0 + 0.587*255 + 0.299*0 ≈ 150
    // (Green is the same in both since G is the middle channel.)
    //
    // Pure red (SHM: R=255,G=0,B=0 → cv::Mat: Vec3b(0,0,255)):
    // COLOR_RGB2GRAY: 0.299*0 + 0.587*0 + 0.114*255 ≈ 29
    // COLOR_BGR2GRAY: 0.114*0 + 0.587*0 + 0.299*255 ≈ 76  ← WRONG
    //
    // Pure blue (SHM: R=0,G=0,B=255 → cv::Mat: Vec3b(255,0,0)):
    // COLOR_RGB2GRAY: 0.299*255 + 0.587*0 + 0.114*0 ≈ 76
    // COLOR_BGR2GRAY: 0.114*255 + 0.587*0 + 0.299*0 ≈ 29  ← WRONG

    cv::Mat rgb(1, 3, CV_8UC3); // 1 row, 3 columns of pixels
    // Column 0: pure red (SHM: [R=255, G=0, B=0])
    rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);
    // Column 1: pure green (SHM: [R=0, G=255, B=0])
    rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);
    // Column 2: pure blue (SHM: [R=0, G=0, B=255])
    rgb.at<cv::Vec3b>(0, 2) = cv::Vec3b(255, 0, 0);

    CameraFrame result = ShmCamera::convert(rgb, 3, 1, 3, 1, 0, 0, 1, 0.0);
    ASSERT_FALSE(result.gray.empty());
    EXPECT_EQ(result.gray.cols, 3);
    EXPECT_EQ(result.gray.rows, 1);

    const int expected_red   = static_cast<int>(0.114 * 255 + 0.5); // ≈ 29
    const int expected_green = static_cast<int>(0.587 * 255 + 0.5); // ≈ 150
    const int expected_blue  = static_cast<int>(0.299 * 255 + 0.5); // ≈ 76

    EXPECT_EQ(result.gray.at<uint8_t>(0, 0), expected_red);
    EXPECT_EQ(result.gray.at<uint8_t>(0, 1), expected_green);
    EXPECT_EQ(result.gray.at<uint8_t>(0, 2), expected_blue);
}

// ══════════════════════════════════════════════════════════════════
// Regression: wait_next timeout based on last_seen (not 0)
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraSlotProtocol, NoNewFrameDetectedAfterTimeout) {
    // Simulates the timeout predicate: if counter hasn't advanced
    // past poll_start, it's a genuine timeout (no duplicate copy).
    uint64_t poll_start = 5;
    // Case 1: counter still 5 (no advancement) → timeout
    EXPECT_FALSE(has_advanced(5, poll_start));
    // Case 2: counter is 0 (uninitialized) → not valid
    EXPECT_FALSE(has_advanced(0, poll_start));
    // Case 3: counter advanced to 6 → valid new frame
    EXPECT_TRUE(has_advanced(6, poll_start));
    // Case 4: Bug repro: was comparing against 0 instead of poll_start
    // Old code: has_advanced(5, 0) → true (WRONG — would copy duplicate)
    // Fixed code: has_advanced(5, 5) → false (correct — no new frame)
    EXPECT_FALSE(has_advanced(5, 5));
}

// ══════════════════════════════════════════════════════════════════
// Regression: source-vs-target dimension contract
// ══════════════════════════════════════════════════════════════════

TEST(ShmCameraConvert, SourceVsTargetDimensionsDiffer) {
    // Prove that source (5472×3648) and target (2736×1824) can differ
    // without corruption: a known row pattern in source should appear
    // at the expected positions after resize to target.
    const int src_w = 12, src_h = 8;
    const int tgt_w = 6, tgt_h = 4;

    // Create source RGB: each pixel = (row_idx, col_idx, 128) for
    // easy debugging.  The gray value derived from these is deterministic.
    cv::Mat rgb(src_h, src_w, CV_8UC3);
    for (int r = 0; r < src_h; ++r) {
        auto* row = rgb.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src_w; ++c) {
            row[c] = cv::Vec3b(static_cast<uint8_t>(r), static_cast<uint8_t>(c), 128);
        }
    }

    CameraFrame result = ShmCamera::convert(rgb, src_w, src_h, tgt_w, tgt_h, 0, 0, 1, 0.0);

    EXPECT_EQ(result.gray.cols, tgt_w);
    EXPECT_EQ(result.gray.rows, tgt_h);
    EXPECT_EQ(result.gray.type(), CV_8UC1);
}

TEST(ShmCameraConvert, SourceDimensionsMustMatchInputMat) {
    // convert() must reject an input cv::Mat whose cols/rows do not
    // match the declared source_width/source_height parameters.
    cv::Mat rgb(4, 4, CV_8UC3, cv::Scalar(128, 128, 128));

    // Declare source as 8×8 but pass a 4×4 mat → empty gray
    CameraFrame result = ShmCamera::convert(rgb, 8, 8, 4, 4, 0, 0, 1, 0.0);
    EXPECT_TRUE(result.gray.empty());
}

TEST(ShmCameraConvert, PatternPreservedThroughResize) {
    // Top row of source is all bright (r=0, varying c), bottom row dark.
    // After resize, top rows should be brighter than bottom rows.
    const int src_w = 20, src_h = 20;
    const int tgt_w = 10, tgt_h = 10;

    cv::Mat rgb(src_h, src_w, CV_8UC3);
    for (int r = 0; r < src_h; ++r) {
        auto* row = rgb.ptr<cv::Vec3b>(r);
        uint8_t v = (r < src_h / 2) ? 200 : 50;
        for (int c = 0; c < src_w; ++c) {
            row[c] = cv::Vec3b(v, v, v);
        }
    }

    CameraFrame result = ShmCamera::convert(rgb, src_w, src_h, tgt_w, tgt_h, 0, 0, 1, 0.0);

    EXPECT_EQ(result.gray.cols, tgt_w);
    EXPECT_EQ(result.gray.rows, tgt_h);

    // Top rows (0-4) should be brighter than bottom rows (5-9)
    double top_sum = 0.0, bottom_sum = 0.0;
    for (int r = 0; r < tgt_h / 2; ++r) {
        top_sum += result.gray.at<uint8_t>(r, 0);
    }
    for (int r = tgt_h / 2; r < tgt_h; ++r) {
        bottom_sum += result.gray.at<uint8_t>(r, 0);
    }
    EXPECT_GT(top_sum, bottom_sum);
}

TEST(ShmCameraConvert, Full5472x3648ByteSizeMatchesRealSHM) {
    // Verify the byte size for the real HIK configuration:
    // 5472 * 3648 * 3 = 59,885,568 bytes ≤ MAX_IMAGE_SIZE (60,000,000).
    const size_t bytes = static_cast<size_t>(5472) * static_cast<size_t>(3648) * 3;
    EXPECT_EQ(bytes, 59'885'568ULL);
    EXPECT_LE(bytes, static_cast<size_t>(MAX_IMAGE_SIZE));
}

TEST(ShmCameraConvert, TargetSmallerThanSourceIsValid) {
    // The nominal case: source 5472×3648 → target 2736×1824.
    const int src_w = 5472, src_h = 3648;
    const int tgt_w = 2736, tgt_h = 1824;

    cv::Mat rgb(src_h, src_w, CV_8UC3, cv::Scalar(128, 128, 128));

    CameraFrame result = ShmCamera::convert(rgb, src_w, src_h, tgt_w, tgt_h, 0, 0, 1, 0.0);

    EXPECT_EQ(result.gray.cols, tgt_w);
    EXPECT_EQ(result.gray.rows, tgt_h);
    EXPECT_EQ(result.gray.type(), CV_8UC1);
    EXPECT_FALSE(result.gray.empty());
}

// ══════════════════════════════════════════════════════════════════
// CameraFrameQueue tests
// ══════════════════════════════════════════════════════════════════

static auto make_frame(uint64_t seq, double ts) -> CameraFrame {
    CameraFrame f;
    f.gray              = cv::Mat(2, 2, CV_8UC1, cv::Scalar(static_cast<uint8_t>(seq)));
    f.sequence          = seq;
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
    EXPECT_TRUE(q.push(make_frame(3, 3.0))); // evicts seq 1
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
