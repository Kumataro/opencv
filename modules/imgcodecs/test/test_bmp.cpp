// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

#include <vector>

namespace opencv_test { namespace {

// See https://github.com/opencv/opencv/issues/27789
// See https://github.com/opencv/opencv/issues/23233
TEST(Imgcodecs_BMP, encode_decode_over1GB_regression27789)
{
    applyTestTag( CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_LONG );

    // Create large Mat over 1GB
    // 20000 px * 18000 px *  24 bpp(3ch) = 1,080,000,000 bytes
    // 1 GiB                              = 1,073,741,824 bytes
    cv::Mat src(20000, 18000, CV_8UC3, cv::Scalar(0,0,0));

    // Encode large BMP file.
    std::vector<uint8_t> buf;
    bool ret = false;
    ASSERT_NO_THROW(ret = cv::imencode(".bmp", src, buf, {}));
    ASSERT_TRUE(ret);

    src.release(); // To reduce usage memory, it is needed.

    // Decode large BMP file.
    cv::Mat dst;
    ASSERT_NO_THROW(dst = cv::imdecode(buf, cv::IMREAD_COLOR));
    ASSERT_FALSE(dst.empty());
}

TEST(Imgcodecs_BMP, write_read_over1GB_regression27789)
{
    // tag CV_TEST_TAG_VERYLONG applied to skip on CI. The test writes ~1GB file.
    applyTestTag( CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_VERYLONG );
    string bmpFilename = cv::tempfile(".bmp"); // To remove it, test must use EXPECT_* instead of ASSERT_*.

    // Create large Mat over 1GB
    // 20000 px * 18000 px *  24 bpp(3ch) = 1,080,000,000 bytes
    // 1 GiB                              = 1,073,741,824 bytes
    cv::Mat src(20000, 18000, CV_8UC3, cv::Scalar(0,0,0));

    // Write large BMP file.
    bool ret = false;
    EXPECT_NO_THROW(ret = cv::imwrite(bmpFilename, src, {}));
    EXPECT_TRUE(ret);

    // Read large BMP file.
    cv::Mat dst;
    EXPECT_NO_THROW(dst = cv::imread(bmpFilename, cv::IMREAD_COLOR));
    EXPECT_FALSE(dst.empty());

    remove(bmpFilename.c_str());
}

// See https://github.com/opencv/opencv/issues/28350

struct BmpOverflowParam {
    int width;
    int height;
    int bpp;
    std::string label;

    // Added constructor for C++11 compatibility in Values()
    BmpOverflowParam(int w, int h, int b, const std::string& l)
        : width(w), height(h), bpp(b), label(l) {}
};

// Function to provide a human-readable name for each test case
struct PrintBmpOverflowParam {
    std::string operator()(const testing::TestParamInfo<BmpOverflowParam>& info) const {
        std::string s = info.param.label;
        // gtest names only allow alphanumeric and underscores
        for (char &c : s) {
            if (!isalnum(c)) c = '_';
        }
        return s;
    }
};

class Imgcodecs_Bmp_Overflow : public testing::TestWithParam<BmpOverflowParam> {};

TEST_P(Imgcodecs_Bmp_Overflow, reject_invalid_header)
{
    const BmpOverflowParam& params = GetParam();

    const int64_t channels64 = (params.bpp + 7) / 8;
    const int64_t pitch64 = ((int64_t)params.width * channels64 + 3) & ~3;
    const int64_t total64 = pitch64 * std::abs((int64_t)params.height);

    // If total exceeds 2GB (OpenCV's practical limit on 32-bit), skip it on 64-bit platforms
    // to avoid trying to allocate massive memory or hitting OS-specific limits.
    if (sizeof(size_t) > 4 && total64 > (int64_t)INT_MAX) {
        throw SkipTestException("Skipping huge image test on 64-bit environment");
    }

    // 1. Create a minimal valid BMP (1x1)
    Mat tiny_img(1, 1, CV_8UC3, Scalar::all(0));
    std::vector<uchar> buf;
    imencode(".bmp", tiny_img, buf);

    if (buf.size() < 30) throw std::runtime_error("Buffer too small");

    // 2. Modify BITMAPINFOHEADER using bit-shifts for Endian-safety
    auto write_le32 = [&](size_t offset, int32_t value) {
        buf[offset + 0] = (uchar)(value & 0xFF);
        buf[offset + 1] = (uchar)((value >> 8) & 0xFF);
        buf[offset + 2] = (uchar)((value >> 16) & 0xFF);
        buf[offset + 3] = (uchar)((value >> 24) & 0xFF);
    };

    auto write_le16 = [&](size_t offset, int16_t value) {
        buf[offset + 0] = (uchar)(value & 0xFF);
        buf[offset + 1] = (uchar)((value >> 8) & 0xFF);
    };

    write_le32(18, params.width);
    write_le32(22, params.height);
    write_le16(28, (int16_t)params.bpp);

    // 3. Decode should return empty Mat due to readHeader validation
    Mat dst = imdecode(buf, IMREAD_UNCHANGED);
    EXPECT_TRUE(dst.empty()) << "Failed to reject: " << params.label;
}

INSTANTIATE_TEST_CASE_P(imgcodecs, Imgcodecs_Bmp_Overflow, ::testing::Values(
    BmpOverflowParam(134217731, 1, 32, "Total size exceeds 32-bit limit"),
    BmpOverflowParam(1000000000, 1, 24, "Pitch exceeds INT_MAX"),
    BmpOverflowParam(0, 1, 24, "Width is zero"),
    BmpOverflowParam(100, 0, 24, "Height is zero"),
    BmpOverflowParam(100, 1, 0, "BPP is zero"),
    BmpOverflowParam(1000000, 1000000, 32, "Extreme total size")
), PrintBmpOverflowParam());


}} // namespace
