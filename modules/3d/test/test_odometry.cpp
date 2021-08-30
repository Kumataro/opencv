// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#define SHOW_DEBUG_IMAGES  0

static
void warpFrame(const Mat& image, const Mat& depth, const Mat& rvec, const Mat& tvec, const Mat& K,
               Mat& warpedImage, Mat& warpedDepth)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1);

    CV_Assert(depth.size() == image.size());
    CV_Assert(depth.type() == CV_32FC1);

    CV_Assert(!rvec.empty());
    CV_Assert(rvec.total() == 3);
    CV_Assert(rvec.type() == CV_64FC1);

    CV_Assert(!tvec.empty());
    CV_Assert(tvec.size() == Size(1, 3));
    CV_Assert(tvec.type() == CV_64FC1);

    warpedImage.create(image.size(), CV_8UC1);
    warpedImage = Scalar(0);
    warpedDepth.create(image.size(), CV_32FC1);
    warpedDepth = Scalar(FLT_MAX);

    Mat cloud;
    depthTo3d(depth, K, cloud);
    Mat Rt = Mat::eye(4, 4, CV_64FC1);
    {
        Mat R, dst;
        cv::Rodrigues(rvec, R);

        dst = Rt(Rect(0,0,3,3));
        R.copyTo(dst);

        dst = Rt(Rect(3,0,1,3));
        tvec.copyTo(dst);
    }
    Mat warpedCloud, warpedImagePoints;
    perspectiveTransform(cloud, warpedCloud, Rt);
    projectPoints(warpedCloud.reshape(3, 1), Mat(3,1,CV_32FC1, Scalar(0)), Mat(3,1,CV_32FC1, Scalar(0)), K, Mat(1,5,CV_32FC1, Scalar(0)), warpedImagePoints);
    warpedImagePoints = warpedImagePoints.reshape(2, cloud.rows);
    Rect r(0, 0, image.cols, image.rows);
    for(int y = 0; y < cloud.rows; y++)
    {
        for(int x = 0; x < cloud.cols; x++)
        {
            Point p = warpedImagePoints.at<Point2f>(y,x);
            if(r.contains(p))
            {
                float curDepth = warpedDepth.at<float>(p.y, p.x);
                float newDepth = warpedCloud.at<Point3f>(y, x).z;
                if(newDepth < curDepth && newDepth > 0)
                {
                    warpedImage.at<uchar>(p.y, p.x) = image.at<uchar>(y,x);
                    warpedDepth.at<float>(p.y, p.x) = newDepth;
                }
            }
        }
    }
    warpedDepth.setTo(std::numeric_limits<float>::quiet_NaN(), warpedDepth > 100);
}

static
void dilateFrame(Mat& image, Mat& depth)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC1);

    CV_Assert(!depth.empty());
    CV_Assert(depth.type() == CV_32FC1);
    CV_Assert(depth.size() == image.size());

    Mat mask(image.size(), CV_8UC1, Scalar(255));
    for(int y = 0; y < depth.rows; y++)
        for(int x = 0; x < depth.cols; x++)
            if(cvIsNaN(depth.at<float>(y,x)) || depth.at<float>(y,x) > 10 || depth.at<float>(y,x) <= FLT_EPSILON)
                mask.at<uchar>(y,x) = 0;

    image.setTo(255, ~mask);
    Mat minImage;
    erode(image, minImage, Mat());

    image.setTo(0, ~mask);
    Mat maxImage;
    dilate(image, maxImage, Mat());

    depth.setTo(FLT_MAX, ~mask);
    Mat minDepth;
    erode(depth, minDepth, Mat());

    depth.setTo(0, ~mask);
    Mat maxDepth;
    dilate(depth, maxDepth, Mat());

    Mat dilatedMask;
    dilate(mask, dilatedMask, Mat(), Point(-1,-1), 1);
    for(int y = 0; y < depth.rows; y++)
        for(int x = 0; x < depth.cols; x++)
            if(!mask.at<uchar>(y,x) && dilatedMask.at<uchar>(y,x))
            {
                image.at<uchar>(y,x) = static_cast<uchar>(0.5f * (static_cast<float>(minImage.at<uchar>(y,x)) +
                                                                  static_cast<float>(maxImage.at<uchar>(y,x))));
                depth.at<float>(y,x) = 0.5f * (minDepth.at<float>(y,x) + maxDepth.at<float>(y,x));
            }
}

class OdometryTest
{
public:
    OdometryTest(const Ptr<Odometry>& _odometry,
                 double _maxError1,
                 double _maxError5,
                 double _idError = DBL_EPSILON) :
        odometry(_odometry),
        maxError1(_maxError1),
        maxError5(_maxError5),
        idError(_idError)
    { }

    void readData(Mat& image, Mat& depth) const;
    static Mat getCameraMatrix()
    {
        float fx = 525.0f, // default
              fy = 525.0f,
              cx = 319.5f,
              cy = 239.5f;
        Matx33f K(fx,  0, cx,
                   0, fy, cy,
                   0,  0,  1);
        return Mat(K);
    }
    static void generateRandomTransformation(Mat& R, Mat& t);

    void run();
    void checkUMats();

    Ptr<Odometry> odometry;
    double maxError1;
    double maxError5;
    double idError;
};


void OdometryTest::readData(Mat& image, Mat& depth) const
{
    std::string dataPath = cvtest::TS::ptr()->get_data_path();
    std::string imageFilename = dataPath + "/cv/rgbd/rgb.png";
    std::string depthFilename = dataPath + "/cv/rgbd/depth.png";

    image = imread(imageFilename,  0);
    depth = imread(depthFilename, -1);

    if(image.empty())
    {
        FAIL() << "Image " << imageFilename.c_str() << " can not be read" << std::endl;
    }
    if(depth.empty())
    {
        FAIL() << "Depth" << depthFilename.c_str() << "can not be read" << std::endl;
    }

    CV_DbgAssert(image.type() == CV_8UC1);
    CV_DbgAssert(depth.type() == CV_16UC1);
    {
        Mat depth_flt;
        depth.convertTo(depth_flt, CV_32FC1, 1.f/5000.f);
        depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth_flt < FLT_EPSILON);
        depth = depth_flt;
    }
}

void OdometryTest::generateRandomTransformation(Mat& rvec, Mat& tvec)
{
    const float maxRotation = (float)(3.f / 180.f * CV_PI); //rad
    const float maxTranslation = 0.02f; //m

    RNG& rng = theRNG();
    rvec.create(3, 1, CV_64FC1);
    tvec.create(3, 1, CV_64FC1);

    randu(rvec, Scalar(-1000), Scalar(1000));
    normalize(rvec, rvec, rng.uniform(0.007f, maxRotation));

    randu(tvec, Scalar(-1000), Scalar(1000));
    normalize(tvec, tvec, rng.uniform(0.008f, maxTranslation));
}

void OdometryTest::checkUMats()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);

    odometry->setCameraMatrix(K);

    Mat calcRt;

    UMat uimage, udepth, umask;
    image.copyTo(uimage);
    depth.copyTo(udepth);
    Mat(image.size(), CV_8UC1, Scalar(255)).copyTo(umask);

    bool isComputed = odometry->compute(uimage, udepth, umask,
                                        uimage, udepth, umask,
                                        calcRt);
    ASSERT_TRUE(isComputed);
    double diff = cv::norm(calcRt, Mat::eye(4, 4, CV_64FC1));
    if (diff > idError)
    {
        FAIL() << "Incorrect transformation between the same frame (not the identity matrix), diff = " << diff << std::endl;
    }
}

void OdometryTest::run()
{
    Mat K = getCameraMatrix();

    Mat image, depth;
    readData(image, depth);

    odometry->setCameraMatrix(K);

    Mat calcRt;

    // 1. Try to find Rt between the same frame (try masks also).
    Mat mask(image.size(), CV_8UC1, Scalar(255));
    bool isComputed = odometry->compute(image, depth, mask, image, depth, mask, calcRt);
    if(!isComputed)
    {
        FAIL() << "Can not find Rt between the same frame" << std::endl;
    }
    double diff = cv::norm(calcRt, Mat::eye(4,4,CV_64FC1));
    if(diff > idError)
    {
        FAIL() << "Incorrect transformation between the same frame (not the identity matrix), diff = " << diff << std::endl;
    }

    // 2. Generate random rigid body motion in some ranges several times (iterCount).
    // On each iteration an input frame is warped using generated transformation.
    // Odometry is run on the following pair: the original frame and the warped one.
    // Comparing a computed transformation with an applied one we compute 2 errors:
    // better_1time_count - count of poses which error is less than ground truth pose,
    // better_5times_count - count of poses which error is 5 times less than ground truth pose.
    int iterCount = 100;
    int better_1time_count = 0;
    int better_5times_count = 0;
    for(int iter = 0; iter < iterCount; iter++)
    {
        Mat rvec, tvec;
        generateRandomTransformation(rvec, tvec);
        Mat warpedImage, warpedDepth;
        warpFrame(image, depth, rvec, tvec, K, warpedImage, warpedDepth);
        dilateFrame(warpedImage, warpedDepth); // due to inaccuracy after warping

        isComputed = odometry->compute(image, depth, mask, warpedImage, warpedDepth, mask, calcRt);
        if(!isComputed)
            continue;

        Mat calcR = calcRt(Rect(0,0,3,3)), calcRvec;
        cv::Rodrigues(calcR, calcRvec);
        calcRvec = calcRvec.reshape(rvec.channels(), rvec.rows);
        Mat calcTvec = calcRt(Rect(3,0,1,3));

#if SHOW_DEBUG_IMAGES
        imshow("image", image);
        imshow("warpedImage", warpedImage);
        Mat resultImage, resultDepth;
        warpFrame(image, depth, calcRvec, calcTvec, K, resultImage, resultDepth);
        imshow("resultImage", resultImage);
        waitKey();
#endif

        // compare rotation
        double rdiffnorm = cv::norm(rvec - calcRvec),
               rnorm = cv::norm(rvec);
        double tdiffnorm = cv::norm(tvec - calcTvec),
               tnorm = cv::norm(tvec);
        if(rdiffnorm < rnorm &&  tdiffnorm < tnorm)
            better_1time_count++;
        if(5. * rdiffnorm < rnorm && 5 * tdiffnorm < tnorm)
            better_5times_count++;

        CV_LOG_INFO(NULL, "Iter " << iter);
        CV_LOG_INFO(NULL, "rdiffnorm " << rdiffnorm << "; rnorm " << rnorm);
        CV_LOG_INFO(NULL, "tdiffnorm " << tdiffnorm << "; tnorm " << tnorm);

        CV_LOG_INFO(NULL, "better_1time_count " << better_1time_count << "; better_5time_count " << better_5times_count);
    }

    if(static_cast<double>(better_1time_count) < maxError1 * static_cast<double>(iterCount))
    {
        FAIL() << "Incorrect count of accurate poses [1st case]: "
            << static_cast<double>(better_1time_count) << " / "
            << maxError1 * static_cast<double>(iterCount) << std::endl;
    }

    if(static_cast<double>(better_5times_count) < maxError5 * static_cast<double>(iterCount))
    {
        FAIL() << "Incorrect count of accurate poses [2nd case]: "
            << static_cast<double>(better_5times_count) << " / "
            << maxError5 * static_cast<double>(iterCount) << std::endl;
    }
}

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST(RGBD_Odometry_Rgbd, algorithmic)
{
    OdometryTest test(cv::Odometry::createFromName("RgbdOdometry"), 0.99, 0.89);
    test.run();
}

TEST(RGBD_Odometry_ICP, algorithmic)
{
    OdometryTest test(cv::Odometry::createFromName("ICPOdometry"), 0.99, 0.99);
    test.run();
}

TEST(RGBD_Odometry_RgbdICP, algorithmic)
{
    OdometryTest test(cv::Odometry::createFromName("RgbdICPOdometry"), 0.99, 0.99);
    test.run();
}

TEST(RGBD_Odometry_FastICP, algorithmic)
{
    OdometryTest test(cv::Odometry::createFromName("FastICPOdometry"), 0.99, 0.99, FLT_EPSILON);
    test.run();
}

TEST(RGBD_Odometry_Rgbd, UMats)
{
    OdometryTest test(cv::Odometry::createFromName("RgbdOdometry"), 0.99, 0.89);
    test.checkUMats();
}

TEST(RGBD_Odometry_ICP, UMats)
{
    OdometryTest test(cv::Odometry::createFromName("ICPOdometry"), 0.99, 0.99);
    test.checkUMats();
}

TEST(RGBD_Odometry_RgbdICP, UMats)
{
    OdometryTest test(cv::Odometry::createFromName("RgbdICPOdometry"), 0.99, 0.99);
    test.checkUMats();
}

TEST(RGBD_Odometry_FastICP, UMats)
{
    OdometryTest test(cv::Odometry::createFromName("FastICPOdometry"), 0.99, 0.99, FLT_EPSILON);
    test.checkUMats();
}


/****************************************************************************************\
*                                Depth to 3d tests                                       *
\****************************************************************************************/

TEST(RGBD_DepthTo3d, compute)
{
    // K from a VGA Kinect
    Mat K = OdometryTest::getCameraMatrix();

    // Create a random depth image
    RNG rng;
    Mat_<float> depth(480, 640);
    rng.fill(depth, RNG::UNIFORM, 0, 100);

    // Create some 3d points on the plane
    int rows = depth.rows, cols = depth.cols;
    Mat_<Vec3f> points3d;
    depthTo3d(depth, K, points3d);

    // Make sure the points belong to the plane
    Mat points = points3d.reshape(1, rows * cols);
    Mat image_points;
    Mat rvec;
    cv::Rodrigues(Mat::eye(3, 3, CV_32F), rvec);
    Mat tvec = (Mat_<float>(1, 3) << 0, 0, 0);
    projectPoints(points, rvec, tvec, K, Mat(), image_points);
    image_points = image_points.reshape(2, rows);

    float avg_diff = 0;
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            avg_diff += (float)cv::norm(image_points.at<Vec2f>(y, x) - Vec2f((float)x, (float)y));

    // Verify the function works
    ASSERT_LE(avg_diff / rows / cols, 1e-4) << "Average error for ground truth is: " << (avg_diff / rows / cols);
}


}} // namespace