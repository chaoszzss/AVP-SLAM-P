// Adapted for IPM semantic segmentation image input
// For compatibility with floop dataset (ipm_semantic_label + gps_odom)
// Author: Adapted from liuguitao's original work

#include <iostream>
#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <mutex>
#include <queue>
#include <cmath>
#include <vector>

// Process IPM semantic segmentation image
// Extract feature point cloud from semantic labels (slot lines, arrows, dash lines)
// Publish point cloud for mapping node

ros::Publisher pubCameraCloudFrame;
ros::Publisher pubCurrentOdom;

typedef pcl::PointXYZRGB PointType;

// IPM image parameters
int ipmWidth = 1000;
int ipmHeight = 1000;
double pixelScale = 0.02;  // meters per pixel
int ipmCenterX = 500;
int ipmCenterY = 500;
double rearAxleOffset = 1.32;  // offset from vehicle center to rear axle

// Point cloud parameters
float pointCloudLeafSize = 0.1;
double farPointThresh = 20.0;
double closePointThresh = 0.1;

// Semantic color thresholds (BGR format)
// Slot lines - Green
int slotColorB = 0;
int slotColorG = 255;
int slotColorR = 0;
int slotColorThresh = 80;

// Arrow lines - Yellow
int arrowColorB = 0;
int arrowColorG = 255;
int arrowColorR = 255;
int arrowColorThresh = 80;

// Pillar/olive-yellow suppression (mis-detections that pollute the map)
int oliveMinR = 150;
int oliveMaxR = 220;
int oliveMinG = 150;
int oliveMaxG = 220;
int oliveMaxB = 80;

// Dash lines - Blue
int dashColorB = 255;
int dashColorG = 0;
int dashColorR = 0;
int dashColorThresh = 80;

// Skip pixels for downsampling
int pixelSkip = 2;

// Message buffers
std::queue<sensor_msgs::ImageConstPtr> ipmImageBuf;
std::queue<nav_msgs::OdometryConstPtr> odomBuf;
std::mutex mBuf;

// Last processed timestamp
double lastProcessTime = 0;
double minProcessInterval = 0.05;  // 20Hz max

// Check if pixel is a feature pixel based on semantic color and position
// Only extract: slot lines (green/yellow-green), arrows (yellow), lane lines (blue)
// Exclude: road background (purple), vehicle (black), speed bumps (cyan), noise (white), FPS text
bool isFeaturePixel(int b, int g, int r, int u, int v) {
    // === REGION EXCLUSION ===
    // Exclude top-left corner where FPS/Frame text is displayed
    // Text area is approximately in the first 200 pixels width and 80 pixels height
    if (u < 220 && v < 100) return false;

    // Also exclude top-right corner (potential timestamp/info area)
    // if (u > ipmWidth - 200 && v < 80) return false;

    // === FEATURES TO EXTRACT ===

    // 0. Explicitly drop olive/dark yellow (parking pillars / blind zones) to avoid map pollution
    // Typical RGB around (192, 191, 0); keep tunable via params.
    if (r >= oliveMinR && r <= oliveMaxR &&
        g >= oliveMinG && g <= oliveMaxG &&
        b < oliveMaxB) {
        return false;
    }

    // === AVM-SLAM dataset semantic colors (BGR) ===
    // Parking slots: magenta (255, 0, 255)
    if (b > 200 && r > 200 && g < 80) return true;
    // Lane lines: blue (255, 0, 0)
    if (b > 200 && g < 80 && r < 80) return true;
    // Arrows: yellow (0, 255, 255) — keep existing rule below

    // 1. Yellow - Arrow/direction markers (very high G and R, low B)
    // RGB approximately (254, 254, 0) - bright yellow arrows
    // Must distinguish from blind zone markers (192, 191, 0) - olive yellow
    // Arrows have R and G > 240, blind zones have R and G around 190
    if (g > 240 && r > 240 && b < 50) return true;

    // 2. Green to Yellow-Green - Slot/parking lines
    // Can be pure green (0, 255, 0) or yellow-green (128, 255, 0)
    // Key: high G, low B, R can vary from low to medium
    if (g > 180 && b < 100 && r < 200) {
        // Exclude yellow (already handled above) by requiring G > R or similar G and R
        // This catches green and yellow-green parking slot lines
        return true;
    }

    // 3. Blue - Lane lines
    // RGB approximately (76, 128, 254) - blue with medium green
    // Key: B is highest (>200), G is medium (100-150), R is low (<100)
    // Must distinguish from cyan (33, 133, 136) where B ≈ G
    if (b > 200 && g > 100 && g < 160 && r < 100) {
        // Ensure B is significantly higher than G (not cyan)
        if (b - g > 80) return true;
    }

    // === FEATURES TO EXCLUDE ===
    // Cyan/Teal (speed bumps, area boundaries) - high B AND high G - EXCLUDED
    // Purple/Magenta (road background) - high R and B, low G - EXCLUDED
    // White (noise/edges) - all channels high - EXCLUDED
    // Black (vehicle area) - all channels low - EXCLUDED

    return false;
}

// Convert IPM pixel coordinate to vehicle coordinate
Eigen::Vector3d ipmToVehicle(int u, int v) {
    Eigen::Vector3d pt;
    // IPM image center is at vehicle center
    // X-axis: left is positive (perpendicular to vehicle heading)
    // Y-axis: forward is positive (vehicle heading direction)
    pt.x() = -(ipmCenterX - u) * pixelScale;
    pt.y() = (ipmCenterY - v) * pixelScale + rearAxleOffset;
    pt.z() = 0.0;  // ground plane
    return pt;
}

// Delete close points
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out, float thres) {
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }
    size_t j = 0;
    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        double dist2 = cloud_in.points[i].x * cloud_in.points[i].x +
                       cloud_in.points[i].y * cloud_in.points[i].y;
        if (dist2 < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size()) {
        cloud_out.points.resize(j);
    }
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

// Delete far points
template <typename PointT>
void removeFarPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                         pcl::PointCloud<PointT> &cloud_out, float thres) {
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }
    size_t j = 0;
    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        double dist2 = cloud_in.points[i].x * cloud_in.points[i].x +
                       cloud_in.points[i].y * cloud_in.points[i].y;
        if (dist2 > thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size()) {
        cloud_out.points.resize(j);
    }
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

// Image callback
void ipmImageHandler(const sensor_msgs::ImageConstPtr& imageMsg) {
    mBuf.lock();
    ipmImageBuf.push(imageMsg);
    mBuf.unlock();
}

// Odometry callback
void odomHandler(const nav_msgs::OdometryConstPtr& odomMsg) {
    mBuf.lock();
    odomBuf.push(odomMsg);
    mBuf.unlock();
}

// Extract feature point cloud from IPM semantic image
void extractFeatureCloud(const cv::Mat& img, pcl::PointCloud<PointType>::Ptr& cloud) {
    cloud->clear();

    int rows = img.rows;
    int cols = img.cols;

    // Update IPM dimensions if different
    if (rows != ipmHeight || cols != ipmWidth) {
        ipmHeight = rows;
        ipmWidth = cols;
        ipmCenterX = cols / 2;
        ipmCenterY = rows / 2;
    }

    for (int v = 0; v < rows; v += pixelSkip) {
        const uchar* rowPtr = img.ptr<uchar>(v);
        for (int u = 0; u < cols; u += pixelSkip) {
            int b = rowPtr[3 * u];
            int g = rowPtr[3 * u + 1];
            int r = rowPtr[3 * u + 2];

            // Skip non-feature pixels (pass pixel coordinates for region filtering)
            if (!isFeaturePixel(b, g, r, u, v)) continue;

            // Convert to vehicle coordinate
            Eigen::Vector3d ptVehicle = ipmToVehicle(u, v);

            // Create point
            PointType pt;
            pt.x = ptVehicle.x();
            pt.y = ptVehicle.y();
            pt.z = ptVehicle.z();
            pt.r = r;
            pt.g = g;
            pt.b = b;

            cloud->push_back(pt);
        }
    }
}

// Process synchronized image and odometry
void processFrame(const sensor_msgs::ImageConstPtr& imgMsg,
                  const nav_msgs::OdometryConstPtr& odomMsg) {

    // Convert ROS image to OpenCV
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(imgMsg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Extract feature point cloud
    pcl::PointCloud<PointType>::Ptr featureCloud(new pcl::PointCloud<PointType>());
    extractFeatureCloud(cv_ptr->image, featureCloud);

    // Remove close and far points
    pcl::PointCloud<PointType>::Ptr filteredCloud(new pcl::PointCloud<PointType>());
    removeClosedPointCloud(*featureCloud, *filteredCloud, closePointThresh);
    removeFarPointCloud(*filteredCloud, *featureCloud, farPointThresh);

    // Voxel grid downsampling
    pcl::PointCloud<PointType>::Ptr downsampledCloud(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> voxelFilter;
    voxelFilter.setInputCloud(featureCloud);
    voxelFilter.setLeafSize(pointCloudLeafSize, pointCloudLeafSize, pointCloudLeafSize);
    voxelFilter.filter(*downsampledCloud);

    std::cout << "IPM feature points: " << downsampledCloud->points.size() << std::endl;

    // Publish point cloud
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg(*downsampledCloud, cloudMsg);
    cloudMsg.header.stamp = imgMsg->header.stamp;
    cloudMsg.header.frame_id = "camera0_link";
    pubCameraCloudFrame.publish(cloudMsg);

    // Forward odometry (optional, for reference)
    nav_msgs::Odometry odomOut = *odomMsg;
    odomOut.header.stamp = imgMsg->header.stamp;
    pubCurrentOdom.publish(odomOut);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "pointCloudFromIPM");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Get parameters from config file
    nh.param<int>("ipmWidth", ipmWidth, 1000);
    nh.param<int>("ipmHeight", ipmHeight, 1000);
    nh.param<double>("pixelScale", pixelScale, 0.02);
    nh.param<int>("ipmCenterX", ipmCenterX, 500);
    nh.param<int>("ipmCenterY", ipmCenterY, 500);
    nh.param<double>("rearAxleOffset", rearAxleOffset, 1.32);

    nh.param<float>("pointCloudLeafSize", pointCloudLeafSize, 0.1);
    nh.param<double>("farPointThresh", farPointThresh, 20.0);
    nh.param<double>("closePointThresh", closePointThresh, 0.1);

    nh.param<int>("slotColorB", slotColorB, 0);
    nh.param<int>("slotColorG", slotColorG, 255);
    nh.param<int>("slotColorR", slotColorR, 0);
    nh.param<int>("slotColorThresh", slotColorThresh, 80);

    nh.param<int>("arrowColorB", arrowColorB, 0);
    nh.param<int>("arrowColorG", arrowColorG, 255);
    nh.param<int>("arrowColorR", arrowColorR, 255);
    nh.param<int>("arrowColorThresh", arrowColorThresh, 80);

    nh.param<int>("oliveMinR", oliveMinR, 150);
    nh.param<int>("oliveMaxR", oliveMaxR, 220);
    nh.param<int>("oliveMinG", oliveMinG, 150);
    nh.param<int>("oliveMaxG", oliveMaxG, 220);
    nh.param<int>("oliveMaxB", oliveMaxB, 80);

    nh.param<int>("dashColorB", dashColorB, 255);
    nh.param<int>("dashColorG", dashColorG, 0);
    nh.param<int>("dashColorR", dashColorR, 0);
    nh.param<int>("dashColorThresh", dashColorThresh, 80);

    nh.param<int>("pixelSkip", pixelSkip, 2);
    nh.param<double>("minProcessInterval", minProcessInterval, 0.05);

    // Subscribe to IPM image and odometry
    std::string ipmTopic, odomTopic;
    nh.param<std::string>("ipm_topic", ipmTopic, "/ipm_semantic_label");
    nh.param<std::string>("odom_topic", odomTopic, "/gps_odom");
    // Prefer node-private params (set inside <node> in launch) over global ones.
    pnh.param<std::string>("ipm_topic", ipmTopic, ipmTopic);
    pnh.param<std::string>("odom_topic", odomTopic, odomTopic);

    ros::Subscriber subIpmImage = nh.subscribe(ipmTopic, 100, ipmImageHandler);
    ros::Subscriber subOdom = nh.subscribe(odomTopic, 100, odomHandler);

    // Publish point cloud
    pubCameraCloudFrame = nh.advertise<sensor_msgs::PointCloud2>("/cameraCloudFrame", 100);
    pubCurrentOdom = nh.advertise<nav_msgs::Odometry>("/ipm_odom", 100);

    std::cout << "pointCloudFromIPM node started" << std::endl;
    std::cout << "Subscribing to: " << ipmTopic << " and " << odomTopic << std::endl;
    std::cout << "IPM parameters: " << ipmWidth << "x" << ipmHeight << ", scale=" << pixelScale << " m/pixel" << std::endl;

    ros::Rate rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        // Process when both image and odometry are available
        if (!ipmImageBuf.empty() && !odomBuf.empty()) {
            double timeImage = ipmImageBuf.front()->header.stamp.toSec();
            double timeOdom = odomBuf.front()->header.stamp.toSec();

            // Synchronize by timestamp
            // Remove old odometry data
            while (!odomBuf.empty() && odomBuf.front()->header.stamp.toSec() < timeImage - 0.1) {
                mBuf.lock();
                odomBuf.pop();
                mBuf.unlock();
            }

            if (odomBuf.empty()) {
                rate.sleep();
                continue;
            }

            timeOdom = odomBuf.front()->header.stamp.toSec();

            // Check time difference
            if (fabs(timeImage - timeOdom) > 0.2) {
                // Time difference too large, skip this image
                if (timeImage < timeOdom) {
                    mBuf.lock();
                    ipmImageBuf.pop();
                    mBuf.unlock();
                }
                rate.sleep();
                continue;
            }

            // Rate limiting
            if (timeImage - lastProcessTime < minProcessInterval) {
                mBuf.lock();
                ipmImageBuf.pop();
                mBuf.unlock();
                rate.sleep();
                continue;
            }

            // Get synchronized messages
            sensor_msgs::ImageConstPtr imgMsg;
            nav_msgs::OdometryConstPtr odomMsg;

            mBuf.lock();
            imgMsg = ipmImageBuf.front();
            ipmImageBuf.pop();
            odomMsg = odomBuf.front();
            // Keep odometry for next frame (don't pop)
            mBuf.unlock();

            // Process frame
            processFrame(imgMsg, odomMsg);
            lastProcessTime = timeImage;
        }

        rate.sleep();
    }

    std::cout << "pointCloudFromIPM node finished" << std::endl;
    return 0;
}
