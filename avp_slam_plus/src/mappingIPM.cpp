// Mapping node for IPM semantic segmentation mode
// Uses GPS odometry directly for pose estimation (same as avp_mapping)
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
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/opencv.hpp>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Dense>
#include <mutex>
#include <queue>
#include <cmath>

// Mapping node using GPS odometry for pose
// Directly transforms point cloud to world frame using GPS pose
// Creates global map by accumulating keyframes

typedef pcl::PointXYZRGB PointType;

ros::Publisher pubCurrentFeature;
ros::Publisher pubCurrentFeatureInWorld;
ros::Publisher pubGlobalFeature;
ros::Publisher pubCurrentPose;
ros::Publisher pubPath;

pcl::PointCloud<PointType>::Ptr currentFeatureCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr currentFeatureCloudInWorld(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalFeatureCloud(new pcl::PointCloud<PointType>());

nav_msgs::Path pathMsg;

// Buffers for synchronization
std::queue<sensor_msgs::PointCloud2ConstPtr> cloudBuf;
std::queue<nav_msgs::OdometryConstPtr> odomBuf;
std::mutex mBuf;

bool systemInitial = false;

double odomKeyFramDisThresh = 0.5;
double odomKeyFrameRotThresh = 0.1;  // radians (~5.7 degrees)

double lastPoseX = 0;
double lastPoseY = 0;
double lastPoseZ = 0;
double lastPoseYaw = 0;

float currentX = 0;
float currentY = 0;
float currentZ = 0;
float currentRoll = 0;
float currentPitch = 0;
float currentYaw = 0;

int frameCount = 0;
int invalidColorThresh = 30;
// Olive/dark-yellow suppression to remove pillar noise from semantic map
int oliveMinR = 150;
int oliveMaxR = 220;
int oliveMinG = 150;
int oliveMaxG = 220;
int oliveMaxB = 80;

std::string mapSaveLocation = "/home/chaoszzss/my_ws/src/AVP-SLAM-PLUS/avp_slam_plus/data/ipmMap.pcd";
float pointCloudLeafSize = 0.1;
bool mapSave = true;

// Transform point cloud according to pose
pcl::PointCloud<PointType>::Ptr transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn,
    const Eigen::Affine3f& transCur) {

    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    for (int i = 0; i < cloudSize; ++i) {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].r = pointFrom.r;
        cloudOut->points[i].g = pointFrom.g;
        cloudOut->points[i].b = pointFrom.b;
    }
    return cloudOut;
}

// Quaternion to Euler angles
void quaternionToEuler(const geometry_msgs::Quaternion& q,
                       double& roll, double& pitch, double& yaw) {
    tf::Quaternion tfQ(q.x, q.y, q.z, q.w);
    tf::Matrix3x3(tfQ).getRPY(roll, pitch, yaw);
}

// Point cloud callback
void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& cloudMsg) {
    mBuf.lock();
    cloudBuf.push(cloudMsg);
    mBuf.unlock();
}

// Odometry callback
void odomHandler(const nav_msgs::OdometryConstPtr& odomMsg) {
    mBuf.lock();
    odomBuf.push(odomMsg);
    mBuf.unlock();
}

// Process synchronized cloud and odometry
void processFrame(const sensor_msgs::PointCloud2ConstPtr& cloudMsg,
                  const nav_msgs::OdometryConstPtr& odomMsg) {

    // Convert point cloud
    pcl::PointCloud<PointType> cloudIn;
    pcl::fromROSMsg(*cloudMsg, cloudIn);

    // Extract feature points (filter out dark/ground pixels)
    currentFeatureCloud->clear();
    for (size_t i = 0; i < cloudIn.points.size(); i++) {
        PointType pi = cloudIn.points[i];
        int r = int(pi.r);
        int g = int(pi.g);
        int b = int(pi.b);
        // Drop olive/dark yellow (parking pillars / blind zones)
        if (r >= oliveMinR && r <= oliveMaxR &&
            g >= oliveMinG && g <= oliveMaxG &&
            b < oliveMaxB) {
            continue;
        }
        // Keep only bright/colored features
        if (r < invalidColorThresh && b < invalidColorThresh && g < invalidColorThresh) {
            continue;
        }
        currentFeatureCloud->push_back(pi);
    }

    // Check if enough points
    if (currentFeatureCloud->points.size() < 10) {
        std::cout << "Not enough feature points: " << currentFeatureCloud->points.size() << std::endl;
        return;
    }

    // Extract pose from GPS odometry
    currentX = odomMsg->pose.pose.position.x;
    currentY = odomMsg->pose.pose.position.y;
    currentZ = 0;  // Keep on ground plane

    double roll, pitch, yaw;
    quaternionToEuler(odomMsg->pose.pose.orientation, roll, pitch, yaw);
    currentRoll = 0;    // Force roll to 0 (ground vehicle)
    currentPitch = 0;   // Force pitch to 0 (ground vehicle)
    currentYaw = yaw;

    // Create transformation matrix from GPS pose
    Eigen::Affine3f transWorldCurrent = pcl::getTransformation(
        currentX, currentY, currentZ, currentRoll, currentPitch, currentYaw);

    // Broadcast current feature in camera frame
    sensor_msgs::PointCloud2 cameraCloudFrameMsg;
    pcl::toROSMsg(*currentFeatureCloud, cameraCloudFrameMsg);
    cameraCloudFrameMsg.header.stamp = cloudMsg->header.stamp;
    cameraCloudFrameMsg.header.frame_id = "camera0_link";
    pubCurrentFeature.publish(cameraCloudFrameMsg);

    // Initialize system
    if (!systemInitial) {
        // Transform first frame to world and add to global map
        *currentFeatureCloudInWorld = *transformPointCloud(currentFeatureCloud, transWorldCurrent);
        *globalFeatureCloud = *globalFeatureCloud + *currentFeatureCloudInWorld;

        lastPoseX = currentX;
        lastPoseY = currentY;
        lastPoseZ = currentZ;
        lastPoseYaw = currentYaw;
        systemInitial = true;

        std::cout << "System initialized at GPS pose: x=" << currentX << " y=" << currentY << " yaw=" << currentYaw << std::endl;
        return;
    }

    // Broadcast TF
    static tf::TransformBroadcaster tfMap2Camera;
    tf::Transform mapToCamera = tf::Transform(
        tf::createQuaternionFromRPY(currentRoll, currentPitch, currentYaw),
        tf::Vector3(currentX, currentY, currentZ));
    tfMap2Camera.sendTransform(tf::StampedTransform(mapToCamera, cloudMsg->header.stamp, "map", "camera0_link"));

    // Broadcast pose
    geometry_msgs::Quaternion cameraPoseQuat = tf::createQuaternionMsgFromRollPitchYaw(currentRoll, currentPitch, currentYaw);
    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "map";
    odomAftMapped.child_frame_id = "camera0_link";
    odomAftMapped.header.stamp = cloudMsg->header.stamp;
    odomAftMapped.pose.pose.orientation = cameraPoseQuat;
    odomAftMapped.pose.pose.position.x = currentX;
    odomAftMapped.pose.pose.position.y = currentY;
    odomAftMapped.pose.pose.position.z = currentZ;
    pubCurrentPose.publish(odomAftMapped);

    // Add to path
    geometry_msgs::PoseStamped poseStamped;
    poseStamped.header = odomAftMapped.header;
    poseStamped.pose = odomAftMapped.pose.pose;
    pathMsg.header = odomAftMapped.header;
    pathMsg.poses.push_back(poseStamped);
    if (pathMsg.poses.size() > 10000) {
        pathMsg.poses.erase(pathMsg.poses.begin());
    }
    pubPath.publish(pathMsg);

    // Transform current frame to world coordinate
    std::cout << "GPS pose: x=" << currentX << " y=" << currentY << " yaw=" << currentYaw << std::endl;
    *currentFeatureCloudInWorld = *transformPointCloud(currentFeatureCloud, transWorldCurrent);

    sensor_msgs::PointCloud2 cameraCloudCurrentInWorldMsg;
    pcl::toROSMsg(*currentFeatureCloudInWorld, cameraCloudCurrentInWorldMsg);
    cameraCloudCurrentInWorldMsg.header.stamp = cloudMsg->header.stamp;
    cameraCloudCurrentInWorldMsg.header.frame_id = "map";
    pubCurrentFeatureInWorld.publish(cameraCloudCurrentInWorldMsg);

    // Check keyframe condition (distance or rotation threshold)
    double disSq = (currentX - lastPoseX) * (currentX - lastPoseX) +
                   (currentY - lastPoseY) * (currentY - lastPoseY);
    double yawDiff = fabs(currentYaw - lastPoseYaw);
    if (yawDiff > M_PI) yawDiff = 2 * M_PI - yawDiff;

    frameCount++;
    if (disSq > odomKeyFramDisThresh * odomKeyFramDisThresh || yawDiff > odomKeyFrameRotThresh) {
        // Add to global map
        *globalFeatureCloud = *globalFeatureCloud + *currentFeatureCloudInWorld;

        // Downsample global map
        pcl::PointCloud<PointType> globalMapDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(globalFeatureCloud);
        downSizeFilter.setLeafSize(pointCloudLeafSize, pointCloudLeafSize, pointCloudLeafSize);
        downSizeFilter.filter(globalMapDS);
        *globalFeatureCloud = globalMapDS;

        frameCount = 0;
        lastPoseX = currentX;
        lastPoseY = currentY;
        lastPoseZ = currentZ;
        lastPoseYaw = currentYaw;

        std::cout << "Keyframe added! Global map size: " << globalFeatureCloud->points.size() << std::endl;
    }

    // Broadcast global map
    sensor_msgs::PointCloud2 cameraCloudGlobalMapMsg;
    pcl::toROSMsg(*globalFeatureCloud, cameraCloudGlobalMapMsg);
    cameraCloudGlobalMapMsg.header.stamp = cloudMsg->header.stamp;
    cameraCloudGlobalMapMsg.header.frame_id = "map";
    pubGlobalFeature.publish(cameraCloudGlobalMapMsg);
}

// Save map callback
void saveMapCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    std::cout << "Map save triggered..." << std::endl;
    if (globalFeatureCloud->points.size() > 0) {
        pcl::io::savePCDFileASCII(mapSaveLocation, *globalFeatureCloud);
        std::cout << "Map saved to: " << mapSaveLocation << std::endl;
        std::cout << "Total points: " << globalFeatureCloud->points.size() << std::endl;
    } else {
        std::cout << "Map is empty, nothing to save." << std::endl;
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "mappingIPM");
    ros::NodeHandle nh;

    // Get parameters
    nh.param<std::string>("mapSaveLocation", mapSaveLocation,
        "/home/chaoszzss/my_ws/src/AVP-SLAM-PLUS/avp_slam_plus/data/ipmMap.pcd");
    nh.param<double>("odomKeyFramDisThresh", odomKeyFramDisThresh, 0.5);
    nh.param<int>("InvalidColorThresh", invalidColorThresh, 30);
    nh.param<int>("oliveMinR", oliveMinR, 150);
    nh.param<int>("oliveMaxR", oliveMaxR, 220);
    nh.param<int>("oliveMinG", oliveMinG, 150);
    nh.param<int>("oliveMaxG", oliveMaxG, 220);
    nh.param<int>("oliveMaxB", oliveMaxB, 80);
    nh.param<float>("pointCloudLeafSize", pointCloudLeafSize, 0.1);
    nh.param<bool>("mapSave", mapSave, true);

    // Subscribe to point cloud and odometry
    ros::Subscriber subCloud = nh.subscribe<sensor_msgs::PointCloud2>("/cameraCloudFrame", 100, cloudHandler);
    ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/ipm_odom", 100, odomHandler);
    ros::Subscriber subSaveMap = nh.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10, saveMapCallback);

    // Publishers
    pubCurrentFeature = nh.advertise<sensor_msgs::PointCloud2>("/currentFeature", 100);
    pubCurrentFeatureInWorld = nh.advertise<sensor_msgs::PointCloud2>("/currentFeatureInWorld", 100);
    pubGlobalFeature = nh.advertise<sensor_msgs::PointCloud2>("/globalFeatureMap", 100);
    pubCurrentPose = nh.advertise<nav_msgs::Odometry>("/currentPose", 100);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100);

    std::cout << "mappingIPM node started" << std::endl;
    std::cout << "Using GPS odometry for pose estimation" << std::endl;
    std::cout << "Keyframe distance threshold: " << odomKeyFramDisThresh << " m" << std::endl;

    ros::Rate rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        // Synchronize cloud and odometry
        if (!cloudBuf.empty() && !odomBuf.empty()) {
            double timeCloud = cloudBuf.front()->header.stamp.toSec();
            double timeOdom = odomBuf.front()->header.stamp.toSec();

            // Remove old odometry
            while (!odomBuf.empty() && odomBuf.front()->header.stamp.toSec() < timeCloud - 0.1) {
                mBuf.lock();
                odomBuf.pop();
                mBuf.unlock();
            }

            if (odomBuf.empty()) {
                rate.sleep();
                continue;
            }

            timeOdom = odomBuf.front()->header.stamp.toSec();

            // Check time sync
            if (fabs(timeCloud - timeOdom) > 0.2) {
                if (timeCloud < timeOdom) {
                    mBuf.lock();
                    cloudBuf.pop();
                    mBuf.unlock();
                }
                rate.sleep();
                continue;
            }

            // Get synchronized messages
            sensor_msgs::PointCloud2ConstPtr cloudMsg;
            nav_msgs::OdometryConstPtr odomMsg;

            mBuf.lock();
            cloudMsg = cloudBuf.front();
            cloudBuf.pop();
            odomMsg = odomBuf.front();
            // Keep odometry for interpolation
            mBuf.unlock();

            // Process frame
            processFrame(cloudMsg, odomMsg);
        }

        rate.sleep();
    }

    // Save map on exit
    if (mapSave && globalFeatureCloud->points.size() > 0) {
        std::cout << "Saving map on exit..." << std::endl;
        pcl::io::savePCDFileASCII(mapSaveLocation, *globalFeatureCloud);
        std::cout << "Map saved to: " << mapSaveLocation << std::endl;
        std::cout << "Total points: " << globalFeatureCloud->points.size() << std::endl;
    }

    std::cout << "mappingIPM node finished" << std::endl;
    return 0;
}
