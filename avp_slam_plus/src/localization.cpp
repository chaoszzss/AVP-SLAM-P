// IPM localization node.
// - Loads prebuilt PCD map from mappingIPM.
// - Uses the same color filtering strategy as mappingIPM for feature consistency.
// - Registers current frame cloud against global map with NDT/ICP.
// - Uses GPS odometry as registration prior (from /ipm_odom or /gps_odom).

#include <cmath>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

typedef pcl::PointXYZRGB PointType;

ros::Publisher pubCurrentFeature;
ros::Publisher pubCurrentFeatureInWorld;
ros::Publisher pubGlobalFeature;
ros::Publisher pubCurrentPose;
ros::Publisher pubGpsPoseInMap;
ros::Publisher pubIcpPath;
ros::Publisher pubGpsPath;

nav_msgs::Path icpPathMsg;
nav_msgs::Path gpsPathMsg;
int pathKeepSize = 10000;
bool saveTumTrajectory = true;
std::string icpTumSavePath = "/home/chaoszzss/my_ws/src/AVP-SLAM-PLUS/avp_slam_plus/output/aloop_estimation.tum";
std::string gpsTumSavePath = "/home/chaoszzss/my_ws/src/AVP-SLAM-PLUS/avp_slam_plus/output/aloop_gps.tum";
std::ofstream icpTumFile;
std::ofstream gpsTumFile;

pcl::PointCloud<PointType>::Ptr currentFeatureCloud(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr currentFeatureCloudInWorld(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalFeatureCloud(new pcl::PointCloud<PointType>());

std::queue<sensor_msgs::PointCloud2ConstPtr> cloudBuf;
std::queue<nav_msgs::OdometryConstPtr> odomBuf;
std::mutex mBuf;

bool systemInitial = false;
bool globalMapLoad = false;

// Feature filtering params (aligned with mappingIPM)
int invalidColorThresh = 30;
int oliveMinR = 150;
int oliveMaxR = 220;
int oliveMinG = 150;
int oliveMaxG = 220;
int oliveMaxB = 80;

// Registration params
bool useICP = false;
bool useNDT = true;
double icpMaxCorrespondenceDistance = 20.0;
int icpMaximumIterations = 100;
double icpTransformationEpsilon = 1e-10;
double icpEuclideanFitnessEpsilon = 0.001;
double icpFitnessScoreThresh = 0.3;
double ndtTransformationEpsilon = 0.01;
double ndtResolution = 1.0;
double ndtFitnessScoreThresh = 0.5;

// Data sync / prior params
bool useGpsPrior = true;
double syncTolerance = 0.2;
double odomLookback = 0.1;
std::string odomTopic = "/ipm_odom";
std::string mapSaveLocation = "/home/chaoszzss/my_ws/src/AVP-SLAM-PLUS/avp_slam_plus/data/ipmMap.pcd";

// Current estimated pose
float currentX = 0.0f;
float currentY = 0.0f;
float currentZ = 0.0f;
float currentRoll = 0.0f;
float currentPitch = 0.0f;
float currentYaw = 0.0f;
Eigen::Affine3f transWorldCurrent = pcl::getTransformation(0, 0, 0, 0, 0, 0);

pcl::PointCloud<PointType>::Ptr transformPointCloud(
    const pcl::PointCloud<PointType>::Ptr &cloudIn,
    const Eigen::Affine3f &transCur) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    cloudOut->resize(cloudIn->size());

    for (size_t i = 0; i < cloudIn->size(); ++i) {
        const auto &pointFrom = cloudIn->points[i];
        auto &pointTo = cloudOut->points[i];
        pointTo.x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        pointTo.y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        pointTo.z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        pointTo.r = pointFrom.r;
        pointTo.g = pointFrom.g;
        pointTo.b = pointFrom.b;
    }
    return cloudOut;
}

std::string getParentDirectory(const std::string &filePath) {
    const size_t slashPos = filePath.find_last_of('/');
    if (slashPos == std::string::npos) {
        return std::string();
    }
    return filePath.substr(0, slashPos);
}

bool ensureDirectoryRecursive(const std::string &dirPath) {
    if (dirPath.empty()) {
        return true;
    }

    std::string currentPath;
    size_t index = 0;
    if (dirPath.front() == '/') {
        currentPath = "/";
        index = 1;
    }

    while (index <= dirPath.size()) {
        const size_t slashPos = dirPath.find('/', index);
        const std::string part = (slashPos == std::string::npos)
                                     ? dirPath.substr(index)
                                     : dirPath.substr(index, slashPos - index);

        if (!part.empty()) {
            if (!currentPath.empty() && currentPath.back() != '/') {
                currentPath += "/";
            }
            currentPath += part;

            if (::mkdir(currentPath.c_str(), 0755) != 0 && errno != EEXIST) {
                ROS_ERROR_STREAM("Failed to create directory: " << currentPath << ", errno=" << errno);
                return false;
            }
        }

        if (slashPos == std::string::npos) {
            break;
        }
        index = slashPos + 1;
    }
    return true;
}

bool initTumWriters() {
    if (!saveTumTrajectory) {
        return true;
    }

    if (!ensureDirectoryRecursive(getParentDirectory(icpTumSavePath))) {
        return false;
    }
    if (!ensureDirectoryRecursive(getParentDirectory(gpsTumSavePath))) {
        return false;
    }

    icpTumFile.open(icpTumSavePath, std::ios::out | std::ios::trunc);
    gpsTumFile.open(gpsTumSavePath, std::ios::out | std::ios::trunc);
    if (!icpTumFile.is_open() || !gpsTumFile.is_open()) {
        ROS_ERROR_STREAM("Failed to open TUM files. ICP: " << icpTumSavePath
                        << ", GPS: " << gpsTumSavePath);
        return false;
    }

    icpTumFile << std::fixed << std::setprecision(9);
    gpsTumFile << std::fixed << std::setprecision(9);
    ROS_INFO_STREAM("Saving ICP TUM to: " << icpTumSavePath);
    ROS_INFO_STREAM("Saving GPS TUM to: " << gpsTumSavePath);
    return true;
}

void writeTumPose(
    std::ofstream &tumFile,
    const ros::Time &stamp,
    const float x,
    const float y,
    const float z,
    const float roll,
    const float pitch,
    const float yaw) {
    if (!saveTumTrajectory || !tumFile.is_open()) {
        return;
    }

    const tf::Quaternion q = tf::createQuaternionFromRPY(roll, pitch, yaw);
    tumFile << stamp.toSec() << " "
            << x << " " << y << " " << z << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << '\n';
    tumFile.flush();
}

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloudMsg) {
    std::lock_guard<std::mutex> lock(mBuf);
    cloudBuf.push(cloudMsg);
}

void odomHandler(const nav_msgs::OdometryConstPtr &odomMsg) {
    std::lock_guard<std::mutex> lock(mBuf);
    odomBuf.push(odomMsg);
}

void initposeHandler(const geometry_msgs::PoseWithCovarianceStampedConstPtr &initpose) {
    double roll, pitch, yaw;
    tf::Quaternion q(
        initpose->pose.pose.orientation.x,
        initpose->pose.pose.orientation.y,
        initpose->pose.pose.orientation.z,
        initpose->pose.pose.orientation.w);
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

    transWorldCurrent = pcl::getTransformation(
        initpose->pose.pose.position.x,
        initpose->pose.pose.position.y,
        initpose->pose.pose.position.z,
        roll,
        pitch,
        yaw);
    systemInitial = true;

    ROS_INFO_STREAM("Set initial pose: x=" << initpose->pose.pose.position.x
                    << " y=" << initpose->pose.pose.position.y
                    << " yaw=" << yaw);
}

bool loadMap() {
    globalFeatureCloud->clear();
    ROS_INFO_STREAM("Loading map from: " << mapSaveLocation);
    if (pcl::io::loadPCDFile(mapSaveLocation, *globalFeatureCloud) < 0) {
        ROS_ERROR_STREAM("Failed to load map: " << mapSaveLocation);
        return false;
    }
    if (globalFeatureCloud->empty()) {
        ROS_ERROR("Loaded map is empty.");
        return false;
    }
    globalMapLoad = true;
    ROS_INFO_STREAM("Map loaded, points: " << globalFeatureCloud->points.size());
    return true;
}

void extractFeatureCloud(const sensor_msgs::PointCloud2ConstPtr &cameraCloudMsg) {
    pcl::PointCloud<PointType> cameraCloudIn;
    pcl::fromROSMsg(*cameraCloudMsg, cameraCloudIn);

    currentFeatureCloud->clear();
    currentFeatureCloud->reserve(cameraCloudIn.points.size());
    for (const auto &pi : cameraCloudIn.points) {
        const int r = static_cast<int>(pi.r);
        const int g = static_cast<int>(pi.g);
        const int b = static_cast<int>(pi.b);
        if (r >= oliveMinR && r <= oliveMaxR &&
            g >= oliveMinG && g <= oliveMaxG &&
            b < oliveMaxB) {
            continue;
        }
        if (r < invalidColorThresh && g < invalidColorThresh && b < invalidColorThresh) {
            continue;
        }
        currentFeatureCloud->push_back(pi);
    }
}

bool getGpsPriorTransform(const nav_msgs::OdometryConstPtr &odomMsg, Eigen::Affine3f &prior) {
    if (!odomMsg) {
        return false;
    }

    double roll, pitch, yaw;
    tf::Quaternion q(
        odomMsg->pose.pose.orientation.x,
        odomMsg->pose.pose.orientation.y,
        odomMsg->pose.pose.orientation.z,
        odomMsg->pose.pose.orientation.w);
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // Keep ground-vehicle assumption consistent with mappingIPM.
    prior = pcl::getTransformation(
        odomMsg->pose.pose.position.x,
        odomMsg->pose.pose.position.y,
        0.0,
        0.0,
        0.0,
        yaw);
    return true;
}

bool registerToMap(const Eigen::Affine3f &guess, Eigen::Affine3f &result) {
    if (!globalMapLoad || globalFeatureCloud->empty()) {
        ROS_ERROR_THROTTLE(1.0, "Global map is not loaded.");
        return false;
    }
    if (currentFeatureCloud->size() < 10) {
        return false;
    }

    Eigen::Affine3f estimate = guess;

    if (useNDT) {
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(ndtTransformationEpsilon);
        ndt.setResolution(ndtResolution);
        ndt.setInputSource(currentFeatureCloud);
        ndt.setInputTarget(globalFeatureCloud);

        pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
        ndt.align(*aligned, estimate.matrix());
        if (!ndt.hasConverged() || ndt.getFitnessScore() > ndtFitnessScoreThresh) {
            ROS_WARN_THROTTLE(1.0, "NDT failed, score=%f", ndt.getFitnessScore());
            return false;
        }
        estimate = ndt.getFinalTransformation();
    }

    if (useICP) {
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(icpMaxCorrespondenceDistance);
        icp.setMaximumIterations(icpMaximumIterations);
        icp.setTransformationEpsilon(icpTransformationEpsilon);
        icp.setEuclideanFitnessEpsilon(icpEuclideanFitnessEpsilon);
        icp.setInputSource(currentFeatureCloud);
        icp.setInputTarget(globalFeatureCloud);

        pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
        icp.align(*aligned, estimate.matrix());
        if (!icp.hasConverged() || icp.getFitnessScore() > icpFitnessScoreThresh) {
            ROS_WARN_THROTTLE(1.0, "ICP failed, score=%f", icp.getFitnessScore());
            return false;
        }
        estimate = icp.getFinalTransformation();
    }

    result = estimate;
    return true;
}

void publishCurrentAndMap(const ros::Time &stamp) {
    sensor_msgs::PointCloud2 currentMsg;
    pcl::toROSMsg(*currentFeatureCloud, currentMsg);
    currentMsg.header.stamp = stamp;
    currentMsg.header.frame_id = "camera0_link";
    pubCurrentFeature.publish(currentMsg);

    if (globalMapLoad) {
        sensor_msgs::PointCloud2 mapMsg;
        pcl::toROSMsg(*globalFeatureCloud, mapMsg);
        mapMsg.header.stamp = stamp;
        mapMsg.header.frame_id = "map";
        pubGlobalFeature.publish(mapMsg);
    }
}

void pushPoseToPath(
    const ros::Time &stamp,
    const float x,
    const float y,
    const float z,
    const float roll,
    const float pitch,
    const float yaw,
    nav_msgs::Path &pathMsg,
    ros::Publisher &pathPub) {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = "map";
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = z;
    pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);

    pathMsg.header = pose.header;
    pathMsg.poses.push_back(pose);
    if (pathMsg.poses.size() > static_cast<size_t>(pathKeepSize)) {
        pathMsg.poses.erase(pathMsg.poses.begin());
    }
    pathPub.publish(pathMsg);
}

void publishGpsPoseInMap(const Eigen::Affine3f &gpsPrior, const ros::Time &stamp) {
    float gpsX = 0.0f;
    float gpsY = 0.0f;
    float gpsZ = 0.0f;
    float gpsRoll = 0.0f;
    float gpsPitch = 0.0f;
    float gpsYaw = 0.0f;
    pcl::getTranslationAndEulerAngles(gpsPrior, gpsX, gpsY, gpsZ, gpsRoll, gpsPitch, gpsYaw);

    nav_msgs::Odometry gpsPoseMsg;
    gpsPoseMsg.header.stamp = stamp;
    gpsPoseMsg.header.frame_id = "map";
    gpsPoseMsg.child_frame_id = "camera0_link";
    gpsPoseMsg.pose.pose.position.x = gpsX;
    gpsPoseMsg.pose.pose.position.y = gpsY;
    gpsPoseMsg.pose.pose.position.z = gpsZ;
    gpsPoseMsg.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0f, 0.0f, gpsYaw);
    pubGpsPoseInMap.publish(gpsPoseMsg);

    writeTumPose(gpsTumFile, stamp, gpsX, gpsY, gpsZ, 0.0f, 0.0f, gpsYaw);
    pushPoseToPath(stamp, gpsX, gpsY, gpsZ, 0.0f, 0.0f, gpsYaw, gpsPathMsg, pubGpsPath);
}

void publishPoseAndCloud(const ros::Time &stamp) {
    pcl::getTranslationAndEulerAngles(
        transWorldCurrent, currentX, currentY, currentZ, currentRoll, currentPitch, currentYaw);

    // Keep planar assumption (same as mappingIPM).
    currentRoll = 0.0f;
    currentPitch = 0.0f;
    transWorldCurrent = pcl::getTransformation(currentX, currentY, currentZ, currentRoll, currentPitch, currentYaw);

    static tf::TransformBroadcaster tfMap2Camera;
    tf::Transform mapToCamera(
        tf::createQuaternionFromRPY(currentRoll, currentPitch, currentYaw),
        tf::Vector3(currentX, currentY, currentZ));
    tfMap2Camera.sendTransform(tf::StampedTransform(mapToCamera, stamp, "map", "camera0_link"));

    nav_msgs::Odometry poseMsg;
    poseMsg.header.stamp = stamp;
    poseMsg.header.frame_id = "map";
    poseMsg.child_frame_id = "camera0_link";
    poseMsg.pose.pose.position.x = currentX;
    poseMsg.pose.pose.position.y = currentY;
    poseMsg.pose.pose.position.z = currentZ;
    poseMsg.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(currentRoll, currentPitch, currentYaw);
    pubCurrentPose.publish(poseMsg);
    writeTumPose(icpTumFile, stamp, currentX, currentY, currentZ, currentRoll, currentPitch, currentYaw);
    pushPoseToPath(stamp, currentX, currentY, currentZ, currentRoll, currentPitch, currentYaw, icpPathMsg, pubIcpPath);

    *currentFeatureCloudInWorld = *transformPointCloud(currentFeatureCloud, transWorldCurrent);
    sensor_msgs::PointCloud2 currentInWorldMsg;
    pcl::toROSMsg(*currentFeatureCloudInWorld, currentInWorldMsg);
    currentInWorldMsg.header.stamp = stamp;
    currentInWorldMsg.header.frame_id = "map";
    pubCurrentFeatureInWorld.publish(currentInWorldMsg);
}

void processFrame(
    const sensor_msgs::PointCloud2ConstPtr &cloudMsg,
    const nav_msgs::OdometryConstPtr &odomMsg) {
    extractFeatureCloud(cloudMsg);
    publishCurrentAndMap(cloudMsg->header.stamp);

    if (currentFeatureCloud->size() < 10) {
        ROS_WARN_THROTTLE(1.0, "Not enough feature points: %zu", currentFeatureCloud->size());
        return;
    }

    Eigen::Affine3f gpsPrior = transWorldCurrent;
    const bool hasGpsPrior = useGpsPrior && getGpsPriorTransform(odomMsg, gpsPrior);
    if (hasGpsPrior) {
        publishGpsPoseInMap(gpsPrior, cloudMsg->header.stamp);
    }

    if (!systemInitial) {
        transWorldCurrent = hasGpsPrior ? gpsPrior : pcl::getTransformation(0, 0, 0, 0, 0, 0);
        systemInitial = true;
        ROS_INFO_STREAM("Localization initialized. "
                        << (hasGpsPrior ? "Using GPS prior." : "Using identity prior."));
    }

    const Eigen::Affine3f guess = hasGpsPrior ? gpsPrior : transWorldCurrent;
    Eigen::Affine3f refined;
    if (registerToMap(guess, refined)) {
        transWorldCurrent = refined;
    } else if (hasGpsPrior) {
        // Keep output continuous even if one registration step fails.
        transWorldCurrent = gpsPrior;
    } else {
        return;
    }

    publishPoseAndCloud(cloudMsg->header.stamp);
}

bool fetchSynchronizedFrame(
    sensor_msgs::PointCloud2ConstPtr &cloudMsg,
    nav_msgs::OdometryConstPtr &odomMsg) {
    std::lock_guard<std::mutex> lock(mBuf);

    if (cloudBuf.empty()) {
        return false;
    }

    if (!useGpsPrior) {
        cloudMsg = cloudBuf.front();
        cloudBuf.pop();
        odomMsg.reset();
        return true;
    }

    if (odomBuf.empty()) {
        return false;
    }

    const double cloudTime = cloudBuf.front()->header.stamp.toSec();
    while (!odomBuf.empty() && odomBuf.front()->header.stamp.toSec() < cloudTime - odomLookback) {
        odomBuf.pop();
    }
    if (odomBuf.empty()) {
        return false;
    }

    const double odomTime = odomBuf.front()->header.stamp.toSec();
    if (std::fabs(cloudTime - odomTime) > syncTolerance) {
        if (cloudTime < odomTime) {
            cloudBuf.pop();
        } else {
            odomBuf.pop();
        }
        return false;
    }

    cloudMsg = cloudBuf.front();
    cloudBuf.pop();
    odomMsg = odomBuf.front();  // keep this odom for interpolation-like reuse
    return true;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "localization");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    nh.param<std::string>(
        "mapSaveLocation",
        mapSaveLocation,
        std::string("/home/chaoszzss/my_ws/src/AVP-SLAM-PLUS/avp_slam_plus/data/ipmMap.pcd"));
    // Prefer node-private params (set inside <node> in launch) over global ones.
    pnh.param<std::string>("mapSaveLocation", mapSaveLocation, mapSaveLocation);

    // Support both legacy and current key styles.
    nh.param<int>("invalidColorThresh", invalidColorThresh, invalidColorThresh);
    nh.param<int>("InvalidColorThresh", invalidColorThresh, invalidColorThresh);
    nh.param<int>("oliveMinR", oliveMinR, oliveMinR);
    nh.param<int>("oliveMaxR", oliveMaxR, oliveMaxR);
    nh.param<int>("oliveMinG", oliveMinG, oliveMinG);
    nh.param<int>("oliveMaxG", oliveMaxG, oliveMaxG);
    nh.param<int>("oliveMaxB", oliveMaxB, oliveMaxB);

    nh.param<bool>("useICP", useICP, useICP);
    nh.param<double>("icpMaxCorrespondenceDistance", icpMaxCorrespondenceDistance, icpMaxCorrespondenceDistance);
    nh.param<double>("IcpMaxCorrespondenceDistance", icpMaxCorrespondenceDistance, icpMaxCorrespondenceDistance);
    nh.param<int>("icpMaximumIterations", icpMaximumIterations, icpMaximumIterations);
    nh.param<int>("IcpMaximumIterations", icpMaximumIterations, icpMaximumIterations);
    nh.param<double>("icpTransformationEpsilon", icpTransformationEpsilon, icpTransformationEpsilon);
    nh.param<double>("IcpTransformationEpsilon", icpTransformationEpsilon, icpTransformationEpsilon);
    nh.param<double>("icpEuclideanFitnessEpsilon", icpEuclideanFitnessEpsilon, icpEuclideanFitnessEpsilon);
    nh.param<double>("IcpEuclideanFitnessEpsilon", icpEuclideanFitnessEpsilon, icpEuclideanFitnessEpsilon);
    nh.param<double>("icpFitnessScoreThresh", icpFitnessScoreThresh, icpFitnessScoreThresh);

    nh.param<bool>("useNDT", useNDT, useNDT);
    nh.param<double>("ndtTransformationEpsilon", ndtTransformationEpsilon, ndtTransformationEpsilon);
    nh.param<double>("ndtResolution", ndtResolution, ndtResolution);
    nh.param<double>("ndtFitnessScoreThresh", ndtFitnessScoreThresh, ndtFitnessScoreThresh);

    nh.param<bool>("useGpsPrior", useGpsPrior, true);
    nh.param<std::string>("odom_topic", odomTopic, std::string("/ipm_odom"));
    nh.param<double>("syncTolerance", syncTolerance, 0.2);
    nh.param<double>("odomLookback", odomLookback, 0.1);
    nh.param<int>("pathKeepSize", pathKeepSize, pathKeepSize);
    nh.param<bool>("saveTumTrajectory", saveTumTrajectory, true);
    nh.param<std::string>("icpTumSavePath", icpTumSavePath, icpTumSavePath);
    nh.param<std::string>("gpsTumSavePath", gpsTumSavePath, gpsTumSavePath);
    // Prefer node-private params (set inside <node> in launch) over global ones.
    pnh.param<bool>("useGpsPrior", useGpsPrior, useGpsPrior);
    pnh.param<std::string>("odom_topic", odomTopic, odomTopic);
    pnh.param<double>("syncTolerance", syncTolerance, syncTolerance);
    pnh.param<double>("odomLookback", odomLookback, odomLookback);
    pnh.param<int>("pathKeepSize", pathKeepSize, pathKeepSize);
    pnh.param<bool>("saveTumTrajectory", saveTumTrajectory, saveTumTrajectory);
    pnh.param<std::string>("icpTumSavePath", icpTumSavePath, icpTumSavePath);
    pnh.param<std::string>("gpsTumSavePath", gpsTumSavePath, gpsTumSavePath);

    ros::Subscriber subCloud =
        nh.subscribe<sensor_msgs::PointCloud2>("/cameraCloudFrame", 200, cloudHandler);
    ros::Subscriber subOdom;
    if (useGpsPrior) {
        subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic, 200, odomHandler);
    }
    ros::Subscriber subInitPoseGive = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>(
        "/initialposeGive", 1, initposeHandler, ros::TransportHints().tcpNoDelay());
    ros::Subscriber subInitPose = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>(
        "/initialpose", 1, initposeHandler, ros::TransportHints().tcpNoDelay());

    pubCurrentFeature = nh.advertise<sensor_msgs::PointCloud2>("/currentFeature", 100);
    pubCurrentFeatureInWorld = nh.advertise<sensor_msgs::PointCloud2>("/currentFeatureInWorld", 100);
    pubGlobalFeature = nh.advertise<sensor_msgs::PointCloud2>("/globalFeatureMap", 10);
    pubCurrentPose = nh.advertise<nav_msgs::Odometry>("/currentPose", 100);
    pubGpsPoseInMap = nh.advertise<nav_msgs::Odometry>("/gpsPoseInMap", 100);
    pubIcpPath = nh.advertise<nav_msgs::Path>("/icpPath", 100);
    pubGpsPath = nh.advertise<nav_msgs::Path>("/gpsPath", 100);

    if (!loadMap()) {
        return 1;
    }
    if (!initTumWriters()) {
        return 1;
    }

    ROS_INFO_STREAM("Localization node started. useGpsPrior=" << (useGpsPrior ? "true" : "false")
                    << ", odom_topic=" << odomTopic);

    ros::Rate rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        // Process a few frames each loop to drain buffered bag playback smoothly.
        for (int i = 0; i < 5; ++i) {
            sensor_msgs::PointCloud2ConstPtr cloudMsg;
            nav_msgs::OdometryConstPtr odomMsg;
            if (!fetchSynchronizedFrame(cloudMsg, odomMsg)) {
                break;
            }
            processFrame(cloudMsg, odomMsg);
        }

        rate.sleep();
    }

    if (icpTumFile.is_open()) {
        icpTumFile.close();
    }
    if (gpsTumFile.is_open()) {
        gpsTumFile.close();
    }

    return 0;
}























