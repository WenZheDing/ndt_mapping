#include <ros/ros.h>
#include <iostream>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Point.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>



using namespace std;

vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> CloudFrames;
vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pointcloud_q;
vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointcloud_t;
Eigen::Quaterniond last_q;
Eigen::Vector3d last_t;
bool pose_received = false;
ros::Publisher pub_map, pub_merged_cloud;
double d_width1 = -40.0, d_width2 = 40.0, d_height1 = -40.0, d_height2 = 40.0, d_z1 = -2.6, d_z2 = 0.2, \
d_ego_left = 1.2, d_ego_right = -1.1, d_ego_front = 2.9, d_ego_back = -1.8, d_ego_top = 0.3, d_ego_bottom = -2.1,\
d_resolution = 0.5, d_height_diff = 0.2, i_max_thresh = 2, i_mid_thresh = 2, i_min_thresh = 2;
#define MAX_OGM std::numeric_limits<float>::max()
#define MIN_OGM -std::numeric_limits<float>::max()

pcl::PointCloud<pcl::PointXYZI>::Ptr transformPointCloud(const Eigen::Quaterniond &q, const Eigen::Vector3d &t, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in)
{   
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>());
    cloud_out->resize(cloud_in->points.size());

    for (uint32_t i = 0; i < cloud_in->points.size(); i++)
    {
        Eigen::Vector3d point(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z);
        Eigen::Vector3d point2 = q * point + t;
        // nan point may cause std::bad_alloc
        if (!std::isfinite(point2[0]) || !std::isfinite(point2[1]) || !std::isfinite(point2[2]) || !std::isfinite(cloud_in->points[i].intensity)) continue;
        pcl::PointXYZI p;
        p.x = point2[0];
        p.y = point2[1];
        p.z = point2[2];
        p.intensity = cloud_in->points[i].intensity;

        cloud_out->points[i] = p;
    }

    cloud_out->height = 1;
    cloud_out->width = cloud_in->points.size();

    return cloud_out;
}

void createLocalMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr merged_cloud)
{
    int i_rows = (d_height2 - d_height1) / d_resolution;
    int i_cols = (d_width2 - d_width1) / d_resolution;
    std::vector<std::vector<float>> vt_maxheight(i_rows);
    std::vector<std::vector<float>> vt_minheight(i_rows);
    std::vector<std::vector<std::vector<float>>> vt_classify_max(i_rows), vt_classify_min(i_rows), vt_classify_mid(i_rows);
    for (int i = 0; i < i_rows; i++) {
        vt_maxheight[i].resize(i_cols);
        vt_minheight[i].resize(i_cols);
        vt_classify_max[i].resize(i_cols);
        vt_classify_min[i].resize(i_cols);
        vt_classify_mid[i].resize(i_cols);
        for (int j = 0; j < i_cols; j++) {
            vt_maxheight[i][j] = MIN_OGM;
            vt_minheight[i][j] = MAX_OGM;
        }
    }
    for (int i = 0; i < merged_cloud->points.size(); i++) {
        Eigen::Vector4f p;
        auto st_point = merged_cloud->points[i];
        if (st_point.x < d_ego_front && st_point.x > d_ego_back && st_point.y < d_ego_left &&
            st_point.y > d_ego_right && st_point.z < d_ego_top && st_point.z > d_ego_bottom) {
            continue;
        }
        if (st_point.x > d_width1+1 && st_point.x < d_width2-1 && st_point.y > d_height1+1 &&
           st_point.y < d_height2-1 && st_point.z > d_z1 && st_point.z < d_z2) {
            int i_x = (st_point.x - d_width1) / d_resolution;
            int i_y = (st_point.y - d_height1) / d_resolution;
            if (st_point.z > vt_maxheight[i_y][i_x]) {
                vt_maxheight[i_y][i_x] = st_point.z;
            }
            if (st_point.z < vt_minheight[i_y][i_x]) {
                vt_minheight[i_y][i_x] = st_point.z;
            }
            // 对范围内的点进行高度上的分类，若三段高度内都有点，认为是人形或车型障碍物;
            if (st_point.z >= d_z1 && st_point.z < (2 * d_z1 + d_z2) / 3.0) {
                vt_classify_min[i_y][i_x].push_back(st_point.z);
            } else if (st_point.z >= (2 * d_z1 + d_z2) / 3.0 && st_point.z < (2 * d_z2 + d_z1) / 3.0) {
                vt_classify_mid[i_y][i_x].push_back(st_point.z);
            } else if (st_point.z >= (2 * d_z2 + d_z1) / 3.0 && st_point.z <= d_z2) {
                vt_classify_max[i_y][i_x].push_back(st_point.z);
            }
        }
    }
    nav_msgs::OccupancyGrid map;
	map.header.frame_id = "conch";
	map.header.stamp = ros::Time::now();
	map.info.resolution = d_resolution;
	map.info.width = i_cols;
	map.info.height = i_rows;
	map.info.origin.position.x = d_width1;
	map.info.origin.position.y = d_height1;
	map.data.resize(i_cols*i_rows);

    for (int i = 0; i < i_rows; i++) {
        for (int j = 0; j < i_cols; j++) {
            int index = j + map.info.width * i;
            map.data[index] = 0;
            if ((vt_maxheight[i][j] > vt_minheight[i][j]) && (vt_maxheight[i][j] - vt_minheight[i][j] > d_height_diff)) {
                // st_map.data[index] = int((maxheight[i][j] - minheight[i][j]) * 100);
                map.data[index] = 50;
                // std::cout << "+";
            }
            // else
            // {
            //     if (i == static_cast<int>(-d_width1 / d_resolution) &&
            //         j == static_cast<int>(-d_height1 / d_resolution))
            //         std::cout << "%";
            //     else
            //         std::cout << ".";
            // }
            if (vt_classify_min[i][j].size() >= i_min_thresh && vt_classify_mid[i][j].size() >= i_mid_thresh &&
                vt_classify_max[i][j].size() >= i_max_thresh) {
                std::cout << "max is " << vt_classify_max[i][j].size()
                          << "\tmid is " << vt_classify_mid[i][j].size()
                          << "\tmin is " << vt_classify_min[i][j].size()
                          << std::endl;
                printf("The height range of this grid is (%.2f, %.2f)", vt_minheight[i][j], vt_maxheight[i][j]);
                map.data[index] = 100;
            }
        }
    }
    pub_map.publish(map);
}

void CallbackLidar(const sensor_msgs::PointCloud2::ConstPtr pcloud)
{
    if(!pose_received) return;
    if (pcloud != NULL)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr plidar(new pcl::PointCloud<pcl::PointXYZI>());
        plidar.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*pcloud, *plidar);
        CloudFrames.push_back(plidar); 
    }
    Eigen::Quaterniond tmp_q;
    Eigen::Matrix3d tmp_R;
    Eigen::Vector3d tmp_t;
    pointcloud_q.push_back(last_q);
    pointcloud_t.push_back(last_t);

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>());
    for(int i = 1; i < 10; i++)
    {
        int id = CloudFrames.size() - i;
        if(id < 0) break;
        tmp_R = pointcloud_q.back().matrix().inverse() * pointcloud_q[id].matrix();
        tmp_q = Eigen::Quaterniond(tmp_R);
        tmp_t = pointcloud_q.back().matrix().inverse() * (pointcloud_t[id] - pointcloud_t.back());
        *tmp += *transformPointCloud(tmp_q, tmp_t, CloudFrames[id]);
    }
    tmp->width=tmp->points.size();
	tmp->height=1;
    sensor_msgs::PointCloud2 output;
	pcl::toROSMsg(*tmp, output);
	output.header.frame_id = "conch";
    pub_merged_cloud.publish(output);
    createLocalMap(tmp);
}

void CallbackPose(const geometry_msgs::Point::ConstPtr point)
{
    Eigen::Matrix3d R; 
    R = Eigen::AngleAxisd((point->z - 90)*M_PI/180, Eigen::Vector3d::UnitZ()) * 
                      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) * 
                      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
    last_q = Eigen::Quaterniond(R);
    last_t = Eigen::Vector3d(point->x, point->y, 0);
    pose_received = true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "localmap");
    ros::NodeHandle nh;

    ros::Subscriber subLidar = nh.subscribe("/lidar_aeb/raw_points", 1, &CallbackLidar);
    ros::Subscriber subPose = nh.subscribe("/imu_pose", 1, &CallbackPose);
	pub_map = nh.advertise<nav_msgs::OccupancyGrid>("/occupancy_grid_map", 1);
    pub_merged_cloud = nh.advertise<sensor_msgs::PointCloud2>("/merged_cloud", 1);
  
    ros::Rate rate(100);
    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}