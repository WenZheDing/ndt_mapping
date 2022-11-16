#include <ros/ros.h>
#include <iostream>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <pcl/filters/voxel_grid.h>
#include <pcl-1.8/pcl/registration/icp.h>
#include <pclomp/ndt_omp.h>
#include "tic_toc.h"


using namespace std;

vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> CloudFrames;
vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pointcloud_q;
vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointcloud_t;
bool pose_received = false;
ros::Publisher pub_map, pub_merged_cloud;
double d_width1 = -40.0, d_width2 = 40.0, d_height1 = -40.0, d_height2 = 40.0, d_z1 = -2.6, d_z2 = 0.2, \
d_ego_left = 1.2, d_ego_right = -1.1, d_ego_front = 2.9, d_ego_back = -1.8, d_ego_top = 0.3, d_ego_bottom = -2.1,\
d_resolution = 0.5, d_height_diff = 0.2, i_max_thresh = 2, i_mid_thresh = 2, i_min_thresh = 2;
vector<Eigen::Quaterniond> map_q_;
vector<Eigen::Vector3d> map_t_;
vector<double> map_time_;
bool init;
double init_time;
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
                // std::cout << "max is " << vt_classify_max[i][j].size()
                //           << "\tmid is " << vt_classify_mid[i][j].size()
                //           << "\tmin is " << vt_classify_min[i][j].size()
                //           << std::endl;
                // printf("The height range of this grid is (%.2f, %.2f)", vt_minheight[i][j], vt_maxheight[i][j]);
                map.data[index] = 100;
            }
        }
    }
    pub_map.publish(map);
}

void CallbackLidar(const sensor_msgs::PointCloud2::ConstPtr pcloud)
{
    double coff = -1;
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setLeafSize(1.0, 1.0, 1.0); 
    if(!pose_received) return;
    if (pcloud != NULL)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr plidar(new pcl::PointCloud<pcl::PointXYZI>());
        plidar.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*pcloud, *plidar);
        CloudFrames.push_back(plidar); 
    }
    Eigen::Quaterniond q, tmp_q;
    Eigen::Matrix3d tmp_R;
    Eigen::Vector3d t, tmp_t;
    int index_before = 1, index_after = 1;
    for(auto iter = map_time_.begin(); iter != map_time_.end(); iter++)
    {
        if (pcloud->header.stamp.toSec() >= (*iter) && pcloud->header.stamp.toSec() <= *(iter+1))
        {
            index_before = static_cast<int>(iter - map_time_.begin());
            index_after = index_before + 1;
            coff = (pcloud->header.stamp.toSec() - map_time_[index_before]) / (map_time_[index_after] - map_time_[index_before]);
            break;
        }
    }
    if (coff < 0 || coff > 1)
    {   
        printf("index_before : %d index_after : %d  map_time_.size(): %d \n", index_before, index_after, map_time_.size());
        q = map_q_.back();
        t = map_t_.back(); 
    }
    else
    {
        Eigen::Quaterniond q_before = map_q_[index_before];
        Eigen::Quaterniond q_after = map_q_[index_after];
        q = q_before.slerp(coff, q_after);
        t = coff * (map_t_[index_after] - map_t_[index_before]) + map_t_[index_before];
    }

    pointcloud_q.push_back(q);
    pointcloud_t.push_back(t);

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr target(new pcl::PointCloud<pcl::PointXYZI>());
    //ndt
    pcl::PointCloud<pcl::PointXYZI>::Ptr palign;
    Eigen::Matrix4f trans(Eigen::Matrix4f::Identity());
    //TicToc ndt_time;
    // pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr anh_ndt;
    // anh_ndt.reset(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
    // anh_ndt->setResolution(0.5);
    // anh_ndt->setMaximumIterations(200);
    // anh_ndt->setStepSize(0.05);
    // anh_ndt->setTransformationEpsilon(1e-6);
    // anh_ndt->setEuclideanFitnessEpsilon(1e-6);
    // anh_ndt->setNumThreads(4);
    // anh_ndt->setNeighborhoodSearchMethod(pclomp::DIRECT26);

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setMaximumIterations(100);
    icp.setMaxCorrespondenceDistance(1.0);
    icp.setEuclideanFitnessEpsilon(1e-2);
    icp.setRANSACIterations(0);
    icp.setTransformationEpsilon(1e-3);

    for(int i = 1; i < 31; i += 15)
    {
        palign.reset(new pcl::PointCloud<pcl::PointXYZI>());
        //init
        int id = CloudFrames.size() - i;
        if(id < 0) break;
        tmp_R = pointcloud_q.back().matrix().inverse() * pointcloud_q[id].matrix();
        tmp_q = Eigen::Quaterniond(tmp_R);
        tmp_t = pointcloud_q.back().matrix().inverse() * (pointcloud_t[id] - pointcloud_t.back());
        Eigen::Vector3d angle = tmp_q.matrix().eulerAngles(2,1,0);
        printf("tmp_t: %.2lf, %.2lf, %.2lf\n", tmp_t(0), tmp_t(1), tmp_t(2));
        printf("tmp_q : %.3lf, %.3lf, %.3lf\n", angle(0), angle(1), angle(2));
        Eigen::Translation3d init_translation(tmp_t(0), tmp_t(1), tmp_t(2));
        Eigen::AngleAxisd init_rotation(tmp_q);
        Eigen::Matrix4d init_guess = (init_translation * init_rotation) * Eigen::Matrix4d::Identity();
        TicToc ndt_time;
        ndt_time.tic();
        downSizeFilter.setInputCloud(CloudFrames[id]);
        downSizeFilter.filter(*source);
        downSizeFilter.setInputCloud(CloudFrames.back());
        downSizeFilter.filter(*target);
        // anh_ndt->setInputSource(CloudFrames[id]);
        // anh_ndt->setInputTarget(CloudFrames.back());
        // anh_ndt->align(*palign, init_guess.cast<float>());
        // trans = anh_ndt->getFinalTransformation();
        // int iteration = anh_ndt->getFinalNumIteration();
        // bool converged = anh_ndt->hasConverged();
        // cout<<"iteration: "<<iteration<<endl;
        // cout<<"converged: "<<converged<<endl;
        // cout<<"ndt time: "<< ndt_time.toc() << endl << endl;

        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.align(*palign, init_guess.cast<float>());

        Eigen::Matrix4f trans(Eigen::Matrix4f::Identity());
        trans = icp.getFinalTransformation();
        cout<<"converged: "<<icp.hasConverged()<<endl;
        cout<<"fitness score: "<<icp.getFitnessScore()<<endl;
        cout<<"time: " << ndt_time.toc() << "  ms"  <<endl << endl;

        Eigen::Vector3d t(trans(0, 3), trans(1, 3), trans(2, 3));
        Eigen::Matrix3d R;
        Eigen::Quaterniond q;
        R << trans(0, 0), trans(0, 1), trans(0, 2), trans(1, 0), trans(1, 1), trans(1, 2), trans(2, 0), trans(2, 1), trans(2, 2);
        q=Eigen::Quaterniond(R);
        pcl::PointCloud<pcl::PointXYZI>::Ptr trans_pcd(new pcl::PointCloud<pcl::PointXYZI>());
        trans_pcd = transformPointCloud(q, t, CloudFrames[id]);
        // trans_pcd = transformPointCloud(tmp_q, tmp_t, CloudFrames[id]);

        *tmp += *trans_pcd;
        // debug : save pcd
        if(CloudFrames.size() == 10)
        {
            string source_pcd = string(ROOT_DIR) + "PCD/" + to_string(id) +".pcd";
            string transformed_source_pcd = string(ROOT_DIR) + "PCD/" + to_string(id) + "_" + to_string(CloudFrames.size() - 1) + "trans.pcd";
            string target_pcd = string(ROOT_DIR) + "PCD/" + to_string(CloudFrames.size() - 1) +".pcd";
            pcl::io::savePCDFileASCII(source_pcd, *CloudFrames[id]);
            pcl::io::savePCDFileASCII(transformed_source_pcd, *trans_pcd);
            pcl::io::savePCDFileASCII(target_pcd, *CloudFrames.back());
        }
    }
   /* for(int i = 1; i < 2; i++)
    {
        int id = CloudFrames.size() - i;
        if(id < 0) break;
        tmp_R = pointcloud_q.back().matrix().inverse() * pointcloud_q[id].matrix();
        tmp_q = Eigen::Quaterniond(tmp_R);
        tmp_t = pointcloud_q.back().matrix().inverse() * (pointcloud_t[id] - pointcloud_t.back());
        *tmp += *transformPointCloud(tmp_q, tmp_t, CloudFrames[id]);
    }*/
    /*
    int i=10;
    int id = CloudFrames.size() - i;
    if(id >=0) {
    tmp_R = pointcloud_q.back().matrix().inverse() * pointcloud_q[id].matrix();
    tmp_q = Eigen::Quaterniond(tmp_R);
    tmp_t = pointcloud_q.back().matrix().inverse() * (pointcloud_t[id] - pointcloud_t.back());
    *tmp = *transformPointCloud(tmp_q, tmp_t, CloudFrames[id]);
    }*/
    tmp->width=tmp->points.size();
	tmp->height=1;
    sensor_msgs::PointCloud2 output;
	pcl::toROSMsg(*tmp, output);
	output.header.frame_id = "conch";
    pub_merged_cloud.publish(output);
    createLocalMap(tmp);
    cout <<"lidar size():  " << CloudFrames.size() << endl;
}

void CallbackPose(const geometry_msgs::PoseStamped::ConstPtr pose)
{
    if(!init) {init_time = pose->header.stamp.toSec();}
    Eigen::Matrix3d R; 
    Eigen::Quaterniond last_q;
    Eigen::Vector3d last_t;
    R = Eigen::AngleAxisd(-pose->pose.position.z*M_PI/180, Eigen::Vector3d::UnitZ()) * 
                      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) * 
                      Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
    last_q = Eigen::Quaterniond(R);
    last_t = Eigen::Vector3d(pose->pose.position.x, pose->pose.position.y, 0);
    map_q_.push_back(last_q);
    map_t_.push_back(last_t);
    map_time_.push_back(pose->header.stamp.toSec());
    pose_received = true;
    cout << "time size(): " <<map_time_.size() << endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "localmap");
    ros::NodeHandle nh;

    ros::Subscriber subLidar = nh.subscribe("/lidar_aeb/raw_points", 1000, &CallbackLidar);
    ros::Subscriber subPose = nh.subscribe("/imu", 1000, &CallbackPose);
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