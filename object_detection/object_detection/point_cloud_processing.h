#pragma once
#include <Eigen/Core>
#include <vtkAutoInit.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include "VzenseNebula_api.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <boost/thread/thread.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/common/angles.h>
#include <pcl/registration/transformation_estimation_svd.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);

// Calculate IoU on the image plane
double get_iou(std::vector<cv::Point> point_set_fitting, std::vector<cv::Point> point_set_gt, int h, int w);

// Get the incline angle of the object plane(unit: radian)
double get_incline_angle(pcl::ModelCoefficients::Ptr &coefficients);

// Get the information about the bounding box
void get_bbox_information(std::vector<pcl::PointXYZRGB> pointSetGT, int boxLength, int boxWidth);

// Reorder the unordered point before drawing the bounding box
void reorder_point(std::vector<cv::Point> &reorder_point_set);

// Roughly calculate the angle before drawing the bounding box
double hough_line_detect_angle(cv::Mat image);

// Get the final score of each object plane
void get_final_score(int k, int box_length, int box_width, pcl::visualization::PCLVisualizer::Ptr &viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr &one_object_cloud, std::vector<cv::Vec3d> colors, cv::Mat image_current_object, cv::Mat image_rgb, VzSensorIntrinsicParameters camera_param);
