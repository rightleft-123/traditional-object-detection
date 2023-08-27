#include "point_cloud_processing.h"


double get_iou(std::vector<cv::Point> point_set_fitting, std::vector<cv::Point> point_set_gt, int h, int w)
{
	cv::Mat canvas1 = cv::Mat::zeros(h, w, CV_8UC1);
	cv::Mat canvas2 = cv::Mat::zeros(h, w, CV_8UC1);
	cv::Mat canvas_1or2 = cv::Mat::zeros(h, w, CV_8UC1);
	cv::Mat canvas_1and2 = cv::Mat::zeros(h, w, CV_8UC1);

	cv::polylines(canvas1, point_set_fitting, true, cv::Scalar(255, 255, 255), 1, 8, 0);
	cv::fillPoly(canvas1, point_set_fitting, cv::Scalar(255, 255, 255));

	cv::polylines(canvas2, point_set_gt, true, cv::Scalar(255, 255, 255), 1, 8, 0);
	cv::fillPoly(canvas2, point_set_gt, cv::Scalar(255, 255, 255));

	canvas_1or2 = canvas1 | canvas2;
	canvas_1and2 = canvas1 & canvas2;

	cv::Mat label, state, centroid;
	int and_point = cv::connectedComponentsWithStats(canvas_1and2, label, state, centroid);
	double and_area = state.at<int>(1, cv::CC_STAT_AREA);
	int or_point = cv::connectedComponentsWithStats(canvas_1or2, label, state, centroid);
	double or_area = state.at<int>(1, cv::CC_STAT_AREA);

	double iou = (and_area / or_area);
	return iou;
}

double get_incline_angle(pcl::ModelCoefficients::Ptr &plane_coefficients)
{
	cv::Point3d plane_normal_line;
	plane_normal_line.x = plane_coefficients->values[0];
	plane_normal_line.y = plane_coefficients->values[1];
	plane_normal_line.z = plane_coefficients->values[2];

	double z_angle = abs(acos(abs(plane_normal_line.z) / abs(sqrt(pow(plane_normal_line.x, 2) + pow(plane_normal_line.y, 2) + pow(plane_normal_line.z, 2)))));
	double x_angle = abs(acos(abs(plane_normal_line.x) / abs(sqrt(pow(plane_normal_line.x, 2) + pow(plane_normal_line.y, 2) + pow(plane_normal_line.z, 2)))));
	double y_angle = abs(acos(abs(plane_normal_line.y) / abs(sqrt(pow(plane_normal_line.x, 2) + pow(plane_normal_line.y, 2) + pow(plane_normal_line.z, 2)))));

	return z_angle;
}

void get_bbox_information(std::vector<pcl::PointXYZRGB> point_set_groundtruth, int box_length, int box_width)
{
	std::vector<double> distance_set;
	for (size_t i = 0; i < point_set_groundtruth.size(); ++i)
	{
		for (size_t j = 0; j < point_set_groundtruth.size(); ++j)
		{
			if (i != j)
			{
				double distance = sqrt(pow((point_set_groundtruth[i].x - point_set_groundtruth[j].x), 2) + pow((point_set_groundtruth[i].y - point_set_groundtruth[j].y), 2) + pow((point_set_groundtruth[i].z - point_set_groundtruth[j].z), 2));
				distance_set.push_back(distance);
			}
		}
	}

	for (int i = 1; i < distance_set.size(); i++) {
		int j;
		if (distance_set[i] < distance_set[i - 1])
		{
			int temp = distance_set[i];
			for (j = i - 1; j >= 0 && temp < distance_set[j]; j--)
			{
				distance_set[j + 1] = distance_set[j];
			}
			distance_set[j + 1] = temp;
		}
	}

	std::cout << "BBOX Length: " << (distance_set[4] + distance_set[5] + distance_set[6] + distance_set[7]) / 4 << '\t' << "Groundtrue Length: " << box_length << '\t' << "Difference: " << (distance_set[4] + distance_set[5] + distance_set[6] + distance_set[7]) / 4 - box_length << endl;
	std::cout << "BBOX Width: " << (distance_set[0] + distance_set[1] + distance_set[2] + distance_set[3]) / 4 << '\t' << "Groundtrue Width: " << box_width << '\t' << "Difference: " << (distance_set[0] + distance_set[1] + distance_set[2] + distance_set[3]) / 4 - box_width << endl;
	std::cout << "BBOX Accumulate difference: " << abs((distance_set[4] + distance_set[5] + distance_set[6] + distance_set[7]) / 4 - box_length) + abs((distance_set[0] + distance_set[1] + distance_set[2] + distance_set[3]) / 4 - box_width) << endl;
}

void reorder_point(std::vector<cv::Point> &reorder_point_set)
{
	const int POLYGON_LINE = 4;

	std::vector<cv::Point> temp;

	double shortest_distance = INT_MAX;
	double shortest_distance_index1 = 0;
	for (int j = 1; j < POLYGON_LINE; j++)
	{
		double distance = sqrt(pow(reorder_point_set[0].x - reorder_point_set[j].x, 2) + pow(reorder_point_set[0].y - reorder_point_set[j].y, 2));
		if (distance < shortest_distance)
		{
			shortest_distance = distance;
			shortest_distance_index1 = j;
		}
	}
	temp.push_back(reorder_point_set[0]);
	temp.push_back(reorder_point_set[shortest_distance_index1]);

	shortest_distance = INT_MAX;
	double shortest_distance_index2 = 0;
	for (int j = 1; j < POLYGON_LINE; j++) {
		if (j != shortest_distance_index1)
		{
			double distance = sqrt(pow(reorder_point_set[shortest_distance_index1].x - reorder_point_set[j].x, 2) + pow(reorder_point_set[shortest_distance_index1].y - reorder_point_set[j].y, 2));
			if (distance < shortest_distance)
			{
				shortest_distance = distance;
				shortest_distance_index2 = j;
			}
		}
	}


	shortest_distance = INT_MAX;
	double shortest_distance_index3 = 0;
	for (int j = 1; j < POLYGON_LINE; j++) {
		if (j != shortest_distance_index1 && j != shortest_distance_index2)
		{
			double distance = sqrt(pow(reorder_point_set[shortest_distance_index2].x - reorder_point_set[j].x, 2) + pow(reorder_point_set[shortest_distance_index2].y - reorder_point_set[j].y, 2));
			if (distance < shortest_distance)
			{
				shortest_distance = distance;
				shortest_distance_index3 = j;
			}
		}
	}
	temp.push_back(reorder_point_set[shortest_distance_index2]);
	temp.push_back(reorder_point_set[shortest_distance_index3]);
	reorder_point_set.clear();
	reorder_point_set = temp;
}

double hough_line_detect_angle(cv::Mat image)
{
	cv::Mat image_edge;
	std::vector<cv::Vec2f> lines;
	cv::Canny(image, image_edge, 10, 255);
	cv::HoughLines(image_edge, lines, 1, CV_PI / 180, 40);

	double avg_angle_postive = 0, num_postive = 0;
	double avg_angle_negative = 0, num_negative = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		if (lines[i][0] > 0)
		{
			avg_angle_postive += lines[i][1];
			num_postive += 1;
		}
		if (lines[i][0] < 0)
		{
			avg_angle_negative += lines[i][1];
			num_negative += 1;
		}
	
	}

	if (num_postive > num_negative)
	{
		avg_angle_postive /= num_postive;
		avg_angle_postive = 180 / CV_PI  * avg_angle_postive;

		return avg_angle_postive;
	}
	else
	{
		avg_angle_negative /= num_negative;
		avg_angle_negative = 180 / CV_PI  * avg_angle_negative;

		return avg_angle_negative;
	}
}

void get_final_score(int k, int box_length, int box_width, pcl::visualization::PCLVisualizer::Ptr &viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr &one_object_cloud, std::vector<cv::Vec3d> colors, cv::Mat image_current_object, cv::Mat image_rgb, VzSensorIntrinsicParameters camera_param)
{
	// Use RANSAC to do plane fitting
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(5);
	seg.setMaxIterations(25);
	seg.setInputCloud(one_object_cloud);
	seg.segment(*inliers, *coefficients);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr one_object_colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*one_object_cloud, *one_object_colored_cloud);

	// Mark the result of segmentation
	double avg_x = 0, avg_y = 0, avg_z = 0;
	for (size_t i = 0; i < inliers->indices.size(); ++i)
	{
		one_object_colored_cloud->points[inliers->indices[i]].r = colors[k][2];
		one_object_colored_cloud->points[inliers->indices[i]].g = colors[k][1];
		one_object_colored_cloud->points[inliers->indices[i]].b = colors[k][0];

		avg_x += one_object_cloud->points[inliers->indices[i]].x;
		avg_y += one_object_cloud->points[inliers->indices[i]].y;
		avg_z += one_object_cloud->points[inliers->indices[i]].z;
	}
	avg_x /= inliers->indices.size();
	avg_y /= inliers->indices.size();
	avg_z /= inliers->indices.size();

	// Four GT corner point
	pcl::PointXYZ gt_vertex1(avg_x - box_length / 2, avg_y + box_width / 2, avg_z);
	pcl::PointXYZ gt_vertex2(avg_x + box_length / 2, avg_y + box_width / 2, avg_z);
	pcl::PointXYZ gt_vertex3(avg_x + box_length / 2, avg_y - box_width / 2, avg_z);
	pcl::PointXYZ gt_vertex4(avg_x - box_length / 2, avg_y - box_width / 2, avg_z);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vertex(new pcl::PointCloud<pcl::PointXYZ>);
	cloud_vertex->push_back(gt_vertex1); cloud_vertex->push_back(gt_vertex2); cloud_vertex->push_back(gt_vertex3); cloud_vertex->push_back(gt_vertex4);

	// Achieve Point Cloud rotation around the fitting plane centroid 
	int best_theta = 0; double best_iou = 0, incline_angle = get_incline_angle(coefficients);

	int shortest_distance_index1, shortest_distance_index2, shortest_distance_index3, shortest_distance_index4;
	cv::Point3d point_cloud_nearest1, point_cloud_nearest2, point_cloud_nearest3, point_cloud_nearest4;
	cv::Point3d point_cloud_groundtruth1, point_cloud_groundtruth2, point_cloud_groundtruth3, point_cloud_groundtruth4;

	cv::Point point_image_final1, point_image_final2, point_image_final3, point_image_final4;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vertex_transformed(new pcl::PointCloud<pcl::PointXYZ>);

	// rotate by Z-axis
	int aproximate_angle = hough_line_detect_angle(image_current_object);
	for (int theta = aproximate_angle - 10; theta <= aproximate_angle + 10; theta++)
	{
		int fixed_theta = theta;
		if (theta <= 90)
			fixed_theta = theta + 90;
		else if (theta >= 90)
			fixed_theta = theta - 90;


		for (int i = 0; i < cloud_vertex->size(); i++)
		{
			pcl::PointXYZ temp;
			temp.x = cloud_vertex->points[i].x - avg_x;
			temp.y = cloud_vertex->points[i].y - avg_y;
			temp.z = cloud_vertex->points[i].z - avg_z;
			cloud_vertex_transformed->push_back(temp);
		}


		Eigen::Matrix4f rotation_z = Eigen::Matrix4f::Identity();
		double angle_z = M_PI / 180 * fixed_theta;
		rotation_z(0, 0) = cos(angle_z);
		rotation_z(0, 1) = -sin(angle_z);
		rotation_z(1, 0) = sin(angle_z);
		rotation_z(1, 1) = cos(angle_z);
		pcl::transformPointCloud(*cloud_vertex_transformed, *cloud_vertex_transformed, rotation_z);

		Eigen::Matrix4f transform_x = Eigen::Matrix4f::Identity();
		transform_x(1, 1) = cos(incline_angle);
		transform_x(1, 2) = -sin(incline_angle);
		transform_x(2, 1) = sin(incline_angle);
		transform_x(2, 2) = cos(incline_angle);
		pcl::transformPointCloud(*cloud_vertex_transformed, *cloud_vertex_transformed, transform_x);

		for (int i = 0; i < cloud_vertex_transformed->size(); i++)
		{
			cloud_vertex_transformed->points[i].x += avg_x;
			cloud_vertex_transformed->points[i].y += avg_y;
			cloud_vertex_transformed->points[i].z += avg_z;
		}
		point_cloud_groundtruth1.x = cloud_vertex_transformed->points[0].x; point_cloud_groundtruth1.y = cloud_vertex_transformed->points[0].y; point_cloud_groundtruth1.z = cloud_vertex_transformed->points[0].z;
		point_cloud_groundtruth2.x = cloud_vertex_transformed->points[1].x; point_cloud_groundtruth2.y = cloud_vertex_transformed->points[1].y; point_cloud_groundtruth2.z = cloud_vertex_transformed->points[1].z;
		point_cloud_groundtruth3.x = cloud_vertex_transformed->points[2].x; point_cloud_groundtruth3.y = cloud_vertex_transformed->points[2].y; point_cloud_groundtruth3.z = cloud_vertex_transformed->points[2].z;
		point_cloud_groundtruth4.x = cloud_vertex_transformed->points[3].x; point_cloud_groundtruth4.y = cloud_vertex_transformed->points[3].y; point_cloud_groundtruth4.z = cloud_vertex_transformed->points[3].z;

		// Find the Nearest Point Cloud
		double shortest_distance1 = INT_MAX;
		double shortest_distance2 = INT_MAX;
		double shortest_distance3 = INT_MAX;
		double shortest_distance4 = INT_MAX;
		int distance_index1, distance_index2, distance_index3, distance_index4;

		for (size_t i = 0; i < inliers->indices.size(); ++i)
		{
			double distance1 = 0;
			distance1 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[0].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[0].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[0].z), 2));
			if (distance1 < shortest_distance1) {
				shortest_distance1 = distance1;
				point_cloud_nearest1.x = one_object_colored_cloud->points[inliers->indices[i]].x;
				point_cloud_nearest1.y = one_object_colored_cloud->points[inliers->indices[i]].y;
				point_cloud_nearest1.z = one_object_colored_cloud->points[inliers->indices[i]].z;
				distance_index1 = inliers->indices[i];
			}

			double distance2 = 0;
			distance2 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[1].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[1].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[1].z), 2));
			if (distance2 < shortest_distance2) {
				shortest_distance2 = distance2;
				point_cloud_nearest2.x = one_object_colored_cloud->points[inliers->indices[i]].x;
				point_cloud_nearest2.y = one_object_colored_cloud->points[inliers->indices[i]].y;
				point_cloud_nearest2.z = one_object_colored_cloud->points[inliers->indices[i]].z;
				distance_index2 = inliers->indices[i];
			}

			double distance3 = 0;
			distance3 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[2].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[2].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[2].z), 2));
			if (distance3 < shortest_distance3) {
				shortest_distance3 = distance3;
				point_cloud_nearest3.x = one_object_colored_cloud->points[inliers->indices[i]].x;
				point_cloud_nearest3.y = one_object_colored_cloud->points[inliers->indices[i]].y;
				point_cloud_nearest3.z = one_object_colored_cloud->points[inliers->indices[i]].z;
				distance_index3 = inliers->indices[i];
			}

			double distance4 = 0;
			distance4 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[3].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[3].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[3].z), 2));
			if (distance4 < shortest_distance4) {
				shortest_distance4 = distance4;
				point_cloud_nearest4.x = one_object_colored_cloud->points[inliers->indices[i]].x;
				point_cloud_nearest4.y = one_object_colored_cloud->points[inliers->indices[i]].y;
				point_cloud_nearest4.z = one_object_colored_cloud->points[inliers->indices[i]].z;
				distance_index4 = inliers->indices[i];
			}
		}

		// Map the four point back to image
		std::vector<cv::Point> point_image_set;
		std::vector<cv::Point> point_image_set_gt;
		std::vector<cv::Point3d> point_cloud_set_current = { point_cloud_nearest1 ,point_cloud_nearest2 ,point_cloud_nearest3 ,point_cloud_nearest4 };
		std::vector<cv::Point3d> point_cloud_set_gt_current = { point_cloud_groundtruth1, point_cloud_groundtruth2, point_cloud_groundtruth3, point_cloud_groundtruth4 };
		for (int i = 0; i < point_cloud_set_current.size(); i++) {
			double fx = camera_param.fx;
			double fy = camera_param.fy;
			double cx = camera_param.cx;
			double cy = camera_param.cy;
			double X = point_cloud_set_current[i].x;
			double Y = point_cloud_set_current[i].y;

			int u = X / point_cloud_set_current[i].z*fx + cx;
			int v = Y / point_cloud_set_current[i].z*fy + cy;

			cv::Point point_image;
			point_image.x = u;
			point_image.y = v;
			point_image_set.push_back(point_image);

			X = point_cloud_set_gt_current[i].x;
			Y = point_cloud_set_gt_current[i].y;

			u = X / point_cloud_set_gt_current[i].z*fx + cx;
			v = Y / point_cloud_set_gt_current[i].z*fy + cy;

			cv::Point point_image_temp;
			point_image_temp.x = u;
			point_image_temp.y = v;
			point_image_set_gt.push_back(point_image_temp);
		}


		// Calculate IoU
		reorder_point(point_image_set);
		reorder_point(point_image_set_gt);
		//cv::imshow("111", image_current_object);
		double iou = get_iou(point_image_set, point_image_set_gt, image_current_object.rows, image_current_object.cols);
		if (iou > best_iou)
		{
			best_iou = iou;
			best_theta = fixed_theta;
			point_image_final1 = point_image_set[0];
			point_image_final2 = point_image_set[1];
			point_image_final3 = point_image_set[2];
			point_image_final4 = point_image_set[3];
			shortest_distance_index1 = distance_index1;
			shortest_distance_index2 = distance_index2;
			shortest_distance_index3 = distance_index3;
			shortest_distance_index4 = distance_index4;
		}
		cloud_vertex_transformed->clear();
	}

	// Deal with houghline wrong case
	if (best_iou < 0.7)
	{
		for (int theta = 0; theta < 180; theta++)
		{
			for (int i = 0; i < cloud_vertex->size(); i++)
			{
				pcl::PointXYZ temp;
				temp.x = cloud_vertex->points[i].x - avg_x;
				temp.y = cloud_vertex->points[i].y - avg_y;
				temp.z = cloud_vertex->points[i].z - avg_z;
				cloud_vertex_transformed->push_back(temp);
			}


			Eigen::Matrix4f rotation_z = Eigen::Matrix4f::Identity();
			double angle_z = M_PI / 180 * theta;
			rotation_z(0, 0) = cos(angle_z);
			rotation_z(0, 1) = -sin(angle_z);
			rotation_z(1, 0) = sin(angle_z);
			rotation_z(1, 1) = cos(angle_z);
			pcl::transformPointCloud(*cloud_vertex_transformed, *cloud_vertex_transformed, rotation_z);

			Eigen::Matrix4f transform_x = Eigen::Matrix4f::Identity();
			transform_x(1, 1) = cos(incline_angle);
			transform_x(1, 2) = -sin(incline_angle);
			transform_x(2, 1) = sin(incline_angle);
			transform_x(2, 2) = cos(incline_angle);
			pcl::transformPointCloud(*cloud_vertex_transformed, *cloud_vertex_transformed, transform_x);

			for (int i = 0; i < cloud_vertex_transformed->size(); i++)
			{
				cloud_vertex_transformed->points[i].x += avg_x;
				cloud_vertex_transformed->points[i].y += avg_y;
				cloud_vertex_transformed->points[i].z += avg_z;
			}
			point_cloud_groundtruth1.x = cloud_vertex_transformed->points[0].x; point_cloud_groundtruth1.y = cloud_vertex_transformed->points[0].y; point_cloud_groundtruth1.z = cloud_vertex_transformed->points[0].z;
			point_cloud_groundtruth2.x = cloud_vertex_transformed->points[1].x; point_cloud_groundtruth2.y = cloud_vertex_transformed->points[1].y; point_cloud_groundtruth2.z = cloud_vertex_transformed->points[1].z;
			point_cloud_groundtruth3.x = cloud_vertex_transformed->points[2].x; point_cloud_groundtruth3.y = cloud_vertex_transformed->points[2].y; point_cloud_groundtruth3.z = cloud_vertex_transformed->points[2].z;
			point_cloud_groundtruth4.x = cloud_vertex_transformed->points[3].x; point_cloud_groundtruth4.y = cloud_vertex_transformed->points[3].y; point_cloud_groundtruth4.z = cloud_vertex_transformed->points[3].z;

			// Find the Nearest Point Cloud
			double shortest_distance1 = INT_MAX;
			double shortest_distance2 = INT_MAX;
			double shortest_distance3 = INT_MAX;
			double shortest_distance4 = INT_MAX;
			int distance_index1, distance_index2, distance_index3, distance_index4;

			for (size_t i = 0; i < inliers->indices.size(); ++i)
			{
				double distance1 = 0;
				distance1 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[0].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[0].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[0].z), 2));
				if (distance1 < shortest_distance1) {
					shortest_distance1 = distance1;
					point_cloud_nearest1.x = one_object_colored_cloud->points[inliers->indices[i]].x;
					point_cloud_nearest1.y = one_object_colored_cloud->points[inliers->indices[i]].y;
					point_cloud_nearest1.z = one_object_colored_cloud->points[inliers->indices[i]].z;
					distance_index1 = inliers->indices[i];
				}

				double distance2 = 0;
				distance2 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[1].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[1].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[1].z), 2));
				if (distance2 < shortest_distance2) {
					shortest_distance2 = distance2;
					point_cloud_nearest2.x = one_object_colored_cloud->points[inliers->indices[i]].x;
					point_cloud_nearest2.y = one_object_colored_cloud->points[inliers->indices[i]].y;
					point_cloud_nearest2.z = one_object_colored_cloud->points[inliers->indices[i]].z;
					distance_index2 = inliers->indices[i];
				}

				double distance3 = 0;
				distance3 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[2].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[2].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[2].z), 2));
				if (distance3 < shortest_distance3) {
					shortest_distance3 = distance3;
					point_cloud_nearest3.x = one_object_colored_cloud->points[inliers->indices[i]].x;
					point_cloud_nearest3.y = one_object_colored_cloud->points[inliers->indices[i]].y;
					point_cloud_nearest3.z = one_object_colored_cloud->points[inliers->indices[i]].z;
					distance_index3 = inliers->indices[i];
				}

				double distance4 = 0;
				distance4 = sqrt(pow((one_object_colored_cloud->points[inliers->indices[i]].x - cloud_vertex_transformed->points[3].x), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].y - cloud_vertex_transformed->points[3].y), 2) + pow((one_object_colored_cloud->points[inliers->indices[i]].z - cloud_vertex_transformed->points[3].z), 2));
				if (distance4 < shortest_distance4) {
					shortest_distance4 = distance4;
					point_cloud_nearest4.x = one_object_colored_cloud->points[inliers->indices[i]].x;
					point_cloud_nearest4.y = one_object_colored_cloud->points[inliers->indices[i]].y;
					point_cloud_nearest4.z = one_object_colored_cloud->points[inliers->indices[i]].z;
					distance_index4 = inliers->indices[i];
				}
			}

			// Map the four point back to image
			std::vector<cv::Point> point_image_set;
			std::vector<cv::Point> point_image_set_gt;
			std::vector<cv::Point3d> point_cloud_set_current = { point_cloud_nearest1 ,point_cloud_nearest2 ,point_cloud_nearest3 ,point_cloud_nearest4 };
			std::vector<cv::Point3d> point_cloud_set_gt_current = { point_cloud_groundtruth1, point_cloud_groundtruth2, point_cloud_groundtruth3, point_cloud_groundtruth4 };
			for (int i = 0; i < point_cloud_set_current.size(); i++) {
				double fx = camera_param.fx;
				double fy = camera_param.fy;
				double cx = camera_param.cx;
				double cy = camera_param.cy;
				double X = point_cloud_set_current[i].x;
				double Y = point_cloud_set_current[i].y;

				int u = X / point_cloud_set_current[i].z*fx + cx;
				int v = Y / point_cloud_set_current[i].z*fy + cy;

				cv::Point point_image;
				point_image.x = u;
				point_image.y = v;
				point_image_set.push_back(point_image);

				X = point_cloud_set_gt_current[i].x;
				Y = point_cloud_set_gt_current[i].y;

				u = X / point_cloud_set_gt_current[i].z*fx + cx;
				v = Y / point_cloud_set_gt_current[i].z*fy + cy;

				cv::Point point_image_temp;
				point_image_temp.x = u;
				point_image_temp.y = v;
				point_image_set_gt.push_back(point_image_temp);
			}


			// Calculate IoU
			reorder_point(point_image_set);
			reorder_point(point_image_set_gt);
			double iou = get_iou(point_image_set, point_image_set_gt, image_current_object.rows, image_current_object.cols);
			if (iou > best_iou)
			{
				best_iou = iou;
				best_theta = theta;
				point_image_final1 = point_image_set[0];
				point_image_final2 = point_image_set[1];
				point_image_final3 = point_image_set[2];
				point_image_final4 = point_image_set[3];
				shortest_distance_index1 = distance_index1;
				shortest_distance_index2 = distance_index2;
				shortest_distance_index3 = distance_index3;
				shortest_distance_index4 = distance_index4;
			}
			cloud_vertex_transformed->clear();
		}
	}


	std::vector<pcl::PointXYZRGB> point_cloud_set;
	point_cloud_set.push_back(one_object_colored_cloud->points[shortest_distance_index1]);
	point_cloud_set.push_back(one_object_colored_cloud->points[shortest_distance_index2]);
	point_cloud_set.push_back(one_object_colored_cloud->points[shortest_distance_index3]);
	point_cloud_set.push_back(one_object_colored_cloud->points[shortest_distance_index4]);

	// get_bbox_information(point_cloud_set, box_length, box_width);

	one_object_colored_cloud->points[shortest_distance_index1].r = 255, one_object_colored_cloud->points[shortest_distance_index1].g = 0, one_object_colored_cloud->points[shortest_distance_index1].b = 30;
	one_object_colored_cloud->points[shortest_distance_index2].r = 255, one_object_colored_cloud->points[shortest_distance_index2].g = 0, one_object_colored_cloud->points[shortest_distance_index2].b = 30;
	one_object_colored_cloud->points[shortest_distance_index3].r = 255, one_object_colored_cloud->points[shortest_distance_index3].g = 0, one_object_colored_cloud->points[shortest_distance_index3].b = 30;
	one_object_colored_cloud->points[shortest_distance_index4].r = 255, one_object_colored_cloud->points[shortest_distance_index4].g = 0, one_object_colored_cloud->points[shortest_distance_index4].b = 30;


	// Visualization
	std::string id_num = "bbox" + std::to_string(k);
	viewer->addCoordinateSystem(120, avg_x, avg_y, avg_z, id_num);

	viewer->addLine(point_cloud_set[0], point_cloud_set[1], 255, 0, 0, "line1" + std::to_string(k));
	viewer->addLine(point_cloud_set[1], point_cloud_set[2], 255, 0, 0, "line2" + std::to_string(k));
	viewer->addLine(point_cloud_set[2], point_cloud_set[3], 255, 0, 0, "line3" + std::to_string(k));
	viewer->addLine(point_cloud_set[3], point_cloud_set[0], 255, 0, 0, "line4" + std::to_string(k));

	std::vector<cv::Point> point_set;
	point_set.push_back(point_image_final1);
	point_set.push_back(point_image_final3);
	point_set.push_back(point_image_final2);
	point_set.push_back(point_image_final4);
	reorder_point(point_set);
	cv::polylines(image_rgb, point_set, true, cv::Scalar(155, 100, 20), 2, 8, 0);

	id_num = "segmented_cloud" + std::to_string(k);
	viewer->addPointCloud<pcl::PointXYZRGB>(one_object_colored_cloud, id_num);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, id_num);

	// Calculate the final IoU
	cv::Mat label, state, centroid;
	cv::connectedComponentsWithStats(image_current_object, label, state, centroid);
	int center_x = centroid.ptr<double>(1)[0];
	int center_y = centroid.ptr<double>(1)[1];

	std::string iou = std::to_string(best_iou);
	iou.resize(4);
	std::string text = "Score: " + iou;
	cv::putText(image_rgb, text, cv::Point(center_x, center_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 140, 255), 2);
	printf("The box %d base information:\n", k);
	printf("The centroid of the box is: %.2lfmm, %.2lfmm, %.2lfmm\n", avg_x, avg_y, avg_z);
	printf("The incline angle is: %.2lf¡ã\n", incline_angle * 180 / M_PI);
	printf("The rotation around Z-axis is: %d¡ã\n", abs(best_theta));
	printf("The IoU is: %.2lf\n\n", best_iou);
}
