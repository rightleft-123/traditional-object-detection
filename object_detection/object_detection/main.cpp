#include <thread>
#include <iostream>
#include "point_cloud_processing.h"
#include "image_preprocessing.h"
#include "vzense_sdk_parameter.h"

using namespace std;
using namespace cv;

cv::Point g_Pos(320, 240);
int g_Slope = 7495;

int main() {

	double boxLength = 245, boxWidth = 155, boxDistance = 1600;
	printf("Enter the box size and the distance from the camera: ");
	scanf("&lf &lf &lf", boxLength, boxWidth, boxDistance);

	uint32_t deviceCount;
	VzDeviceInfo* pDeviceListInfo = NULL;
	VzDeviceHandle deviceHandle = 0;
	VzReturnStatus status = VzRetOthers;
	VzFrameReady FrameReady = { 0 };

	status = VZ_Initialize();
GET:
	status = VZ_GetDeviceCount(&deviceCount);
	if (status != VzReturnStatus::VzRetOK) {
		cout << "VzGetDeviceCount failed! make sure the DCAM is connected" << endl;
		this_thread::sleep_for(chrono::seconds(1));
		goto GET;
	}
	cout << "Get device count: " << deviceCount << endl;
	if (0 == deviceCount) {
		this_thread::sleep_for(chrono::seconds(1));
		goto GET;
	}

	pDeviceListInfo = new VzDeviceInfo[deviceCount];
	status = VZ_GetDeviceInfoList(deviceCount, pDeviceListInfo);
	status = VZ_OpenDeviceByUri(pDeviceListInfo[0].uri, &deviceHandle);
	cout << "open device successful,status :" << status << endl;

	status = VZ_StartStream(deviceHandle);

	// Set filters
	VzConfidenceFilterParams confidenceParameter;
	confidenceParameter.threshold = 75;
	confidenceParameter.enable = true;
	VZ_SetConfidenceFilterParams(deviceHandle, confidenceParameter);
	VZ_SetSpatialFilterEnabled(deviceHandle, true);
	VZ_SetFillHoleFilterEnabled(deviceHandle, true);

	// Switch RGB Resolution
	int resolution_w = 640;
	int resolution_h = 480;
	status = VZ_SetColorResolution(deviceHandle, resolution_w, resolution_h);
	status = VZ_SetTransformDepthImgToColorSensorEnabled(deviceHandle, true);

	cv::Mat imageDepth;
	cv::Mat imageRGB;
	VzSensorIntrinsicParameters cameraParam = {};
	VZ_GetSensorIntrinsicParameters(deviceHandle, VzColorSensor, &cameraParam);
	vector<cv::Mat> multiImage;

	for (;;)
	{
		VzFrame rgbFrame = { 0 };
		VzFrame depthFrame = { 0 };
		status = VZ_GetFrameReady(deviceHandle, 1200, &FrameReady);

		if (1 == FrameReady.transformedDepth)
		{
			// Calculate frame number
			status = VZ_GetFrame(deviceHandle, VzTransformDepthImgToColorSensorFrame, &depthFrame);
			if (status == VzReturnStatus::VzRetOK && depthFrame.pFrameData != NULL)
			{
				static int index = 0;
				static float fps = 0;
				static int64 start = cv::getTickCount();

				int64 current = cv::getTickCount();
				int64 diff = current - start;
				index++;
				if (diff > cv::getTickFrequency())
				{
					fps = index * cv::getTickFrequency() / diff;
					index = 0;
					start = current;
				}

				// Read depth image
				Get_Depth_Image(g_Slope, depthFrame.height, depthFrame.width, depthFrame.pFrameData, imageDepth, g_Pos);

				// Muti-frame combine
				cv::Mat imageCurrentFrame = imageDepth.clone();
				if (index < 8) {
					cv::cvtColor(imageCurrentFrame, imageCurrentFrame, cv::COLOR_BGR2GRAY);
					cv::threshold(imageCurrentFrame, imageCurrentFrame, 80, 255, cv::THRESH_BINARY);
					multiImage.push_back(imageCurrentFrame);
				}
				else if (index == 8) {
					// Record the program Beginning time
					auto beforeTime = std::chrono::steady_clock::now();
					printf("\n\nStart detection on one frame\n");

					// Combine multi-depth-frame
					get_multi_frame_combine(multiImage, imageDepth);

					// Read rgb image
					Get_RGB_Image(rgbFrame, deviceHandle, imageRGB);

					// Limit the ROI by manual
					limit_ROI(boxDistance, depthFrame, imageDepth, imageRGB, imageCurrentFrame);

					// Connected Component  analysis
					cv::Mat label, state, centroid;
					int nccomps = cv::connectedComponentsWithStats(imageCurrentFrame, label, state, centroid);

					// Advocate the label area with specific color && Eliminate the small area
					vector<cv::Vec3d> colors(nccomps);
					srand(1);
					for (int i = 1; i < nccomps; i++) {
						if (state.at<int>(i, cv::CC_STAT_AREA) > 400)
							colors[i] = cv::Vec3d(rand() % 255, rand() % 255, rand() % 255);
					}

					// Record the point cloud according to the Connectivity
					pcl::PointCloud<pcl::PointXYZ>::Ptr entireCloud(new pcl::PointCloud<pcl::PointXYZ>);
					pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Segmentation Viewer"));
					viewer->setBackgroundColor(0, 0, 0);

					VzFrame &srcFrame = depthFrame;
					const uint16_t* pDepthFrameData = (uint16_t*)srcFrame.pFrameData;

					for (int k = 0; k < nccomps; k++)
					{
						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
						pcl::PointXYZ point;
						cv::Mat imageCurrentObject = cv::Mat::zeros(imageCurrentFrame.size(), CV_8UC1);

						if (state.at<int>(k, cv::CC_STAT_AREA) > 400)
						{
							// Get the point cloud according to connected component analysis
							for (int i = 0, offset = i * srcFrame.width; i < srcFrame.height; i++)
							{
								for (int j = 0; j < srcFrame.width; j++)
								{
									VzDepthVector3 depthPoint = { j, i, pDepthFrameData[offset + j] };
									VzVector3f worldV = {};
									VZ_ConvertDepthToPointCloud(deviceHandle, &depthPoint, &worldV, 1, &cameraParam);

									// Match the label
									int labels = label.at<int>(i, j);
									if (labels == k && k > 0)
									{
										point.x = worldV.x;
										point.y = worldV.y;
										point.z = worldV.z;
										imageCurrentObject.at<uchar>(i, j) = 255;
										cloud->push_back(point);
									}
									// Save the point cloud background
									if (k == 0 && worldV.z < 7000)
									{
										point.x = worldV.x;
										point.y = worldV.y;
										point.z = worldV.z;
										entireCloud->push_back(point);
									}
								}
								offset += srcFrame.width;
							}

							if (k > 0) {
								get_final_score(k, boxLength, boxWidth, viewer, cloud, colors, imageCurrentObject, imageRGB, cameraParam);
							}
							else {
								viewer->addPointCloud<pcl::PointXYZ>(entireCloud, "background_cloud");
								viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "background_cloud");
							}
						}
					}

					// Show the final result
					viewer->addCoordinateSystem(200, 0, 0, 0, "origin");
					cv::rectangle(imageRGB, cv::Point(200, 200), cv::Point(450, 450), cv::Scalar(255, 245, 125), 2);

					cv::imshow("Depth Image", imageCurrentFrame);
					cv::imshow("Detect Result", imageRGB);

					auto afterTime = std::chrono::steady_clock::now();
					double durationTime = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
					printf("Finshed detection on one frame\nTotal time: %.2lfms\n\n", durationTime);
					while (!viewer->wasStopped())
					{
						viewer->spinOnce(100);
					}
				}
			}
		}
	}

	status = VZ_StopStream(deviceHandle);
	status = VZ_CloseDevice(&deviceHandle);
	status = VZ_Shutdown();

	return 0;
}
