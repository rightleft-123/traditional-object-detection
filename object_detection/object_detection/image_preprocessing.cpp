#include "image_preprocessing.h"


int get_depth(int height, int width, uint8_t*pData, cv::Mat dispImg, cv::Point point)
{
	dispImg = cv::Mat(height, width, CV_16UC1, pData);
	cv::Point2d pointxy = point;
	int val = dispImg.at<ushort>(pointxy);
	return val;
}

void get_multi_frame_combine(std::vector<cv::Mat> &multi_image_container, cv::Mat &image)
{
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::threshold(image, image, 80, 255, cv::THRESH_BINARY);
	cv::Mat fuse_image = image.clone();

	for (int i = 0; i < multi_image_container.size(); i++) {
		fuse_image += multi_image_container[i];
	}

	fuse_image /= (multi_image_container.size() + 1);
	multi_image_container.clear();
	cv::threshold(fuse_image, image, 10, 255, cv::THRESH_BINARY);
}

void limit_ROI(double distance, const VzFrame frame, cv::Mat image_depth, cv::Mat image_binary_rgb, cv::Mat &image_new_depth)
{
	// Binarization RGB
	image_new_depth = image_depth.clone();
	cv::cvtColor(image_binary_rgb, image_binary_rgb, cv::COLOR_BGR2GRAY);
	cv::adaptiveThreshold(image_binary_rgb, image_binary_rgb, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, -3);

	distance = (int)distance;
	// Restrict depth and ROI by parameter
	for (int row = 0; row < frame.height; row++) {
		for (int column = 0; column < frame.width; column++) {
			if ((200 < row && row < 450) && (200 < column && column < 450)) {
				cv::Point g_Pos1(column, row);
				int depth = get_depth(frame.height, frame.width, frame.pFrameData, image_depth, g_Pos1);
				if (depth > distance || depth < 1200) {
					image_new_depth.ptr<uchar>(row)[column] = 0;
					image_binary_rgb.ptr<uchar>(row)[column] = 0;
				}
			}
			else {
				image_new_depth.ptr<uchar>(row)[column] = 0;
				image_binary_rgb.ptr<uchar>(row)[column] = 0;
			}
		}
	}

	// Make the ROI more clear by combine RGB and Depth image
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(5, 5));
	cv::erode(image_new_depth, image_new_depth, element);
	image_new_depth = image_binary_rgb | image_new_depth;
	cv::medianBlur(image_new_depth, image_new_depth, 5);
}
