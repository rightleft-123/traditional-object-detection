#include "vzense_sdk_parameter.h"


void Get_Depth_Image(uint32_t slope, int height, int width, uint8_t*pData, cv::Mat& dispImg, cv::Point point)
{
	dispImg = cv::Mat(height, width, CV_16UC1, pData);
	cv::Point2d pointxy = point;
	int val = dispImg.at<ushort>(pointxy);
	char text[20];
#ifdef _WIN32
	sprintf_s(text, "%d", val);
#else
	snprintf(text, sizeof(text), "%d", val);
#endif
	dispImg.convertTo(dispImg, CV_8U, 255.0 / slope);
	applyColorMap(dispImg, dispImg, cv::COLORMAP_RAINBOW);
	int color;
	if (val > 2500)
		color = 0;
	else
		color = 4096;
	circle(dispImg, pointxy, 4, cv::Scalar(color, color, color), -1, 8, 0);
	putText(dispImg, text, pointxy, cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar(color, color, color));
}

void Get_RGB_Image(VzFrame rgb_frame, VzDeviceHandle devicehandle, cv::Mat &image_rgb)
{
	// Read RGB image
	VzReturnStatus status = VZ_GetFrame(devicehandle, VzColorFrame, &rgb_frame);
	if (status == VzReturnStatus::VzRetOK && rgb_frame.pFrameData != NULL) {
		image_rgb = cv::Mat(rgb_frame.height, rgb_frame.width, CV_8UC3, rgb_frame.pFrameData);
	}
}
