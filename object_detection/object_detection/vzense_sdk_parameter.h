#pragma once
#include "VzenseNebula_api.h"
#include <opencv2/opencv.hpp>

void Get_Depth_Image(uint32_t slope, int height, int width, uint8_t*pData, cv::Mat& dispImg, cv::Point point);

void Get_RGB_Image(VzFrame rgb_frame, VzDeviceHandle devicehandle, cv::Mat &image_rgb);
