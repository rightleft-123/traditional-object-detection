#pragma once
#include <vector>
#include <iostream>
#include "VzenseNebula_api.h"
#include <opencv2/opencv.hpp>

// Get the depth value of depth image
int get_depth(int height, int width, uint8_t*pData, cv::Mat dispImg, cv::Point point);

// Restrict the ROI by hand
void limit_ROI(double distance, const VzFrame frame, cv::Mat image_depth, cv::Mat image_binary_rgb, cv::Mat &image_new_depth);

// Combine multiple frame to hold a more stable result
void get_multi_frame_combine(std::vector<cv::Mat> &multi_image_container, cv::Mat &image_depth);
