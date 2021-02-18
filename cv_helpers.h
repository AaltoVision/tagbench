#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include <functional>
#include <sstream>

using mat2x4 = Eigen::Matrix<double, 2, 4>;

// Fixed-size Eigen types' allocation must be aligned
template<typename T>
using e_vec = std::vector<T, Eigen::aligned_allocator<T>>;

void view_images(std::function<cv::Mat(int)> const& get_image, int size);

void put_text_lines(cv::Mat& image, std::stringstream& text, int y);

void visualize_projections(std::function<cv::Mat(int)> const& get_image,
                           size_t image_count,
                           e_vec<mat2x4> const& detections,
                           e_vec<mat2x4> const& projections
                           );
