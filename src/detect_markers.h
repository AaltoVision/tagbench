#pragma once

#include <vector>
#include <array>

namespace cv
{
    class Mat;
}

using tag_corners = std::array<std::array<double, 2>, 4>;

// Use the apriltag library to detect apriltag corners in the image
std::vector<tag_corners> detect_markers(cv::Mat const& image);