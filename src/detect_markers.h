#pragma once

#include <vector>
#include <array>

namespace cv
{
    class Mat;
}

using tag_corners = std::array<std::array<double, 2>, 4>;

// Use the apriltag library to detect apriltag corners in the image.
// For now, supporting just the 36h11 tag family.
std::vector<tag_corners> detect_markers(cv::Mat const& image);

// Helper for scaling marker corner positions, in case they are detected
// from a downscaled image, or need to be shown with downscaled images.
void scale_markers(std::vector<tag_corners>& markers, double scale_factor);
