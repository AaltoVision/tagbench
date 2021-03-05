#include "detect_markers.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

std::vector<tag_corners> detect_markers(cv::Mat const& image)
{
    // The detector expects grayscale images
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    apriltag_detector_t* detector = apriltag_detector_create();
    auto tag_family = tag36h11_create();
    apriltag_detector_add_family(detector, tag_family);

    auto im = image_u8_t { image_gray.cols, image_gray.rows,
                           image_gray.cols, image_gray.data };
    zarray_t *detections = apriltag_detector_detect(detector, &im);

    auto markers = std::vector<tag_corners>{};
    for (int i = 0; i < zarray_size(detections); ++i)
    {
        apriltag_detection_t* detection;
        zarray_get(detections, i, &detection);
        markers.emplace_back();
        static_assert(sizeof(markers.back()) == sizeof(detection->p));
        for (size_t c = 0; c < 4; ++c)
        {
            markers.back()[c][0] = detection->p[c][0];
            markers.back()[c][1] = detection->p[c][1];
        }
    }

    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(detector);
    tag36h11_destroy(tag_family);
    return markers;
}

void scale_markers(std::vector<tag_corners>& markers, double scale_factor)
{
    for (auto& marker : markers)
    {
        for (auto& corner : marker)
        {
            // Scale marker corner positions to match original frame size
            corner[0] *= scale_factor;
            corner[1] *= scale_factor;
        }
    }
}
