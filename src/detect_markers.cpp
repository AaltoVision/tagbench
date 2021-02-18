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
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    apriltag_detector_t* detector = apriltag_detector_create();
    auto tag_family = tag36h11_create();
    apriltag_detector_add_family(detector, tag_family);

    auto im = image_u8_t { image_gray.cols, image_gray.rows,
                           image_gray.cols, image_gray.data };
    zarray_t *detections = apriltag_detector_detect(detector, &im);

    auto corners = std::vector<tag_corners>{};
    for (int i = 0; i < zarray_size(detections); ++i)
    {
        apriltag_detection_t* detection;
        zarray_get(detections, i, &detection);
        corners.emplace_back();
        static_assert(sizeof(corners.back()) == sizeof(detection->p));
        for (size_t c = 0; c < 4; ++c)
        {
            corners.back()[c][0] = detection->p[c][0];
            corners.back()[c][1] = detection->p[c][1];
        }
    }

    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(detector);
    tag36h11_destroy(tag_family);
    return corners;
}
