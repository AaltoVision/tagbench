#include "cv_helpers.h"

#include <functional>
#include <sstream>
#include <iomanip>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

// Helper for configurable image loading
// - When plotting the projections, caching images instead of reloading from disk each time
//   makes browsing between images much smoother
// - However, if the dataset is very large, you cannot cache images since you might run out of memory
std::function<cv::Mat(int)> make_image_loader(bool cache_images, int image_downscale_factor,
    std::vector<std::string> const& frame_paths)
{
    auto load_scaled_frame = [&frame_paths, image_downscale_factor](int i)
    {
        auto image = cv::Mat{};
        auto temp_image = cv::imread(frame_paths[i]);
        cv::resize(temp_image, image, temp_image.size() / image_downscale_factor);
        return image;
    };

    if (!cache_images)
    {
        return load_scaled_frame;
    }
    else
    {
        auto loaded_images = std::map<int, cv::Mat>{};
        return [load_scaled_frame, loaded_images, &frame_paths](int i) mutable
        {
            auto it = loaded_images.find(i);
            if (it == loaded_images.end())
            {
                loaded_images[i] = load_scaled_frame(i);
                return loaded_images[i].clone();
            }
            else
            {
                return it->second.clone();
            }
        };

    }
}

void view_images(std::function<cv::Mat(int)> const& get_image, int size)
{
    int i = 0;
    while (true)
    {
        cv::imshow("projection", get_image(i));
        int key = cv::waitKey() & 0xFF;
        if (key == 27) { break; } // Esc to close
        if (key == 'r') { i = 0; }
        else if (key == 'p') { i--; i = std::max(0, i); }
        else { i++; i = std::min(size - 1, i); };
    }
}

void put_text_lines(cv::Mat& image, std::stringstream& text, int y)
{
    std::string line;
    while (std::getline(text, line, '\n'))
    {
        cv::putText(image, line, { 20, y }, cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 0, 0), 3, CV_AA);
        cv::putText(image, line, { 20, y }, cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 255, 255), 1, CV_AA);
        int baseline = 0;
        auto text_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.6, 3, &baseline);
        y += text_size.height + baseline;
    }
}

void visualize_projections(std::function<cv::Mat(int)> const& get_image,
                           size_t image_count,
                           e_vec<mat2x4> const& detections,
                           e_vec<mat2x4> const& projections
                           )
{
    auto get_labeled_image = [&](int i)
    {
        cv::Mat image_with_projections = get_image(i).clone();
        for (auto j = 0; j < 4; ++j)
        {
            auto detected_point = cv::Point2d{ detections[i](0, j), detections[i](1, j) };
            cv::drawMarker(image_with_projections, detected_point, CV_RGB(255,0,0), 0, 20, 2);

            auto projected_point = cv::Point2d{ projections[i](0, j), projections[i](1, j) };
            cv::drawMarker(image_with_projections, projected_point, CV_RGB(0,255,0), 1, 20, 2);
        }
        std::stringstream label;
        label << std::setprecision(3);
        label << "Image: " << i << "\n";
        label << "Red: groundtruth\n";
        label << "Green: projections\n";

        put_text_lines(image_with_projections, label, 20);
        return image_with_projections;
    };
    view_images(get_labeled_image, (int)image_count);
}
