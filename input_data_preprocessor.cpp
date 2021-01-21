#include <json.hpp>
#include <Eigen/Dense>
#include <TagDetector.h>
#include <DebugImage.h>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <array>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using json = nlohmann::json;

static auto timing = [](auto const& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e-3f;
    return dt;
};

// hacky, TODO some better way to contain the frame<->index info
int parse_frame_number_from_path(std::filesystem::path const& path)
{
    return std::stoi(path.filename().stem().string().c_str());
}

using tag_corners = std::array<std::array<float, 2>, 4>;

std::vector<tag_corners> detect_marker_corners(cv::Mat& image)
{
    TagDetectorParams params;
    auto tag_family = TagFamily(std::string("Tag36h11"));
    TagDetector detector(tag_family, params);
    TagDetectionArray detections;

    if (image.empty())
        exit(1);
    auto temp_image = cv::Mat{};
    cv::resize(image, temp_image, image.size() / 2);
    image = temp_image;

    // cv::Point2d optical_center(principal_point_x, principal_point_y);
    detector.process(image, cv::Point2i{image.size().width, image.size().height}, detections);
    auto dt = timing([&] {
        detector.process(image, cv::Point2i{image.size().width, image.size().height}, detections);
    });

    std::vector<tag_corners> marker_corners;
    for (auto const &d : detections)
    {
        marker_corners.emplace_back();
        memcpy(&marker_corners.back(), d.p, sizeof(d.p));
#if 0
        for (int j=0; j<4; ++j)
        {
            // cv::Mat img2 = image*0.5f + 127.0f;
            cv::Mat img2 = image;
            cv::line( img2, d.p[j], d.p[(j+1)%4], CV_RGB(255,0,0), 2, CV_AA);
            cv::imshow("marker debug", img2);
            cv::waitKey();
        }
#endif
    }
    return marker_corners;
}

// Prepare input data from given directory into .jsonl that 'tagbench' program expects
// Output in jsonl format:
//
//      {
//          "frameIndex": 1,
//          "framePath": "frame0001.png",
//          "cameraIntrinsics": {focal lengths, principal point...},
//          "cameraExtrinsics": {pos, rotation...},
//          "markers": [{"id":0,"corners":[[p0x,p0y],[p1x,p1y]...]}, {"id":1...}]
//      }
//
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Missing input data path, exiting" << std::endl;
        return 1;
    }

    // Write to given file path, otherwise to stdout
    FILE* out = stdout;
    if (argc > 2)
    {
        if (fopen_s(&out, argv[2], "w"))
        {
            std::cerr << "Failed to open output file" << std::endl;
            return 2;
        }
    }

    auto const input_dir = std::filesystem::path{ argv[1] };
    auto const input_frames_dir = input_dir / "frames";
    auto input_sensor_readings = std::ifstream{ input_dir / "data.jsonl" };

    // Read in frame paths with their indices
    auto frame_paths = std::map<int, std::filesystem::path>{};
    auto frame_directory = std::filesystem::directory_iterator(input_frames_dir);
    try {
        std::transform(frame_directory, std::filesystem::directory_iterator{}, std::inserter(frame_paths, frame_paths.end()),
            [](auto const& path)
            {
                return std::make_pair(parse_frame_number_from_path(path), path);
            });
    } catch (std::exception& e)
    {
        std::printf("Failed to parse frame numbers\n");
        std::printf("%s\n", e.what());
        return 3;
    }

    // Parse time->frame_index mapping
    auto time_to_frame_index = std::unordered_map<float, int>{};
    auto line = std::string{};
    while (std::getline(input_sensor_readings, line))
    {
        auto line_json = json::parse(line);

        if (line_json.contains("frames"))
        {
            auto time = line_json["time"].get<float>();
            auto frame_index = line_json["number"].get<int>();
            time_to_frame_index[time] = frame_index;
        }
    }

    // Parse all camera parameters for frames
    auto camera_parameters = std::unordered_map<int, json>{};
    input_sensor_readings = std::ifstream{ input_dir / "data.jsonl" };
    while (std::getline(input_sensor_readings, line))
    {
        auto line_json = json::parse(line);

        if (line_json.contains("arcore"))
        {
            auto const time = line_json["time"].get<float>();
            auto const frame_index_it = time_to_frame_index.find(time);
            if (frame_index_it != time_to_frame_index.end())
            {
                auto const frame_index = frame_index_it->second;
                if (camera_parameters[frame_index].is_null())
                    camera_parameters[frame_index] = json::object();
                camera_parameters[frame_index]["cameraExtrinsics"] = line_json["arcore"];
            }
        }
        if (line_json.contains("frames"))
        {
            auto const frame_index = line_json["number"].get<int>();
            if (camera_parameters[frame_index].is_null())
                camera_parameters[frame_index] = json::object();
            camera_parameters[frame_index]["cameraIntrinsics"] = line_json["frames"][0]["cameraParameters"];
        }
    }

    // Detect markers and write out results and other preprocessed data
    for (auto const& f : frame_paths)
    {
        auto const frame_index = f.first;
        auto const& frame_path = f.second;

        auto& p = camera_parameters[frame_index];

        if (p.contains("cameraIntrinsics") && p.contains("cameraExtrinsics"))
        {
            auto j = json{};
            j["cameraIntrinsics"] = p["cameraIntrinsics"];
            j["cameraExtrinsics"] = p["cameraExtrinsics"];

            j["frameIndex"] = frame_index;
            j["framePath"] = frame_path.string();
            // TODO: j["frameTime"] perhaps

            auto frame = cv::imread(frame_path.string());
            j["markers"] = detect_marker_corners(frame);

            fprintf(out, "%s\n", j.dump().c_str());
        }
    }


    if (out && (out != stdout)) fclose(out);
}

