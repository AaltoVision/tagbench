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

#include <nlohmann/json.hpp>
#include <cxxopts.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "detect_markers.h"

using json = nlohmann::json;

// hacky, TODO some better way to contain the frame<->index info
int parse_frame_number_from_path(fs::path const& path)
{
    return std::stoi(path.filename().stem().string().c_str());
}


// Prepare input data from given directory into .jsonl that 'tagbench' program expects
// Example output in jsonl format:
//
//      {
//          "frameIndex": 1,
//          "framePath": "frame0001.png",
//          "cameraIntrinsics": {focal lengths, principal point...},
//          "cameraExtrinsics": {position, rotation...},
//          "markers": [{"id":0,"corners":[[p0x,p0y],[p1x,p1y]...]}, {"id":1...}]
//      }
//
int main(int argc, char* argv[])
{
    auto input_dir_option = std::string{};
    auto pose_data_name = std::string{};
    auto output_file_path = std::string{};
    auto image_downscale_factor = 1;
    auto frames_with_vio_only = true;

    cxxopts::Options options(argv[0], "");
    options.add_options()
        ("i,input", "Path to the input data directory", cxxopts::value(input_dir_option))
        ("o,output", "Output path (.jsonl file) (default: stdout)", cxxopts::value(output_file_path))
        ("frames_with_vio_only", "Only output frames which have a VIO reading at the same timestamp (otherwise, output also frames between VIO readings)", cxxopts::value(frames_with_vio_only))
        ("n,pose_data_name", "Name of pose data in the input (for example arcore)", cxxopts::value(pose_data_name))
        ("d,image_downscale_factor", "Image downscaling factor", cxxopts::value(image_downscale_factor)->default_value("1"))
        ("h,help", "Show this help message")
        ;

    auto parsed_args = options.parse(argc, argv);
    if (parsed_args.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }
    if (parsed_args.count("input") == 0)
    {
        std::cerr << "Missing argument: input_data_path." << std::endl;
        std::cerr << options.help() << std::endl;
        std::cerr << "See README.md for more instructions." << std::endl;
        return 1;
    }
    if (parsed_args.count("pose_data_name") == 0)
    {
        std::cerr << "Missing argument: pose_data_name." << std::endl;
        std::cerr << options.help() << std::endl;
        std::cerr << "See README.md for more instructions." << std::endl;
        return 1;
    }

    // Write to given file path, otherwise to stdout
    std::ofstream output_file_stream;
    if (parsed_args.count("output"))
    {
        output_file_stream.open(output_file_path);
    }
    std::ostream& out = parsed_args.count("output")
        ? (std::ostream&)output_file_stream
        : std::cout;

    auto const input_dir = fs::path{ input_dir_option };
    auto const input_frames_dir = input_dir / "frames";
    auto const input_sensor_readings_path = input_dir / "data.jsonl";
    auto input_sensor_readings = std::ifstream{ input_sensor_readings_path };
    if (!input_sensor_readings)
    {
        std::cerr << "Failed to open input file (does it exist?): " << input_sensor_readings_path;
        return 1;
    }

    // Read in frame paths with their indices
    auto frame_paths = std::map<int, fs::path>{};
    auto frame_directory = fs::directory_iterator(input_frames_dir);
    try {
        std::transform(frame_directory, fs::directory_iterator{}, std::inserter(frame_paths, frame_paths.end()),
            [](auto const& path)
            {
                return std::make_pair(parse_frame_number_from_path(path), path);
            });
    } catch (std::exception& e)
    {
        std::cerr << "Failed to parse frame numbers" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Parse time->frame_index mapping
    auto time_to_frame_index = std::unordered_map<float, int>{};
    auto frame_index_to_time = std::unordered_map<int, float>{};
    auto line = std::string{};
    while (std::getline(input_sensor_readings, line))
    {
        auto line_json = json::parse(line);

        if (line_json.contains("frames"))
        {
            auto time = line_json["time"].get<float>();
            auto frame_index = line_json["number"].get<int>();
            time_to_frame_index[time] = frame_index;
            frame_index_to_time[frame_index] = time;
        }
    }

    // Parse all camera parameters for frames
    auto camera_parameters = std::unordered_map<int, json>{};
    input_sensor_readings = std::ifstream{ input_dir / "data.jsonl" };
    while (std::getline(input_sensor_readings, line))
    {
        auto line_json = json::parse(line);

        if (line_json.contains(pose_data_name))
        {
            auto const time = line_json["time"].get<float>();
            auto const frame_index_it = time_to_frame_index.find(time);
            if (frame_index_it != time_to_frame_index.end())
            {
                auto const frame_index = frame_index_it->second;
                if (camera_parameters[frame_index].is_null())
                    camera_parameters[frame_index] = json::object();
                camera_parameters[frame_index]["cameraExtrinsics"] = line_json[pose_data_name];
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
            j["time"] = frame_index_to_time[frame_index];

            // Note: marker corner positions are exported in original resolution,
            // not downscaled resolution
            auto frame = cv::imread(frame_path.string());
            auto scaled_frame = cv::Mat{};
            cv::resize(frame, scaled_frame, frame.size() / image_downscale_factor);
            auto markers = detect_markers(scaled_frame);
            scale_markers(markers, image_downscale_factor);
            j["markers"] = markers;

            // Useful for flipping marker position y-values in tagbench
            j["frameHeight"] = frame.size().height;

            out << j.dump() << std::endl;
        }
    }
}

