#include <json.hpp>
#include <Eigen/Dense>
#include <TagDetector.h>
#include <DebugImage.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

using json = nlohmann::json;

static auto timing = [](auto& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e-3f;
    return dt;
};

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Missing sensor readings path, exiting" << std::endl;
        return 1;
    }
    if (argc < 3)
    {
        std::cerr << "Missing video frame path, exiting" << std::endl;
        return 2;
    }
    std::ifstream sensor_readings(argv[1]);
    if (!sensor_readings)
    {
        std::cerr << "Failed to read input file, may not exist, exiting" << std::endl;
    }
    std::string line;

    Eigen::Matrix4f projection_matrix = Eigen::Matrix4f::Zero();
    Eigen::Matrix4f model_view_matrix = Eigen::Matrix4f::Zero();
float principal_point_x;
float principal_point_y;

    while (std::getline(sensor_readings, line))
    {
        nlohmann::json j = nlohmann::json::parse(line);
        if (j.contains("sensor"))
        {
        }
        if (j.contains("frames"))
        {
            auto const& json_camera_parameters = j["frames"][0]["cameraParameters"];
            auto const focal_length_x = json_camera_parameters["focalLengthX"].get<float>();
            auto const focal_length_y = json_camera_parameters["focalLengthY"].get<float>();
            // auto const principal_point_x = json_camera_parameters["principalPointX"].get<float>();
            // auto const principal_point_y = json_camera_parameters["principalPointY"].get<float>();
            principal_point_x = json_camera_parameters["principalPointX"].get<float>();
            principal_point_y = json_camera_parameters["principalPointY"].get<float>();

            // OpenGL near and far clip plane distances (meters)
            constexpr float zNear = 0.01f;
            constexpr float zFar = 20.0f;

            projection_matrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
            projection_matrix(2, 3) = -2 * zFar * zNear / (zFar - zNear);
            projection_matrix(3, 2) = -1;

            Eigen::Matrix4f intrinsic_matrix = Eigen::Matrix4f::Zero();
            intrinsic_matrix(0, 0) = focal_length_x;
            intrinsic_matrix(1, 1) = focal_length_y;
            intrinsic_matrix(0, 2) = principal_point_x;
            intrinsic_matrix(1, 2) = principal_point_y;
            intrinsic_matrix(2, 2) = 1.0f;

            // rotated image: flip X & Y
            // projection_matrix(0, 1) = -2 * focalLength / width;
            // projection_matrix(1, 0) = 2 * focalLength / height;

            // std::cout << projection_matrix << std::endl;
            // std::cout << intrinsic_matrix << std::endl;
            // exit(0);
        }
        if (j.contains("arcore"))
        {
            auto const& json_position = j["arcore"]["position"];
            Eigen::Vector3f p = {
                json_position["x"].get<float>(),
                json_position["y"].get<float>(),
                json_position["z"].get<float>(),
            };
            auto const& json_orientation = j["arcore"]["orientation"];
            Eigen::Quaternionf q = {
                json_orientation["w"].get<float>(),
                json_orientation["x"].get<float>(),
                json_orientation["y"].get<float>(),
                json_orientation["z"].get<float>(),
            };
            Eigen::Matrix3f R = q.toRotationMatrix();

            model_view_matrix.block<3, 3>(0, 0) = R;
            model_view_matrix.block<3, 1>(0, 3) = -R * p;
            model_view_matrix(3, 3) = 1;

            // Intrinsic matrix...
        }
    }

    TagDetectorParams params;
    auto tag_family = TagFamily(std::string("Tag36h11"));
    TagDetector detector(tag_family, params);
    TagDetectionArray detections;

    auto const frames = argc - 2;

    // printf("Finished in %.2fs\n", dt);

    // cv::putText(img, "hello", cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
    //     CV_RGB(0,0,0), 3, CV_AA);
    // cv::putText(img, "hello", cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
    //     CV_RGB(255,255,255), 1, CV_AA);
    // cv::imshow("window name", img);
    // cv::waitKey();

    // printf("Superimposed in %.2f\n", timing([&]{tag_family.superimposeDetections(img, detections);}));

    // First step for now: print pixel coords for tags for each frame
    // Use jsonl and stdout, easy to pipe somewhere
    for (int i = 0; i < frames; ++i)
    {
        auto img = cv::imread(argv[2+i]);
        if (img.empty()) exit(1);
        auto temp_img = cv::Mat{};
        cv::resize(img, temp_img, img.size()/2);
        img = temp_img;

        // cv::Point2d optical_center(principal_point_x, principal_point_y);

        detector.process(img, cv::Point2i{ img.size().width, img.size().height }, detections);
        auto dt = timing([&] {
            detector.process(img, cv::Point2i{ img.size().width, img.size().height }, detections);
        });

        using frame_marker_detections = std::vector<float[4][2]>;
        json line;
        line["detected_markers"] = frame_marker_detections{};
        for (auto const& d : detections)
        {
            float corners[4][2];
            memcpy(corners, d.p, sizeof(corners));
            line["detected_markers"] = corners;

            #if 1
            for (int j=0; j<4; ++j) {
                // cv::Mat img2 = img*0.5f + 127.0f;
                cv::Mat img2 = img;
                cv::line( img2, d.p[j], d.p[(j+1)%4], CV_RGB(255,0,0), 2, CV_AA);
                cv::imshow("marker debug", img2);
                cv::waitKey();
            }
            #endif
        }

        printf("%s\n", line.dump().c_str());
    }

    // cv::solvePnP();


    // Detect corners in image ((sub)pixel) to produce accurate ground truth for the
    // corner bearings w.r.t. camera (in frames with not too much motion blur)


    // 1. Place "virtual checkerboard" in the AR coordinate system
    // 2. Project its vertices to each frame using the recorded camera matrices
    // 3. Compute the error w.r.t. ground truth corners
    //      - We don't necessarily have a good ground truth position for the checkerboard
    //        in the AR coords, but we could just select the pose that minimizes total RMSE of corner projections



    // Project with camera matrix

    // Least RMSE



}