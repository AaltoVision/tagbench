#define _USE_MATH_DEFINES
#include <cmath>

#define NOMINMAX
#include <json.hpp>
#include <Eigen/Dense>
#include <TagDetector.h>
#include <DebugImage.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <algorithm>
#include <exception>

#include "pose_optimizer.h"

using json = nlohmann::json;

using mat2x4 = Eigen::Matrix<double, 2, 4>;
using mat3x4 = Eigen::Matrix<double, 3, 4>;
using mat4 = Eigen::Matrix4d;

static auto view_images = [](auto& get_image, int size)
{
    int i = 0;
    while (true)
    {
        cv::imshow("projection", get_image(i));
        int key = cv::waitKey() & 0xFF;
        if (key == 27) { break; } // Esc to close
        if (key == 'p') { i--; i = std::max(0, i); }
        else { i++; i = std::min(size - 1, i); };
    }
};

void parse_camera_intrinsics(json const &camera_intrinsics,
                             Eigen::Matrix4d& intrinsic_matrix)
{
    auto const focal_length_x = camera_intrinsics["focalLengthX"].get<float>();
    auto const focal_length_y = camera_intrinsics["focalLengthY"].get<float>();
    auto const principal_point_x = camera_intrinsics["principalPointX"].get<float>();
    auto const principal_point_y = camera_intrinsics["principalPointY"].get<float>();

    intrinsic_matrix = Eigen::Matrix4d::Zero();
    intrinsic_matrix(0, 0) = focal_length_x;
    intrinsic_matrix(1, 1) = focal_length_y;
    intrinsic_matrix(0, 2) = principal_point_x;
    intrinsic_matrix(1, 2) = principal_point_y;
    intrinsic_matrix(2, 2) = 1.0f;
}

void parse_camera_extrinsics(json const& camera_extrinsics, Eigen::Matrix4d& view_matrix)
{
    auto const& json_position = camera_extrinsics["position"];
    Eigen::Vector3d p = {
        json_position["x"].get<double>(),
        json_position["y"].get<double>(),
        json_position["z"].get<double>(),
    };
    auto const& json_orientation = camera_extrinsics["orientation"];
    Eigen::Quaterniond q = {
        json_orientation["w"].get<double>(),
        json_orientation["x"].get<double>(),
        json_orientation["y"].get<double>(),
        json_orientation["z"].get<double>(),
    };
    Eigen::Matrix3d R = q.toRotationMatrix();

    // TODO: maybe keep the original p and q handy as well

    view_matrix = Eigen::Matrix4d::Zero();
    view_matrix.block<3, 3>(0, 0) = R;
    view_matrix.block<3, 1>(0, 3) = -R * p;
    view_matrix(3, 3) = 1;
}

void put_text_lines(cv::Mat& image, std::stringstream& text)
{
    std::string line;
    int y = 20;
    while (std::getline(text, line, '\n'))
    {
        cv::putText(image, line, { 20, y }, cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 0, 0), 3, CV_AA);
        cv::putText(image, line, { 20, y }, cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 255, 255), 1, CV_AA);
        int baseline = 0;
        auto text_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.6, 3, &baseline);
        y += text_size.height + baseline;
    }
}

void visualize_projections(cv::InputArrayOfArrays images,
                           std::vector<mat2x4> const& detections,
                           std::vector<mat2x4> const& projections
                        //    std::vector<cv::Vec3f> const& Ts,
                        //    std::vector<cv::Matx33f> const& Rs
                           )
{
    auto get_image = [&](int i)
    {
        cv::Mat image_with_projections = images.getMat(i).clone();
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
        // label << "T: " << 10000* Ts[i] << "\n";
        // label << "R: " << Rs[i] << "\n";

        // cv::Ptr<cv::Formatter> formatMat = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
        // formatMat->set64fPrecision(3);
        // formatMat->set32fPrecision(3);
        // label << "R:\n" << formatMat->format( cv::Mat(Rs[i]) ) << std::endl;

        put_text_lines(image_with_projections, label);
        return image_with_projections;
    };
    view_images(get_image, images.size().width);
}


// Create synthetic dataset of extrinsic (Vs) matrices, and groundtruth marker corner projections (Ys),
// as well as the correct M expected from optimization.
// Input is Z, which contains marker corners in object space as its columns, as well as P.
// Working optimizer should be able to find an (at least locally) optimal pose from this data, while recorded camera/VIO data may
// have drift between matrices and images (same image may be reported with different V later).
void create_synthetic_dataset(mat4 const& Z,
                              mat3x4 const& P,
                              std::vector<mat4>& Vs,
                              std::vector<mat2x4>& Ys,
                              mat4 const& M)
{
    // Rotate 30 degrees left-right around Y axis
    Eigen::VectorXd angles = Eigen::ArrayXd::LinSpaced(50, -M_PI/6, M_PI/6);
    for (size_t i = 0; i < 50; ++i)
    {
        Eigen::Vector3d t = Eigen::Vector3d{ 0, 0, -4 };
        Eigen::Matrix3d R = Eigen::AngleAxisd(angles[i], Eigen::Vector3d::UnitY()).toRotationMatrix();
        Vs.emplace_back();

        // Vs.back() = make_pose_matrix(R, t);
        Vs.back() = Eigen::Matrix4d::Zero();
        Vs.back().block<3, 3>(0, 0) = R;
        Vs.back().block<3, 1>(0, 3) = -R * t;
        Vs.back()(3, 3) = 1;
    }
    for (size_t i = 0; i < 50; ++i)
    {
        Eigen::Vector3d t = Eigen::Vector3d{ 0, 0, -4 };
        Eigen::Matrix3d R = Eigen::AngleAxisd(angles[i], Eigen::Vector3d::UnitX()).toRotationMatrix();
        Vs.emplace_back();

        // Vs.back() = make_pose_matrix(R, t);
        Vs.back() = Eigen::Matrix4d::Zero();
        Vs.back().block<3, 3>(0, 0) = R;
        Vs.back().block<3, 1>(0, 3) = -R * t;
        Vs.back()(3, 3) = 1;
    }

    // Translate in a circle around Z axis, looking towards Z-
    // double d_angle = 2.0 * M_PI / 100;
    // Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    // double t_magnitude = 10; // TODO: relate to pixels (through s and resolution)
    // for (size_t i = 0; i < 100; ++i)
    // {
    //     Eigen::Vector3d t = Eigen::AngleAxisd(i * d_angle, -Eigen::Vector3d::UnitZ()) * Eigen::Vector3d{ 1, 0, 0 } * t_magnitude;
    //     Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    //     R(2, 2) = -1; // Look at Z-
    //     Vs.emplace_back();
    //     Vs.back() = make_pose_matrix(R, t);
    // }

    // Create groundtruth projections
    auto PVs = std::vector<mat3x4>(Vs.size());
    std::transform(Vs.begin(), Vs.end(),
                    PVs.begin(), [&](auto const& V) -> mat3x4 { return P * V; });
    Ys = project_corners(PVs, M, Z);

    // Add noise in screen space x,y
    for (auto& y : Ys)
    {
        y += mat2x4::Random() * 3;
        // y.row(0) += Eigen::Matrix<double, 1, 4>::Random() * 3;
    }
}


// Input in jsonl format (file or stdin) (file not supported currently):
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
    std::istream& input = std::cin;

    std::string line;
    using tag_corners = std::array<std::array<float, 2>, 4>;
    struct frame_info
    {
        Eigen::Matrix4d intrinsic_matrix;
        Eigen::Matrix4d view_matrix;
        std::vector<tag_corners> detections;
        std::string frame_path;
    };
    auto frames = std::vector<frame_info>{};
    size_t total_input_frames = 0;
    auto input_parse_time = timing([&]{
        while (std::getline(input, line))
        {
            ++total_input_frames;
            nlohmann::json j = nlohmann::json::parse(line);

            auto f = frame_info{};

            parse_camera_intrinsics(j["cameraIntrinsics"], /*f.projection_matrix,*/ f.intrinsic_matrix);
            // We will scale images to half size, so have to adjust these focal lengths and principal point as well
            f.intrinsic_matrix(0, 0) /= 2;
            f.intrinsic_matrix(1, 1) /= 2;
            f.intrinsic_matrix(0, 2) /= 2;
            f.intrinsic_matrix(1, 2) /= 2;
            parse_camera_extrinsics(j["cameraExtrinsics"], f.view_matrix);
            f.frame_path = j["framePath"];

            if (j.contains("markers"))
            {
                f.detections = j["markers"].get<std::vector<tag_corners>>();
            }
            else
            {
                auto temp_image = cv::imread(j["framePath"]);
                auto image = cv::Mat{};
                cv::resize(temp_image, image, temp_image.size() / 2);

                TagDetectorParams params;
                auto tag_family = TagFamily(std::string("Tag36h11"));
                TagDetector detector(tag_family, params);
                TagDetectionArray detections;

                detector.process(image, cv::Point2i{image.size().width, image.size().height}, detections);

                for (auto const &d : detections)
                {
                    f.detections.emplace_back();
                    memcpy(&f.detections.back(), d.p, sizeof(d.p));
                }
            }

            // For now, only consider frames where exactly only one Apriltag was detected
            if (f.detections.size() == 1)
            {
                frames.push_back(f);
            }
        }
    });
    std::printf("Parsed input in %.2fs\n", input_parse_time);

    auto setup_start = std::chrono::steady_clock::now();

    // Tag on the screen is 19.8cm (in arcore-7-1-single-2 data, where tag is shown on screen)
    auto const s = 0.198f;
    std::vector<cv::Point3f> cv_Z = {
        { -s/2, -s/2, 0 }, // bottom-left
        { s/2, -s/2, 0 }, // bottom-right
        { s/2, s/2, 0 }, // top-right
        { -s/2, s/2, 0 }, // top-left
    };

    auto Ys = std::vector<mat2x4>{};
    auto Vs = std::vector<cv::Matx44f>{};
    auto Rs = std::vector<cv::Matx33f>{};
    auto Ts = std::vector<cv::Vec3f>{};
    for (auto& f : frames)
    {
        auto const& d = f.detections[0];

        auto cv_Y = std::vector<cv::Point2f>{
            cv::Point2f{ d[0][0], d[0][1] }, // bottom-left
            cv::Point2f{ d[1][0], d[1][1] }, // bottom-right
            cv::Point2f{ d[2][0], d[2][1] }, // top-right
            cv::Point2f{ d[3][0], d[3][1] }, // top-left
        };
        mat2x4 Y;
        Y << d[0][0], d[1][0], d[2][0], d[3][0],
             d[0][1], d[1][1], d[2][1], d[3][1];
        Ys.push_back(Y);

        cv::Matx33f K = {
            (float)f.intrinsic_matrix(0, 0), (float)f.intrinsic_matrix(0, 1), (float)f.intrinsic_matrix(0, 2),
            (float)f.intrinsic_matrix(1, 0), (float)f.intrinsic_matrix(1, 1), (float)f.intrinsic_matrix(1, 2),
            (float)f.intrinsic_matrix(2, 0), (float)f.intrinsic_matrix(2, 1), (float)f.intrinsic_matrix(2, 2),
        };

        Vs.push_back({
            (float)f.view_matrix(0, 0), (float)f.view_matrix(0, 1), (float)f.view_matrix(0, 2), (float)f.view_matrix(0, 3),
            (float)f.view_matrix(1, 0), (float)f.view_matrix(1, 1), (float)f.view_matrix(1, 2), (float)f.view_matrix(0, 3),
            (float)f.view_matrix(2, 0), (float)f.view_matrix(2, 1), (float)f.view_matrix(2, 2), (float)f.view_matrix(0, 3),
            (float)f.view_matrix(3, 0), (float)f.view_matrix(3, 1), (float)f.view_matrix(3, 2), (float)f.view_matrix(3, 3),
        });

        cv::Vec3f r;
        Ts.push_back(cv::Vec3f{});
        auto& T = Ts.back();
        cv::solvePnP(cv_Z, cv_Y, K, cv::Vec4f{ 0, 0, 0, 0 }, r, T);

        Rs.push_back({});
        auto& R = Rs.back();
        cv::Rodrigues(r, R);
    }

    // TODO: [0, s] or [-s/2, s/2] ?
    // Probably does not affect solution, as long as we are consistent
    mat4 Z = mat4 {
        { -s/2, -s/2, 0, 1, }, // bottom-left
        { s/2, -s/2, 0, 1, }, // bottom-right
        { s/2, s/2, 0, 1, }, // top-right
        { -s/2, s/2, 0, 1, }, // top-left
    };
    Z.transposeInPlace();
    Z *= 1000;

    // Test fitting synthetic data
    {
        std::vector<mat4> Vs;
        std::vector<mat2x4> Ys;
        mat3x4 P = frames[0].intrinsic_matrix.block<3, 4>(0, 0).cast<double>();

        // TODO: try with more complicated M
        mat4 synthetic_M = mat4::Identity();
        // synthetic_M.col(0) *= -1;
        // synthetic_M.col(3) = 100*Eigen::Vector4d{ 0.5, -1, -2, 1 };
        create_synthetic_dataset(Z, P, Vs, Ys, synthetic_M);

        auto PVs = std::vector<mat3x4>(Vs.size());
        std::transform(Vs.begin(), Vs.end(),
                       PVs.begin(), [&](auto const& V) -> mat3x4 { return P * V; });
        
        // TODO: maybe start with better initial estimate
        mat4 M0 = mat4::Identity();
        mat4 M = optimize_pose(PVs, Ys, Z, M0); 
        throw_if_nan_or_inf(M);
        auto projections = project_corners(PVs, M, Z);
        auto images = std::vector<cv::Mat>(PVs.size());
        for (auto& image : images) {
            image = cv::Mat(cv::Size2d{ 2*P(0, 2), 2*P(1, 2) }, CV_8UC3, cv::Scalar(255, 255, 255));
        }

        std::for_each(PVs.begin(), PVs.end(), throw_if_nan_or_inf);
        std::for_each(Ys.begin(), Ys.end(), throw_if_nan_or_inf);
        std::for_each(projections.begin(), projections.end(), throw_if_nan_or_inf);

        visualize_projections(images, Ys, projections);
    }

    // Prepare some of the data into easier form
    auto images = std::vector<cv::Mat>{};
    auto Cs = std::vector<mat4>{};
    auto Ps = std::vector<mat3x4>{};
    for (auto i = 0u; i < frames.size(); ++i)
    {
        auto const& frame = frames[i];
        auto temp_image = cv::imread(frame.frame_path);
        images.push_back({});
        auto& image = images.back();
        cv::resize(temp_image, image, temp_image.size() / 2);

        Ps.push_back(frame.intrinsic_matrix.block<3, 4>(0, 0).cast<double>());
        auto const& R = Rs[i];
        auto const& T = Ts[i];

        mat4 C;
        C << R(0, 0), R(0, 1), R(0, 2), T(0),
            R(1, 0), R(1, 1), R(1, 2), T(1),
            R(2, 0), R(2, 1), R(2, 2), T(2),
            0, 0, 0, 1;
        Cs.push_back(C);
    }

    auto setup_end = std::chrono::steady_clock::now();
    auto setup_dt = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start).count() * 1e-3f;

    std::printf("Setup done in %.2fs\n", setup_dt);
    std::vector<mat3x4> PVs;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        PVs.push_back(Ps[i] * frames[i].view_matrix);
    }

    mat4 M0 = frames[0].view_matrix.inverse() * Cs[0];
    mat4 optimized_M;
    auto optimization_time = timing([&]{
        optimized_M = optimize_pose(PVs, Ys, Z, M0);
    });
    std::printf("Total optimization time: %.2fs\n", optimization_time);

    auto optimized_M_projected_points = project_corners(PVs, optimized_M, Z);
    // TODO: consider get_image kind of thing here, because loading and resizing 400 images upfront takes 7.5s in release build...
    visualize_projections(images, Ys, optimized_M_projected_points);

    // Final score
    auto mse = calculate_mse(optimized_M_projected_points, Ys);
    // auto mse = 0.0;
    // auto average_pixel_distance = 0.0;
    // for (size_t j = 0; j < frames.size(); ++j)
    // {
    //     mat2x4 residuals = optimized_M_projected_points[j] - Ys[j];
    //     auto r2 = (residuals.transpose() * residuals).diagonal();
    //     average_pixel_distance += r2.cwiseSqrt().sum();
    //     auto frame_mse = r2.sum();
    //     mse += frame_mse;
    //     // std::printf("e_%zu = %.2f\n", j, frame_mse);
    // }
    // average_pixel_distance /= (frames.size() * 4);
    std::printf("Total input frames: %zu\n", total_input_frames);
    std::printf("Frames considered (single marker detected): %zu\n", frames.size());
    std::printf("E(M) = %.2f (Error for optimized M)\n", mse);
    // std::printf("E(M)/n_corners = %.2f\n", mse / (frames.size() * 4));
    // std::printf("Average pixel distance = %.2f\n", average_pixel_distance);

}
