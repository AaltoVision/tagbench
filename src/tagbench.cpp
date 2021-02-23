#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <nlohmann/json.hpp>
#include <Eigen/Dense>
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
#include <functional>

#include "cv_helpers.h"
#include "detect_markers.h"
#include "pose_optimizer.h"

using json = nlohmann::json;

using mat2x4 = Eigen::Matrix<double, 2, 4>;
using mat3x4 = Eigen::Matrix<double, 3, 4>;
using mat3 = Eigen::Matrix3d;
using mat4 = Eigen::Matrix4d;
using vec2 = Eigen::Vector2d;
using vec3 = Eigen::Vector3d;
using vec4 = Eigen::Vector4d;

// Fixed-size Eigen types' allocation must be aligned
template<typename T>
using e_vec = std::vector<T, Eigen::aligned_allocator<T>>;

std::function<cv::Mat(int)> make_image_loader(bool cache_images, int image_minify_factor, std::vector<std::string> const& frame_paths)
{
    auto load_scaled_frame = [&frame_paths, image_minify_factor](int i)
    {
        auto image = cv::Mat{};
        auto temp_image = cv::imread(frame_paths[i]);
        cv::resize(temp_image, image, temp_image.size() / image_minify_factor);
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

// Mainly for debugging; change view matrices Vs s.t. they are relative to V
void relate_views_to(e_vec<mat4>& Vs, mat4 const& V_inv)
{
    for (auto& V : Vs)
    {
        V = V * V_inv;
    };
}

void parse_camera_intrinsics(json const &camera_intrinsics,
                             mat3x4& intrinsic_matrix)
{
    auto const focal_length_x = camera_intrinsics["focalLengthX"].get<float>();
    auto const focal_length_y = camera_intrinsics["focalLengthY"].get<float>();
    auto const principal_point_x = camera_intrinsics["principalPointX"].get<float>();
    auto const principal_point_y = camera_intrinsics["principalPointY"].get<float>();

    // NOTE: opengl view matrix has camera looking at z-; this intrinsic matrix makes projections stay on the correct side instead of flipping on screen
    // NOTE: after division by w, Z will be 1 instead of -1, but is discarded anyway so it does not matter
    intrinsic_matrix = mat3x4::Zero();
    intrinsic_matrix(0, 0) = focal_length_x;
    intrinsic_matrix(1, 1) = focal_length_y;
    intrinsic_matrix(0, 2) = -principal_point_x;
    intrinsic_matrix(1, 2) = -principal_point_y;
    intrinsic_matrix(2, 2) = -1.0f;
}

void parse_camera_extrinsics(json const& camera_extrinsics, mat4& view_matrix, vec3& p)
{
    auto const& json_position = camera_extrinsics["position"];
    p = {
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
    mat3 R = q.toRotationMatrix();
    double det = R.determinant();
    if (std::abs(det - 1.0) > 0.01) throw;

    view_matrix = mat4::Zero();
    view_matrix.block<3, 3>(0, 0) = R;
    view_matrix.block<3, 1>(0, 3) = -R * p; // Restore original p, which was recorded as -R.t()*p
    view_matrix(3, 3) = 1;
}

Eigen::MatrixX4d corner_pixel_distances(e_vec<mat2x4> const& projected, e_vec<mat2x4> const& y)
{
    auto distances = Eigen::MatrixX4d(projected.size(), 4);
    for (size_t i = 0; i < projected.size(); ++i)
    {
        distances.row(i) = (projected[i] - y[i]).colwise().norm();
    }
    return distances;
};

e_vec<mat4> solve_homographies(e_vec<mat3x4> const& Ps, e_vec<mat2x4> const& Ys, mat4 const& Z)
{
    auto Cs = e_vec<mat4>{};
    auto cv_Z = std::vector<cv::Point3d>{
        cv::Point3d{ Z(0, 0), Z(1, 0), Z(2, 0) },
        cv::Point3d{ Z(0, 1), Z(1, 1), Z(2, 1) },
        cv::Point3d{ Z(0, 2), Z(1, 2), Z(2, 2) },
        cv::Point3d{ Z(0, 3), Z(1, 3), Z(2, 3) },
    };
    for (size_t i = 0; i < Ys.size(); ++i)
    {
        auto cv_Y = std::vector<cv::Point2d>{
            cv::Point2d{ Ys[i](0, 0), Ys[i](1, 0) }, // bottom-left
            cv::Point2d{ Ys[i](0, 1), Ys[i](1, 1) }, // bottom-right
            cv::Point2d{ Ys[i](0, 2), Ys[i](1, 2) }, // top-right
            cv::Point2d{ Ys[i](0, 3), Ys[i](1, 3) }, // top-left
        };
        cv::Matx33d K = {
            Ps[i](0, 0), Ps[i](0, 1), -Ps[i](0, 2),
            Ps[i](1, 0), Ps[i](1, 1), -Ps[i](1, 2),
            Ps[i](2, 0), Ps[i](2, 1), -Ps[i](2, 2),
        };
        cv::Vec3d r;
        cv::Vec3d T;
        cv::solvePnP(cv_Z, cv_Y, K, cv::Vec4d{ 0, 0, 0, 0 }, r, T);
        cv::Matx33d R;
        cv::Rodrigues(r, R);
        mat4 C;
        C << R(0, 0), R(0, 1), R(0, 2), T(0),
            R(1, 0), R(1, 1), R(1, 2), T(1),
            R(2, 0), R(2, 1), R(2, 2), T(2),
            0, 0, 0, 1;
        Cs.push_back(C);
    }
    return Cs;
}

void flip_y(mat2x4& corner_points, double image_height)
{
    corner_points(1, 0) = image_height - corner_points(1, 0) - 1;
    corner_points(1, 1) = image_height - corner_points(1, 1) - 1;
    corner_points(1, 2) = image_height - corner_points(1, 2) - 1;
    corner_points(1, 3) = image_height - corner_points(1, 3) - 1;
}

e_vec<mat2x4> with_flipped_ys(e_vec<mat2x4> const& corner_points, std::vector<double> const& image_heights)
{
    auto flipped = corner_points;
    for (size_t i = 0; i < flipped.size(); ++i)
    {
        flip_y(flipped[i], image_heights[i]);
    }
    return flipped;
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
    auto settings = json{};
    settings["tag_side_length"] = 0.198;
    settings["image_minify_factor"] = 2;
    settings["show_projections"] = true;
    settings["preload_images"] = false;
    settings["cache_images"] = false;
    settings["optimizer_max_steps"] = 100;
    settings["optimizer_stop_threshold"] = 0.01;
    settings["optimizer_print_steps"] = true;

    std::istream& input = std::cin;

    auto Ps = e_vec<mat3x4>{};
    auto Vs = e_vec<mat4>{};
    auto Ts = e_vec<vec3>{};
    auto cv_Ys = e_vec<mat2x4>{}; // original y-coordinates (grow down) (opencv coordinate system)
    auto Ys = e_vec<mat2x4>{}; // flipped y-coordinates (grow up) (opengl coordinate system)
    auto frame_paths = std::vector<std::string>{};
    auto image_heights = std::vector<double>{}; // needed for flipping y-coordinates

    std::string line;
    size_t total_input_frames = 0;
    auto input_parse_time = timing([&]{
        while (std::getline(input, line))
        {
            nlohmann::json j = nlohmann::json::parse(line);

            if (j.contains("settings"))
            {
                settings.update(j["settings"]);
                continue;
            }

            ++total_input_frames;

            mat3x4 P;
            parse_camera_intrinsics(j["cameraIntrinsics"], P);
            auto const image_minify_factor = settings["image_minify_factor"].get<int>();

            // We will scale images down, so have to adjust these focal lengths and principal point as well
            P(0, 0) /= image_minify_factor;
            P(1, 1) /= image_minify_factor;
            P(0, 2) /= image_minify_factor;
            P(1, 2) /= image_minify_factor;
            mat4 V;
            vec3 t;

            parse_camera_extrinsics(j["cameraExtrinsics"], V, t);

            auto corners = std::vector<tag_corners>{};
            if (j.contains("markers"))
            {
                corners = j["markers"].get<std::vector<tag_corners>>();
            }
            else
            {
                auto temp_image = cv::imread(j["framePath"]);
                auto image = cv::Mat{};
                cv::resize(temp_image, image, temp_image.size() / image_minify_factor);
                corners = detect_markers(image);
            }

            // For now, only consider frames where exactly only one Apriltag was detected
            if (corners.size() == 1)
            {
                Vs.push_back(V);
                Ps.push_back(P);
                Ts.push_back(t);
                auto const& d = corners[0];
                mat2x4 cv_Y;
                cv_Y << d[0][0], d[1][0], d[2][0], d[3][0],
                    d[0][1], d[1][1], d[2][1], d[3][1];
                cv_Ys.push_back(cv_Y);

                // Flip y-coordinates
                mat2x4 flipped_Y = cv_Y;
                double original_image_height = j.contains("frameHeight")
                    ? j["frameHeight"].get<double>()
                    : cv::imread(j["framePath"].get<std::string>()).size().height;
                double image_height = original_image_height / image_minify_factor;
                image_heights.push_back(image_height);
                flip_y(flipped_Y, image_height);
                Ys.push_back(flipped_Y);

                frame_paths.push_back(j["framePath"].get<std::string>());
            }
        }
    });
    std::printf("Parsed input in %.2fs\n", input_parse_time);

    double s = settings["tag_side_length"].get<double>();
    mat4 Z;
    Z.col(0) = vec4{ -s/2, -s/2, 0, 1, }; // bottom-left
    Z.col(1) = vec4{ s/2, -s/2, 0, 1, }; // bottom-right
    Z.col(2) = vec4{ s/2, s/2, 0, 1, }; // top-right
    Z.col(3) = vec4{ -s/2, s/2, 0, 1, }; // top-left

    e_vec<mat3x4> PVs;
    for (size_t i = 0; i < Vs.size(); ++i)
    {
        PVs.push_back(Ps[i] * Vs[i]);
    }

    // Initial guess M0 that puts corners on the correct side of camera (camera looks at Z- in view space)
    // mat4 M0 = mat4::Identity();
    // M0(2, 3) = -1;
    // M0 = (Vs[0].inverse() * M0).eval();

    // Solve homography from marker corners from first image to initialize M0
    // Note: OpenGL and OpenCV have different coordinate systems, hence cv_to_ogl
    auto Cs = solve_homographies(Ps, cv_Ys, Z);
    mat4 cv_to_ogl = vec4{ 1, -1, -1, 1 }.asDiagonal();
    mat4 C0 = cv_to_ogl * Cs[0];
    mat4 M0 = (Vs[0].inverse() * C0);

    mat4 optimized_M;
    try
    {
        auto optimization_time = timing([&]{
            optimized_M = optimize_pose(PVs, Ys, Z, M0,
                settings["optimizer_max_steps"],
                settings["optimizer_stop_threshold"],
                !settings["optimizer_print_steps"]);
        });
        std::printf("Optimized in %.2fs\n", optimization_time);
    }
    catch (...)
    {
        std::cerr << "Failed to optimize" << std::endl;
        return 1;
    }

    auto const M_points = project_corners(PVs, optimized_M, Z);
    auto corner_ds = corner_pixel_distances(M_points, Ys);
    auto avg_ds = corner_ds.rowwise().mean();

    auto image_loader = make_image_loader(settings["cache_images"], settings["image_minify_factor"], frame_paths);

    if (settings["preload_images"])
    {
        for (size_t i = 0; i < frame_paths.size(); ++i)
        {
            image_loader((int)i);
        }
    }

    auto scaled_frame_with_info = [&](int i)
    {
        auto image = image_loader(i);
        auto label = std::stringstream();
        label.precision(3);
        label << "Average corner distance: " << avg_ds(i) << "px\n";
        label << "Frame: " << frame_paths[i] << "\n";
        put_text_lines(image, label, 100);
        return image;
    };

    if (settings["show_projections"].get<bool>())
    {
        auto cv_M_points = with_flipped_ys(M_points, image_heights);
        visualize_projections(scaled_frame_with_info, Vs.size(), cv_Ys, cv_M_points);
    }

    // Final score
    auto mse = calculate_mse(M_points, Ys);
    std::printf("Total input frames: %zu\n", total_input_frames);
    std::printf("Frames considered (single marker detected): %zu\n", Vs.size());
    std::printf("E(M) = %.2f (Error for optimized M)\n", mse);
    std::printf("E(M) per frame = %.2f\n", mse / frame_paths.size());

}