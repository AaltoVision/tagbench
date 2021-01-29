#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#ifndef NOMINMAX
#define NOMINMAX
#endif

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
using mat3 = Eigen::Matrix3d;
using mat4 = Eigen::Matrix4d;
using vec3 = Eigen::Vector3d;
using vec4 = Eigen::Vector4d;

// Fixed-size Eigen types' allocation must be aligned
template<typename T>
using e_vec = std::vector<T, Eigen::aligned_allocator<T>>;

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

void parse_camera_extrinsics(json const& camera_extrinsics, Eigen::Matrix4d& view_matrix, Eigen::Vector3d& p)
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
    Eigen::Matrix3d R = q.toRotationMatrix();
    double det = R.determinant();
    if (std::abs(det - 1.0) > 0.01) throw;

    view_matrix = Eigen::Matrix4d::Zero();
    view_matrix.block<3, 3>(0, 0) = R;
    view_matrix.block<3, 1>(0, 3) = -R * p; // Restore original p, which was recorded as -R.t()*p
    view_matrix(3, 3) = 1;
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

void visualize_projections(std::function<cv::Mat(int)> get_image, size_t image_count,
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
            Ps[i](0, 0), Ps[i](0, 1), Ps[i](0, 2),
            Ps[i](1, 0), Ps[i](1, 1), Ps[i](1, 2),
            Ps[i](2, 0), Ps[i](2, 1), Ps[i](2, 2),
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

// Create synthetic dataset of extrinsic (Vs) matrices, and groundtruth marker corner projections (Ys),
// as well as the correct M expected from optimization.
// Input is Z, which contains marker corners in object space as its columns, as well as P.
// Working optimizer should be able to find an (at least locally) optimal pose from this data, while recorded camera/VIO data may
// have drift between matrices and images (same image may be reported with different V later).
void create_synthetic_dataset(mat4 const& Z,
                              mat3x4 const& P,
                              e_vec<mat4>& Vs,
                              e_vec<mat2x4>& Ys,
                              mat4 const& M)
{
    auto const look_z_minus = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
    auto const angles = Eigen::VectorXd(Eigen::ArrayXd::LinSpaced(32, -M_PI/6, M_PI/6));

    // Rotate 30 degrees right-left around Y axis
    // for (size_t i = 0; i < 32; ++i)
    for (auto angle : angles)
    {
        Eigen::Vector3d t = Eigen::Vector3d{ 0, 0, 4 };
        Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix() * look_z_minus;
        Vs.push_back(make_view_matrix(R, t));
        // Just a sanity check; check origin is not behind camera (same as V(2,3)<0)
        if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
    }
    // // Rotate 30 degrees up-down
    // for (size_t i = 0; i < 32; ++i)
    // {
    //     Eigen::Vector3d t = Eigen::Vector3d{ 0, 0, 4 };
    //     Eigen::Matrix3d R = Eigen::AngleAxisd(angles[i], Eigen::Vector3d::UnitX()).toRotationMatrix() * look_z_minus;
    //     Vs.push_back(make_view_matrix(R, t));
    //     if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
    // }
    // // Rotate -30 to 30 degrees around Z- (rolling left ro right)
    // for (size_t i = 0; i < 32; ++i)
    // {
    //     Eigen::Vector3d t = Eigen::Vector3d{ 0, 0, 4 };
    //     Eigen::Matrix3d R = Eigen::AngleAxisd(angles[i], Eigen::Vector3d::UnitZ()).toRotationMatrix() * look_z_minus;
    //     Vs.push_back(make_view_matrix(R, t));
    //     if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
    // }

    // Translate in a circle around Z axis, looking towards Z-
    {
        double d_angle = 2.0 * M_PI / 32;
        double t_magnitude = 0.5; // TODO: relate to pixels (through s and resolution)
        for (size_t i = 0; i < 32; ++i)
        {
            Eigen::Vector3d t = Eigen::AngleAxisd(i * d_angle, -Eigen::Vector3d::UnitZ()) * Eigen::Vector3d::UnitX() * t_magnitude;
            // t(2) = -4;
            t(2) = 4;
            // Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            // R(2, 2) = -1; // Look at Z-
            Eigen::Matrix3d R = look_z_minus.transpose();
            Vs.push_back(make_view_matrix(R, t));
        }
        if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
    }

    // Zoom out
    for (size_t i = 0; i < 32; ++i)
    {
        // Looking at Z-
        Eigen::Matrix3d R = Eigen::Vector3d{ 1, 1, -1 }.asDiagonal();
        // Moving towards Z+
        Eigen::Vector3d t = Eigen::Vector3d::UnitZ() * ((int)i + 1);
        Vs.push_back(make_view_matrix(R, t));
        if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
    }

    // Zoom out, looking left
    for (size_t i = 0; i < 32; ++i)
    {
        // Looking at 30' left from Z-
        Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 6, Eigen::Vector3d::UnitY()) * look_z_minus;
        // Moving towards Z+
        Eigen::Vector3d t = Eigen::Vector3d::UnitZ() * ((int)i + 1);
        Vs.push_back(make_view_matrix(R, t));
        if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
    }

    // Rotate 30 degrees left-right around Y axis, while moving in circle around Z axis
    {
        double d_angle = 2.0 * M_PI / 32;
        double t_magnitude = 0.2; // TODO: relate to pixels (through s and resolution)
        Eigen::VectorXd angles = Eigen::ArrayXd::LinSpaced(32, -M_PI / 12, M_PI / 12);
        for (size_t i = 0; i < 32; ++i)
        {
            Eigen::Vector3d t = Eigen::AngleAxisd(i * d_angle, -Eigen::Vector3d::UnitZ()) * Eigen::Vector3d::UnitX() * t_magnitude;
            t(2) = -(((int)i) * 1.2 + 1);
            Eigen::Matrix3d R = Eigen::AngleAxisd(angles[i], Eigen::Vector3d::UnitY()).toRotationMatrix();
            Vs.push_back(make_view_matrix(R, t));
            if ((Vs.back() * Eigen::Vector3d::Zero().homogeneous()).z() < 0) throw;
        }
    }

    // Create groundtruth projections
    auto PVs = e_vec<mat3x4>(Vs.size());
    std::transform(Vs.begin(), Vs.end(),
                    PVs.begin(), [&](auto const& V) -> mat3x4 { return P * V; });
    Ys = project_corners(PVs, M, Z);
}

void test_synthetic_case(mat3x4 const& P, mat4 const& Z, bool show_visualization)
{
    e_vec<mat4> Vs;
    e_vec<mat2x4> Ys;

    // quad at y=1, with 45' rotation around Y-axis
    mat4 synthetic_M = mat4::Identity();
    synthetic_M.col(0) = Eigen::Vector4d{ 1, 0, -1, 0 }.normalized();
    synthetic_M.col(2) = Eigen::Vector4d{ 1, 0, 1, 0 }.normalized();
    synthetic_M.col(3) = Eigen::Vector4d{ 0, 1, 0, 1 };
    create_synthetic_dataset(Z, P, Vs, Ys, synthetic_M);

    auto PVs = e_vec<mat3x4>(Vs.size());
    std::transform(Vs.begin(), Vs.end(),
                    PVs.begin(), [&](auto const& V) -> mat3x4 { return P * V; });

    // Add noise in screen space x,y
    auto noisy_Ys = Ys;
    for (auto& y : noisy_Ys)
    {
        y += mat2x4::Random() * 2;
    }

    mat4 M0 = mat4::Identity();
    mat4 M = optimize_pose(PVs, noisy_Ys, Z, M0);
    // mat4 M = optimize_pose(PVs, Ys, Z, M0);
    std::stringstream label;
    label.precision(3);
    label << "Optimized M for synthetic case:\n" << M << std::endl;
    label << "Diff from synthetic_M\n" << (M - synthetic_M).cwiseAbs().format(Eigen::StreamPrecision) << std::endl;
    throw_if_nan_or_inf(M);
    auto projections = project_corners(PVs, M, Z);
    auto images = std::vector<cv::Mat>(PVs.size());
    for (auto& image : images) {
        image = cv::Mat(cv::Size2d{ 2*P(0, 2), 2*P(1, 2) }, CV_8UC3, cv::Scalar{ 200, 200, 200 });
        auto l = std::stringstream{ label.str() };
        put_text_lines(image, l, 100);
    }

    std::for_each(PVs.begin(), PVs.end(), throw_if_nan_or_inf);
    std::for_each(Ys.begin(), Ys.end(), throw_if_nan_or_inf);
    std::for_each(projections.begin(), projections.end(), throw_if_nan_or_inf);

    if (show_visualization)
    {
        visualize_projections([&](int i) { return images[i]; }, images.size(), noisy_Ys, projections);
        // visualize_projections([&](int i) { return images[i]; }, images.size(), Ys, projections);
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
    auto settings = json{};
    settings["synthetic_optimize"] = false;
    settings["synthetic_vis"] = false;
    settings["real_optimize"] = true;
    settings["real_vis"] = true;
    settings["tag_side_length"] = 0.198;

    std::istream& input = std::cin;

    auto Ps = e_vec<mat3x4>{};
    auto Vs = e_vec<mat4>{};
    auto Ts = e_vec<Eigen::Vector3d>{};
    auto Ys = e_vec<mat2x4>{};
    auto frame_paths = std::vector<std::string>{};

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

            mat4 P;
            parse_camera_intrinsics(j["cameraIntrinsics"], P);
            // We will scale images to half size, so have to adjust these focal lengths and principal point as well
            P(0, 0) /= 2;
            P(1, 1) /= 2;
            P(0, 2) /= 2;
            P(1, 2) /= 2;
            mat4 V;
            Eigen::Vector3d t;
            parse_camera_extrinsics(j["cameraExtrinsics"], V, t);

            using tag_corners = std::array<std::array<float, 2>, 4>;
            auto corners = std::vector<tag_corners>{};
            if (j.contains("markers"))
            {
                corners = j["markers"].get<std::vector<tag_corners>>();
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
                    corners.emplace_back();
                    memcpy(&corners.back(), d.p, sizeof(d.p));
                }
            }

            // For now, only consider frames where exactly only one Apriltag was detected
            if (corners.size() == 1)
            {
                Vs.push_back(V);
                Ps.push_back(P.block<3, 4>(0, 0).cast<double>());
                Ts.push_back(t);
                auto const& d = corners[0];
                mat2x4 Y;
                Y << d[0][0], d[1][0], d[2][0], d[3][0],
                    d[0][1], d[1][1], d[2][1], d[3][1];
                Ys.push_back(Y);
                frame_paths.push_back(j["framePath"].get<std::string>());
            }
        }
    });
    std::printf("Parsed input in %.2fs\n", input_parse_time);

    // TODO: [0, s] or [-s/2, s/2] ?
    // Probably does not affect solution, as long as we are consistent
    // Tag on the screen is 19.8cm (in arcore-7-1-single-2 data, where tag is shown on screen)
    double s = settings["tag_side_length"].get<double>();
    mat4 Z;
    Z.col(0) = Eigen::Vector4d{ -s/2, -s/2, 0, 1, }; // bottom-left
    Z.col(1) = Eigen::Vector4d{ s/2, -s/2, 0, 1, }; // bottom-right
    Z.col(2) = Eigen::Vector4d{ s/2, s/2, 0, 1, }; // top-right
    Z.col(3) = Eigen::Vector4d{ -s/2, s/2, 0, 1, }; // top-left

    // Test fitting synthetic data
    if (settings["synthetic_optimize"].get<bool>())
    {
        test_synthetic_case(Ps[0], Z, settings["synthetic_vis"].get<bool>());
    }

    if (!settings["real_optimize"]) return 0;

    // // Turn V0 into identity, and other Vs relative to that
    // for (size_t i = 0; i < Vs.size(); ++i)
    // {
    //     mat4 inv = Vs[0].inverse();
    //     mat4 V = Vs[i];
    //     Vs[i] = inv * V;
    // }

    auto Cs = solve_homographies(Ps, Ys, Z);

    // // Debugging: use Cs instead of Vs, and add a random transformation, which optimizer should find
    // Vs = Cs;
    // mat4 hidden_M = mat4::Identity();
    // vec3 T = vec3::Random() * 2.;
    // mat3 R = Eigen::AngleAxisd(M_PI/6-0.1, vec3::UnitY()).toRotationMatrix();
    // hidden_M.block<3, 3>(0, 0) = R;
    // hidden_M.col(3).head(3) = R*T;
    // for (size_t i = 0; i < Vs.size(); ++i)
    // {
    //     Vs[i] = (Vs[i] * hidden_M).eval();
    // }

    e_vec<mat3x4> PVs;
    for (size_t i = 0; i < Vs.size(); ++i)
    {
        PVs.push_back(Ps[i] * Vs[i]);
    }

    // mat4 M0 = mat4::Identity();
    mat4 M0 = Vs[0].inverse() * Cs[0];

    mat4 optimized_M;
    auto optimization_time = timing([&]{
        optimized_M = optimize_pose(PVs, Ys, Z, M0);
    });

    if (optimized_M.block<3, 3>(0, 0).determinant() < 0.99) throw;
    std::printf("Total optimization time: %.2fs\n", optimization_time);

    auto optimized_M_projected_points = project_corners(PVs, optimized_M, Z);
    auto corner_ds = corner_pixel_distances(optimized_M_projected_points, Ys);
    auto avg_ds = corner_ds.rowwise().mean();

    auto get_scaled_frame = [&](int i) {
        auto image = cv::Mat{};
        auto temp_image = cv::imread(frame_paths[i]);
        cv::resize(temp_image, image, temp_image.size() / 2);
        return image;
    };

    auto scaled_frame_with_distances = [&](int i)
    {
        auto image = get_scaled_frame(i);
        auto label = std::stringstream();
        label.precision(3);
        label << "Average corner distance: " << avg_ds(i) << "\n";
        label << "P[i]:\n" << Ps[i] << "\n";
        label << "Image size: " << image.size() << "\n\n";
        label << "Optimized M:\n" << optimized_M << std::endl;
        put_text_lines(image, label, 100);
        return image;
    };

    if (settings["real_vis"].get<bool>())
    {
        visualize_projections(scaled_frame_with_distances, Vs.size(), Ys, optimized_M_projected_points);
    }

    // Final score
    auto mse = calculate_mse(optimized_M_projected_points, Ys);
    std::printf("Total input frames: %zu\n", total_input_frames);
    std::printf("Frames considered (single marker detected): %zu\n", Vs.size());
    std::printf("E(M) = %.2f (Error for optimized M)\n", mse);

}
