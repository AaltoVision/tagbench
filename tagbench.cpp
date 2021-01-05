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
#include <string>
#include <chrono>
#include <array>
#include <algorithm>

using json = nlohmann::json;

template<typename F>
void image_viewer(F get_image, int size)
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
}

static auto timing = [](auto& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e-3f;
    return dt;
};

void parse_camera_intrinsics(json const &camera_intrinsics,
                            //  Eigen::Matrix4f& projection_matrix,
                             Eigen::Matrix4f& intrinsic_matrix)
{
    auto const focal_length_x = camera_intrinsics["focalLengthX"].get<float>();
    auto const focal_length_y = camera_intrinsics["focalLengthY"].get<float>();
    auto const principal_point_x = camera_intrinsics["principalPointX"].get<float>();
    auto const principal_point_y = camera_intrinsics["principalPointY"].get<float>();

    // // OpenGL near and far clip plane distances (meters)
    // constexpr float zNear = 0.01f;
    // constexpr float zFar = 20.0f;

    // projection_matrix = Eigen::Matrix4f::Zero();
    // projection_matrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
    // projection_matrix(2, 3) = -2 * zFar * zNear / (zFar - zNear);
    // projection_matrix(3, 2) = -1;

    // rotated image: flip X & Y
    // projection_matrix(0, 1) = -2 * focalLength / width;
    // projection_matrix(1, 0) = 2 * focalLength / height;

    intrinsic_matrix = Eigen::Matrix4f::Zero();
    intrinsic_matrix(0, 0) = focal_length_x;
    intrinsic_matrix(1, 1) = focal_length_y;
    intrinsic_matrix(0, 2) = principal_point_x;
    intrinsic_matrix(1, 2) = principal_point_y;
    intrinsic_matrix(2, 2) = 1.0f;
}

void parse_camera_extrinsics(json const& camera_extrinsics, Eigen::Matrix4f& view_matrix)
{
    auto const& json_position = camera_extrinsics["position"];
    Eigen::Vector3f p = {
        json_position["x"].get<float>(),
        json_position["y"].get<float>(),
        json_position["z"].get<float>(),
    };
    auto const& json_orientation = camera_extrinsics["orientation"];
    Eigen::Quaternionf q = {
        json_orientation["w"].get<float>(),
        json_orientation["x"].get<float>(),
        json_orientation["y"].get<float>(),
        json_orientation["z"].get<float>(),
    };
    Eigen::Matrix3f R = q.toRotationMatrix();

    view_matrix = Eigen::Matrix4f::Zero();
    view_matrix.block<3, 3>(0, 0) = R;
    view_matrix.block<3, 1>(0, 3) = -R * p;
    view_matrix(3, 3) = 1;
}

void visualize_projections(cv::InputArrayOfArrays images,
                           std::vector<std::array<cv::Point2f, 4>> const& detections,
                           std::vector<std::array<cv::Point2f, 4>> const& projections,
                           cv::Vec3f& T
                           )
{
    auto get_image = [&](int i)
    {
        cv::Mat image_with_projections = images.getMat(i).clone();
        for (auto j = 0; j < 4; ++j)
        {
            cv::drawMarker(image_with_projections, detections[i][j], CV_RGB(255,0,0), 0, 20, 2);
            cv::drawMarker(image_with_projections, projections[i][j], CV_RGB(0,255,0), 1, 20, 2);
        }
        std::stringstream label;
        label << "Image: " << i << "\n";
        label << "T: " << T << "\n";
        cv::putText(image_with_projections, label.str(), { 20, 20 }, 0, 0.6, CV_RGB(0, 0, 0), 3);
        cv::putText(image_with_projections, label.str(), { 20, 20 }, 0, 0.6, CV_RGB(255, 255, 255), 2);
        printf("image %d\n", i);
        return image_with_projections;
    };
    image_viewer(get_image, images.size().width);
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
    struct frame_info
    {
        Eigen::Matrix4f intrinsic_matrix;
        Eigen::Matrix4f view_matrix;
        TagDetectionArray detections;
        std::string frame_path;
    };
    auto frames = std::vector<frame_info>{};
    while (std::getline(input, line))
    {
        nlohmann::json j = nlohmann::json::parse(line);

        // auto& f = frames.emplace_back();
        auto f = frame_info{};

        parse_camera_intrinsics(j["cameraIntrinsics"], /*f.projection_matrix,*/ f.intrinsic_matrix);
        parse_camera_extrinsics(j["cameraExtrinsics"], f.view_matrix);
        f.frame_path = j["framePath"];

        TagDetectorParams params;
        auto tag_family = TagFamily(std::string("Tag36h11"));
        TagDetector detector(tag_family, params);

        auto temp_image = cv::imread(j["framePath"]);
        auto image = cv::Mat{};
        cv::resize(temp_image, image, temp_image.size() / 2);
        detector.process(image, cv::Point2i{image.size().width, image.size().height}, f.detections);

        if (f.detections.size() > 0)
        {
            frames.push_back(f);
        }
    }

    auto error = [&](Eigen::Matrix4f const &P, Eigen::Matrix4f const &V, Eigen::Matrix4f const &M,
                     Eigen::Vector4f const &z, Eigen::Vector4f const &y) {
        Eigen::Vector4f PVMz = P*V*M*z;
        PVMz[0] /= PVMz[3];
        PVMz[1] /= PVMz[3];
        return ((PVMz - y).transpose()*(PVMz - y))[0];
    };

    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    auto z0 = Eigen::Vector4f(0, 0, 0, 1); // top-left?
    auto py = frames[0].detections[0].p[0]; // adjust range from [-.5, .5] to [0, 1]?
    auto E = error(frames[0].intrinsic_matrix, frames[0].view_matrix, M,
                    z0, Eigen::Vector4f(py.x, py.y, 0, 1));
    std::printf("E(M): %.2f\n", E);

    // Initial M

    auto const& f = frames[0];

    auto const s = 0.241f;
    std::vector<cv::Point3f> Z = {
        { -s/2, -s/2, 0 },
        { s/2, -s/2, 0 },
        { -s/2, s/2, 0 },
        { s/2, s/2, 0 },
    };

    auto const& d = f.detections[0].p;
    std::vector<cv::Point2f> Y = {
        cv::Point2f{ d[0].x, d[0].y },
        cv::Point2f{ d[1].x, d[1].y },
        cv::Point2f{ d[2].x, d[2].y },
        cv::Point2f{ d[3].x, d[3].y },
    };

    cv::Matx33f K;
    {
        auto &c0 = f.intrinsic_matrix.col(0);
        auto &c1 = f.intrinsic_matrix.col(1);
        auto &c2 = f.intrinsic_matrix.col(2);
        K = {
            c0.x(), c0.y(), c0.z(),
            c1.x(), c1.y(), c1.z(),
            c2.x(), c2.y(), c2.z(),
        };
    }

    cv::Matx44f V;
    {
        auto& c0 = f.view_matrix.col(0);
        auto& c1 = f.view_matrix.col(1);
        auto& c2 = f.view_matrix.col(2);
        auto& c3 = f.view_matrix.col(3);
        V = {
            c0.x(), c0.y(), c0.z(), c0.w(),
            c1.x(), c1.y(), c1.z(), c1.w(),
            c2.x(), c2.y(), c2.z(), c2.w(),
            c3.x(), c3.y(), c3.z(), c3.w(),
        };
    }

    cv::Vec3f r;
    cv::Vec3f T;
    cv::solvePnP(Z, Y, K, cv::Vec4f{ 0, 0, 0, 0 }, r, T);

    std::cout << "r:\n" << r << "\n";
    std::cout << "t:\n" << T << "\n";

    // cv::Mat R;
    cv::Matx33f R;
    cv::Rodrigues(r, R);
    std::cout << "R:\n" << R << "\n";

    // P == K?
    auto error_cv = [&](cv::Matx44f const &P, cv::Matx44f const &V, cv::Matx44f const &M,
                     cv::Vec4f const &z, cv::Vec4f const &y) {
        cv::Vec4f PVMz = P*V*M*z;
        std::cout << "PVMz:\n" << PVMz << "\n";
        PVMz[0] /= PVMz[3];
        PVMz[1] /= PVMz[3];
        return ((PVMz - y).t()*(PVMz - y))[0];
    };
    cv::Matx44f K44 = {
        K(0, 0), K(0, 1), K(0, 2), 0,
        K(1, 0), K(1, 1), K(1, 2), 0,
        K(2, 0), K(2, 1), K(2, 2), 0,
        0, 0, 0, 1,
    };
    cv::Matx44f M_test = {
        R(0, 0), R(0, 1), R(0, 2), T(0),
        R(1, 0), R(1, 1), R(1, 2), T(1),
        R(2, 0), R(2, 1), R(2, 2), T(2),
        0, 0, 0, 1,
    };
    cv::Vec4f z00 = { -s/2, -s/2, 0, 1 };
    cv::Vec4f y0 = { f.detections[0].p[0].x, f.detections[0].p[0].y, 0, 1 };
    auto E2 = error_cv(K44, V, M_test, z00, y0);
    printf("E2: %.2f\n", E2);

    // Gauss-Newton optimization for M

    auto images = std::vector<cv::Mat>{};
    auto detected_points = std::vector<std::array<cv::Point2f, 4>>{};
    auto projected_points = std::vector<std::array<cv::Point2f, 4>>{};
    for (auto const& frame : frames)
    {
        auto temp_image = cv::imread(frame.frame_path);
        auto& image = images.emplace_back();
        cv::resize(temp_image, image, temp_image.size() / 2);
        detected_points.push_back({
            cv::Point2f{frame.detections[0].p[0].x, frame.detections[0].p[0].y},
            cv::Point2f{frame.detections[0].p[1].x, frame.detections[0].p[1].y},
            cv::Point2f{frame.detections[0].p[2].x, frame.detections[0].p[2].y},
            cv::Point2f{frame.detections[0].p[3].x, frame.detections[0].p[3].y},
        });

        auto const s = 0.241f; // tag on the screen is 24.1cm (in arcore-31-single-tag data, where tag is shown on screen)
        std::vector<cv::Vec4f> Z4 = {
            { -s/2, -s/2, 0, 1 },
            { s/2, -s/2, 0, 1 },
            { -s/2, s/2, 0, 1 },
            { s/2, s/2, 0, 1 },
        };
        auto& proj = projected_points.emplace_back();
        for (auto iz = 0; iz < 4; ++iz)
        {
            // M is (tag object space coords) -> (camera coords)
            // M.inv is (camera coords) -> (tag object space coords)

            //                           M                  K
            // (tag object space coords) -> (camera coords) -> (screen coords)

            cv::Vec4f proj_h = K44 * M_test * Z4[iz];
            proj_h[0] /= proj_h[3];
            proj_h[1] /= proj_h[3];

            proj[iz] = { proj_h[0], proj_h[1] };
        }
    }
    // visualize_projections(M_test, { K44 }, { V }, images, detected_points);
    // visualize_projections(images, detected_points, projected_points);
    visualize_projections(images, detected_points, projected_points, T);
}
