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
#include <chrono>
#include <array>
#include <algorithm>

using json = nlohmann::json;

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

    // TODO: maybe keep the original p and q handy as well

    view_matrix = Eigen::Matrix4f::Zero();
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
                           std::vector<std::array<cv::Point2f, 4>> const& detections,
                           std::vector<std::array<cv::Point2f, 4>> const& projections,
                           std::vector<cv::Vec3f> const& Ts,
                        //    std::vector<cv::Vec3f> const& VIO_Ts
                           std::vector<cv::Matx33f> const& Rs
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
        label << std::setprecision(3);
        label << "Image: " << i << "\n";
        label << "T: " << 10000* Ts[i] << "\n";
        // label << "R: " << Rs[i] << "\n";

cv::Ptr<cv::Formatter> formatMat = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);
formatMat->set64fPrecision(3);
formatMat->set32fPrecision(3);
label << "R:\n" << formatMat->format( cv::Mat(Rs[i]) ) << std::endl;

        // label << "VIO_T: " << 10000* VIO_Ts[i];
        put_text_lines(image_with_projections, label);
        return image_with_projections;
    };
    view_images(get_image, images.size().width);
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

    // Tag on the screen is 19.8cm (in arcore-7-1-single-2 data, where tag is shown on screen)
    auto const s = 0.198f;
    std::vector<cv::Point3f> Z = {
        { -s/2, -s/2, 0 }, // bottom-left
        { s/2, -s/2, 0 }, // bottom-right
        { s/2, s/2, 0 }, // top-right
        { -s/2, s/2, 0 }, // top-left
    };
    std::vector<cv::Vec4f> Z4 = {
        { -s/2, -s/2, 0, 1, }, // bottom-left
        { s/2, -s/2, 0, 1, }, // bottom-right
        { s/2, s/2, 0, 1, }, // top-right
        { -s/2, s/2, 0, 1, }, // top-left
    };

    auto Ys = std::vector<std::vector<cv::Point2f>>{};
    auto Ks = std::vector<cv::Matx33f>{};
    auto Vs = std::vector<cv::Matx44f>{};
    auto Rs = std::vector<cv::Matx33f>{};
    auto Ts = std::vector<cv::Vec3f>{};
    for (auto& f : frames)
    {
        auto const& d = f.detections[0].p;

        // TODO: check order of corners vs. Z
        Ys.push_back({
            cv::Point2f{ d[0].x, d[0].y },
            cv::Point2f{ d[1].x, d[1].y },
            cv::Point2f{ d[2].x, d[2].y },
            cv::Point2f{ d[3].x, d[3].y },
        });

        Ks.push_back({
            f.intrinsic_matrix(0, 0), f.intrinsic_matrix(0, 1), f.intrinsic_matrix(0, 2),
            f.intrinsic_matrix(1, 0), f.intrinsic_matrix(1, 1), f.intrinsic_matrix(1, 2),
            f.intrinsic_matrix(2, 0), f.intrinsic_matrix(2, 1), f.intrinsic_matrix(2, 2),
        });

        Vs.push_back({
            f.view_matrix(0, 0), f.view_matrix(0, 1), f.view_matrix(0, 2), f.view_matrix(0, 3),
            f.view_matrix(1, 0), f.view_matrix(1, 1), f.view_matrix(1, 2), f.view_matrix(0, 3),
            f.view_matrix(2, 0), f.view_matrix(2, 1), f.view_matrix(2, 2), f.view_matrix(0, 3),
            f.view_matrix(3, 0), f.view_matrix(3, 1), f.view_matrix(3, 2), f.view_matrix(3, 3),
        });

        cv::Vec3f r;
        auto& T = Ts.emplace_back();
        cv::solvePnP(Z, Ys.back(), Ks.back(), cv::Vec4f{ 0, 0, 0, 0 }, r, T);

        std::cout << "Z:\n" << Z << "\n";
        std::cout << "Y:\n" << Ys.back() << "\n";
        std::cout << "K:\n" << Ks.back() << "\n";
        std::cout << "V:\n" << Vs.back() << "\n";

        auto& R = Rs.emplace_back();
        cv::Rodrigues(r, R);
    }

    // auto error_cv = [&](cv::Matx44f const &P, cv::Matx44f const &V, cv::Matx44f const &M,
    //                  cv::Vec4f const &z, cv::Vec4f const &y) {
    //     cv::Vec4f PVMz = P*V*M*z;
    //     std::cout << "PVMz:\n" << PVMz << "\n";
    //     PVMz[0] /= PVMz[3];
    //     PVMz[1] /= PVMz[3];
    //     return ((PVMz - y).t()*(PVMz - y))[0];
    // };

    auto images = std::vector<cv::Mat>{};
    auto detected_points = std::vector<std::array<cv::Point2f, 4>>{};
    auto projected_points = std::vector<std::array<cv::Point2f, 4>>{};
    for (auto i = 0u; i < frames.size(); ++i)
    {
        auto const& frame = frames[i];
        auto temp_image = cv::imread(frame.frame_path);
        auto& image = images.emplace_back();
        cv::resize(temp_image, image, temp_image.size() / 2);
        detected_points.push_back({
            cv::Point2f{frame.detections[0].p[0].x, frame.detections[0].p[0].y},
            cv::Point2f{frame.detections[0].p[1].x, frame.detections[0].p[1].y},
            cv::Point2f{frame.detections[0].p[2].x, frame.detections[0].p[2].y},
            cv::Point2f{frame.detections[0].p[3].x, frame.detections[0].p[3].y},
        });

        auto const& K = Ks[i];
        cv::Matx44f K44 = {
            K(0, 0), K(0, 1), K(0, 2), 0,
            K(1, 0), K(1, 1), K(1, 2), 0,
            K(2, 0), K(2, 1), K(2, 2), 0,
            0, 0, 0, 1,
        };
        auto const& R = Rs[i];
        auto const& T = Ts[i];
        cv::Matx44f M_test = {
            R(0, 0), R(0, 1), R(0, 2), T(0),
            R(1, 0), R(1, 1), R(1, 2), T(1),
            R(2, 0), R(2, 1), R(2, 2), T(2),
            0, 0, 0, 1,
        };

        auto& proj = projected_points.emplace_back();
        for (auto iz = 0; iz < 4; ++iz)
        {
            // M is (tag object space coords) -> (camera coords)
            // M.inv is (camera coords) -> (tag object space coords)

            //                           M                  K
            // (tag object space coords) -> (camera coords) -> (screen coords)

            cv::Vec4f proj_h = K44 * M_test * Z4[iz];
            // proj_h[0] /= proj_h[2];
            // proj_h[1] /= proj_h[2];
            // TODO: remove the Z element? (so Z is x,y,1, and Rt is r1 r2 (ar3+t) or something)

            proj[iz] = { proj_h[0], proj_h[1] };
        }
    }
    // visualize_projections(M_test, { K44 }, { V }, images, detected_points);
    // visualize_projections(images, detected_points, projected_points);
    visualize_projections(images, detected_points, projected_points, Ts, Rs);

    // TODO: Gauss-Newton optimization for M

}
