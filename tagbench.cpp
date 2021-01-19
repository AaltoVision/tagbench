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
#include <chrono>
#include <array>
#include <algorithm>
#include <exception>

using json = nlohmann::json;

using mat2x4 = Eigen::Matrix<double, 2, 4>;
using mat3x4 = Eigen::Matrix<double, 3, 4>;
using mat4 = Eigen::Matrix4d;

static auto throw_if_nan = [](auto const& m)
{
    if (m.hasNaN())
    {
        throw std::range_error("NaN encountered");
    }
};

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


Eigen::Matrix3d quat2rmat(const Eigen::Vector4d& q) {
    Eigen::Matrix3d R;
    R <<
        q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2],
        2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1],
        2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    return R;
}

// Derivatives of the rotation matrix w.r.t. the quaternion of the quat2rmat() function.
Eigen::Matrix3d quat2rmat_d(const Eigen::Vector4d& q, Eigen::Matrix3d(&dR)[4]) {
    dR[0] <<
        2*q(0), -2*q(3),  2*q(2),
        2*q(3),  2*q(0), -2*q(1),
        -2*q(2),  2*q(1),  2*q(0);
    dR[1] <<
        2*q(1),  2*q(2),  2*q(3),
        2*q(2), -2*q(1), -2*q(0),
        2*q(3),  2*q(0), -2*q(1);
    dR[2] <<
        -2*q(2),  2*q(1),  2*q(0),
        2*q(1),  2*q(2),  2*q(3),
        -2*q(0),  2*q(3), -2*q(2);
    dR[3] <<
        -2*q(3), -2*q(0),  2*q(1),
        2*q(0), -2*q(3),  2*q(2),
        2*q(1),  2*q(2),  2*q(3);
    return quat2rmat(q);
}

Eigen::Matrix4d make_pose_matrix(Eigen::Matrix3d const &R, Eigen::Vector3d const &t) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Zero();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    return pose;
};

// Create synthetic dataset of extrinsic (Vs) matrices, and groundtruth marker corner projections (Ys),
// as well as the correct M expected from optimization.
// Input is Z, which contains marker corners in object space as its columns, as well as P.
// Working optimizer should be able to find an (at least locally) optimal pose from this data, while recorded camera/VIO data may
// have drift between matrices and images (same image may be reported with different V later).
void create_synthetic_dataset(mat4 const& Z,
                              mat3x4 const& P,
                              std::vector<mat4>& Vs,
                              std::vector<mat2x4>& Ys,
                              mat4& expected_M)
{
    mat4 M = mat4::Identity();
    M.col(0) *= -1;
    M.col(3) = 100*Eigen::Vector4d{ 0.5, -1, -2, 1 };

    // Rotate 45 degrees left-right around Y axis
    auto angles = Eigen::ArrayXd::LinSpaced(100, -M_PI/4, M_PI/4);
    for (size_t i = 0; i < 100; ++i)
    {
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R = Eigen::AngleAxisd(angles[i], Eigen::Vector3d::UnitY()).toRotationMatrix();
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

    // Create the correct projections
    for (size_t i = 0; i < Vs.size(); ++i)
    {
        Ys.emplace_back();
        for (size_t k = 0; k < 4; ++k)
        {
            Eigen::Vector3d p = P * Vs[i] * M * Z.col(k);
            Ys.back().col(k) = Eigen::Vector2d{ p(0)/p(2), p(1)/p(2) };
        }
    }

    expected_M = M;
}

Eigen::Vector<double, 7> optimize_step(
    std::vector<mat3x4> const& PVs,
    std::vector<mat2x4> const& Ys,
    Eigen::Matrix4d const& Z,
    Eigen::Vector3d const& t,
    Eigen::Vector4d const& q)
{
    Eigen::Matrix4d M = make_pose_matrix(quat2rmat(q), t);

    // Accumulate A and b over all frames and tag corners
    Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Vector<double, 7> b = Eigen::Vector<double, 7>::Zero();
    for (size_t j = 0; j < PVs.size(); ++j)
    {
        Eigen::Matrix<double, 3, 4> PV = PVs[j];

        // TODO: simplify, recompute stuff much less
        for (size_t k = 0; k < 4; ++k)
        {
            // Compute translation and orientation derivatives
            Eigen::Matrix4d dMdXi[7];
            // t1, t2, t3
            for (size_t i = 0; i < 3; ++i)
            {
                dMdXi[i] = Eigen::Matrix4d::Zero();
                dMdXi[i](i, 3) = 1;
            }
            // q1, q2, q3, q4
            Eigen::Matrix3d dRdq[4];
            quat2rmat_d(q, dRdq);
            for (size_t i = 0; i < 4; ++i)
            {
                dMdXi[3 + i] = Eigen::Matrix4d::Zero();
                dMdXi[3 + i].block<3, 3>(0, 0) = dRdq[i];
            }

            // Projection of z_k with current M
            Eigen::Vector3d xyw = PV * M * Z.col(k);

            // Jg
            double w2 = xyw(2) * xyw(2);
            Eigen::Matrix<double, 2, 3> Jg = Eigen::Matrix<double, 2, 3>{
                {1.0f / xyw(2), 0, -xyw(0) / w2},
                {0, 1.0f / xyw(2), -xyw(1) / w2},
            };

            Eigen::Matrix<double, 2, 4> Jg_P_V = Jg * PV;
            Eigen::Matrix<double, 2, 7> J_jk;
            for (size_t i = 0; i < 7; ++i)
            {
                J_jk.col(i) = Jg_P_V * dMdXi[i] * Z.col(k);
            }

            A += J_jk.transpose() * J_jk;
            Eigen::Vector2d xy = { xyw(0) / xyw(2), xyw(1) / xyw(2) };
            Eigen::Vector2d xy_detected = Eigen::Vector2d{ (double)Ys[j](0, k), (double)Ys[j](1, k) };
            Eigen::Vector2d residual = xy - xy_detected;
            b -= J_jk.transpose() * residual;

            throw_if_nan(A);
            throw_if_nan(b);
        }
    }

    Eigen::Vector<double, 7> dx = A.colPivHouseholderQr().solve(b);
    throw_if_nan(dx);
    return dx;
};

double calculate_mse(std::vector<mat2x4> const& p, std::vector<mat2x4> const& y)
{
    auto mse = 0.0;
    for (size_t j = 0; j < p.size(); ++j)
    {
        mat2x4 residuals = p[j] - y[j];
        auto r2 = (residuals.transpose() * residuals).diagonal();
        auto frame_mse = r2.sum();
        mse += frame_mse;
    }
    return mse;
}

std::vector<mat2x4> project_corners(std::vector<mat3x4> const& PVs, mat4 const& M, mat4 const& Z)
{
    auto projected = std::vector<mat2x4>{};
    for (size_t i = 0; i < PVs.size(); ++i)
    {
        projected.push_back({});
        auto& proj = projected.back();
        for (auto iz = 0; iz < 4; ++iz)
        {
            Eigen::Vector3d proj_h = PVs[i] * M * Z.col(iz);
            proj.col(iz) = Eigen::Vector2d{
                proj_h(0) / proj_h(2),
                proj_h(1) / proj_h(2),
            };
        }
    }
    return projected;
}

Eigen::Matrix4d optimize_pose(
    std::vector<mat3x4> const& PVs,
    std::vector<mat2x4> const& Ys,
    Eigen::Matrix4d const& Z,
    Eigen::Matrix4d const& M0
    )
{
    Eigen::Matrix4d M = M0;
    Eigen::Vector3d t = M.block<3, 1>(0, 3);
    Eigen::Quaterniond qq(Eigen::AngleAxisd(M.block<3, 3>(0, 0)));

    // TODO: check if quat2rmat and quat2rmat_d expect q to be wxyz or xyzw
    Eigen::Vector4d q = { qq.x(), qq.y(), qq.z(), qq.w(), };

    // TODO: actual threshold etc.
    for (size_t step = 0; step < 100; ++step)
    {
        Eigen::Vector<double, 7> dx;
        auto step_time = timing([&]{ dx = optimize_step(PVs, Ys, Z, t, q); });
        t += dx.block<3, 1>(0, 0);
        q += dx.block<4, 1>(3, 0);
        qq = Eigen::Quaterniond{ q.x(), q.y(), q.z(), q.w() };
        qq.normalize();
        q = { qq.x(), qq.y(), qq.z(), qq.w(), };

        M = make_pose_matrix(quat2rmat(q), t);
        std::cout << "Step " << step << ": |dx| = " << dx.norm();
        // std::printf("\t\tE(M) = %.2f", calculate_mse(project_corners(PVs, M, Z), Ys));
        std::printf("\t\tE(M) = %.6e", calculate_mse(project_corners(PVs, M, Z), Ys));
        std::printf("\t\t(step time: %.2fs)", step_time);
        std::cout << std::endl;
    }
    // M = make_pose_matrix(quat2rmat(q), t);
    return M;
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

    // Test fitting synthetic data
    {
        std::vector<mat4> Vs;
        std::vector<mat2x4> Ys;
        mat4 synthetic_M;
        mat3x4 P = frames[0].intrinsic_matrix.block<3, 4>(0, 0).cast<double>();
        create_synthetic_dataset(Z, P, Vs, Ys, synthetic_M);

        auto PVs = std::vector<mat3x4>(Vs.size());
        std::transform(Vs.begin(), Vs.end(),
                       PVs.begin(), [&](auto const& V) -> mat3x4 { return P * V; });
        
        mat4 M0 = mat4::Identity();
        mat4 M = optimize_pose(PVs, Ys, Z, M0); 
        if (M.hasNaN())
        {
            std::cout << "Pose estimation failed, solution is NaN" << std::endl;
            return 1;
        }
        auto projections = project_corners(PVs, M, Z);
        auto images = std::vector<cv::Mat>(PVs.size());
        for (auto& image : images) {
            image = cv::Mat(cv::Size2d{ P(0, 2), P(1, 2) }, CV_8UC3, cv::Scalar(255, 255, 255));
        }
        for (auto& p : projections)
        {
            throw_if_nan(p);
        }
        // visualize_projections(images, Ys, projections);
        // return 0;
    }

    // Prepare some of the data into easier form...
    auto images = std::vector<cv::Mat>{};
    auto Cs = std::vector<cv::Matx44f>{};
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
        Cs.push_back({
            R(0, 0), R(0, 1), R(0, 2), T(0),
            R(1, 0), R(1, 1), R(1, 2), T(1),
            R(2, 0), R(2, 1), R(2, 2), T(2),
            0, 0, 0, 1,
        });
    }

    auto setup_end = std::chrono::steady_clock::now();
    auto setup_dt = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start).count() * 1e-3f;

    std::printf("Setup done in %.2fs\n", setup_dt);
    std::vector<mat3x4> PVs;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        PVs.push_back(Ps[i] * frames[i].view_matrix);
    }

    cv::Matx44f cv_M0 = Vs[0].inv() * Cs[0];
    Eigen::Matrix4d M0;
    M0 <<
        cv_M0(0, 0), cv_M0(0, 1), cv_M0(0, 2), cv_M0(0, 3),
        cv_M0(1, 0), cv_M0(1, 1), cv_M0(1, 2), cv_M0(1, 3),
        cv_M0(2, 0), cv_M0(2, 1), cv_M0(2, 2), cv_M0(2, 3),
        cv_M0(3, 0), cv_M0(3, 1), cv_M0(3, 2), cv_M0(3, 3);

    mat4 optimized_M = mat4::Identity();

    auto optimization_time = timing([&]{
        optimized_M = optimize_pose(PVs, Ys, Z, M0);
    });
    std::printf("Total optimization time: %.2fs\n", optimization_time);

    auto optimized_M_projected_points = project_corners(PVs, optimized_M, Z);
    // TODO: consider get_image kind of thing here, because loading 400 images (and resizing) takes 7.5s in release build...
    // visualize_projections(images, Ys, optimized_M_projected_points);

    // Final score
    auto mse = 0.0;
    auto average_pixel_distance = 0.0;
    for (size_t j = 0; j < frames.size(); ++j)
    {
        mat2x4 residuals = optimized_M_projected_points[j] - Ys[j];
        auto r2 = (residuals.transpose() * residuals).diagonal();
        average_pixel_distance += r2.cwiseSqrt().sum();
        auto frame_mse = r2.sum();
        mse += frame_mse;
        // std::printf("e_%zu = %.2f\n", j, frame_mse);
    }
    average_pixel_distance /= (frames.size() * 4);
    std::printf("Total input frames: %zu\n", total_input_frames);
    std::printf("Frames considered (single marker detected): %zu\n", frames.size());
    std::printf("E(M) = %.2f (Error for optimized M)\n", mse);
    std::printf("E(M)/n_corners = %.2f\n", mse / (frames.size() * 4));
    std::printf("Average pixel distance = %.2f\n", average_pixel_distance);

}
