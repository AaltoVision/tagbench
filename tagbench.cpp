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

        put_text_lines(image_with_projections, label);
        return image_with_projections;
    };
    view_images(get_image, images.size().width);
}


// Eigen::Matrix3d quat2rmat(const Eigen::Vector4d& q) {
//     Eigen::Matrix3d R;
Eigen::Matrix3f quat2rmat(const Eigen::Vector4f& q) {
    Eigen::Matrix3f R;
    R <<
        q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2],
        2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1],
        2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    return R;
}

// Derivatives of the rotation matrix w.r.t. the quaternion of the quat2rmat() function.
// Eigen::Matrix3d quat2rmat_d(const Eigen::Vector4d& q, Eigen::Matrix3d(&dR)[4]) {
Eigen::Matrix3f quat2rmat_d(const Eigen::Vector4f& q, Eigen::Matrix3f(&dR)[4]) {
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

Eigen::Matrix4f make_pose_matrix(Eigen::Matrix3f const &R, Eigen::Vector3f const &t) {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Zero();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    return pose;
};

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
        // We will scale images to half size, so have to adjust these focal lengths and principal point as well
        f.intrinsic_matrix(0, 0) /= 2;
        f.intrinsic_matrix(1, 1) /= 2;
        f.intrinsic_matrix(0, 2) /= 2;
        f.intrinsic_matrix(1, 2) /= 2;
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
        assert(f.detections.size() < 2);
    }

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

        Ys.push_back({
            cv::Point2f{ d[0].x, d[0].y }, // bottom-left
            cv::Point2f{ d[1].x, d[1].y }, // bottom-right
            cv::Point2f{ d[2].x, d[2].y }, // top-right
            cv::Point2f{ d[3].x, d[3].y }, // top-left
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
        Ts.push_back(cv::Vec3f{});
        auto& T = Ts.back();
        cv::solvePnP(Z, Ys.back(), Ks.back(), cv::Vec4f{ 0, 0, 0, 0 }, r, T);

        // std::cout << "Z:\n" << Z << "\n";
        // std::cout << "Y:\n" << Ys.back() << "\n";
        // std::cout << "K:\n" << Ks.back() << "\n";
        // std::cout << "V:\n" << Vs.back() << "\n";

        Rs.push_back({});
        auto& R = Rs.back();
        cv::Rodrigues(r, R);
    }

    // Prepare some of the data into easier form...
    auto images = std::vector<cv::Mat>{};
    auto detected_points = std::vector<std::array<cv::Point2f, 4>>{};
    auto Cs = std::vector<cv::Matx44f>{};
    auto K44s = std::vector<cv::Matx44f>{};
    auto K34s = std::vector<cv::Matx34f>{};
    for (auto i = 0u; i < frames.size(); ++i)
    {
        auto const& frame = frames[i];
        auto temp_image = cv::imread(frame.frame_path);
        images.push_back({});
        auto& image = images.back();
        cv::resize(temp_image, image, temp_image.size() / 2);
        detected_points.push_back({
            cv::Point2f{frame.detections[0].p[0].x, frame.detections[0].p[0].y},
            cv::Point2f{frame.detections[0].p[1].x, frame.detections[0].p[1].y},
            cv::Point2f{frame.detections[0].p[2].x, frame.detections[0].p[2].y},
            cv::Point2f{frame.detections[0].p[3].x, frame.detections[0].p[3].y},
        });

        auto const& K = Ks[i];
        K44s.push_back({
            K(0, 0), K(0, 1), K(0, 2), 0,
            K(1, 0), K(1, 1), K(1, 2), 0,
            K(2, 0), K(2, 1), K(2, 2), 0,
            0, 0, 0, 1,
        });
        K34s.push_back({
            K(0, 0), K(0, 1), K(0, 2), 0,
            K(1, 0), K(1, 1), K(1, 2), 0,
            0, 0, 0, 1,
        });
        auto const& R = Rs[i];
        auto const& T = Ts[i];
        Cs.push_back({
            R(0, 0), R(0, 1), R(0, 2), T(0),
            R(1, 0), R(1, 1), R(1, 2), T(1),
            R(2, 0), R(2, 1), R(2, 2), T(2),
            0, 0, 0, 1,
        });
    }

    // Show projected points when just using different (optimal) M for each image (just a sanity check for apriltag pose, it should match well)
    // auto tag_projected_points = std::vector<std::array<cv::Point2f, 4>>{};
    // for (auto i = 0u; i < frames.size(); ++i)
    // {
    //     tag_projected_points.push_back({});
    //     auto& proj = tag_projected_points.back();
    //     for (auto iz = 0; iz < 4; ++iz)
    //     {
    //         auto const& K = K44s[i];
    //         auto const& C = Cs[i];
    //         cv::Vec4f proj_h = K * C * Z4[iz]; // == K * Vs[i] * Vs[i].inv() * C * Z4[iz]
    //         proj_h[0] /= proj_h[2];
    //         proj_h[1] /= proj_h[2];
    //         proj[iz] = { proj_h[0], proj_h[1] };
    //     }
    // }
    // visualize_projections(images, detected_points, tag_projected_points, Ts, Rs);

    // Show projected points when just using initial M (M0) for each image
    // auto M0_projected_points = std::vector<std::array<cv::Point2f, 4>>{};
    // auto M0 = Vs[0].inv() * Cs[0];
    // for (auto i = 0u; i < frames.size(); ++i)
    // {
    //     M0_projected_points.push_back({});
    //     auto& proj = M0_projected_points.back();
    //     for (auto iz = 0; iz < 4; ++iz)
    //     {
    //         auto const& K = K44s[i];
    //         cv::Vec4f proj_h = K * Vs[i] * M0 * Z4[iz];
    //         proj_h[0] /= proj_h[2];
    //         proj_h[1] /= proj_h[2];
    //         proj[iz] = { proj_h[0], proj_h[1] };
    //     }
    // }
    // visualize_projections(images, detected_points, M0_projected_points, Ts, Rs);

    // Show projected points when using optimized M for each image
    // TODO
    // auto M_projected_points = std::vector<std::array<cv::Point2f, 4>>{};
    // auto M = cv::Matx44f::zeros();
    // for (auto i = 0u; i < frames.size(); ++i)
    // {
    //     M += Vs[i].inv() * Cs[i];
    // }
    // M /= (float)frames.size();
    // for (auto i = 0u; i < frames.size(); ++i)
    // {
    //     M_projected_points.push_back({});
    //     auto& proj = M_projected_points.back();
    //     for (auto iz = 0; iz < 4; ++iz)
    //     {
    //         auto const& K = K44s[i];
    //         cv::Vec4f proj_h = K * Vs[i] * M * Z4[iz];
    //         // auto const& K = K34s[i];
    //         // cv::Vec3f proj_h = K * Vs[i] * M * Z4[iz];
    //         proj_h[0] /= proj_h[2];
    //         proj_h[1] /= proj_h[2];
    //         proj[iz] = { proj_h[0], proj_h[1] };
    //         assert(std::abs(proj_h[3] - 1.0f) < 0.01);
    //     }
    // }
    // visualize_projections(images, detected_points, M_projected_points, Ts, Rs);

    // TODO: could just pass in Z and Y arrays and compute all 4 corner projection errors at once
    auto projection_error = [](cv::Matx34f const &K, cv::Matx44f const &V, cv::Matx44f const &M,
                     cv::Vec4f const &z, cv::Vec2f const &y) {
        // cv::Vec4f PVMz = K*V*M*z;
        cv::Vec3f PVMz = K*V*M*z;
        PVMz[0] /= PVMz[2];
        PVMz[1] /= PVMz[2];
        auto PVMz_xy = cv::Vec2f{ PVMz[0], PVMz[1] };
        auto d = PVMz_xy - y;
        return d.dot(d);
    };

    // auto full_mse = [](cv::Matx44f const &K, std::vector<cv::Matx44f> const &V, cv::Matx44f const &M,
    //                    cv::Vec4f const &z, std::vector<cv::Vec2f> const &y) {
                           
    // };

    // {
    //     auto mse = 0.0f;
    //     for (auto j = 0u; j < frames.size(); ++j)
    //     {
    //         auto frame_mse = 0.0f;
    //         for (auto k = 0u; k < 4; ++k)
    //         {
    //             auto y = cv::Vec2f{ Ys[j][k].x, Ys[j][k].y };
    //             // frame_mse += projection_error(K44s[j], Vs[j], Vs[j].inv() * Cs[j], Z4[k], y);
    //             frame_mse += projection_error(K34s[j], Vs[j], Vs[j].inv() * Cs[j], Z4[k], y);
    //         }
    //         std::printf("e_%u = %.2f\n", j, frame_mse);
    //         mse += frame_mse;
    //     }
    //     mse /= (frames.size() * 4);
    //     std::printf("E(M_jk) = %.2f (Reference error; projecting with per-frame April tag pose)\n", mse);
    // }

    // {
    //     auto mse = 0.0f;
    //     cv::Matx44f M0 = Vs[0].inv() * Cs[0]; // initialize with frame 0 (arbitrary)
    //     for (auto j = 0u; j < frames.size(); ++j)
    //     {
    //         auto frame_mse = 0.0f;
    //         for (auto k = 0u; k < 4; ++k)
    //         {
    //             auto y = cv::Vec2f{ Ys[j][k].x, Ys[j][k].y };
    //             // frame_mse += projection_error(K44s[j], Vs[j], M0, Z4[k], y);
    //             frame_mse += projection_error(K34s[j], Vs[j], M0, Z4[k], y);
    //         }
    //         std::printf("e_%u = %.2f\n", j, frame_mse);
    //         mse += frame_mse;
    //     }
    //     mse /= (frames.size() * 4);
    //     std::printf("E(M_0) = %.2f (Error for M0 initialized from first frame, no optimization)\n", mse);
    // }

    // TODO: Gauss-Newton optimization for M
    cv::Matx44f optimized_M = cv::Matx44f::eye();

    // TODO check all column/row initializations again

    // TODO: single- or double-precision floats?
    {
        // TODO: [0, s] or [-s/2, s/2] ?
        // Probably does not affect solution, as long as we are consistent
        Eigen::Matrix4f Z = Eigen::Matrix4f {
            { -s/2, -s/2, 0, 1, }, // bottom-left
            { s/2, -s/2, 0, 1, }, // bottom-right
            { s/2, s/2, 0, 1, }, // top-right
            { -s/2, s/2, 0, 1, }, // top-left
        };
        Z.transposeInPlace();

        Eigen::Matrix4f M0;
        {
            auto cv_M0 = Vs[0].inv() * Cs[0];
            M0 <<
                cv_M0(0, 0), cv_M0(0, 1), cv_M0(0, 2), cv_M0(0, 3),
                cv_M0(1, 0), cv_M0(1, 1), cv_M0(1, 2), cv_M0(1, 3),
                cv_M0(2, 0), cv_M0(2, 1), cv_M0(2, 2), cv_M0(2, 3),
                cv_M0(3, 0), cv_M0(3, 1), cv_M0(3, 2), cv_M0(3, 3);
            M0.transposeInPlace();
        }

        auto optimize_step = [&](Eigen::Vector3f const& t, Eigen::Vector4f const& q)
        {
            Eigen::Matrix4f M = make_pose_matrix(quat2rmat(q), t);

            // Accumulate A and b over all frames and tag corners
            Eigen::Matrix<float, 7, 7> A = Eigen::Matrix<float, 7, 7>::Zero();
            Eigen::Vector<float, 7> b = Eigen::Vector<float, 7>::Zero();
            for (size_t j = 0; j < frames.size(); ++j)
            {
                // TODO: simplify, recompute stuff much less
                for (size_t k = 0; k < 4; ++k)
                {
                    Eigen::Matrix<float, 2, 7> J_jk;

                    // Compute translation and orientation derivatives
                    Eigen::Matrix4f dMdXi[7];
                    // t1, t2, t3
                    for (size_t i = 0; i < 3; ++i)
                    {
                        dMdXi[i] = Eigen::Matrix4f::Zero();
                        dMdXi[i](i, 3) = 1;
                    }
                    // q1, q2, q3, q4
                    Eigen::Matrix3f dRdq[4];
                    quat2rmat_d(q, dRdq);
                    for (size_t i = 0; i < 4; ++i)
                    {
                        dMdXi[3 + i] = Eigen::Matrix4f::Zero();
                        dMdXi[3 + i].block<3, 3>(0, 0) = dRdq[i];
                        dMdXi[3 + i](3, 3) = 1;
                    }

                    // Projection of z_k with current M
                    Eigen::Matrix<float, 3, 4> P = frames[j].intrinsic_matrix.block<3, 4>(0, 0);
                    auto const& V = frames[j].view_matrix;
                    Eigen::Vector3f xyw = P * V * M * Z.col(k);

                    // Jg
                    float w2 = xyw(2) * xyw(2);
                    auto const Jg = Eigen::Matrix<float, 2, 3>{
                        {1.0f / xyw(2), 0, -xyw(0) / w2},
                        {0, 1.0f / xyw(2), -xyw(1) / w2},
                    };

                    Eigen::Matrix<float, 2, 4> Jg_P_V = Jg * P * V;
                    for (size_t i = 0; i < 7; ++i)
                    {
                        J_jk.block<2, 1>(0, i) = Jg_P_V * dMdXi[i] * Z.col(k);
                    }

                    A += J_jk.transpose() * J_jk;
                    Eigen::Vector2f xy = { xyw(0) / xyw(2), xyw(1) / xyw(2) };
                    auto const& p = frames[j].detections[0].p[k];
                    Eigen::Vector2f xy_true = Eigen::Vector2f{ p.x, p.y };
                    Eigen::Vector2f residual = xy - xy_true;
                    b -= J_jk.transpose() * residual;
                }
            }

            Eigen::Matrix<float, 7, 7> A_inv;
            Eigen::Vector<float, 7> dx = A_inv * b;

            return dx;
        };

        Eigen::Matrix4f M = M0;
        Eigen::Vector3f t = M.block<3, 1>(0, 3);
        Eigen::Quaternionf qq(Eigen::AngleAxisf(M.block<3, 3>(0, 0)));
        Eigen::Vector4f q = { qq.x(), qq.y(), qq.z(), qq.w(), };

        for (size_t step = 0; step < 100; ++step)
        {
            auto dx = optimize_step(t, q);
            t += dx.block<3, 1>(0, 0);
            q += dx.block<4, 1>(3, 0);
            qq = Eigen::Quaternionf{ q.x(), q.y(), q.z(), q.w() };
            qq.normalize(); // TODO: Make sure this is the quaternion normalization, not vec4 normalization
            q = { qq.x(), qq.y(), qq.z(), qq.w(), };
            std::cout << "q: " << q << "\n";
        }
        M = make_pose_matrix(quat2rmat(q), t);
        optimized_M = {
            M(0, 0), M(0, 1), M(0, 2), M(0, 3),
            M(1, 0), M(1, 1), M(1, 2), M(1, 3),
            M(2, 0), M(2, 1), M(2, 2), M(2, 3),
            M(3, 0), M(3, 1), M(3, 2), M(3, 3),
        };
    }

    // Final score
    {
        auto mse = 0.0f;
        for (auto j = 0u; j < frames.size(); ++j)
        {
            auto frame_mse = 0.0f;
            for (auto k = 0u; k < 4; ++k)
            {
                auto y = cv::Vec2f{ Ys[j][k].x, Ys[j][k].y };
                // frame_mse += projection_error(K44s[j], Vs[j], optimized_M, Z4[k], y);
                frame_mse += projection_error(K34s[j], Vs[j], optimized_M, Z4[k], y);
            }
            std::printf("e_%u = %.2f\n", j, frame_mse);
            mse += frame_mse;
        }
        mse /= (frames.size() * 4);
        std::printf("E(M) = %.2f (Error for optimized M)\n", mse);
    }

}
