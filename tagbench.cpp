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
        // auto const& d = f.detections[0].p;

        auto const& d = f.detections[0];

        Ys.push_back({
            cv::Point2f{ d[0][0], d[0][1] }, // bottom-left
            cv::Point2f{ d[1][0], d[1][1] }, // bottom-right
            cv::Point2f{ d[2][0], d[2][1] }, // top-right
            cv::Point2f{ d[3][0], d[3][1] }, // top-left
        });

        Ks.push_back({
            (float)f.intrinsic_matrix(0, 0), (float)f.intrinsic_matrix(0, 1), (float)f.intrinsic_matrix(0, 2),
            (float)f.intrinsic_matrix(1, 0), (float)f.intrinsic_matrix(1, 1), (float)f.intrinsic_matrix(1, 2),
            (float)f.intrinsic_matrix(2, 0), (float)f.intrinsic_matrix(2, 1), (float)f.intrinsic_matrix(2, 2),
        });

        Vs.push_back({
            (float)f.view_matrix(0, 0), (float)f.view_matrix(0, 1), (float)f.view_matrix(0, 2), (float)f.view_matrix(0, 3),
            (float)f.view_matrix(1, 0), (float)f.view_matrix(1, 1), (float)f.view_matrix(1, 2), (float)f.view_matrix(0, 3),
            (float)f.view_matrix(2, 0), (float)f.view_matrix(2, 1), (float)f.view_matrix(2, 2), (float)f.view_matrix(0, 3),
            (float)f.view_matrix(3, 0), (float)f.view_matrix(3, 1), (float)f.view_matrix(3, 2), (float)f.view_matrix(3, 3),
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
    auto cv_Ps = std::vector<cv::Matx34f>{};
    auto Ps = std::vector<Eigen::Matrix<double, 3, 4>>{};
    for (auto i = 0u; i < frames.size(); ++i)
    {
        auto const& frame = frames[i];
        auto temp_image = cv::imread(frame.frame_path);
        images.push_back({});
        auto& image = images.back();
        cv::resize(temp_image, image, temp_image.size() / 2);
        detected_points.push_back({
            cv::Point2f{frame.detections[0][0][0], frame.detections[0][0][1]},
            cv::Point2f{frame.detections[0][1][0], frame.detections[0][1][1]},
            cv::Point2f{frame.detections[0][2][0], frame.detections[0][2][1]},
            cv::Point2f{frame.detections[0][3][0], frame.detections[0][3][1]},
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
        cv_Ps.push_back({
            K(0, 0), K(0, 1), K(0, 2), 0,
            K(1, 0), K(1, 1), K(1, 2), 0,
            K(2, 0), K(2, 1), K(2, 2), 0,
        });
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

    cv::Matx44f optimized_M = cv::Matx44f::eye();

    // TODO check all column/row initializations again

    auto setup_end = std::chrono::steady_clock::now();
    auto setup_dt = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start).count() * 1e-3f;

    std::printf("Setup done in %.2fs\n", setup_dt);

    // TODO: single- or double-precision floats?
    {
        // TODO: [0, s] or [-s/2, s/2] ?
        // Probably does not affect solution, as long as we are consistent
        Eigen::Matrix4d Z = Eigen::Matrix4d {
            { -s/2, -s/2, 0, 1, }, // bottom-left
            { s/2, -s/2, 0, 1, }, // bottom-right
            { s/2, s/2, 0, 1, }, // top-right
            { -s/2, s/2, 0, 1, }, // top-left
        };
        Z.transposeInPlace();

        std::vector<Eigen::Matrix<double, 3, 4>> PVs;
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
        // Eigen::Matrix4d M0 = Eigen::Matrix4d::Identity();

        optimized_M = cv_M0;

        auto optimize_step = [&](Eigen::Vector3d const& t, Eigen::Vector4d const& q)
        {
            Eigen::Matrix4d M = make_pose_matrix(quat2rmat(q), t);

            // Accumulate A and b over all frames and tag corners
            Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Zero();
            Eigen::Vector<double, 7> b = Eigen::Vector<double, 7>::Zero();
            for (size_t j = 0; j < frames.size(); ++j)
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
                        J_jk.block<2, 1>(0, i) = Jg_P_V * dMdXi[i] * Z.col(k);
                    }

                    A += J_jk.transpose() * J_jk;
                    Eigen::Vector2d xy = { xyw(0) / xyw(2), xyw(1) / xyw(2) };
                    Eigen::Vector2d xy_detected = Eigen::Vector2d{ (double)Ys[j][k].x, (double)Ys[j][k].y };
                    Eigen::Vector2d residual = xy - xy_detected;
                    b -= J_jk.transpose() * residual;
                }
            }

            Eigen::Vector<double, 7> dx = A.colPivHouseholderQr().solve(b);
            return dx;
        };

        Eigen::Matrix4d M = M0;
        Eigen::Vector3d t = M.block<3, 1>(0, 3);
        Eigen::Quaterniond qq(Eigen::AngleAxisd(M.block<3, 3>(0, 0)));

        // TODO: check if quat2rmat and quat2rmat_d expect q to be wxyz or xyzw
        Eigen::Vector4d q = { qq.x(), qq.y(), qq.z(), qq.w(), };

        auto optimization_time = timing([&]{
            for (size_t step = 0; step < 100; ++step)
            {
                Eigen::Vector<double, 7> dx;
                auto step_time = timing([&]{ dx = optimize_step(t, q); });
                t += dx.block<3, 1>(0, 0);
                q += dx.block<4, 1>(3, 0);
                // TODO: check if shuffle goes correctly for normalization
                qq = Eigen::Quaterniond{ q.y(), q.z(), q.w(), q.x() };
                // qq = Eigen::Quaterniond{ q.x(), q.y(), q.z(), q.w() };
                qq.normalize();
                q = { qq.x(), qq.y(), qq.z(), qq.w(), };
                std::cout << "Step " << step << ": |dx| = " << dx.norm();
                std::printf("\t\t(step time: %.2fs)", step_time);
                std::cout << std::endl;
            }
            M = make_pose_matrix(quat2rmat(q), t);
            Eigen::Matrix4f Mf = M.cast<float>();
            optimized_M = {
                Mf(0, 0), Mf(0, 1), Mf(0, 2), Mf(0, 3),
                Mf(1, 0), Mf(1, 1), Mf(1, 2), Mf(1, 3),
                Mf(2, 0), Mf(2, 1), Mf(2, 2), Mf(2, 3),
                Mf(3, 0), Mf(3, 1), Mf(3, 2), Mf(3, 3),
            };
        });
        std::printf("Total optimization time: %.2fs\n", optimization_time);
    }


    std::vector<std::array<cv::Point2f, 4>> optimized_M_projected_points;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        optimized_M_projected_points.push_back({});
        auto& proj = optimized_M_projected_points.back();
        for (auto iz = 0; iz < 4; ++iz)
        {
            cv::Vec3f proj_h = cv_Ps[i] * Vs[i] * optimized_M * Z4[iz];
            proj_h[0] /= proj_h[2];
            proj_h[1] /= proj_h[2];
            proj[iz] = { proj_h[0], proj_h[1] };
        }
    }
    visualize_projections(images, detected_points, optimized_M_projected_points, Ts, Rs);

    // Final score
    {
        auto mse = 0.0;
        for (size_t j = 0; j < frames.size(); ++j)
        {
            auto frame_mse = 0.0f;
            for (size_t k = 0; k < 4; ++k)
            {
                auto y = cv::Vec2f{ Ys[j][k].x, Ys[j][k].y };
                frame_mse += projection_error(cv_Ps[j], Vs[j], optimized_M, Z4[k], y);
            }
            // std::printf("e_%zu = %.2f\n", j, frame_mse);
            mse += frame_mse;
        }
        std::printf("Total input frames: %zu\n", total_input_frames);
        std::printf("Frames considered (single marker detected): %zu\n", frames.size());
        std::printf("E(M) = %.2f (Error for optimized M)\n", mse);
        mse /= (frames.size() * 4);
        std::printf("average error = %.2f (Error for optimized M)\n", mse);
    }

}
