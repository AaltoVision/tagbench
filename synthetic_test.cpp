#include "pose_optimizer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d.hpp>

#include "cv_helpers.h"

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
    auto const angles = Eigen::VectorXd(Eigen::ArrayXd::LinSpaced(32, -M_PI/6, M_PI/6));

    // Rotate 30 degrees right-left around Y axis
    for (auto angle : angles)
    {
        vec3 t = vec3{ 0, 0, 4 };
        mat3 R = Eigen::AngleAxisd(angle, vec3::UnitY()).toRotationMatrix();
        Vs.push_back(make_view_matrix(R, t));
        // Just a sanity check; check origin is not behind camera (same as V(2,3)<0)
        if ((Vs.back() * vec3::Zero().homogeneous()).z() > 0) throw;
    }

    // Translate in a circle around Z axis, looking towards Z-
    {
        double d_angle = 2.0 * M_PI / 32;
        double t_magnitude = 0.5; // TODO: relate to pixels (through s and resolution)
        for (size_t i = 0; i < 32; ++i)
        {
            vec3 t = Eigen::AngleAxisd(i * d_angle, -vec3::UnitZ()) * vec3::UnitX() * t_magnitude;
            t(2) = 4;
            mat3 R = mat3::Identity();
            Vs.push_back(make_view_matrix(R, t));
        }
        if ((Vs.back() * vec3::Zero().homogeneous()).z() > 0) throw;
    }

    // Zoom out
    for (size_t i = 0; i < 32; ++i)
    {
        // Looking at Z-
        mat3 R = mat3::Identity();
        // Moving towards Z+
        vec3 t = vec3::UnitZ() * ((int)i + 1);
        Vs.push_back(make_view_matrix(R, t));
        if ((Vs.back() * vec3::Zero().homogeneous()).z() > 0) throw;
    }

    // Zoom out, looking left
    for (size_t i = 0; i < 32; ++i)
    {
        // Looking at 30' left from Z-
        mat3 R = Eigen::AngleAxisd(M_PI / 6, vec3::UnitY()).toRotationMatrix();
        // Moving towards Z+
        vec3 t = vec3::UnitZ() * ((int)i + 1);
        Vs.push_back(make_view_matrix(R, t));
        if ((Vs.back() * vec3::Zero().homogeneous()).z() > 0) throw;
    }

    // Rotate 30 degrees left-right around Y axis, while moving in circle around Z axis
    {
        double d_angle = 2.0 * M_PI / 32;
        double t_magnitude = 0.2;
        Eigen::VectorXd angles = Eigen::ArrayXd::LinSpaced(32, -M_PI / 12, M_PI / 12);
        for (size_t i = 0; i < 32; ++i)
        {
            vec3 t = Eigen::AngleAxisd(i * d_angle, -vec3::UnitZ()) * vec3::UnitX() * t_magnitude;
            t(2) = ((int)i) * 1.2 + 1;
            mat3 R = Eigen::AngleAxisd(angles[i], vec3::UnitY()).toRotationMatrix();
            Vs.push_back(make_view_matrix(R, t));
            if ((Vs.back() * vec3::Zero().homogeneous()).z() > 0) throw;
        }
    }

    // Create groundtruth projections
    auto PVs = e_vec<mat3x4>(Vs.size());
    std::transform(Vs.begin(), Vs.end(),
                    PVs.begin(), [&](auto const& V) -> mat3x4 { return P * V; });
    Ys = project_corners(PVs, M, Z);
}

void test_synthetic_case(bool show_visualization)
{
    e_vec<mat4> Vs;
    e_vec<mat2x4> Ys;

    double s = 0.198;
    mat4 Z;
    Z.col(0) = vec4{ -s/2, -s/2, 0, 1, }; // bottom-left
    Z.col(1) = vec4{ s/2, -s/2, 0, 1, }; // bottom-right
    Z.col(2) = vec4{ s/2, s/2, 0, 1, }; // top-right
    Z.col(3) = vec4{ -s/2, s/2, 0, 1, }; // top-left

    double w = 1920, h = 1080;
    double fx = 1445.514404296875, fy = 1451.21630859375;
    double px = 950.2744140625, py = 538.8798217773438;
    w /= 2; h /= 2; fx /= 2; fy /= 2;  px /= 2; py /= 2;
    mat3x4 P;
    P.row(0) = vec4{ fx, 0, -px, 0 };
    P.row(1) = vec4{ 0, fy, -py, 0 };
    P.row(2) = vec4{ 0, 0, -1, 0 };

    // quad at y=1, with 45' rotation around Y-axis
    mat4 synthetic_M = mat4::Identity();
    synthetic_M.col(0) = vec4{ 1, 0, -1, 0 }.normalized();
    synthetic_M.col(2) = vec4{ 1, 0, 1, 0 }.normalized();
    synthetic_M.col(3) = vec4{ 0, 1, 0, 1 };
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
    mat4 M = optimize_pose(PVs, noisy_Ys, Z, M0, 100, 0, false);
    std::stringstream label;
    label.precision(3);
    label << "Optimized M for synthetic case:\n" << M << std::endl;
    label << "Diff from synthetic_M\n" << (M - synthetic_M).cwiseAbs().format(Eigen::StreamPrecision) << std::endl;
    throw_if_nan_or_inf(M);
    auto projections = project_corners(PVs, M, Z);
    auto images = std::vector<cv::Mat>(PVs.size());
    for (auto& image : images) {
        image = cv::Mat(cv::Size2d{ w, h }, CV_8UC3, cv::Scalar{ 200, 200, 200 });
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

int main()
{
    test_synthetic_case(true);
}