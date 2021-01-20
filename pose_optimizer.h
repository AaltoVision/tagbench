#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <chrono>

using mat2x4 = Eigen::Matrix<double, 2, 4>;
using mat3x4 = Eigen::Matrix<double, 3, 4>;
using mat4 = Eigen::Matrix4d;

static auto throw_if_nan_or_inf = [](auto const& M)
{
    // if (M.hasNaN() || !std::isfinite(M))
    if (M.hasNaN() || !( (M - M).array() == (M - M).array()).all())
    {
        throw std::range_error("NaN encountered");
    }
};

static auto timing = [](auto& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e-3f;
    return dt;
};

Eigen::Matrix3d quat2rmat(const Eigen::Vector4d& q);

// Derivatives of the rotation matrix w.r.t. the quaternion of the quat2rmat() function.
Eigen::Matrix3d quat2rmat_d(const Eigen::Vector4d& q, Eigen::Matrix3d(&dR)[4]);

double calculate_mse(std::vector<mat2x4> const& p, std::vector<mat2x4> const& y);

std::vector<mat2x4> project_corners(std::vector<mat3x4> const& PVs, mat4 const& M, mat4 const& Z);

Eigen::Matrix4d make_pose_matrix(Eigen::Matrix3d const &R, Eigen::Vector3d const &t);

Eigen::Vector<double, 7> optimize_step(
    std::vector<mat3x4> const& PVs,
    std::vector<mat2x4> const& Ys,
    Eigen::Matrix4d const& Z,
    Eigen::Vector3d const& t,
    Eigen::Vector4d const& q);

Eigen::Matrix4d optimize_pose(
    std::vector<mat3x4> const& PVs,
    std::vector<mat2x4> const& Ys,
    Eigen::Matrix4d const& Z,
    Eigen::Matrix4d const& M0
    );
