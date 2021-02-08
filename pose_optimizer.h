#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <vector>

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

static auto throw_if_nan_or_inf = [](auto const& M)
{
    if (M.hasNaN() || M.array().isInf().any())
    {
        throw std::range_error("NaN or Inf encountered");
    }
};

static auto timing = [](auto const& f) {
    auto start = std::chrono::steady_clock::now();
    f();
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e-3f;
    return dt;
};

mat4 make_pose_matrix(mat3 const &R, vec3 const &t);

mat4 make_view_matrix(mat3 const& R, vec3 const& t);

mat3 quat2rmat(const vec4& q);

// Derivatives of the rotation matrix w.r.t. the quaternion of the quat2rmat() function.
mat3 quat2rmat_d(const vec4& q, mat3(&dR)[4]);

double calculate_mse(e_vec<mat2x4> const& p, e_vec<mat2x4> const& y);

e_vec<mat2x4> project_corners(e_vec<mat3x4> const& PVs, mat4 const& M, mat4 const& Z);

mat4 make_pose_matrix(mat3 const &R, vec3 const &t);

Eigen::Vector<double, 7> optimize_step(
    e_vec<mat3x4> const& PVs,
    e_vec<mat2x4> const& Ys,
    mat4 const& Z,
    vec3 const& t,
    vec4 const& q);

mat4 optimize_pose(
    e_vec<mat3x4> const& PVs,
    e_vec<mat2x4> const& Ys,
    mat4 const& Z,
    mat4 const& M0
    );
