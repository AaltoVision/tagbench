#include "pose_optimizer.h"

mat3 quat2rmat(const vec4& q) {
    mat3 R;
    R <<
        q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2],
        2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1],
        2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    return R;
}

// Derivatives of the rotation matrix w.r.t. the quaternion of the quat2rmat() function.
mat3 quat2rmat_d(const vec4& q, mat3(&dR)[4]) {
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

double calculate_mse(e_vec<mat2x4> const& p, e_vec<mat2x4> const& y)
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

e_vec<mat2x4> project_corners(e_vec<mat3x4> const& PVs, mat4 const& M, mat4 const& Z)
{
    auto projected = e_vec<mat2x4>(PVs.size());
    for (size_t i = 0; i < PVs.size(); ++i)
    {
        projected[i] = mat2x4::Zero();
        for (auto iz = 0; iz < 4; ++iz)
        {
            vec3 proj_h = PVs[i] * M * Z.col(iz);
            projected[i].col(iz) = vec2{
                proj_h(0) / proj_h(2),
                proj_h(1) / proj_h(2),
            };
        }
    }
    return projected;
}

mat4 make_pose_matrix(mat3 const &R, vec3 const &t)
{
    mat4 pose = mat4::Zero();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    pose(3, 3) = 1;
    return pose;
}

mat4 make_view_matrix(mat3 const& R, vec3 const& t)
{
    mat4 V = mat4::Zero();
    V.block<3, 3>(0, 0) = R.transpose();
    V.block<3, 1>(0, 3) = -R.transpose() * t;
    V(3, 3) = 1;
    return V;
}

// Solve 'dx', update of state vector 'x'
Eigen::Vector<double, 7> optimize_step(
    e_vec<mat3x4> const& PVs,
    e_vec<mat2x4> const& Ys,
    mat4 const& Z,
    vec3 const& t,
    vec4 const& q,
    double residual_norm
    )
{
    mat4 M = make_pose_matrix(quat2rmat(q), t);

    // Calculate norm of projection residuals for thresholding
    residual_norm = 0.0;

    // Accumulate A and b over all frames and tag corners
    Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Vector<double, 7> b = Eigen::Vector<double, 7>::Zero();
    for (size_t j = 0; j < PVs.size(); ++j)
    {
        mat3x4 PV = PVs[j];

        // Compute translation and orientation derivatives
        mat4 dMdXi[7];
        // t1, t2, t3
        for (size_t i = 0; i < 3; ++i)
        {
            dMdXi[i] = mat4::Zero();
            dMdXi[i](i, 3) = 1;
        }
        // q1, q2, q3, q4
        mat3 dRdq[4];
        quat2rmat_d(q, dRdq);
        for (size_t i = 0; i < 4; ++i)
        {
            dMdXi[3 + i] = mat4::Zero();
            dMdXi[3 + i].block<3, 3>(0, 0) = dRdq[i];
        }

        for (size_t k = 0; k < 4; ++k)
        {
            // Projection of corner point z_k with current M
            vec3 xyw = PV * M * Z.col(k);

            // NOTE: xyw(2) is '-Z'. Z should be negative to be in front of camera, so -Z should be positive
            if (xyw(2) < 0)
            {
                // TODO: possibly penalize projecting points behind camera in a meaningful way instead of giving up
                throw "Failed to optimize pose, marker corner projected behind camera";
            }

            // Jg (Jacobian of g(x,y,w) = (x/w, y/w))
            double w2 = xyw(2) * xyw(2);
            Eigen::Matrix<double, 2, 3> Jg = Eigen::Matrix<double, 2, 3>{
                {1.0f / xyw(2), 0, -xyw(0) / w2},
                {0, 1.0f / xyw(2), -xyw(1) / w2},
            };

            mat2x4 Jg_P_V = Jg * PV;
            Eigen::Matrix<double, 2, 7> J_jk;
            for (size_t i = 0; i < 7; ++i)
            {
                J_jk.col(i) = Jg_P_V * dMdXi[i] * Z.col(k);
            }

            A += J_jk.transpose() * J_jk;
            vec2 xy = { xyw(0) / xyw(2), xyw(1) / xyw(2) };
            vec2 xy_groundtruth = Ys[j].col(k);
            vec2 residual = xy - xy_groundtruth;
            b -= J_jk.transpose() * residual;

            throw_if_nan_or_inf(J_jk);
            throw_if_nan_or_inf(residual);
            throw_if_nan_or_inf(A);
            throw_if_nan_or_inf(b);

            residual_norm += residual.transpose() * residual;
        }
    }
    residual_norm = std::sqrt(residual_norm);

    Eigen::Vector<double, 7> dx = A.colPivHouseholderQr().solve(b);
    throw_if_nan_or_inf(dx);
    return dx;
}

mat4 optimize_pose(
    e_vec<mat3x4> const& PVs,
    e_vec<mat2x4> const& Ys,
    mat4 const& Z,
    mat4 const& M0,
    int max_steps,
    double stop_threshold,
    bool silent
    )
{
    mat4 M = M0;
    vec3 t = M.block<3, 1>(0, 3);
    Eigen::Quaterniond qq(Eigen::AngleAxisd(M.block<3, 3>(0, 0)));
    vec4 q = { qq.w(), qq.x(), qq.y(), qq.z(), };

    if (!silent)
    {
        std::printf("Initial error: \t\t\t\tE(M0) = %.6e\n", calculate_mse(project_corners(PVs, M, Z), Ys));
    }

    auto step = 0;
    for (; (step < max_steps); ++step)
    {
        Eigen::Vector<double, 7> dx;
        auto residual_norm = 0.0;
        auto step_time = timing([&]{ dx = optimize_step(PVs, Ys, Z, t, q, residual_norm); });
        if (residual_norm < stop_threshold)
        {
            step++;
            break;
        }
        t += dx.block<3, 1>(0, 0);
        q += dx.block<4, 1>(3, 0);
        qq = Eigen::Quaterniond{ q.x(), q.y(), q.z(), q.w() };
        qq.normalize();
        q = { qq.w(), qq.x(), qq.y(), qq.z(), };

        M = make_pose_matrix(quat2rmat(q), t);
        throw_if_nan_or_inf(M);

        if (!silent)
        {
            std::printf("Step %i: |dx| = %.8f\n", (int)step, dx.norm());
            std::printf("\t\tE(M) = %.6e", calculate_mse(project_corners(PVs, M, Z), Ys));
            std::printf("\t\t(step time: %.2fs)", step_time);
            std::cout << std::endl;
        }
    }
    if (!silent)
    {
        std::printf("Finished after %d steps", step+1);
        std::cout << std::endl;
    }

    return M;
}

