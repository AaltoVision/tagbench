#include "pose_optimizer.h"

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
            Eigen::Vector3d proj_h = PVs[i] * M * Z.col(iz);
            projected[i].col(iz) = Eigen::Vector2d{
                proj_h(0) / proj_h(2),
                proj_h(1) / proj_h(2),
            };
        }
        throw_if_nan_or_inf(projected[i]);
    }
    return projected;
}

mat4 make_pose_matrix(Eigen::Matrix3d const &R, Eigen::Vector3d const &t)
{
    mat4 pose = mat4::Zero();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    pose(3, 3) = 1;
    return pose;
};

mat4 make_view_matrix(Eigen::Matrix3d const& R, Eigen::Vector3d const& t)
{
    mat4 V = Eigen::Matrix4d::Zero();
    V.block<3, 3>(0, 0) = R;
    V.block<3, 1>(0, 3) = -R * t;
    V(3, 3) = 1;
    return V;
};

Eigen::Vector<double, 7> optimize_step(
    e_vec<mat3x4> const& PVs,
    e_vec<mat2x4> const& Ys,
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

        for (size_t k = 0; k < 4; ++k)
        {
            // Projection of z_k with current M
            Eigen::Vector3d xyw = PV * M * Z.col(k);

            // Jg
            if (xyw(2) == 0)
            {
                // TODO
                throw;
                break;
            }
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

            throw_if_nan_or_inf(J_jk);
            throw_if_nan_or_inf(residual);
            throw_if_nan_or_inf(A);
            throw_if_nan_or_inf(b);
        }
    }

    Eigen::Vector<double, 7> dx = A.colPivHouseholderQr().solve(b);
    throw_if_nan_or_inf(dx);
    return dx;
};

Eigen::Matrix4d optimize_pose(
    e_vec<mat3x4> const& PVs,
    e_vec<mat2x4> const& Ys,
    Eigen::Matrix4d const& Z,
    Eigen::Matrix4d const& M0
    )
{
    Eigen::Matrix4d M = M0;
    Eigen::Vector3d t = M.block<3, 1>(0, 3);
    Eigen::Quaterniond qq(Eigen::AngleAxisd(M.block<3, 3>(0, 0)));
    Eigen::Vector4d q = { qq.w(), qq.x(), qq.y(), qq.z(), };

    // TODO: actual threshold etc.
    for (size_t step = 0; step < 10; ++step)
    {
        Eigen::Vector<double, 7> dx;
        auto step_time = timing([&]{ dx = optimize_step(PVs, Ys, Z, t, q); });
        t += dx.block<3, 1>(0, 0);
        q += dx.block<4, 1>(3, 0);
        qq = Eigen::Quaterniond{ q.x(), q.y(), q.z(), q.w() };
        qq.normalize();
        q = { qq.w(), qq.x(), qq.y(), qq.z(), };

        M = make_pose_matrix(quat2rmat(q), t);

        throw_if_nan_or_inf(M);
        throw_if_nan_or_inf(PVs[0]);
        throw_if_nan_or_inf(Z);
        throw_if_nan_or_inf(Ys[0]);

        std::cout << "Step " << step << ": |dx| = " << dx.norm();
        std::printf("\t\tE(M) = %.6e", calculate_mse(project_corners(PVs, M, Z), Ys));
        std::printf("\t\t(step time: %.2fs)", step_time);
        std::cout << std::endl;
    }
    return M;
}

