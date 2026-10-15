/*
 * underwater_allocator.c —— 逐次线性化 + 有界最小二乘控制分配实现
 *
 * 参考 control/cm_uav/{geometry,constrained_allocation}.py 的公式，只使用
 * float / 定点友好的运算，可在飞控 MCU 上以固定周期运行。
 */
#include "underwater_allocator.h"

#include <math.h>
#include <string.h>

#define UW_EPS 1e-9f

static const float UW_Z_HAT[3] = {0.0f, 0.0f, 1.0f};

static float uw_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static void uw_cross(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/* 推力方向 n_i(theta) = cos(theta) z + sin(theta) d_i */
static void uw_thrust_direction(const uw_vehicle_t *v, int i, float theta, float n[3])
{
    const float c = cosf(theta);
    const float s = sinf(theta);
    for (int k = 0; k < 3; ++k) {
        n[k] = c * UW_Z_HAT[k] + s * v->tilt_directions[i][k];
    }
}

/* 方向对倾转角的导数 dn_i/dtheta */
static void uw_thrust_direction_deriv(const uw_vehicle_t *v, int i, float theta, float dn[3])
{
    const float c = cosf(theta);
    const float s = sinf(theta);
    for (int k = 0; k < 3; ++k) {
        dn[k] = -s * UW_Z_HAT[k] + c * v->tilt_directions[i][k];
    }
}

/* 由推力/倾转角/效能计算机体系合力与合力矩。 */
static void uw_body_wrench(const uw_vehicle_t *v, const float T[UW_NUM_ROTORS],
                           const float theta[UW_NUM_ROTORS], const float health[UW_NUM_ROTORS],
                           float out[UW_NUM_WRENCH])
{
    float force[3] = {0.0f, 0.0f, 0.0f};
    float torque[3] = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        float n[3], ti[3], moment[3];
        const float thrust = T[i] * health[i];
        uw_thrust_direction(v, i, theta[i], n);
        for (int k = 0; k < 3; ++k) {
            ti[k] = thrust * n[k];
            force[k] += ti[k];
        }
        uw_cross(v->positions[i], ti, moment);
        const float reaction = v->spin_signs[i] * v->reaction_torque_coeff * thrust;
        for (int k = 0; k < 3; ++k) {
            torque[k] += moment[k] + reaction * n[k];
        }
    }
    for (int k = 0; k < 3; ++k) {
        out[k] = force[k];
        out[3 + k] = torque[k];
    }
}

/* 控制效率矩阵 B = d(wrench)/du，(6,8)，列 0..3 已含效能。 */
static void uw_effectiveness(const uw_vehicle_t *v, const float T[UW_NUM_ROTORS],
                             const float theta[UW_NUM_ROTORS], const float health[UW_NUM_ROTORS],
                             float B[UW_NUM_WRENCH][UW_NUM_ACTUATORS])
{
    memset(B, 0, sizeof(float) * UW_NUM_WRENCH * UW_NUM_ACTUATORS);

    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        float n[3], dn[3], col[3], dcol[3];
        const float thrust = T[i] * health[i];
        uw_thrust_direction(v, i, theta[i], n);
        uw_thrust_direction_deriv(v, i, theta[i], dn);

        /* 对推力的偏导 */
        uw_cross(v->positions[i], n, col);
        for (int k = 0; k < 3; ++k) {
            B[k][i] = n[k] * health[i];
            B[3 + k][i] = (col[k] + v->spin_signs[i] * v->reaction_torque_coeff * n[k]) * health[i];
        }

        /* 对倾转角的偏导 */
        uw_cross(v->positions[i], dn, dcol);
        for (int k = 0; k < 3; ++k) {
            B[k][4 + i] = thrust * dn[k];
            B[3 + k][4 + i] = thrust * (dcol[k] + v->spin_signs[i] * v->reaction_torque_coeff * dn[k]);
        }
    }
}

/* 6x8 矩阵的数值秩（高斯消元 + 部分主元）。 */
static int uw_matrix_rank(const float B[UW_NUM_WRENCH][UW_NUM_ACTUATORS])
{
    float m[UW_NUM_WRENCH][UW_NUM_ACTUATORS];
    memcpy(m, B, sizeof(m));

    float scale = 0.0f;
    for (int r = 0; r < UW_NUM_WRENCH; ++r) {
        for (int c = 0; c < UW_NUM_ACTUATORS; ++c) {
            const float a = fabsf(m[r][c]);
            if (a > scale) {
                scale = a;
            }
        }
    }
    const float tol = 1e-6f * (scale > 0.0f ? scale : 1.0f);

    int rank = 0;
    for (int c = 0; c < UW_NUM_ACTUATORS && rank < UW_NUM_WRENCH; ++c) {
        int pivot = -1;
        float best = tol;
        for (int r = rank; r < UW_NUM_WRENCH; ++r) {
            const float a = fabsf(m[r][c]);
            if (a > best) {
                best = a;
                pivot = r;
            }
        }
        if (pivot < 0) {
            continue;
        }
        if (pivot != rank) {
            for (int k = 0; k < UW_NUM_ACTUATORS; ++k) {
                const float tmp = m[rank][k];
                m[rank][k] = m[pivot][k];
                m[pivot][k] = tmp;
            }
        }
        const float inv = 1.0f / m[rank][c];
        for (int r = 0; r < UW_NUM_WRENCH; ++r) {
            if (r == rank) {
                continue;
            }
            const float factor = m[r][c] * inv;
            if (factor == 0.0f) {
                continue;
            }
            for (int k = c; k < UW_NUM_ACTUATORS; ++k) {
                m[r][k] -= factor * m[rank][k];
            }
        }
        ++rank;
    }
    return rank;
}

/* 正规方程求解：A x = b（8x8，部分主元高斯-约当）。奇异时返回 -1。 */
static int uw_solve8(float A[UW_NUM_ACTUATORS][UW_NUM_ACTUATORS],
                     const float b[UW_NUM_ACTUATORS], float x[UW_NUM_ACTUATORS])
{
    float aug[UW_NUM_ACTUATORS][UW_NUM_ACTUATORS + 1];
    for (int r = 0; r < UW_NUM_ACTUATORS; ++r) {
        for (int c = 0; c < UW_NUM_ACTUATORS; ++c) {
            aug[r][c] = A[r][c];
        }
        aug[r][UW_NUM_ACTUATORS] = b[r];
    }

    for (int c = 0; c < UW_NUM_ACTUATORS; ++c) {
        int pivot = c;
        for (int r = c + 1; r < UW_NUM_ACTUATORS; ++r) {
            if (fabsf(aug[r][c]) > fabsf(aug[pivot][c])) {
                pivot = r;
            }
        }
        if (fabsf(aug[pivot][c]) < 1e-12f) {
            return -1;
        }
        if (pivot != c) {
            for (int k = c; k <= UW_NUM_ACTUATORS; ++k) {
                const float tmp = aug[c][k];
                aug[c][k] = aug[pivot][k];
                aug[pivot][k] = tmp;
            }
        }
        const float inv = 1.0f / aug[c][c];
        for (int k = c; k <= UW_NUM_ACTUATORS; ++k) {
            aug[c][k] *= inv;
        }
        for (int r = 0; r < UW_NUM_ACTUATORS; ++r) {
            if (r == c) {
                continue;
            }
            const float factor = aug[r][c];
            if (factor == 0.0f) {
                continue;
            }
            for (int k = c; k <= UW_NUM_ACTUATORS; ++k) {
                aug[r][k] -= factor * aug[c][k];
            }
        }
    }

    for (int r = 0; r < UW_NUM_ACTUATORS; ++r) {
        x[r] = aug[r][UW_NUM_ACTUATORS];
    }
    return 0;
}

void uw_allocator_init(uw_allocator_t *a, const uw_vehicle_t *vehicle)
{
    a->vehicle = vehicle;
    uw_allocator_reset(a);
}

void uw_allocator_reset(uw_allocator_t *a)
{
    memset(a->previous, 0, sizeof(a->previous));
    a->has_previous = 0;
}

void uw_allocate(uw_allocator_t *a, const float target[UW_NUM_WRENCH],
                 uw_medium_t medium, float dt, const float health_in[UW_NUM_ROTORS],
                 float air_fraction, uw_allocation_t *out)
{
    const uw_vehicle_t *v = a->vehicle;

    float health[UW_NUM_ROTORS];
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        health[i] = health_in ? uw_clampf(health_in[i], 0.0f, 1.0f) : 1.0f;
    }
    if (dt <= 0.0f) {
        dt = 1e-3f;
    }
    air_fraction = uw_clampf(air_fraction, 0.0f, 1.0f);

    const float thrust_max = (medium == UW_MEDIUM_AIR) ? v->thrust_max_air : v->thrust_max_water;

    /* ---- 1. 可达区间：物理边界 + 变化率 + 空中授权回中 ---- */
    float low[UW_NUM_ACTUATORS], high[UW_NUM_ACTUATORS];
    float physical_low[UW_NUM_ACTUATORS], physical_high[UW_NUM_ACTUATORS];
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        physical_low[i] = 0.0f;
        physical_high[i] = thrust_max;
        physical_low[4 + i] = -v->tilt_limit;
        physical_high[4 + i] = v->tilt_limit;
    }
    float previous[UW_NUM_ACTUATORS];
    for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
        previous[j] = a->has_previous ? uw_clampf(a->previous[j], physical_low[j], physical_high[j])
                                      : uw_clampf(0.0f, physical_low[j], physical_high[j]);
    }
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        const float thrust_step = v->thrust_rate_n_s * dt;
        const float tilt_step = v->tilt_rate_limit * dt;
        low[i] = fmaxf(physical_low[i], previous[i] - thrust_step);
        high[i] = fminf(physical_high[i], previous[i] + thrust_step);
        low[4 + i] = fmaxf(physical_low[4 + i], previous[4 + i] - tilt_step);
        high[4 + i] = fminf(physical_high[4 + i], previous[4 + i] + tilt_step);
    }
    if (air_fraction >= 0.999f) {
        for (int i = 0; i < UW_NUM_ROTORS; ++i) {
            const float centered = uw_clampf(0.0f, low[4 + i], high[4 + i]);
            low[4 + i] = centered;
            high[4 + i] = centered;
        }
    }

    float anchor[UW_NUM_ACTUATORS];
    float u[UW_NUM_ACTUATORS];
    for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
        anchor[j] = previous[j];
        u[j] = uw_clampf(anchor[j], low[j], high[j]);
    }

    /* ---- 2. 归一化权重与平滑惩罚 ---- */
    float scale6[UW_NUM_WRENCH];
    for (int k = 0; k < 3; ++k) {
        scale6[k] = v->force_scale_n;
        scale6[3 + k] = v->torque_scale_nm;
    }
    float weights[UW_NUM_WRENCH];
    for (int k = 0; k < UW_NUM_WRENCH; ++k) {
        weights[k] = v->axis_priority[k] / scale6[k];
    }
    float penalty[UW_NUM_ACTUATORS];
    const float sqrt_smooth = sqrtf(v->smoothness);
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        penalty[i] = sqrt_smooth / thrust_max;
        penalty[4 + i] = sqrt_smooth / v->tilt_limit;
    }

    /* ---- 3. 逐次线性化 ---- */
    float T[UW_NUM_ROTORS], theta[UW_NUM_ROTORS];
    int iterations = 0;
    for (iterations = 1; iterations <= v->max_iterations; ++iterations) {
        for (int i = 0; i < UW_NUM_ROTORS; ++i) {
            T[i] = u[i];
            theta[i] = u[4 + i];
        }

        float B[UW_NUM_WRENCH][UW_NUM_ACTUATORS];
        float w0[UW_NUM_WRENCH];
        uw_effectiveness(v, T, theta, health, B);
        uw_body_wrench(v, T, theta, health, w0);

        float A[UW_NUM_ACTUATORS][UW_NUM_ACTUATORS];
        float rhs[UW_NUM_ACTUATORS];
        for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
            rhs[j] = 0.0f;
            for (int k = 0; k < UW_NUM_ACTUATORS; ++k) {
                A[j][k] = 0.0f;
            }
        }
        for (int r = 0; r < UW_NUM_WRENCH; ++r) {
            const float bw = weights[r] * (target[r] - w0[r]);
            for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
                const float jw = weights[r] * B[r][j];
                rhs[j] += jw * bw;
                for (int k = j; k < UW_NUM_ACTUATORS; ++k) {
                    A[j][k] += jw * weights[r] * B[r][k];
                }
            }
        }
        for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
            for (int k = 0; k < j; ++k) {
                A[j][k] = A[k][j];
            }
            A[j][j] += penalty[j] * penalty[j] + 1e-6f;
            rhs[j] -= penalty[j] * penalty[j] * (u[j] - anchor[j]);
        }

        float step[UW_NUM_ACTUATORS];
        if (uw_solve8(A, rhs, step) != 0) {
            break;
        }
        for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
            step[j] = uw_clampf(step[j], low[j] - u[j], high[j] - u[j]);
        }

        /* 代价：加权力误差 + 平滑正则 */
        float old_cost = 0.0f;
        float err[UW_NUM_WRENCH];
        for (int r = 0; r < UW_NUM_WRENCH; ++r) {
            err[r] = weights[r] * (w0[r] - target[r]);
            old_cost += err[r] * err[r];
        }
        for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
            const float ch = penalty[j] * (u[j] - anchor[j]);
            old_cost += ch * ch;
        }

        int accepted = 0;
        static const float fractions[4] = {1.0f, 0.5f, 0.25f, 0.125f};
        for (int f = 0; f < 4 && !accepted; ++f) {
            float candidate[UW_NUM_ACTUATORS];
            float Tt[UW_NUM_ROTORS], th[UW_NUM_ROTORS];
            float wc[UW_NUM_WRENCH];
            float cost = 0.0f;
            for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
                candidate[j] = uw_clampf(u[j] + fractions[f] * step[j], low[j], high[j]);
            }
            for (int i = 0; i < UW_NUM_ROTORS; ++i) {
                Tt[i] = candidate[i];
                th[i] = candidate[4 + i];
            }
            uw_body_wrench(v, Tt, th, health, wc);
            for (int r = 0; r < UW_NUM_WRENCH; ++r) {
                const float e = weights[r] * (wc[r] - target[r]);
                cost += e * e;
            }
            for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
                const float ch = penalty[j] * (candidate[j] - anchor[j]);
                cost += ch * ch;
            }
            if (cost < old_cost - 1e-12f) {
                for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
                    u[j] = candidate[j];
                }
                accepted = 1;
            }
        }

        float step_norm = 0.0f;
        for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
            step_norm += step[j] * step[j];
        }
        if (!accepted || step_norm < 1e-14f) {
            break;
        }
    }

    /* ---- 4. 结果与诊断 ---- */
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        T[i] = u[i];
        theta[i] = u[4 + i];
    }
    float B[UW_NUM_WRENCH][UW_NUM_ACTUATORS];
    uw_effectiveness(v, T, theta, health, B);
    uw_body_wrench(v, T, theta, health, out->achieved_wrench);

    float norm = 0.0f;
    for (int r = 0; r < UW_NUM_WRENCH; ++r) {
        out->residual[r] = target[r] - out->achieved_wrench[r];
        const float e = weights[r] * out->residual[r];
        norm += e * e;
    }
    out->normalized_error = sqrtf(norm);
    out->local_rank = uw_matrix_rank(B);

    int active = 0;
    for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
        if (u[j] <= low[j] + 1e-6f || u[j] >= high[j] - 1e-6f) {
            ++active;
        }
        out->command[j] = u[j];
    }
    out->active_bounds = active;
    out->iterations = iterations > v->max_iterations ? v->max_iterations : iterations;

    int projected = 0;
    for (int j = 0; j < UW_NUM_ACTUATORS; ++j) {
        if (fabsf(anchor[j] - previous[j]) > 1e-9f) {
            projected = 1;
            break;
        }
    }
    out->forced_projection = projected;

    memcpy(a->previous, u, sizeof(u));
    a->has_previous = 1;
}
