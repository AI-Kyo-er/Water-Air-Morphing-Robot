/*
 * test_underwater.c —— INAV 水下运控模块的主机侧验证
 *
 * 构建（在仓库根目录）：
 *   gcc -std=c11 -Wall -Wextra -O2 \
 *       -I firmware/inav/src/main/underwater \
 *       firmware/inav/src/main/underwater/underwater_allocator.c \
 *       firmware/inav/src/main/underwater/underwater_mode.c \
 *       firmware/inav/src/main/underwater/underwater_config.c \
 *       firmware/inav/tests/test_underwater.c -lm -o underwater_test
 *   ./underwater_test
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "underwater_allocator.h"
#include "underwater_mode.h"
#include "underwater_params.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int approx(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

static void test_air_hover(void)
{
    uw_allocator_t alloc;
    uw_allocator_init(&alloc, &uw_vehicle_default);

    const float target[UW_NUM_WRENCH] = {0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f};
    uw_allocation_t res;

    /* 跑 2 秒收敛，空中授权强制倾转回中 */
    for (int k = 0; k < 50; ++k) {
        uw_allocate(&alloc, target, UW_MEDIUM_AIR, 0.04f, NULL, 1.0f, &res);
    }

    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        assert(approx(res.command[4 + i], 0.0f, 1e-4f)); /* 空中倾转锁定中位 */
        assert(approx(res.command[i], 5.0f, 0.3f));      /* 20N / 4 ≈ 5N */
    }
    assert(res.local_rank <= 5); /* 平行倾转构型无直接横向推力 */
    printf("[ok] 空中悬停：倾转回中，单桨 ~%.2f N\n", res.command[0]);
}

static void test_water_forward(void)
{
    uw_allocator_t alloc;
    uw_allocator_init(&alloc, &uw_vehicle_default);

    const float target[UW_NUM_WRENCH] = {4.0f, 0.0f, 12.0f, 0.0f, 0.0f, 0.0f};
    uw_allocation_t res;
    for (int k = 0; k < 80; ++k) {
        uw_allocate(&alloc, target, UW_MEDIUM_WATER, 0.04f, NULL, 0.0f, &res);
    }

    const float theta_expected = atan2f(4.0f, 12.0f);
    assert(res.command[4] > 0.1f);
    assert(res.command[4] <= uw_vehicle_default.tilt_limit + 1e-4f);
    assert(fabsf(res.command[4] - theta_expected) < 0.25f);
    assert(res.normalized_error < 2.0f);
    assert(res.local_rank <= 5);
    printf("[ok] 水中前推：倾转角 ~%.1f deg（期望 %.1f deg）\n",
           res.command[4] * 180.0f / (float)M_PI,
           theta_expected * 180.0f / (float)M_PI);
}

static void test_rate_limit(void)
{
    uw_allocator_t alloc;
    uw_allocator_init(&alloc, &uw_vehicle_default);

    const float target[UW_NUM_WRENCH] = {0.0f, 0.0f, 60.0f, 0.0f, 0.0f, 0.0f};
    uw_allocation_t res;
    float prev = 0.0f;
    const float dt = 0.04f;
    for (int k = 0; k < 20; ++k) {
        uw_allocate(&alloc, target, UW_MEDIUM_WATER, dt, NULL, 0.0f, &res);
        if (k > 0) {
            /* 单周期推力增量不超过 thrust_rate * dt */
            assert(res.command[0] - prev <= uw_vehicle_default.thrust_rate_n_s * dt + 1e-3f);
        }
        prev = res.command[0];
    }
    printf("[ok] 推力变化率约束：单周期增量 <= %.2f N\n",
           uw_vehicle_default.thrust_rate_n_s * dt);
}

static void test_effectiveness_loss(void)
{
    uw_allocator_t alloc;
    uw_allocator_init(&alloc, &uw_vehicle_default);

    const float target[UW_NUM_WRENCH] = {0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f};
    const float health[UW_NUM_ROTORS] = {0.5f, 1.0f, 1.0f, 1.0f};
    uw_allocation_t res;
    for (int k = 0; k < 80; ++k) {
        uw_allocate(&alloc, target, UW_MEDIUM_WATER, 0.04f, health, 0.0f, &res);
    }
    /* 已知一阶效能下降：为维持合力，总指令推力必须高于无损失时的 20 N */
    float total = 0.0f;
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        assert(res.command[i] >= -1e-4f);
        assert(res.command[i] <= uw_vehicle_default.thrust_max_water + 1e-3f);
        total += res.command[i];
    }
    assert(total > 21.0f);
    assert(res.normalized_error < 1.0f);
    printf("[ok] 已知效能下降：总指令推力 %.2f N > 20 N（补偿受损桨）\n", total);
}

static void test_medium_transition(void)
{
    uw_supervisor_t sup;
    uw_supervisor_init(&sup, &uw_default_gain_table, 2.0f, uw_vehicle_default.tilt_rate_limit);
    assert(uw_supervisor_update(&sup, UW_MEDIUM_AIR, 0) == 0);
    assert(approx(uw_supervisor_air_fraction(&sup), 1.0f, 1e-6f));

    /* 切换到水中：空中授权应单调下降，0.5 s 后到 0 */
    int changed = 0;
    float last = 1.0f;
    for (uint32_t t = 10; t <= 600; t += 10) {
        changed += uw_supervisor_update(&sup, UW_MEDIUM_WATER, t);
        const float af = uw_supervisor_air_fraction(&sup);
        assert(af <= last + 1e-6f);
        last = af;
    }
    assert(changed == 1);
    assert(approx(uw_supervisor_air_fraction(&sup), 0.0f, 1e-6f));
    assert(uw_supervisor_is_water(&sup));
    /* 融合增益应等于水中增益 */
    assert(approx(uw_supervisor_gains(&sup)->p[0], uw_default_gain_table.water.p[0], 1e-3f));

    /* 倾转指令限速，保证介质切换指令连续 */
    const float tilt_target[UW_NUM_ROTORS] = {1.0f, 1.0f, 1.0f, 1.0f};
    float tilt_out[UW_NUM_ROTORS];
    uw_supervisor_slew_tilt(&sup, tilt_target, 0.1f, tilt_out);
    assert(approx(tilt_out[0], uw_vehicle_default.tilt_rate_limit * 0.1f, 1e-4f));
    printf("[ok] 介质切换：无扰过渡 air_fraction 1->0，倾转限速 %.3f rad/step\n", tilt_out[0]);
}

int main(void)
{
    test_air_hover();
    test_water_forward();
    test_rate_limit();
    test_effectiveness_loss();
    test_medium_transition();
    printf("通过 INAV 水下运控模块测试：分配、约束、效能与介质过渡\n");
    return 0;
}
