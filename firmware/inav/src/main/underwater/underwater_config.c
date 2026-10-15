/*
 * underwater_config.c —— 默认整机参数与双套增益的实际定义
 */
#include "underwater_params.h"

const uw_vehicle_t uw_vehicle_default = {
    .positions = {
        {0.30f, 0.30f, 0.0f},
        {-0.30f, 0.30f, 0.0f},
        {-0.30f, -0.30f, 0.0f},
        {0.30f, -0.30f, 0.0f},
    },
    .spin_signs = {1.0f, -1.0f, 1.0f, -1.0f},
    .tilt_directions = {
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
    },
    .thrust_max_air = 13.7f,
    .thrust_max_water = 16.0f,
    .reaction_torque_coeff = 0.02f,
    .tilt_limit = 1.04719756f,     /* 60 deg */
    .tilt_rate_limit = 1.57079633f,/* 90 deg/s */
    .force_scale_n = 12.0f,
    .torque_scale_nm = 1.0f,
    .axis_priority = {1.0f, 0.0f, 3.0f, 3.0f, 3.0f, 1.5f},
    .thrust_rate_n_s = 80.0f,
    .smoothness = 0.002f,
    .max_iterations = 5,
};

const uw_gain_table_t uw_default_gain_table = {
    .air = {
        .p = {45.0f, 45.0f, 45.0f},
        .i = {30.0f, 30.0f, 45.0f},
        .d = {20.0f, 20.0f, 0.0f},
    },
    .water = {
        /* 水中附加质量与阻尼显著，采用更低的速率环增益 */
        .p = {30.0f, 30.0f, 35.0f},
        .i = {25.0f, 25.0f, 35.0f},
        .d = {10.0f, 10.0f, 0.0f},
    },
};
