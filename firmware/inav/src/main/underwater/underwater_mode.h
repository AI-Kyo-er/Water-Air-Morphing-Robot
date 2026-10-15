/*
 * underwater_mode.h —— 介质模式监督器（双套 PID + 无扰切换 + 倾转限速）
 *
 * 飞控以遥控通道判定「空中 / 水下」模态；本模块负责：
 *   1. 维护当前目标介质并输出空中授权 air_fraction（1=空中，0=水中）；
 *   2. 在介质切换时按速率交叉淡入双套 PID 增益，避免指令跳变（bumpless）；
 *   3. 对倾转角指令做速率限制，并保留上周期状态以支持切换连续性。
 */
#pragma once

#include <stdint.h>

#include "underwater_params.h"

typedef struct uw_supervisor_s {
    const uw_gain_table_t *table;
    uw_medium_t medium;
    int initialized;
    float air_fraction;   /* 1=完全空中，0=完全水中 */
    float transition_rate;/* 每秒过渡比例 */
    float max_tilt_rate;  /* 倾转角指令限速 rad/s */
    uw_pid_gains_t gains; /* 当前（融合后）增益 */
    float tilt_hold[UW_NUM_ROTORS];
    int tilt_hold_valid;
    uint32_t last_ms;
} uw_supervisor_t;

void uw_supervisor_init(uw_supervisor_t *s, const uw_gain_table_t *table,
                        float transition_rate, float max_tilt_rate);
void uw_supervisor_reset(uw_supervisor_t *s);

/* 按目标介质推进一个控制周期；返回 1 表示本周期介质发生变化。 */
int uw_supervisor_update(uw_supervisor_t *s, uw_medium_t commanded, uint32_t now_ms);

float uw_supervisor_air_fraction(const uw_supervisor_t *s);
int uw_supervisor_is_water(const uw_supervisor_t *s);
const uw_pid_gains_t *uw_supervisor_gains(const uw_supervisor_t *s);

/* 对四路倾转角指令做速率限制，输出平滑且连续的舵机目标。 */
void uw_supervisor_slew_tilt(uw_supervisor_t *s, const float tilt_target[UW_NUM_ROTORS],
                             float dt, float tilt_out[UW_NUM_ROTORS]);
