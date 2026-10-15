/*
 * underwater_mode.c —— 介质模式监督器实现
 */
#include "underwater_mode.h"

#include <math.h>
#include <string.h>

static float uw_blend(float air, float water, float af)
{
    return air * af + water * (1.0f - af);
}

void uw_supervisor_init(uw_supervisor_t *s, const uw_gain_table_t *table,
                        float transition_rate, float max_tilt_rate)
{
    s->table = table;
    s->transition_rate = transition_rate > 0.0f ? transition_rate : 1.0f;
    s->max_tilt_rate = max_tilt_rate > 0.0f ? max_tilt_rate : 1.0f;
    uw_supervisor_reset(s);
}

void uw_supervisor_reset(uw_supervisor_t *s)
{
    s->medium = UW_MEDIUM_AIR;
    s->initialized = 0;
    s->air_fraction = 1.0f;
    s->tilt_hold_valid = 1;
    s->last_ms = 0;
    memset(s->tilt_hold, 0, sizeof(s->tilt_hold));
    s->gains = s->table ? s->table->air : (uw_pid_gains_t){0};
}

int uw_supervisor_update(uw_supervisor_t *s, uw_medium_t commanded, uint32_t now_ms)
{
    float dt = 0.0f;
    if (s->initialized && now_ms >= s->last_ms) {
        dt = (float)(now_ms - s->last_ms) * 1e-3f;
    }
    s->last_ms = now_ms;
    s->initialized = 1;

    const int changed = (commanded != s->medium);
    s->medium = commanded;

    /* 空中授权按过渡速率向目标（空中=1，水中=0）线性逼近 */
    const float target = (commanded == UW_MEDIUM_AIR) ? 1.0f : 0.0f;
    const float max_delta = s->transition_rate * dt;
    float delta = target - s->air_fraction;
    if (delta > max_delta) {
        delta = max_delta;
    } else if (delta < -max_delta) {
        delta = -max_delta;
    }
    s->air_fraction += delta;
    if (s->air_fraction > 1.0f) {
        s->air_fraction = 1.0f;
    } else if (s->air_fraction < 0.0f) {
        s->air_fraction = 0.0f;
    }

    /* 融合双套增益：介质切换时连续变化，等效无扰过渡 */
    if (s->table) {
        for (int i = 0; i < 3; ++i) {
            s->gains.p[i] = uw_blend(s->table->air.p[i], s->table->water.p[i], s->air_fraction);
            s->gains.i[i] = uw_blend(s->table->air.i[i], s->table->water.i[i], s->air_fraction);
            s->gains.d[i] = uw_blend(s->table->air.d[i], s->table->water.d[i], s->air_fraction);
        }
    }
    return changed;
}

float uw_supervisor_air_fraction(const uw_supervisor_t *s)
{
    return s->air_fraction;
}

int uw_supervisor_is_water(const uw_supervisor_t *s)
{
    return s->medium == UW_MEDIUM_WATER;
}

const uw_pid_gains_t *uw_supervisor_gains(const uw_supervisor_t *s)
{
    return &s->gains;
}

void uw_supervisor_slew_tilt(uw_supervisor_t *s, const float tilt_target[UW_NUM_ROTORS],
                             float dt, float tilt_out[UW_NUM_ROTORS])
{
    if (dt <= 0.0f) {
        dt = 1e-3f;
    }
    const float max_step = s->max_tilt_rate * dt;

    if (!s->tilt_hold_valid) {
        memcpy(s->tilt_hold, tilt_target, sizeof(s->tilt_hold));
        s->tilt_hold_valid = 1;
    }

    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        float delta = tilt_target[i] - s->tilt_hold[i];
        if (delta > max_step) {
            delta = max_step;
        } else if (delta < -max_step) {
            delta = -max_step;
        }
        s->tilt_hold[i] += delta;
        tilt_out[i] = s->tilt_hold[i];
    }
}
