/*
 * underwater.c —— INAV 集成层实现（水下运控模态）
 *
 * 说明：本文件依赖 INAV 7.1.2 的既有接口（rc_modes / runtime_config / pid /
 * servo_sbus / rx）。纯算法逻辑位于同目录前三个文件名中，可独立在主机上测试。
 */
#include "platform.h"

#include "build/build_config.h"

#include "common/maths.h"
#include "common/time.h"

#include "fc/config.h"
#include "fc/rc_controls.h"
#include "fc/rc_modes.h"
#include "fc/runtime_config.h"

#include "flight/pid.h"

#include "io/servo_sbus.h"

#include "rx/rx.h"

#include "underwater/underwater.h"
#include "underwater/underwater_allocator.h"
#include "underwater/underwater_mode.h"
#include "underwater/underwater_params.h"

/*---- 静态状态 ---------------------------------------------------------*/
static uw_supervisor_t uwSupervisor;
static uw_allocator_t uwAllocator;
static bool uwInitialized = false;

/* PID 主循环周期（250 Hz）。 */
#define UW_LOOP_DT_S 0.004f

/* SBUS 倾转通道起始序号（对应 ESP32 侧倾转读取通道）。 */
#define UW_SBUS_TILT_BASE 6

/* 把倾转角（弧度）线性映射到 1000..2000 us，零点 1500 us。 */
static uint16_t uwTiltToPulse(float theta)
{
    const float pulse = 1500.0f + theta * (180.0f / (float)M_PI) * (100.0f / 9.0f);
    return (uint16_t)constrainf(pulse, 1000.0f, 2000.0f);
}

static void uwEnsureInit(void)
{
    if (uwInitialized) {
        return;
    }
    uw_supervisor_init(&uwSupervisor, &uw_default_gain_table,
                       2.0f, uw_vehicle_default.tilt_rate_limit);
    uw_allocator_init(&uwAllocator, &uw_vehicle_default);
    uwInitialized = true;
}

/* 把融合后的增益写入当前 PID profile；双套增益共享同一调参口径。 */
static void uwApplyGains(const uw_pid_gains_t *gains)
{
    pidProfileMutable()->pid[PID_ROLL].P = (uint16_t)gains->p[0];
    pidProfileMutable()->pid[PID_ROLL].I = (uint16_t)gains->i[0];
    pidProfileMutable()->pid[PID_ROLL].D = (uint16_t)gains->d[0];
    pidProfileMutable()->pid[PID_PITCH].P = (uint16_t)gains->p[1];
    pidProfileMutable()->pid[PID_PITCH].I = (uint16_t)gains->i[1];
    pidProfileMutable()->pid[PID_PITCH].D = (uint16_t)gains->d[1];
    pidProfileMutable()->pid[PID_YAW].P = (uint16_t)gains->p[2];
    pidProfileMutable()->pid[PID_YAW].I = (uint16_t)gains->i[2];
    pidProfileMutable()->pid[PID_YAW].D = (uint16_t)gains->d[2];
}

void underwaterUpdateMode(void)
{
    uwEnsureInit();

    const uw_medium_t commanded = IS_RC_MODE_ACTIVE(BOXUNDERWATER) ? UW_MEDIUM_WATER
                                                                   : UW_MEDIUM_AIR;
    if (uw_supervisor_update(&uwSupervisor, commanded, millis())) {
        /* 介质切换：置/清飞行模式标志，供 OSD、混控与日志使用 */
        if (commanded == UW_MEDIUM_WATER) {
            ENABLE_FLIGHT_MODE(UNDERWATER_MODE);
        } else {
            DISABLE_FLIGHT_MODE(UNDERWATER_MODE);
        }
    }

    uwApplyGains(uw_supervisor_gains(&uwSupervisor));
}

void underwaterMixUpdate(void)
{
    if (!uwInitialized || !FLIGHT_MODE(UNDERWATER_MODE)) {
        return;
    }

    const float dt = UW_LOOP_DT_S;

    /* 由遥控意图构造期望广义力（手动/增稳模式）；
     * 全自主模式下应由位置/姿态控制器输出的广义力替代此处。 */
    const float forward = (float)rcCommand[PITCH] / 500.0f * uw_vehicle_default.force_scale_n;
    const float vertical = 2.0f * (float)rcCommand[THROTTLE] / 1000.0f * uw_vehicle_default.thrust_max_water;
    const float roll_torque = (float)rcCommand[ROLL] / 500.0f * uw_vehicle_default.torque_scale_nm;
    const float target[UW_NUM_WRENCH] = {
        forward, 0.0f, vertical, roll_torque, 0.0f, 0.0f
    };

    uw_allocation_t result;
    uw_allocate(&uwAllocator, target, UW_MEDIUM_WATER, dt, NULL,
                uw_supervisor_air_fraction(&uwSupervisor), &result);

    /* 倾转角限速后下发到 SBUS 舵机通道，保持介质切换指令连续 */
    float tiltOut[UW_NUM_ROTORS];
    uw_supervisor_slew_tilt(&uwSupervisor, result.command + 4, dt, tiltOut);
    for (int i = 0; i < UW_NUM_ROTORS; ++i) {
        sbusServoUpdate(UW_SBUS_TILT_BASE + i, uwTiltToPulse(tiltOut[i]));
    }
}
