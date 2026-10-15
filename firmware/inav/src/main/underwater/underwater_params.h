/*
 * underwater_params.h —— 水下运控模态的集中参数
 *
 * 本文件是 INAV 二次开发模块的唯一参数入口，数值与 control/configs 下的
 * 参考模型保持一致（单位：m, kg, s, rad, N, N·m）。禁止在算法代码中散落魔数。
 */
#pragma once

#define UW_NUM_ROTORS 4
#define UW_NUM_ACTUATORS 8 /* [T1..T4, theta1..theta4] */
#define UW_NUM_WRENCH 6    /* [Fx, Fy, Fz, Mx, My, Mz] */

typedef enum uw_medium_e {
    UW_MEDIUM_AIR = 0,
    UW_MEDIUM_WATER = 1
} uw_medium_t;

typedef struct uw_vehicle_s {
    float positions[UW_NUM_ROTORS][3];      /* 推进单元机体系安装位置 */
    float spin_signs[UW_NUM_ROTORS];        /* 旋翼自转方向 */
    float tilt_directions[UW_NUM_ROTORS][3];/* 倾转时推力趋向方向（已归一化） */
    float thrust_max_air;                   /* 单桨空中最大推力 N */
    float thrust_max_water;                 /* 单桨水中最大推力 N */
    float reaction_torque_coeff;            /* 反扭矩系数 k_Q */
    float tilt_limit;                       /* 倾转角幅值上限 rad */
    float tilt_rate_limit;                  /* 倾转角速率上限 rad/s */

    float force_scale_n;                    /* 力归一化尺度 */
    float torque_scale_nm;                  /* 力矩归一化尺度 */
    float axis_priority[UW_NUM_WRENCH];     /* 任务轴优先级权重 */
    float thrust_rate_n_s;                  /* 推力变化率上限 N/s */
    float smoothness;                       /* 指令平滑正则权重 */
    int max_iterations;                     /* 逐次线性化迭代上限 */
} uw_vehicle_t;

/* 与 control/configs/vehicle_default.yaml + research.yaml 对齐的默认整机参数。 */
extern const uw_vehicle_t uw_vehicle_default;

/* 双套 PID 增益（对应 INAV 空中 / 水下两个 PID profile 槽位）。
 * 数值采用 INAV 原生量纲（P/I/D ×1000），由补丁写入 pidProfile。 */
typedef struct uw_pid_gains_s {
    float p[3]; /* roll, pitch, yaw */
    float i[3];
    float d[3];
} uw_pid_gains_t;

typedef struct uw_gain_table_s {
    uw_pid_gains_t air;
    uw_pid_gains_t water;
} uw_gain_table_t;

/* 与 firmware 内配置文件对应的双套增益初值，介质切换时无扰过渡。 */
extern const uw_gain_table_t uw_default_gain_table;
