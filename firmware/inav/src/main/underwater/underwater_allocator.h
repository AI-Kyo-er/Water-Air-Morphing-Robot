/*
 * underwater_allocator.h —— 逐次线性化 + 有界最小二乘的联合推力/倾转角分配
 *
 * 与 control/cm_uav/constrained_allocation.py 采用同一模型与代价：控制效率矩阵
 * 由推进几何（geometry.py）唯一确定，分配器在任务轴权重、介质推力边界、指令
 * 幅值/变化率约束与已知推力效能下求解执行量。
 *
 * 该实现在定点机上以「加权正规方程 + 投影有界步 + 回溯线搜索」近似 Python 的
 * 带界最小二乘（BVLS），不依赖 SVD / 动态内存。
 */
#pragma once

#include "underwater_params.h"

typedef struct uw_allocation_s {
    float command[UW_NUM_ACTUATORS];      /* 实际指令 [T1..T4, th1..th4] */
    float achieved_wrench[UW_NUM_WRENCH]; /* 模型实现的广义力 */
    float residual[UW_NUM_WRENCH];        /* target - achieved */
    float normalized_error;               /* 加权残差范数 */
    int local_rank;                        /* 局部效率矩阵秩（<=5） */
    int active_bounds;                     /* 触界的执行量个数 */
    int iterations;                        /* 实际迭代次数 */
    int forced_projection;                 /* 本周期是否因边界变化被投影 */
} uw_allocation_t;

typedef struct uw_allocator_s {
    const uw_vehicle_t *vehicle;
    float previous[UW_NUM_ACTUATORS];      /* 上周期指令 */
    int has_previous;
} uw_allocator_t;

void uw_allocator_init(uw_allocator_t *a, const uw_vehicle_t *vehicle);

/* 清零历史指令，重新回到中位。 */
void uw_allocator_reset(uw_allocator_t *a);

/*
 * 求解本周期执行量。
 *   target       : 期望广义力 (6,)
 *   medium       : 当前介质（决定推力边界）
 *   dt           : 周期间隔 s
 *   health       : 四路已知推力效能 [0,1]（NULL 视为全 1）
 *   air_fraction : 空中授权，1=完全空中（倾转回中），0=完全水中
 *   out          : 输出分配结果
 */
void uw_allocate(uw_allocator_t *a, const float target[UW_NUM_WRENCH],
                 uw_medium_t medium, float dt, const float health[UW_NUM_ROTORS],
                 float air_fraction, uw_allocation_t *out);
