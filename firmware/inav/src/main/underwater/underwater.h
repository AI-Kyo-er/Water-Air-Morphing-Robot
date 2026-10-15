/*
 * underwater.h —— INAV 集成层接口
 *
 * 该层把纯算法模块（underwater_allocator / underwater_mode）接入 INAV 主循环：
 *   - underwaterUpdateMode()：读取 BOXUNDERWATER，维护模式标志与双套 PID 增益；
 *   - underwaterMixUpdate() ：在水中模态下运行约束分配并把倾转角下发到 SBUS 舵机。
 */
#pragma once

/* 在遥控/模式更新之后调用（fc_core.c processRx）。 */
void underwaterUpdateMode(void);

/* 在混控之后调用（fc_core.c taskMainPidLoop）。 */
void underwaterMixUpdate(void);
