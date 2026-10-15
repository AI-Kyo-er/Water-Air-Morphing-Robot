#pragma once

#include <Arduino.h>

#include "config.h"
#include "sbus_input.h"

namespace cmuav {

// 机体系期望力 / 力矩。
struct BodyWrench {
  float fx = 0.0f;
  float fy = 0.0f;
  float fz = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
  float tz = 0.0f;
};

// 执行量：四桨推力（归一化 0..1）+ 四倾转角（弧度）。
struct ActuatorCommand {
  float thrust[cfg::kNumRotors] = {0, 0, 0, 0};
  float tilt[cfg::kNumRotors] = {0, 0, 0, 0};
};

// 手动共倾转映射参考。
//
// 完整的水下联合分配（逐次线性化 + 有界最小二乘）在飞控侧实现，见
// firmware/inav/src/main/underwater/underwater_allocator.c；本固件负责把倾转角
// 指令落地为四路舵机 PWM，并在失联/切换时保证执行连续。
//
// 本类的结构化映射用于无飞控分配时的降级手动模式：
//   1. 共倾转角 theta = atan2(Fx, Fz)，把总推力矢量指向前向/垂直需求；
//   2. 四桨差动推力产生滚转 / 俯仰 / 偏航力矩。
class ControlAllocator {
 public:
  // 由期望力/力矩计算执行量。water_mode=false 时锁定倾转角（空中）。
  ActuatorCommand allocate(const BodyWrench& w, bool water_mode) const;

  // 由遥控指令直接生成执行量（手动飞行 / 航行）。
  ActuatorCommand fromRc(const RcCommand& rc, bool water_mode) const;

 private:
  static float clampf(float v, float lo, float hi);
};

}  // namespace cmuav
