#include "control_allocator.h"

#include <cmath>

namespace cmuav {

float ControlAllocator::clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

ActuatorCommand ControlAllocator::allocate(const BodyWrench& w,
                                           bool water_mode) const {
  ActuatorCommand out;
  const float tilt_limit = cfg::kTiltLimitDeg * (float)M_PI / 180.0f;

  // ---- 1. 共倾转角：由前向 / 垂直力方向确定 ----
  float theta = 0.0f;
  if (water_mode) {
    theta = std::atan2(w.fx, fmaxf(w.fz, 1e-3f));
    theta = clampf(theta, -tilt_limit, tilt_limit);
  }

  // ---- 2. 公共推力基底与差动 ----
  const float t_total = std::sqrt(w.fx * w.fx + w.fz * w.fz);
  const float t_max = water_mode ? cfg::kThrustMaxWater : cfg::kThrustMaxAir;
  const float base = t_total / cfg::kNumRotors;

  // 倾转后垂直力臂会按 cos(theta) 缩放，须做补偿；cos 过小处限幅防止发散
  const float cos_t = fmaxf(std::cos(theta), 0.2f);
  const float d_roll = w.tx / (4.0f * cfg::kArmY * cos_t);
  const float d_pitch = -w.ty / (4.0f * cfg::kArmX * cos_t);
  const float d_yaw = w.tz / (4.0f * cfg::kReactionTorqueCoeff);

  // X 型混控：位置 (±a, ±a)
  out.thrust[0] = base + d_roll + d_pitch + d_yaw;
  out.thrust[1] = base + d_roll - d_pitch - d_yaw;
  out.thrust[2] = base - d_roll - d_pitch + d_yaw;
  out.thrust[3] = base - d_roll + d_pitch - d_yaw;

  for (int i = 0; i < cfg::kNumRotors; ++i) {
    out.thrust[i] = clampf(out.thrust[i], 0.0f, t_max);
    out.tilt[i] = theta;
  }
  return out;
}

ActuatorCommand ControlAllocator::fromRc(const RcCommand& rc,
                                         bool water_mode) const {
  ActuatorCommand out;

  if (!water_mode) {
    // 空中：标准四旋翼混控，旋翼保持水平
    const float base = rc.throttle;
    const float d_roll = rc.roll * cfg::kRollGain;
    const float d_pitch = rc.pitch * cfg::kRollGain;
    const float d_yaw = rc.yaw * cfg::kYawGain;
    out.thrust[0] = clampf(base + d_roll + d_pitch + d_yaw, 0.0f, cfg::kThrustMaxAir);
    out.thrust[1] = clampf(base + d_roll - d_pitch - d_yaw, 0.0f, cfg::kThrustMaxAir);
    out.thrust[2] = clampf(base - d_roll - d_pitch + d_yaw, 0.0f, cfg::kThrustMaxAir);
    out.thrust[3] = clampf(base - d_roll + d_pitch - d_yaw, 0.0f, cfg::kThrustMaxAir);
    for (int i = 0; i < cfg::kNumRotors; ++i) out.tilt[i] = 0.0f;
    return out;
  }

  // 水中：俯仰量 -> 共倾转角（前推/后退）；滚转、偏航 -> 差动推力
  float theta = rc.pitch * cfg::kTiltGainRad;
  const float tilt_limit = cfg::kTiltLimitDeg * (float)M_PI / 180.0f;
  theta = clampf(theta, -tilt_limit, tilt_limit);

  const float base = rc.throttle * cfg::kThrustMaxWater;
  const float d_roll = rc.roll * cfg::kRollGain;
  const float d_yaw = rc.yaw * cfg::kYawGain;
  out.thrust[0] = clampf(base + d_roll + d_yaw, 0.0f, cfg::kThrustMaxWater);
  out.thrust[1] = clampf(base + d_roll - d_yaw, 0.0f, cfg::kThrustMaxWater);
  out.thrust[2] = clampf(base - d_roll + d_yaw, 0.0f, cfg::kThrustMaxWater);
  out.thrust[3] = clampf(base - d_roll - d_yaw, 0.0f, cfg::kThrustMaxWater);
  for (int i = 0; i < cfg::kNumRotors; ++i) out.tilt[i] = theta;
  return out;
}

}  // namespace cmuav
