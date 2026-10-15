#pragma once

#include <cmath>
#include <cstdint>

#include "config.h"

namespace cmuav {

// 倾转指令最后一层：限幅、限速、镜像安装校准，单位 rad / us。
class ServoCommandGuard {
 public:
  void update(const float target_rad[4], uint32_t now_ms) {
    if (!initialized_) {
      initialized_ = true;
      last_ms_ = now_ms;
      return;
    }
    uint32_t elapsed = now_ms - last_ms_;
    if (elapsed > cfg::kGuardMaxStepMs) elapsed = cfg::kGuardMaxStepMs;
    last_ms_ = now_ms;
    const float rad_per_deg = 3.14159265358979323846f / 180.0f;
    const float limit = cfg::kTiltLimitDeg * rad_per_deg;
    const float step = cfg::kTiltRateDegS * rad_per_deg * elapsed / 1000.0f;
    for (int i = 0; i < 4; ++i) {
      const float target = std::isfinite(target_rad[i])
                               ? clamp(target_rad[i], -limit, limit) : 0.0f;
      angle_[i] += clamp(target - angle_[i], -step, step);
    }
  }

  float angle(int index) const { return angle_[index]; }

  int pulseUs(int index) const {
    const float deg = angle_[index] * 180.0f / 3.14159265358979323846f;
    const float pulse = cfg::kServoNeutralUs[index] +
                        cfg::kServoDirection[index] * cfg::kServoUsPerDeg * deg;
    return static_cast<int>(clamp(pulse, cfg::kServoMinUs, cfg::kServoMaxUs));
  }

 private:
  static float clamp(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
  }

  float angle_[4] = {0, 0, 0, 0};
  uint32_t last_ms_ = 0;
  bool initialized_ = false;
};

}  // namespace cmuav
