#pragma once

#include <cstdint>

#include "config.h"

namespace cmuav {

enum class FlightMode { kStandby, kAir, kWater };

// 模态管理与失效保护。
//
// 根据遥控三段开关确定目标模态；在信号丢失超过阈值时进入待命（安全）状态。
class ModeManager {
 public:
  // valid_frame 表示本周期有效新帧；旧缓存不能刷新失联计时。
  FlightMode update(int switch_pos, bool valid_frame, uint32_t now_ms) {
    if (valid_frame) {
      last_valid_ms_ = now_ms;
      signal_ok_ = true;
    } else if (now_ms - last_valid_ms_ > cfg::kFailsafeHoldMs) {
      signal_ok_ = false;
    }

    if (!signal_ok_) {
      mode_ = FlightMode::kStandby;
      return mode_;
    }

    switch (switch_pos) {
      case 2:
        mode_ = FlightMode::kAir;
        break;
      case 0:
        mode_ = FlightMode::kWater;
        break;
      default:
        mode_ = FlightMode::kStandby;
        break;
    }
    return mode_;
  }

  FlightMode mode() const { return mode_; }
  bool isWater() const { return mode_ == FlightMode::kWater; }
  bool isAir() const { return mode_ == FlightMode::kAir; }

 private:
  FlightMode mode_ = FlightMode::kStandby;
  bool signal_ok_ = false;
  uint32_t last_valid_ms_ = 0;
};

}  // namespace cmuav
