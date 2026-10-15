#pragma once

#include <Arduino.h>
#include <sbus.h>

#include "config.h"

namespace cmuav {

// 归一化遥控输入：各通道映射到 [-1, 1]（节流为 [0,1]）。
struct RcCommand {
  float roll = 0.0f;
  float pitch = 0.0f;
  float throttle = 0.0f;
  float yaw = 0.0f;
  int modeSwitch = 1;  // 0=低(水中) 1=中(待命) 2=高(空中)
  bool valid = false;
  uint32_t lastUpdateMs = 0;
};

// SBUS 遥控接收与解析。
class SbusInput {
 public:
  SbusInput();

  void begin();

  // 尝试读取一帧；成功则更新命令并返回 true。
  bool poll();

  const RcCommand& command() const { return cmd_; }

  // 距最近一次有效帧的时间（毫秒）。
  uint32_t ageMs(uint32_t now_ms) const { return now_ms - cmd_.lastUpdateMs; }

 private:
  bfs::SbusRx sbus_rx_;
  RcCommand cmd_;
};

}  // namespace cmuav
