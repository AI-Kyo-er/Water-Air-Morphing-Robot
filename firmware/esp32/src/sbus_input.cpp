#include "sbus_input.h"

namespace cmuav {

namespace {

// SBUS 原始值 [172, 1811] -> [-1, 1]
inline float normalize(int raw) {
  constexpr float kSpan = 820.0f;
  float v = (raw - cfg::kChannelMid) / kSpan;
  if (v > 1.0f) v = 1.0f;
  if (v < -1.0f) v = -1.0f;
  return v;
}

// 三段开关判据 -> {0,1,2}
inline int classifySwitch(int raw) {
  if (raw > cfg::kSwitchLowMin && raw < cfg::kSwitchLowMax) return 0;
  if (raw > cfg::kSwitchHighMin && raw < cfg::kSwitchHighMax) return 2;
  return 1;
}

}  // namespace

SbusInput::SbusInput()
    : sbus_rx_(&Serial2, cfg::kSbusRxPin, cfg::kSbusTxPin, true) {}

void SbusInput::begin() { sbus_rx_.Begin(); }

bool SbusInput::poll() {
  if (!sbus_rx_.Read()) {
    return false;
  }
  const bfs::SbusData& d = sbus_rx_.data();
  if (d.failsafe || d.lost_frame) {
    cmd_.valid = false;
    return false;
  }

  cmd_.roll = normalize(d.ch[cfg::kChRoll]);
  cmd_.pitch = normalize(d.ch[cfg::kChPitch]);
  cmd_.yaw = normalize(d.ch[cfg::kChYaw]);
  // 节流：[172,1811] -> [0,1]
  float thr = (d.ch[cfg::kChThrottle] - 172) / (1811.0f - 172.0f);
  cmd_.throttle = thr < 0.0f ? 0.0f : (thr > 1.0f ? 1.0f : thr);
  cmd_.modeSwitch = classifySwitch(d.ch[cfg::kChMode]);
  cmd_.valid = true;
  cmd_.lastUpdateMs = millis();
  return true;
}

}  // namespace cmuav
