// 主机侧正式测试：g++ -std=c++17 -I firmware/esp32/include 此文件 -o 临时输出
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include "command_guard.h"
#include "mode_manager.h"

int main() {
  cmuav::ModeManager manager;
  assert(manager.update(0, false, 0) == cmuav::FlightMode::kStandby);
  assert(manager.update(0, true, 100) == cmuav::FlightMode::kWater);
  assert(manager.update(0, false, 200) == cmuav::FlightMode::kWater);
  assert(manager.update(0, false, 401) == cmuav::FlightMode::kStandby);
  cmuav::ModeManager rollover;
  rollover.update(0, true, UINT32_MAX - 100);
  assert(rollover.update(0, false, 250) == cmuav::FlightMode::kStandby);

  cmuav::ServoCommandGuard guard;
  const float target[4] = {1, 1, 1, 1};
  guard.update(target, 0);
  assert(guard.pulseUs(0) == 1500);
  guard.update(target, 10);
  assert(std::abs(guard.angle(0) - 0.015707963f) < 1e-6f);
  assert(guard.pulseUs(0) > 1500);
  assert(guard.pulseUs(2) < 1500);
  const float before = guard.angle(0);
  guard.update(target, 1010);
  assert(guard.angle(0) - before < 0.032f);
  float invalid[4] = {std::numeric_limits<float>::quiet_NaN(), 0, 0, 0};
  guard.update(invalid, 1020);
  assert(std::isfinite(guard.angle(0)));
  for (uint32_t t = 1030; t < 3000; t += 10) guard.update(target, t);
  assert(guard.pulseUs(0) <= cfg::kServoMaxUs);
  assert(guard.pulseUs(2) >= cfg::kServoMinUs);
  std::cout << "通过固件保护测试：帧新鲜度、计时回绕、限速、镜像与限幅\n";
}
