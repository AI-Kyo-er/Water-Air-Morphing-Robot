// =====================================================================
// ESP32 矢量推进倾转控制固件 —— 主程序
//
// 职责：
//   1. 解析遥控 SBUS 信号；
//   2. 依据三段开关在「空中 / 水中 / 待命」间切换；
//   3. 调用手动指令映射器计算四个
//      倾转舵机的目标角度，输出 50 Hz PWM 驱动舵机。
//
// 说明：电机推力分配由飞控（INAV）负责；本固件负责矢量倾转通道。
//       水中模态下，俯仰摇杆映射为共倾转角，实现四桨同向矢量前推。
// =====================================================================

#include <Arduino.h>

#include "config.h"
#include "command_guard.h"
#include "control_allocator.h"
#include "mode_manager.h"
#include "sbus_input.h"

using namespace cmuav;

namespace {

SbusInput g_sbus;
ControlAllocator g_allocator;
ModeManager g_mode;
ServoCommandGuard g_guard;

// 脉宽（微秒）-> LEDC 占空比（0..65535，50 Hz / 16 bit）
uint32_t pulseToDuty(int pulse_us) {
  return (uint32_t)((float)pulse_us / 20000.0f * 65535.0f);
}

void writeTilt(const ActuatorCommand& cmd) {
  g_guard.update(cmd.tilt, millis());
  for (int i = 0; i < cfg::kNumRotors; ++i) {
    ledcWrite(i + 1, pulseToDuty(g_guard.pulseUs(i)));
  }
}

void setupServos() {
  for (int i = 0; i < cfg::kNumRotors; ++i) {
    pinMode(cfg::kServoPins[i], OUTPUT);
    ledcSetup(i + 1, cfg::kServoFreqHz, cfg::kServoResolutionBits);
    ledcAttachPin(cfg::kServoPins[i], i + 1);
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  setupServos();
  g_sbus.begin();

  // 上电默认回到中位，避免舵机乱动
  for (int i = 0; i < cfg::kNumRotors; ++i) {
    ledcWrite(i + 1, pulseToDuty(cfg::kServoMidUs));
  }
  Serial.println("CM-UAV vector tilt controller ready.");
}

void loop() {
  const bool fresh = g_sbus.poll();
  const RcCommand& rc = g_sbus.command();
  const FlightMode mode = g_mode.update(rc.modeSwitch, fresh && rc.valid, millis());

  ActuatorCommand cmd;
  switch (mode) {
    case FlightMode::kAir:
    case FlightMode::kWater:
      cmd = g_allocator.fromRc(rc, mode == FlightMode::kWater);
      break;
    case FlightMode::kStandby:
    default:
      // 安全：倾转回中位
      for (int i = 0; i < cfg::kNumRotors; ++i) cmd.tilt[i] = 0.0f;
      break;
  }

  writeTilt(cmd);

  if (fresh) {
    Serial.printf("mode=%d tilt=%.1fdeg thr=%.2f\n", (int)mode,
                  cmd.tilt[0] * 180.0f / (float)M_PI, cmd.thrust[0]);
  }
  delay(2);
}
