#pragma once

#include <cstdint>

// =====================================================================
// ESP32 矢量推进控制固件 —— 集中配置
// 所有可调参数集中于此，禁止在业务代码中硬编码魔数。
// =====================================================================

// ---- 舵机 / PWM ----
namespace cfg {

// 四个倾转舵机的 PWM 输出引脚（与硬件接线一致）
constexpr int kServoPins[4] = {13, 12, 14, 27};
// 模式指示 / 备用通道
constexpr int kAuxPin = 4;

constexpr int kServoFreqHz = 50;
constexpr int kServoResolutionBits = 16;
constexpr int kServoMinUs = 500;    // 0°
constexpr int kServoMaxUs = 2500;   // 180°
constexpr int kServoMidUs = 1500;   // 90°（机械中位）

// 倾转角限幅（度），与水空控制算法一致
constexpr float kTiltLimitDeg = 60.0f;
constexpr float kTiltRateDegS = 90.0f;
// 原始实机代码中，第三、第四舵机的安装方向与前两个相反。
constexpr float kServoDirection[4] = {1.0f, 1.0f, -1.0f, -1.0f};
constexpr int kServoNeutralUs[4] = {1500, 1500, 1500, 1500};
constexpr float kServoUsPerDeg = 100.0f / 9.0f;
constexpr uint32_t kGuardMaxStepMs = 20;

// ---- SBUS ----
constexpr int kSbusRxPin = 16;
constexpr int kSbusTxPin = 17;

// ---- 遥控通道映射（0-based，对应 CH1..）----
constexpr int kChRoll = 0;
constexpr int kChPitch = 1;
constexpr int kChThrottle = 2;
constexpr int kChYaw = 3;
constexpr int kChMode = 9;  // 三段开关：空中 / 水中 / 待命

// 三段开关判据（原始 SBUS 值 172..1811）
constexpr int kSwitchLowMin = 100;
constexpr int kSwitchLowMax = 300;
constexpr int kSwitchHighMin = 1700;
constexpr int kSwitchHighMax = 1900;
constexpr int kChannelMid = 992;

// ---- 控制分配 ----
constexpr int kNumRotors = 4;
// 力臂（米）
constexpr float kArmX = 0.30f;
constexpr float kArmY = 0.30f;
// 反扭矩系数
constexpr float kReactionTorqueCoeff = 0.02f;

// 推力限幅（归一化 0..1 的电机指令上限）
constexpr float kThrustMaxAir = 0.85f;
constexpr float kThrustMaxWater = 1.00f;

// 水中前推映射：遥控俯仰量 -> 共倾转角
constexpr float kTiltGainRad = 0.9f;   // 满舵时的倾转角（弧度）
constexpr float kYawGain = 0.35f;      // 差动推力偏航增益
constexpr float kRollGain = 0.25f;

// ---- 失效保护 ----
constexpr int kFailsafeFrames = 10;    // 连续丢帧阈值
constexpr int kFailsafeHoldMs = 300;   // 丢信号后保持时间

}  // namespace cfg
