# INAV 水下运控模态 —— 集成与构建

本目录是项目在 **INAV 7.1.2** 上的二次开发产物，为同一套飞控增加**水下运控模态**：
沿用 INAV 的姿态估计、遥控解析与执行链路，仅在水下工况下切换控制律与矢量分配，
支持介质切换时的**指令连续过渡（bumpless）**。

## 1. 目录结构

```
firmware/inav/
├── APPLY.md                     # 本文件
├── config/underwater_cli.txt    # 模式开关 / SBUS 舵机 / 双套 PID 的 CLI 模板
├── integration/
│   └── 0001-underwater-control-mode.patch   # 相对 INAV 7.1.2 的集成补丁
└── src/main/underwater/
    ├── underwater.c/.h              # INAV 集成层（模式标志、增益写入、SBUS 下发）
    ├── underwater_allocator.c/.h    # 逐次线性化 + 有界最小二乘联合分配
    ├── underwater_mode.c/.h         # 介质监督器（双套 PID + 无扰过渡 + 倾转限速）
    ├── underwater_config.c          # 默认整机参数与双套增益定义
    └── underwater_params.h          # 集中参数
```

纯算法部分（`underwater_allocator` / `underwater_mode` / `underwater_config` /
`underwater_params.h`）不依赖 INAV，可在主机上独立测试。

## 2. 主机侧验证

```bash
gcc -std=c11 -Wall -Wextra -O2 \
    -I firmware/inav/src/main/underwater \
    firmware/inav/src/main/underwater/underwater_allocator.c \
    firmware/inav/src/main/underwater/underwater_mode.c \
    firmware/inav/src/main/underwater/underwater_config.c \
    firmware/inav/tests/test_underwater.c -lm -o underwater_test
./underwater_test
```

覆盖：空中倾转回中、水中前推倾角、推力变化率约束、已知效能下降补偿、介质无扰过渡。

## 3. 集成到 INAV 7.1.2

```bash
# 1) 取得官方源码
git clone --branch 7.1.2 https://github.com/iNavFlight/inav.git inav-7.1.2
cd inav-7.1.2

# 2) 拷入新模块
cp -r /path/to/repo/firmware/inav/src/main/underwater src/main/underwater

# 3) 应用集成补丁（新增 BOXUNDERWATER / UNDERWATER_MODE，
#    在 fc_core.c 挂接模式与混控钩子，并把新文件加入 CMake 源列表）
git apply /path/to/repo/firmware/inav/integration/0001-underwater-control-mode.patch

# 4) 编译（以 SPEEDYBEEF7V3 为例）
mkdir -p build && cd build
export PATH=/path/to/xpack-arm-none-eabi-gcc/bin:$PATH
cmake ..
make SPEEDYBEEF7V3
```

补丁改动点：

| 文件 | 改动 |
|---|---|
| `fc/rc_modes.h` | 新增 `BOXUNDERWATER = 56` |
| `fc/fc_msp_box.c` | 注册盒名 `"UNDERWATER"`（permanentId 65） |
| `fc/runtime_config.h` | 新增 `UNDERWATER_MODE` 飞行模式标志 |
| `fc/fc_core.c` | `processRx()` 调用 `underwaterUpdateMode()`；`taskMainPidLoop()` 在 `mixTable()` 后调用 `underwaterMixUpdate()` |
| `CMakeLists.txt` | 纳入 `underwater/` 四个源文件 |

## 4. 使能

在 CLI 中按 `config/underwater_cli.txt` 分配 `UNDERWATER` 模式开关、打开 SBUS 舵机输出，
并写入空中/水下两套 PID 初值后 `save`。

## 5. 与参考模型的关系

`underwater_allocator.c` 与 `control/cm_uav/constrained_allocation.py` 采用同一控制效率
模型与代价（任务轴权重、介质推力边界、指令幅值/变化率、已知效能、平滑正则）。差异仅在
数值求解：飞控端以「加权正规方程 + 投影有界步 + 回溯线搜索」替代 Python 的 BVLS，
避免 SVD 与动态内存。
