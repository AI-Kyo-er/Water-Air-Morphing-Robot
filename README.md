<h1 align="center">潜空智航</h1>
<p align="center">Water-Air Morphing Robot</p>
<p align="center"><strong>水空跨介质机器人 · 倾转推进 · 自主感知与控制</strong></p>
<p align="center">武汉大学创新研究基金 & 国家级大创立项 · 刘胜院士团队指导</p>

<p align="center">
  <a href="#项目团队">项目团队</a> ·
  <a href="#系统与算法">系统与算法</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#机械与硬件">机械与硬件</a> ·
  <a href="#快速开始">快速开始</a>
</p>

## 项目团队

**刘胜**：中国科学院院士，武汉大学工业科学研究院执行院长、微电子学院副院长，IEEE Fellow、ASME Fellow，长期从事集成电路、LED 与微传感器封装及可靠性研究，担任本项目学术指导。

**韦先轶**：本项目团队负责人，武汉大学弘毅学堂计算机科学与技术专业，负责系统总体设计与核心算法实现。

<p align="center">
  <img src="docs/figures/team_photo.png" alt="项目团队合照" width="680">
</p>
<p align="center"><sub>项目团队</sub></p>

## 系统与算法

### 分层控制架构

系统按“大脑—小脑—执行器”分工。树莓派将传感器观测转换为任务与导航目标；飞控根据平台状态计算所需推力和力矩；分配器联合求解四路电机推力与四路倾转角。

| 层级 | 计算任务 | 输出 |
|---|---|---|
| 大脑 · 树莓派 | 目标检测、证据融合、任务调度与路径规划 | 模式、速度、航向、高度/深度目标 |
| 小脑 · INAV 飞控 | 姿态估计、横滚/俯仰/偏航控制、约束分配 | 四路推力与四路倾转角 |
| 执行器 · 电调与 ESP32 | 电机驱动、SBUS 解析、舵机校准与限速 | 桨叶转速与推进方向 |

### 姿态控制与约束分配

控制模型采用**位置—速度—姿态级联**：外环由位置、速度误差计算推进力，姿态环由姿态误差与角速度计算校正力矩。四套推进单元通过差动推力和倾转方向控制横滚、俯仰与偏航。

空中主要通过机体倾斜产生水平推力，旋翼保持回中；水下通过旋翼倾转提供前向推力，减小机体俯仰。介质切换时，连续授权系数调节两种推进方式的占比，空中与水下 PID 增益同步过渡。

分配器根据推进单元位置、推力方向及旋翼反扭矩建立合力与力矩映射，在上一周期指令附近**逐次线性化，以有界最小二乘求解**。推力上限、倾转角、变化率与推进效能共同限定可用执行空间，平滑项抑制电机和舵机指令突变。

飞控在 INAV 7.1.2 上增加水下运控模态，复用姿态估计和执行链路。倾转指令经 SBUS 传至 ESP32，完成镜像校准、角度限速与舵机脉宽映射。

### 视觉感知与任务决策

树莓派运行 **YOLO + ONNX Runtime / OpenCV** 检测链，输出目标类别、检测框与置信度。多帧跟踪保持目标身份，画面位置用于计算方位角；清晰度和曝光参与图像质量评价，视觉与声磁证据按质量、新鲜度和相关性融合。

任务状态机依次组织空中抵达、入水巡检、目标确认和出水返航。目标持续满足确认条件后触发记录与返航，深度、到达状态及电量参与阶段切换。导航采用占据地图与 A* 搜索生成路径，动态窗口方法给出局部速度和偏航率。

<details>
<summary>算法与固件源码</summary>

| 模块 | 代码入口 |
|---|---|
| 级联控制与介质调度 | [controller.py](control/cm_uav/controller.py) |
| 推进几何与约束分配 | [geometry.py](control/cm_uav/geometry.py) · [constrained_allocation.py](control/cm_uav/constrained_allocation.py) |
| INAV 水下运控 | [underwater/](firmware/inav/src/main/underwater/) · [集成与构建](firmware/inav/APPLY.md) |
| ESP32 倾转执行 | [firmware/esp32/](firmware/esp32/) |
| 视觉检测与跟踪 | [perception/vision/](perception/vision/) |
| 可信融合 | [perception/fusion/](perception/fusion/) |
| 任务协调 | [brain.py](autonomy/brain.py) · [mission.py](autonomy/mission.py) |
| 路径规划与局部导航 | [autonomy/navigation/](autonomy/navigation/) |

</details>

## 实验结果

### 飞行与起降

搭载传感器与电子系统的整机完成户外起飞、空中飞行与降落。下图为连续飞行视频中的三个阶段。

<p align="center">
  <img src="docs/figures/flight_takeoff.jpg" alt="整机起飞" width="32%">
  <img src="docs/figures/flight_airborne.jpg" alt="整机空中飞行" width="32%">
  <img src="docs/figures/flight_landing.jpg" alt="整机降落" width="32%">
</p>
<p align="center"><sub>起飞 / 空中飞行 / 降落</sub></p>

### 水中运动与跨介质出水

整机在水池内完成位移与转向，遥控指令经飞控驱动推进单元，密封电子舱保护控制电路。

<p align="center">
  <img src="docs/figures/prototype_pool.jpg" alt="水池测试环境" width="48%">
  <img src="docs/figures/pool_motion_test.jpg" alt="整机水中运动" width="48%">
</p>
<p align="center"><sub>水池测试环境 / 水中运动实拍</sub></p>

出水过程包括水中运行、桨叶加速、突破水面和离水起飞。双速推进单元通过换档适应水中负载与空中转速需求，同一套推进系统完成水空转换。

<p align="center">
  <img src="docs/figures/cross_medium_water.jpg" alt="出水前的水中运行" width="32%">
  <img src="docs/figures/cross_medium_surface.jpg" alt="突破水面" width="32%">
  <img src="docs/figures/cross_medium_airborne.jpg" alt="离水起飞" width="32%">
</p>
<p align="center"><sub>水中运行 / 突破水面 / 离水起飞</sub></p>

### 视觉识别

树莓派完成相机调用、图像显示与 YOLOv5-Lite 实时识别，输出人员、椅子、笔记本电脑等目标的类别、检测框与置信度。

<p align="center">
  <img src="docs/figures/raspberry_camera_capture.png" alt="树莓派相机采集画面" width="48%">
  <img src="docs/figures/vision_realtime_test.jpg" alt="YOLOv5-Lite 实时检测" width="48%">
</p>
<p align="center"><sub>相机采集 / YOLOv5-Lite 目标检测</sub></p>

### 多传感器采集

水听器与磁传感器经 DAQ122 采集模拟量，上位机显示波形与通道读数。

<p align="center">
  <img src="docs/figures/hydrophone_capture.png" alt="水听器采集波形" width="48%">
  <img src="docs/figures/magnetic_capture.png" alt="磁传感器波形与通道读数" width="48%">
</p>
<p align="center"><sub>水声波形 / 磁场信号</sub></p>

P360 声呐在水池内扫描，按距离和方位显示回波，反映周围边界与物体分布。

<p align="center">
  <img src="docs/figures/sonar_capture.jpg" alt="P360 声呐扫描画面" width="48%">
  <img src="docs/figures/sonar_scan_test.jpg" alt="P360 声呐扫描回波" width="48%">
</p>
<p align="center"><sub>声呐扫描 / 实时回波</sub></p>

ESP32 读取深度计，树莓派显示温度、压力与深度信息。

<p align="center">
  <img src="docs/figures/depth_capture.png" alt="深度、压力与温度读数" width="360">
</p>
<p align="center"><sub>树莓派端环境状态读数</sub></p>

### 算法对比与回放

控制分配对比伪逆限幅、共倾转约束分配与本文方法，覆盖常规机动、偏航、效能下降、快速指令和介质切换，并消融速率约束与效能信息。可信融合对比质量与时延变化下的目标判定，视觉—任务—导航链回放跟踪与任务切换过程。

<details>
<summary>查看控制分配、感知融合与任务回放结果</summary>

<p align="center">
  <img src="experiments/control/allocation_benchmark/figures/allocation_comparison.png" alt="控制分配对比" width="780">
</p>
<p align="center"><sub>约束控制分配：基线对比与消融</sub></p>

<p align="center">
  <img src="experiments/perception/reliability_benchmark/figures/reliability_comparison.png" alt="可信感知融合对比" width="780">
</p>
<p align="center"><sub>可信感知融合：质量与时延条件对比</sub></p>

<p align="center">
  <img src="experiments/system/vision_navigation/figures/tracked_marker_029.jpg" alt="视觉跟踪与任务导航回放" width="320">
</p>
<p align="center"><sub>视觉跟踪与任务回放</sub></p>

</details>

## 机械与硬件

面向水空跨介质巡检的倾转旋翼机器人。四套双速推进单元在空气与水中共用，通过电机换向切换传动档位、舵机倾转改变推力方向。树莓派承担感知与任务决策，INAV 飞控完成姿态控制，ESP32 驱动倾转机构。

<p align="center">
  <img src="docs/figures/prototype_assembly.png" alt="潜空智航整机装配" width="720">
</p>
<p align="center"><sub>整机原型：倾转机架、双速推进单元与密封电子舱</sub></p>

### 倾转机架

“工”字形机架以碳方管和 5 mm 碳板构成主体，四套推进单元安装在可转动的碳圆管机臂上。标准碳管切割、开孔，碳板 CNC 加工，连接件与部分支架采用 3D 打印。

舵机经齿轮组驱动碳管，轴承支撑转轴，并为电机线缆留出穿线空间。空中旋翼支撑机体，水下四套推进单元均可倾转参与前向推进。

<p align="center">
  <img src="docs/figures/airframe_model.png" alt="工字形倾转机架模型" width="640">
</p>
<p align="center"><sub>整机机架与四套倾转推进单元</sub></p>

<p align="center">
  <img src="docs/figures/tilt_module_model.png" alt="倾转模块模型" height="150">
  <img src="docs/figures/tilt_module_exploded.png" alt="倾转模块装配分解" height="150">
</p>
<p align="center"><sub>倾转模块 / 轴承与碳管装配</sub></p>

### 双速推进单元

**小齿差齿轮组与单向轴承**构成两档传动。电机正转时直驱，提供空中所需的高转速；反转时切入减速支路，降低桨速、增加扭矩。两条支路交替锁止与释放，使电机换向后桨叶仍保持原旋转方向。

三代结构迭代将大单向轴承由偏心运动改为同心布局，再以两个小轴承减小体积和质量。非标齿轮件采用铝合金 3D 打印，轴类零件 CNC 加工，部分外壳采用 PLA 打印。

<p align="center">
  <img src="docs/figures/gearbox_section.png" alt="双速减速箱剖面" height="260">
  <img src="docs/figures/gearbox_prototype.jpeg" alt="推进单元装配实物" height="260">
</p>
<p align="center"><sub>双速减速箱剖面 / 推进单元实物</sub></p>

### 电子舱与密封

电子舱采用亚克力盖板与 3D 打印壳体，便于拆装和观察内部状态。盖板通过环形槽中的 O 型圈压紧密封，螺丝孔位配合防水硅脂。

出线孔采用硅胶管翻边、压片固定与双组分硅胶灌封，分别密封管壁与壳体、导线与管内间隙。

<p align="center">
  <img src="docs/figures/waterproof_housing_model.png" alt="防水壳体模型" height="200">
  <img src="docs/figures/cable_seal_assembly.jpeg" alt="出线密封组件实物" height="200">
</p>
<p align="center"><sub>电子舱壳体 / 出线密封组件</sub></p>

### 控制硬件与载荷

INAV 飞控连接接收机、GPS 与电调；支持正反转的独立电调配合双速减速箱换档，ESP32 解析 SBUS 并驱动四套倾转舵机。树莓派通过 DAQ122、ESP32 与供电通信 PCB 接入感知载荷。

| 平台参数 | 配置 |
|---|---|
| 整机质量 | 3.65 kg |
| 单推进单元最大空气推力 | 约 1.4 kgf |
| 电池 | 4S 5300 mAh 锂电池 |
| 遥控链路 | 915 MHz，完成约 0.5 m 以内浅水控制测试 |

| 感知载荷 | 采集信息 |
|---|---|
| 水听器 + DAQ122 | 水声波形 |
| 磁力计 + DAQ122 | 磁场变化 |
| 摄像头 | 图像与视觉目标 |
| P360 扫描声呐 | 周围边界与物体回波 |
| MS5837 深度计 + ESP32 | 深度、压力与温度 |

## 目录结构

```text
control/       推进几何、跨介质动力学、约束控制分配与仿真
perception/    视觉检测、声磁检测与可信融合
autonomy/      树莓派任务决策、导航与分层协调
firmware/      INAV 水下运控二次开发与 ESP32 倾转执行固件
hardware/      平台参数与来源
experiments/   控制分配、可信感知与视觉导航结果
docs/          立项申报书与图件
scripts/       测试、结果生成与遥测导入
```

## 快速开始

Python 3.10+，安装依赖：

```bash
pip install -r requirements.txt          # 视觉扩展另加 requirements-vision.txt
python scripts/run_checks.py             # 单元测试与最小验证
python scripts/run_research_suite.py     # 生成控制/感知基准与结果图
```

<details>
<summary>飞控与 ESP32 主机测试命令</summary>

```bash
# INAV 水下运控模块主机测试（详见 firmware/inav/APPLY.md）
gcc -std=c11 -Wall -Wextra -O2 -I firmware/inav/src/main/underwater \
    firmware/inav/src/main/underwater/underwater_allocator.c \
    firmware/inav/src/main/underwater/underwater_mode.c \
    firmware/inav/src/main/underwater/underwater_config.c \
    firmware/inav/tests/test_underwater.c -lm -o underwater_test && ./underwater_test

# ESP32 倾转保护逻辑主机测试
g++ -std=c++17 -Wall -Wextra -I firmware/esp32/include \
    firmware/esp32/tests/test_command_guard.cpp -o command_guard_test && ./command_guard_test
```

</details>

## 说明

- 视觉默认接入本地 YOLO（ONNX Runtime / OpenCV DNN / Ultralytics），未放置权重时回退到
  颜色基线，推理接口与跟踪、质量、证据输出保持一致。
- 平台参数与来源见 `hardware/platform_parameters.md`；立项背景与研究内容见
  `docs/proposal/project_proposal.md`。
