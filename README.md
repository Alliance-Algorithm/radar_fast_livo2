# radar_fast_livo2

[FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)（LiDAR-Inertial-Visual
Odometry）移植到 ROS2 Jazzy，适配全局曝光（global-shutter）面阵固态激光雷达 +
高频 IMU + 卷帘快门相机。LIO（LiDAR-IMU）部分已在真实硬件上完成静止漂移和
走圈闭环验证；VIO（视觉直接法）部分代码完整实现，尚未接入真实相机硬件测试。

![radar_fast_livo2 architecture](../../../docs/architecture/radar-fast-livo2-architecture.svg)

上图展示独立的 LiDAR-IMU-Visual 估计模块；它与主系统的 `radar_lidar` →
`radar_fusion` 定位/跟踪链路分开运行。

最终硬件验证结果（40m 走圈）：

```
静止10s漂移:  2.3 mm
闭环误差:     19.4 mm
路径长度:     40754 mm
漂移率:       0.05%   (优秀，< 1%)
```

## 目录结构

```
include/radar_fast_livo2/
  common_lib.hpp      公共类型（PointType/MeasureGroup/LidarType 等）
  esikf_state.hpp      ESIKF 状态定义（StatesGroup，18维: rot/pos/vel/bg/ba/gravity）
  imu_processing.hpp    IMU 处理器接口
  preprocess.hpp        点云预处理器接口
  shm_camera.hpp        HIK 相机 SHM 适配器（CameraFrame + ShmCamera）
  camera_frame_queue.hpp 相机帧队列（at-most-once、限长、最近匹配消费）
  voxel_map.hpp          体素地图管理器接口
  vio.hpp                视觉直接法管理器接口
src/
  livmapper_node.cpp    ROS2 节点主入口，管线编排
  imu_processing.cpp    IMU 静止初始化 + 正向传播 + 去畸变
  preprocess.cpp         点云预处理（含传感器类型分支）
  shm_camera.cpp         SHM 相机适配器实现（BGR8→gray 转换）
  camera_frame_queue.cpp  相机帧队列实现
  voxel_map.cpp          ESIKF 状态估计 + 增量体素地图
  vio.cpp                 视觉直接法（patch 投影/仿射变换/NCC/光度误差）
test/
  test_shm_camera.cpp   ShmCamera 单元测试（几何/灰度/时间戳/队列语义）
config/odin_livo2.yaml   参数配置（本文件名保留原始适配对象命名，内容通用）
```

## 使用方法

```bash
# 先确保 hikcamera SDK 已编译安装（third-party/hikcamera_sdk）
colcon build --packages-select radar_fast_livo2
source install/setup.bash
ros2 run radar_fast_livo2 radar_fast_livo2_node --ros-args \
    --params-file src/radar_fast_livo2/config/odin_livo2.yaml
```

topic：

| 方向 | Topic | 类型 | 说明 |
|---|---|---|---|
| 订阅 | `lidar_topic`（参数） | `sensor_msgs/PointCloud2` | 原始点云 |
| 订阅 | `imu_topic`（参数） | `sensor_msgs/Imu` | IMU（建议 ≥200Hz） |
| 共享内存 | `camera/shm_name`（参数，仅 LIVO 模式） | POSIX SHM (hikcamera SDK) | 相机 BGR8 → gray CV_8UC1 @2736×1824 |
| 发布 | `/fast_livo2/odom` | `nav_msgs/Odometry` | 位姿 + 速度 + 协方差 |
| 发布 | `/fast_livo2/cloud_world` | `sensor_msgs/PointCloud2` | 世界系当前帧点云 |
| 发布 | `/fast_livo2/path` | `nav_msgs/Path` | 累积轨迹 |

> **相机输入不再使用 ROS Image topic。** LIVO 模式通过 `hikcamera::SharedFrameReader`
> 从 POSIX 共享内存直接读取 HIK 相机帧（需先启动 `hikcamera_ros_driver` 或
> writer 进程创建 SHM 段）。默认 SHM 名 `/hikcamera_shm`，全分辨率 5472×3648
> BGR8 输入，自动转灰度并降采样到 2736×1824 CV_8UC1。
> 相机内参 `cam_fx/fy/cx/cy` 请用目标分辨率标定值或全分辨率值 ×0.5 填入。
>
> **时间域**: `host_monotonic_ns` 是 `std::chrono::steady_clock` epoch，
> 不是 Odin 设备时钟或 ROS clock。LIVO 模式下相机帧时间戳（`steady_clock`）
> 与 LiDAR/IMU 时间戳（Odin 设备时钟域）不在同一时钟域，需要通过
> `img_time_offset` 做域间映射。默认值 `0.0` 仅为占位，在接入真实相机前
> 必须实测标定该偏移量。不使用 device_timestamp_ticks 做时间转换。

关键参数（`slam_mode`: 1=ONLY_LIO 纯雷达惯性，2=LIVO 视觉+雷达+惯性融合）见
`config/odin_livo2.yaml` 内注释。**首次接入新传感器强烈建议先用 `slam_mode: 1`
跑通 LIO，确认里程计稳定后再切 `slam_mode: 2` 接视觉。**

## 设计思想

**状态所有权单一化**：`StatesGroup`（ESIKF 状态）只有一份，由
`LivMapperNode` 持有，`ImuProcess`/`VoxelMapManager`/`VIOManager` 都是
"原地读写调用方传入的状态引用"，不持有自己的状态副本。原版 FAST-LIVO2
（ROS1）里 IMU 处理模块和主循环各有一份状态，通过手动同步保持一致——
这在单线程场景下没问题，但容易在维护中引入"改了一处忘改另一处"的隐藏
bug。移植时把这层间接去掉，状态流转变成一条直线：IMU 正向传播 → ESIKF
修正 → （可选）VIO 光度修正 → 写回地图，每一步都直接改同一个对象。

**分层清晰**：预处理（`Preprocess`）、IMU 处理（`ImuProcess`）、地图管理
（`VoxelMapManager`）、视觉直接法（`VIOManager`）四个模块互不感知对方内部
实现，只通过 `MeasureGroup`/`StatesGroup`/`pointWithVar` 等数据结构交换信息，
移植到不同传感器组合时只需要改 `Preprocess`（新增一个 handler 分支）和
`config/*.yaml`，核心 ESIKF/建图逻辑不用动。

## 我们做的重构和优化

以下问题均在真实硬件上复现、定位、修复，附带控制论/估计论层面的解释，
不是"看起来应该这样改"的猜测性修复。

### 1. 传感器时间戳模型必须匹配物理曝光方式

原版 FAST-LIVO2 支持的旋转式扫描雷达（Livox AVIA / Velodyne / Ouster）每个
点在扫描周期内有真实的曝光时间差，`curvature` 字段（复用来存帧内时间偏移）
非零是正确的——用这个时间差在 `undistort_pcl()` 里做逐点运动补偿。但对于
**全局曝光的面阵传感器**（所有像素同时曝光，没有逐点扫描时间差，原版对
Intel L515 就是这样处理的），如果驱动层虚构了一个"逐点时间戳"字段喂给这
套逐点去畸变逻辑，运动中会把点云沿着错误的旋转轴"拧散"——静止时角速度
≈0，`exp_so3(w, dt)` 退化为单位矩阵，无论 dt 算错成什么值都没有影响，
所以这个 bug 只在运动时才暴露，非常容易被误诊为别的问题。

**教训**：接入新传感器时，第一件事是确认它的曝光模型（scanning vs.
global-shutter），而不是假设所有点云都需要逐点去畸变。

### 2. ESIKF 修正步的门控只能来自协方差本身，不能是人为计数阈值

调试运动发散问题时，一度加过"有效特征数 < 500 就跳过本帧修正"的门限，
逻辑是"信息量不足时宁可信任 IMU"。但这个直觉是错的：ESIKF 的 Kalman
gain 本身就会在信息不足的方向自动收缩到 0——

```
H 的行数（有效特征数）越少 → H^T·H 越小 → K_1=(H^T·H+P^-1)^-1 越接近 P
→ G=K_1·H^T·H 越接近 0 → 修正量 solution 里的 G·vec 项消失
→ 修正趋于 0，等价于纯信任 IMU 先验
```

这正是正确行为，不需要额外的计数门限去"帮忙"。计数门限反而混淆了两个
不同的问题：**可观测性**（该由协方差自动处理）和**异常值鲁棒性**（该靠
逐点的平面方差加权 `R = f(sigma_num, plane_var)` 处理）。留着这个门限的
真实代价：真实场景里（走廊、空旷区）有效特征数跌破阈值是合法情况，门限
会把这些本该被"弱修正"的帧直接改成"零修正"，逼状态纯 IMU 漂移——这
恰恰是它想防止的发散模式。

**教训**：怀疑 ESIKF 修正在特定条件下"行为异常"时，先检查协方差/噪声
模型本身对不对，而不是加计数阈值去拦截。计数阈值几乎总是在掩盖别处的
真实 bug。

### 3. IMU 传播窗口必须与实际消费进度对齐，不能是固定时间窗

最初的实现里，每帧从 IMU 缓冲区截取 `[frame_beg - margin, frame_end +
margin]` 的固定窗口喂给积分。问题：如果上一帧因为处理耗时超过传感器周期
被跳过（更早版本还有个丢帧保护，同样有这个问题），下一帧的固定窗口会
漏掉两帧之间的 IMU 数据，`undistort_pcl()` 只能用最后一个 IMU 样本"零阶
保持"去填这段空隙——运动中零阶保持 100-200ms 会累积 5-10° 姿态误差。

修复方式是把窗口下界从"当前帧开始时间"改成"上一次积分实际结束的时间
戳"（`last_prop_end_time_`），这样无论上一帧处理花了多久，这一帧的积分
窗口总是精确衔接上一次结束的位置，不会有间隙也不会重复积分。

### 4. 初始化阶段的时间锚点不能停在默认值

`last_prop_end_time_` 默认初始化成 0.0，静止初始化阶段（IMU 静止初始化
只做零偏估计，不做 ESIKF）没有更新它。第一次真正调用 `undistort_pcl()`
时，`0.0` 远早于任何真实时间戳，导致"跳过早于积分起点的样本"这个门控
形同虚设——整个静止初始化阶段积累的全部 IMU 样本（可能几百个）会在
第一帧被完整重新积分一次，把协方差在第一次 ESIKF 修正生效前就撑大了
几百倍。修复：初始化完成的瞬间把这个锚点显式推进到最后一个初始化用
IMU 样本的时间戳。

**教训**：任何"上一次结束位置"式的状态变量，都要检查它在所有初始化/
模式切换路径上是否被正确设置——默认值往往是"看起来无害"的 0，但一旦
真实时间戳远大于 0，这个默认值就会在下游产生一个隐蔽但巨大的窗口 bug。

### 5. 静态地图导出：热身帧与后处理分工

建图节点会累积每帧的世界系点云用于导出静态地图。首帧建图（`BuildVoxelMap`）
之后，体素地图里的平面约束非常稀疏，接下来几帧 ESIKF 还在收敛过程中
（实测 residual 从 0.024 降到 0.019 大约需要 5-7 帧）。如果从这几帧就
开始累积，同一块静止表面会用几个还没收敛、逐帧变化的位姿投影到世界系，
叠加起来在导出地图里表现成从传感器原点发散的"射线"伪影。

修复分两层：**源头**跳过前 N 帧（约 3 秒）的地图累积，等 ESIKF 收敛稳定
再开始记录；**后处理**再做一次 voxel 降采样去重 + 网格近邻计数式离群点
剔除（不依赖 KDTree/scipy，用 27 邻域向量化查表实现，可处理百万级点云）。
两层各管一段：源头修复解决"系统性偏移导致的射线"，后处理解决"个别帧
残留的散点噪声"——只做后处理不够，因为射线上的点空间上本就稀疏，voxel
去重命中不到它们；只做源头修复也不够，因为总会有个别帧的匹配质量偶发
下降。

### 6. 手动触发保存不能用信号处理器

需要一个"不停止建图、随时导出当前累积地图"的机制。最初想用 `SIGUSR1`
信号处理器直接调用保存函数，但 PCL 的 `savePCDFileBinary` 内部有动态内存
分配和文件 I/O，这些操作在 POSIX signal handler 里执行是未定义行为——
分配器不是 async-signal-safe 的。改成 1Hz 定时器轮询一个触发文件（约定
类似很多硬件厂商 SDK 常见的"写文件触发命令"模式），在正常执行路径上
（不是信号上下文）检测到触发后再调用保存函数，同时不阻塞主处理循环。

### 7. 手写体素降采样规避第三方库 ABI 冲突

`pcl::VoxelGrid` 在某些编译环境组合下会有内部 `malloc` 与
`Eigen::aligned_allocator::deallocate` 的 ABI 不匹配问题（不同编译单元/
优化选项下 Eigen 对齐分配器的实现细节不兼容）。改用手写哈希表做体素去重
（每个体素保留第一个落入的点），避免了这层依赖，副作用是降采样策略从
"体素内取质心"变成"体素内取首个点"——对大多数 SLAM 场景精度影响可
忽略，换来的是编译期就能规避掉一整类运行时 ABI 崩溃。

## 致谢

基于 [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)
（Chunran Zheng et al.）移植和重构，原始论文与算法设计归功于原作者。
