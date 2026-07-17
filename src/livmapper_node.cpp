// livmapper_node.cpp - FAST-LIVO2 ROS2 节点主入口
// 替换原 ROS1 的 LIVMapper.cpp + main()
//
// 订阅:
//   /odin1/cloud_raw  (sensor_msgs::msg::PointCloud2)  - Odin1 原始点云
//   /odin1/imu        (sensor_msgs::msg::Imu)           - 400Hz IMU
//   /camera/image_raw (sensor_msgs::msg::Image)         - HIK 相机图像 (仅 LIVO 模式)
//
// 发布:
//   /fast_livo2/odom        (nav_msgs::msg::Odometry)
//   /fast_livo2/cloud_world (sensor_msgs::msg::PointCloud2) - 世界坐标系点云
//   /fast_livo2/path        (nav_msgs::msg::Path)
//
// 核心流程 (ONLY_LIO):
//   1. LiDAR 帧到达 → 预处理 (odin1_handler)
//   2. 收集 IMU 时间窗口数据 → 静止初始化 / 正向传播 + 逐点去畸变
//   3. 降采样 + 变换到世界坐标系
//   4. 首帧: VoxelMapManager::BuildVoxelMap() 初始化哈希体素地图
//   5. 非首帧: VoxelMapManager::StateEstimation() ESIKF 迭代匹配
//   6. VoxelMapManager::UpdateVoxelMap() 增量更新地图
//   7. 发布 odom / cloud_world / path / TF

#include "radar_fast_livo2/common_lib.hpp"
#include "radar_fast_livo2/imu_processing.hpp"
#include "radar_fast_livo2/preprocess.hpp"
#include "radar_fast_livo2/esikf_state.hpp"
#include "radar_fast_livo2/voxel_map.hpp"
#include "radar_fast_livo2/vio.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <cv_bridge/cv_bridge.hpp>
#include <unordered_map>
#include <cmath>

#include "radar_fast_livo2/voxel_map.hpp"

#include <array>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <filesystem>
#include <chrono>

namespace radar::fast_livo2 {

class LivMapperNode : public rclcpp::Node {
public:
    explicit LivMapperNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("radar_fast_livo2_node", options) {
        declare_parameters();
        load_parameters();
        init_subscribers();
        init_publishers();
        // HACK: 手动 SLAM 保存触发用轮询文件（约定同 Odin1 厂商驱动的
        // "echo 'set save_map 1' > /tmp/odin_command.txt"），而非 ROS2
        // Service/信号处理器——PCL 的 savePCDFileBinary 内部有动态内存
        // 分配和文件 IO，在 POSIX signal handler（如 SIGUSR1）里直接调用
        // 是未定义行为（分配器不是异步信号安全的），必须放到普通执行路径
        // 里跑。用 1Hz 定时器轮询触发文件比信号处理器更安全，也比新增
        // Service 接口更轻量（.script/odin-map-save 直接 touch 文件即可，
        // 不需要额外的 ROS2 client 依赖）。
        save_trigger_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {
            check_save_trigger();
        });
        RCLCPP_INFO(get_logger(), "radar_fast_livo2 node started (mode=%d)", slam_mode_);
    }

    ~LivMapperNode() override {
        save_pcd();
    }

private:
    // ── 参数声明 ────────────────────────────────────────────────
    void declare_parameters() {
        // Topics
        declare_parameter("lidar_topic",  "/odin1/cloud_raw");
        declare_parameter("imu_topic",    "/odin1/imu");
        declare_parameter("img_topic",    "/camera/image_raw");

        // Sensor config
        declare_parameter("lidar_type",          static_cast<int>(LidarType::ODIN1));
        declare_parameter("point_filter_num",    4);
        declare_parameter("blind",               0.1);
        declare_parameter("max_range",           30.0);
        declare_parameter("confidence_threshold", 35);

        // SLAM mode: 1=ONLY_LIO, 2=LIVO
        declare_parameter("slam_mode", static_cast<int>(SlamMode::LIVO));
        declare_parameter("img_en",    true);

        // IMU
        declare_parameter("imu_en",           true);
        declare_parameter("init_imu_num",      20);
        // 陀螺仪/加速度计噪声协方差 (rad/s、m/s^2，单位 per sqrt(Hz))。
        // 默认值 0.1 是 imu_processing.hpp 里未针对 Odin1 400Hz IMU 调过的占位值，
        // 在大场景(有效特征数千+)、帧耗时变长(~150ms)时会让 ESIKF 先验协方差
        // 过大，几乎完全依赖 LiDAR 观测，一旦匹配点数振荡就失去正则化，
        // 引发静止状态下的发散（已用 Oracle 诊断确认）。
        declare_parameter("gyr_cov", 0.01);
        declare_parameter("acc_cov", 0.01);
        // bias 随机游走协方差（bias 变化速率，与上面的测量白噪声物理含义
        // 不同）。原版 FAST-LIVO2 也是硬编码 0.0001（LIVMapper.cpp 里
        // set_gyr_bias_cov/set_acc_bias_cov 从不读 yaml），比测量噪声小
        // 1000 倍是标准 ESIKF/INS 做法（bias 是缓慢温漂，不应被当成瞬时
        // 噪声一样快速跟踪）。此处开放为参数仅为方便调参，默认值不改。
        declare_parameter("gyr_bias_cov", 0.0001);
        declare_parameter("acc_bias_cov", 0.0001);
        // 默认0：Odin1 在 use_host_ros_time=0 下各 topic 已共享统一时钟，
        // 详见 config/odin_livo2.yaml 内说明，通常不需要调整。
        declare_parameter("imu_time_offset",   0.0);
        declare_parameter("img_time_offset",   0.0);

        // Extrinsics: LiDAR w.r.t. IMU
        declare_parameter("extrinsic_T", std::vector<double>{0.0, 0.0, 0.0});
        declare_parameter("extrinsic_R", std::vector<double>{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        });
        // Extrinsics: Camera w.r.t. LiDAR (Rcl, Pcl)
        declare_parameter("Pcl", std::vector<double>{0.0, 0.0, 0.0});
        declare_parameter("Rcl", std::vector<double>{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        });
        declare_parameter("extrinsic_est_en", true);

        // Camera intrinsics (pinhole)
        declare_parameter("cam_fx", 500.0);
        declare_parameter("cam_fy", 500.0);
        declare_parameter("cam_cx", 640.0);
        declare_parameter("cam_cy", 360.0);
        declare_parameter("cam_width",  1280);
        declare_parameter("cam_height", 720);
        declare_parameter("img_scale",  0.5);

        // VIO
        declare_parameter("patch_size",          8);
        declare_parameter("patch_pyramid_level", 4);

        // Map (downsampling + voxel map config)
        declare_parameter("filter_size_surf",    0.1);
        declare_parameter("filter_size_map",     0.15);
        declare_parameter("cube_side_length",    1000.0);

        // Voxel Map config (从 FAST-LIVO2 voxel_map.cpp loadVoxelConfig 移植)
        declare_parameter("voxel_map/max_layer",         1);
        declare_parameter("voxel_map/voxel_size",        0.5);
        declare_parameter("voxel_map/min_eigen_value",   0.01);
        declare_parameter("voxel_map/sigma_num",         3.0);
        declare_parameter("voxel_map/beam_err",          0.02);
        declare_parameter("voxel_map/dept_err",          0.05);
        declare_parameter("voxel_map/layer_init_num",
            std::vector<int64_t>{5LL, 5LL, 5LL, 5LL, 5LL});
        declare_parameter("voxel_map/max_points_num",    50);
        declare_parameter("voxel_map/max_iterations",    5);
        declare_parameter("voxel_map/map_sliding_en",    false);
        declare_parameter("voxel_map/half_map_size",     100);
        declare_parameter("voxel_map/sliding_thresh",    8.0);

        // PCD save
        declare_parameter("pcd_save_en",      false);
        declare_parameter("pcd_save_interval", -1);
        declare_parameter("map_save_path",    std::string("/tmp/fast_livo2_map.pcd"));
        declare_parameter("map_save_trigger", std::string("/tmp/fast_livo2_save_map"));
    }

    void load_parameters() {
        lidar_topic_ = get_parameter("lidar_topic").as_string();
        imu_topic_   = get_parameter("imu_topic").as_string();
        img_topic_   = get_parameter("img_topic").as_string();
        slam_mode_   = get_parameter("slam_mode").as_int();
        img_en_      = get_parameter("img_en").as_bool() && (slam_mode_ == SlamMode::LIVO);

        // 预处理器参数
        preprocess_.lidar_type           = get_parameter("lidar_type").as_int();
        preprocess_.point_filter_num     = get_parameter("point_filter_num").as_int();
        preprocess_.blind                = get_parameter("blind").as_double();
        preprocess_.max_range            = get_parameter("max_range").as_double();
        preprocess_.confidence_threshold = get_parameter("confidence_threshold").as_int();

        // IMU 处理器参数
        imu_process_.imu_en       = get_parameter("imu_en").as_bool();
        imu_process_.init_imu_num = get_parameter("init_imu_num").as_int();
        const double gyr_cov = get_parameter("gyr_cov").as_double();
        const double acc_cov = get_parameter("acc_cov").as_double();
        imu_process_.set_gyr_cov(Eigen::Vector3d(gyr_cov, gyr_cov, gyr_cov));
        imu_process_.set_acc_cov(Eigen::Vector3d(acc_cov, acc_cov, acc_cov));
        const double gyr_bias_cov = get_parameter("gyr_bias_cov").as_double();
        const double acc_bias_cov = get_parameter("acc_bias_cov").as_double();
        imu_process_.set_gyr_bias_cov(Eigen::Vector3d(gyr_bias_cov, gyr_bias_cov, gyr_bias_cov));
        imu_process_.set_acc_bias_cov(Eigen::Vector3d(acc_bias_cov, acc_bias_cov, acc_bias_cov));

        // 外参加载
        auto ext_t = get_parameter("extrinsic_T").as_double_array();
        auto ext_r = get_parameter("extrinsic_R").as_double_array();
        Eigen::Vector3d t_li(ext_t[0], ext_t[1], ext_t[2]);
        Eigen::Matrix3d r_li;
        r_li << ext_r[0], ext_r[1], ext_r[2],
                ext_r[3], ext_r[4], ext_r[5],
                ext_r[6], ext_r[7], ext_r[8];
        imu_process_.set_extrinsic(t_li, r_li);

        img_time_offset_ = get_parameter("img_time_offset").as_double();
        imu_time_offset_ = get_parameter("imu_time_offset").as_double();
        img_scale_       = get_parameter("img_scale").as_double();

        // ── VIOManager 初始化（仅 LIVO 模式）──
        if (img_en_) {
            auto pcl_ = get_parameter("Pcl").as_double_array();
            auto rcl_ = get_parameter("Rcl").as_double_array();
            V3D pcl(pcl_[0], pcl_[1], pcl_[2]);
            M3D rcl;
            rcl << rcl_[0], rcl_[1], rcl_[2],
                   rcl_[3], rcl_[4], rcl_[5],
                   rcl_[6], rcl_[7], rcl_[8];

            const double cam_fx = get_parameter("cam_fx").as_double();
            const double cam_fy = get_parameter("cam_fy").as_double();
            const double cam_cx = get_parameter("cam_cx").as_double();
            const double cam_cy = get_parameter("cam_cy").as_double();
            const int    cam_width  = get_parameter("cam_width").as_int();
            const int    cam_height = get_parameter("cam_height").as_int();
            const int    patch_size = get_parameter("patch_size").as_int();
            const int    patch_pyramid_level = get_parameter("patch_pyramid_level").as_int();

            vio_manager_.init(cam_fx, cam_fy, cam_cx, cam_cy, cam_width, cam_height,
                               rcl, pcl, r_li, t_li,
                               patch_size, patch_pyramid_level, /*grid_size=*/20,
                               /*normal_en=*/true, /*ncc_en=*/true,
                               /*img_point_cov=*/100.0, /*ncc_thre=*/0.6,
                               /*max_iterations=*/5);
        }

        // 降采样参数
        filter_size_surf_ = get_parameter("filter_size_surf").as_double();

        // ── VoxelMapConfig 初始化 ──
        voxel_config_.max_layer_         = get_parameter("voxel_map/max_layer").as_int();
        voxel_config_.max_voxel_size_    = get_parameter("voxel_map/voxel_size").as_double();
        voxel_config_.planner_threshold_ = get_parameter("voxel_map/min_eigen_value").as_double();
        voxel_config_.sigma_num_         = get_parameter("voxel_map/sigma_num").as_double();
        voxel_config_.beam_err_          = get_parameter("voxel_map/beam_err").as_double();
        voxel_config_.dept_err_          = get_parameter("voxel_map/dept_err").as_double();
        voxel_config_.max_points_num_    = get_parameter("voxel_map/max_points_num").as_int();
        voxel_config_.max_iterations_    = get_parameter("voxel_map/max_iterations").as_int();
        voxel_config_.map_sliding_en     = get_parameter("voxel_map/map_sliding_en").as_bool();
        voxel_config_.half_map_size      = get_parameter("voxel_map/half_map_size").as_int();
        voxel_config_.sliding_thresh     = get_parameter("voxel_map/sliding_thresh").as_double();

        auto layer_init_raw = get_parameter("voxel_map/layer_init_num").as_integer_array();
        voxel_config_.layer_init_num_.clear();
        for (auto v : layer_init_raw) voxel_config_.layer_init_num_.push_back(static_cast<int>(v));
        while (voxel_config_.layer_init_num_.size() < 5) {
            voxel_config_.layer_init_num_.push_back(5);
        }

        // 将 voxel_config 复制到 VoxelMapManager
        voxel_map_.config_setting_ = voxel_config_;

        // 设置 VoxelMapManager 外参
        voxel_map_.extR_ = r_li;
        voxel_map_.extT_ = t_li;

        // PCD
        pcd_save_en_       = get_parameter("pcd_save_en").as_bool();
        map_save_path_     = get_parameter("map_save_path").as_string();
        save_trigger_path_ = get_parameter("map_save_trigger").as_string();

        RCLCPP_INFO(get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(get_logger(), "IMU   topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(get_logger(), "VoxelMap: max_layer=%d, voxel_size=%.3f, max_iter=%d, plane_thresh=%.4f",
                    voxel_config_.max_layer_, voxel_config_.max_voxel_size_,
                    voxel_config_.max_iterations_, voxel_config_.planner_threshold_);
        if (img_en_) {
            RCLCPP_INFO(get_logger(), "Image topic: %s (scale=%.2f)", img_topic_.c_str(), img_scale_);
        }
    }

    // ── 订阅 ────────────────────────────────────────────────────
    void init_subscribers() {
        auto qos = rclcpp::SensorDataQoS();

        sub_lidar_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, qos,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                on_lidar(msg);
            });

        sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::QoS(rclcpp::KeepLast(4000)).reliable(),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                on_imu(msg);
            });

        if (img_en_) {
            sub_img_ = create_subscription<sensor_msgs::msg::Image>(
                img_topic_, qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    on_image(msg);
                });
        }
    }

    void init_publishers() {
        pub_odom_  = create_publisher<nav_msgs::msg::Odometry>("/fast_livo2/odom", 10);
        pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/fast_livo2/cloud_world", 10);
        pub_path_  = create_publisher<nav_msgs::msg::Path>("/fast_livo2/path", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // ── 回调 ────────────────────────────────────────────────────
    void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(imu_buf_mutex_);
            ImuData d;
            d.timestamp = rclcpp::Time(msg->header.stamp).seconds() + imu_time_offset_;
            d.acc  = {msg->linear_acceleration.x,
                      msg->linear_acceleration.y,
                      msg->linear_acceleration.z};
            d.gyro = {msg->angular_velocity.x,
                      msg->angular_velocity.y,
                      msg->angular_velocity.z};
            imu_buf_.push_back(d);
            // FIXME (Oracle 架构裁决): 固定 2s wall-clock 窗口对齐上游"处理
            // 完就 pop"语义后发现有隐患——若处理卡顿超过 2s（比如探索大片
            // 新区域时 UpdateVoxelMap 耗时飙升），会把 imu_process_ 尚未消
            // 费到的样本一起裁掉，造成永久积分 gap（等价于强行丢帧，且比
            // 显式丢帧更隐蔽）。改为按 imu_process_.last_prop_end_time()
            // 裁剪：只删已经被 undistort_pcl 消费过的样本，留一点 margin
            // 防止 undistort_pcl 内部桥接用的 last_imu_ 被误删。
            const double consumed_before = imu_process_.last_prop_end_time() - 0.1;
            auto it = imu_buf_.begin();
            while (it != imu_buf_.end() && it->timestamp < consumed_before) ++it;
            if (it != imu_buf_.begin()) imu_buf_.erase(imu_buf_.begin(), it);
        }
        // 若上一帧因 IMU 未追上而挂在 lidar_buf_ 里等待重试，这里顺带
        // 尝一次（IMU 400Hz 到达，比等下一个 10Hz LiDAR 帧触发重试快得
        // 多）。process_frame() 内部在 lidar_buf_ 为空或 IMU 仍不足时
        // 会快速 return，正常路径下这个额外调用几乎零开销。
        process_frame();
    }

    void on_lidar(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // WARN: 锁的作用域必须严格限定在缓冲区操作内，绝不能带锁进入 process_frame()——
        // process_frame() 内部会再次对 lidar_buf_mutex_ 上锁（非递归锁，同线程
        // 重复 lock() 是自死锁，曾导致节点收到第一帧后永久冻结、后续任何回调
        // 都不再被单线程 executor 调度）。
        // FIXME (Oracle 架构裁决): 曾在 buffer>3 时 pop_front 丢最旧帧，对齐
        // 上游 FAST-LIVO2 sync_packages() 后发现这是错误设计——上游从不丢帧，
        // IMU 未追上时只是 return false 重试，帧留在 deque 里等，只有延迟没有
        // 丢失。丢帧对 ESIKF 是破坏性的：丢的不是"跳过一次修正"（那样估计仍
        // 无偏，只是协方差变大），而是打断了 IMU 积分锚点连续性——下一帧处理
        // 时 process_frame() 用的是 frame_beg，不会自动把丢失区间的 IMU 积分
        // 补上。改为只入队不丢弃，配合 process_frame() 里的 peek-not-pop
        // 重试逻辑，让处理跟不上时自然产生延迟而不是数据丢失。
        {
            std::lock_guard<std::mutex> lock(lidar_buf_mutex_);
            lidar_buf_.push_back(msg);
        }
        // 同步处理（单线程，LiDAR 帧到达时触发；此时已不持有 lidar_buf_mutex_）
        process_frame();
    }

    void on_image(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(img_buf_mutex_);
        img_buf_.push_back(msg);
        if (img_buf_.size() > 5) img_buf_.pop_front();
    }

    // 从 img_buf_ 中取出时间上离 target_time 最近的一帧，转换为灰度图。
    // 找不到足够接近的帧（超过 0.05s）时返回 false，跳过本帧 VIO。
    bool get_nearest_image(double target_time, cv::Mat& out_gray) {
        sensor_msgs::msg::Image::SharedPtr best;
        double best_dt = std::numeric_limits<double>::max();
        {
            std::lock_guard<std::mutex> lock(img_buf_mutex_);
            for (const auto& img_msg : img_buf_) {
                double t = rclcpp::Time(img_msg->header.stamp).seconds() + img_time_offset_;
                double dt = std::abs(t - target_time);
                if (dt < best_dt) { best_dt = dt; best = img_msg; }
            }
        }
        if (!best || best_dt > 0.05) return false;
        auto cv_ptr = cv_bridge::toCvCopy(best, "mono8");
        out_gray = cv_ptr->image;
        return true;
    }

    // ══════════════════════════════════════════════════════════════
    // process_frame — ESIKF 主循环（完整 LIO 管线）
    //
    // 对应 FAST-LIVO2/src/LIVMapper.cpp handleLIO() 第 336-482 行
    //
    // 流程:
    //   1. 取出 LiDAR 帧 → 预处理
    //   2. 收集 IMU 时间窗口数据 → 去畸变
    //   3. 降采样 → 变换到世界系
    //   4. 首帧: BuildVoxelMap() 初始化哈希体素地图
    //   5. 非首帧: StateEstimation() ESIKF 迭代匹配 + 协方差更新
    //   6. UpdateVoxelMap() 增量更新体素地图
    //   7. 发布 odom / cloud_world / path / TF
    //   8. PCD 累积（可选）
    //
    // 简化（相对原版）:
    //   - 跳过 VIO/img 路径（ONLY_LIO 模式）
    //   - 跳过曝光时间估计 (inv_expo_time)
    //   - 跳过平面可视化发布 (pubVoxelMap)
    //   - 跳过 pose_output txt 日志
    // ══════════════════════════════════════════════════════════════

    void process_frame() {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();

        // ── 1. 窥视（不弹出）LiDAR 帧 ──
        // FIXME (Oracle 架构裁决): 对齐上游 sync_packages() 的 retry 语义——
        // IMU 数据不够时只 return（帧留在 lidar_buf_ 里），不弹出、不丢弃，
        // 下一次任意回调（IMU/LiDAR 到达）触发 process_frame() 时会重新
        // 尝试同一帧。只有 IMU 数据集齐之后才真正 pop_front，此时帧数据
        // 和 IMU 积分窗口才算"消费成功"，避免丢帧打断 IMU 积分连续性。
        sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg;
        {
            std::lock_guard<std::mutex> lock(lidar_buf_mutex_);
            if (lidar_buf_.empty()) return;
            lidar_msg = lidar_buf_.front();  // 窥视，暂不 pop
        }

        // ── 2. 预处理: ROS PointCloud2 → PointCloudT (raw) ──
        auto raw_cloud = std::make_shared<PointCloudT>();
        preprocess_.process(lidar_msg, raw_cloud);
        if (raw_cloud->empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Empty point cloud after preprocessing");
            // 空点云无法恢复，此帧真正丢弃（非 IMU 未就绪的可重试场景）
            std::lock_guard<std::mutex> lock(lidar_buf_mutex_);
            if (!lidar_buf_.empty()) lidar_buf_.pop_front();
            return;
        }

        // ── 3. 收集 IMU 数据 ──
        double frame_beg = rclcpp::Time(lidar_msg->header.stamp).seconds();
        double frame_end = frame_beg;
        for (size_t i = 0; i < raw_cloud->points.size(); ++i) {
            double pt_time = frame_beg + raw_cloud->points[i].curvature / 1000.0;
            if (pt_time > frame_end) frame_end = pt_time;
        }
        if (frame_end <= frame_beg) frame_end = frame_beg + 0.05;

        MeasureGroup meas;
        meas.lidar_beg_time = frame_beg;
        meas.lidar_end_time = frame_end;
        meas.lidar = raw_cloud;

        {
            std::lock_guard<std::mutex> lock(imu_buf_mutex_);
            // IMU 窗口取所有 <= frame_end+margin 的样本（不设下界），由
            // undistort_pcl 内部的 prop_beg_time 门控自动跳过早于上帧末
            // （last_prop_end_time_）的部分，从而覆盖 [last_prop_end_time_,
            // frame_end] 的完整积分区间，不依赖 frame_beg 本身。
            const double margin = 0.02;
            meas.imu.reserve(512);
            for (size_t i = 0; i < imu_buf_.size(); ++i) {
                const auto& imu = imu_buf_[i];
                if (imu.timestamp <= frame_end + margin) {
                    meas.imu.push_back(imu);
                }
            }
            // IMU 还没追上这一帧的结束时间：不丢帧，直接 return 重试
            // （帧还在 lidar_buf_.front()，下次任意回调触发时会重新窥视）。
            if (imu_buf_.empty() || imu_buf_.back().timestamp < frame_end + margin) {
                return;
            }
            if (meas.imu.size() < 2) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Insufficient IMU data (%zu samples) for frame [%.3f, %.3f]",
                    meas.imu.size(), frame_beg, frame_end);
                return;
            }
        }

        // IMU 数据已集齐，正式消费这一帧
        {
            std::lock_guard<std::mutex> lock(lidar_buf_mutex_);
            if (!lidar_buf_.empty()) lidar_buf_.pop_front();
        }

        // ── 4. IMU 去畸变 ──
        auto feats_undistort = std::make_shared<PointCloudT>();
        bool imu_ok = imu_process_.process(meas, state_, feats_undistort);
        if (!imu_ok) { return; }

        if (feats_undistort->empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Empty undistorted point cloud");
            return;
        }

        // ── 5. 降采样 ──
        // HACK: 手动体素降采样，避免 pcl::VoxelGrid 内部 malloc 与
        // Eigen::aligned_allocator::deallocate 的 ABI 冲突。
        auto feats_down_body = std::make_shared<PointCloudT>();
        {
            const double inv_leaf = 1.0 / filter_size_surf_;
            std::unordered_map<uint64_t, size_t> voxel_map;
            for (size_t i = 0; i < feats_undistort->size(); ++i) {
                const auto& pt = feats_undistort->points[i];
                int64_t ix = static_cast<int64_t>(std::floor(pt.x * inv_leaf));
                int64_t iy = static_cast<int64_t>(std::floor(pt.y * inv_leaf));
                int64_t iz = static_cast<int64_t>(std::floor(pt.z * inv_leaf));
                uint64_t key = (static_cast<uint64_t>(ix + 32768) << 42)
                             | (static_cast<uint64_t>(iy + 32768) << 21)
                             | (static_cast<uint64_t>(iz + 32768));
                if (voxel_map.emplace(key, feats_down_body->size()).second) {
                    feats_down_body->push_back(pt);
                }
            }
            feats_down_body->header = feats_undistort->header;
            feats_down_body->is_dense = false;
        }

        int feats_down_size = static_cast<int>(feats_down_body->points.size());
        auto t_down = clock::now();

        // ── 6. 设置 VoxelMapManager 上下文 ──
        voxel_map_.feats_undistort_ = feats_undistort;
        voxel_map_.feats_down_body_ = feats_down_body;
        voxel_map_.feats_down_size_ = feats_down_size;

        // 变换到世界坐标系
        auto feats_down_world = std::make_shared<PointCloudT>();
        voxel_map_.TransformLidar(
            state_.rot_end, state_.pos_end, feats_down_body, feats_down_world);
        voxel_map_.feats_down_world_ = feats_down_world;

        // ── 7. 首帧建图 ──
        if (!lidar_map_inited_) {
            lidar_map_inited_ = true;
            voxel_map_.state_ = state_;
            voxel_map_.BuildVoxelMap();
            RCLCPP_INFO(get_logger(),
                "First frame: built voxel map with %d points (%ld root voxels)",
                feats_down_size, voxel_map_.voxel_map_.size());
            return;
        }

        auto t1 = clock::now();

        // ── 8. ESIKF 状态估计 ──
        // state_ 此时已是 imu_process_.process() 原地传播后的先验（含 C2 协方差传播），
        // 迭代起点与 state_propagat 保持同一先验值（Oracle M1 修复）。
        StatesGroup state_propagat = state_;

        voxel_map_.state_ = state_propagat;
        voxel_map_.StateEstimation(state_propagat);
        state_ = voxel_map_.state_;

        auto t2 = clock::now();

        // ── 9. 增量更新体素地图 ──
        // 用更新后的状态重新变换点云，计算世界帧协方差
        auto world_lidar = std::make_shared<PointCloudT>();
        voxel_map_.TransformLidar(
            state_.rot_end, state_.pos_end, feats_down_body, world_lidar);

        for (size_t i = 0; i < static_cast<size_t>(feats_down_size); i++) {
            voxel_map_.pv_list_[i].point_w = V3D(
                world_lidar->points[i].x,
                world_lidar->points[i].y,
                world_lidar->points[i].z);

            M3D point_crossmat = voxel_map_.cross_mat_list_[i];
            M3D var = voxel_map_.body_cov_list_[i];
            var = (state_.rot_end * voxel_map_.extR_) * var
                  * (state_.rot_end * voxel_map_.extR_).transpose()
                  + (-point_crossmat) * state_.cov.block<3, 3>(0, 0)
                    * (-point_crossmat).transpose()
                  + state_.cov.block<3, 3>(3, 3);
            voxel_map_.pv_list_[i].var = var;
        }

        // HACK: 曾在有效特征过低时跳过地图更新，防止 state_ 不可信时把
        // 错误点/平面参数写进 voxel_map_ 形成发散循环（修复过静止场景下
        // frame 234+ 的振荡发散）。但该保护在"移动到地图未覆盖新区域"场景
        // 下会死锁：新区域本来就没有匹配点 → effective 特征天然低于阈值
        // → 保护触发冻住地图 → 地图永远无法扩展到新区域 → 特征永远起不来
        // （三次实测复现：无论怎么调外参/IMU协方差，只要挪到新区域必然
        // 归零发散，说明这是死锁而非参数问题）。静止振荡发散已经用
        // gyr_cov/acc_cov/sigma_num/协方差地板四项参数修复，不再需要这层
        // 保护，故移除，让地图始终随着移动扩展覆盖范围。
        voxel_map_.UpdateVoxelMap(voxel_map_.pv_list_);

        auto t3 = clock::now();

        // ── 9.5. VIO 光度更新（仅 LIVO 模式，且找到时间匹配的图像帧）──
        if (img_en_) {
            cv::Mat gray;
            if (get_nearest_image(frame_end + img_time_offset_, gray)) {
                vio_manager_.state_          = &state_;
                vio_manager_.state_propagat_ = &state_propagat;
                vio_manager_.processFrame(gray, voxel_map_.pv_list_, voxel_map_.voxel_map_);
            }
        }

        // ── 10. 地图滑动 ──
        if (voxel_config_.map_sliding_en) {
            voxel_map_.mapSliding();
        }

        auto t4 = clock::now();

        // ── 11. 发布 odometry ──
        publish_odometry(frame_end);

        // ── 12. 发布世界系点云 ──
        publish_cloud_world(world_lidar, lidar_msg->header.stamp);

        // ── 13. 发布 path ──
        publish_path(lidar_msg->header.stamp);

        // ── 14. 发布 TF ──
        publish_tf(lidar_msg->header.stamp);

        // ── 15. PCD 累积 ──
        if (pcd_save_en_) {
            for (const auto& pt : world_lidar->points) {
                pcd_accumulated_.points.push_back(pt);
            }
        }

        // ── 16. 计时日志 ──
        frame_count_++;
        auto t_total = std::chrono::duration<double>(t4 - t0).count();
        avg_time_ = avg_time_ * (frame_count_ - 1) / frame_count_
                    + t_total / frame_count_;

        RCLCPP_INFO(get_logger(),
            "[ LIO ] frame %d | down: %.1fms | ICP: %.1fms | update: %.1fms"
            " | total: %.1fms (avg: %.1fms) | pts: %d/%d/%d",
            frame_count_,
            std::chrono::duration<double>(t_down - t0).count() * 1000,
            std::chrono::duration<double>(t2 - t1).count() * 1000,
            std::chrono::duration<double>(t3 - t2).count() * 1000,
            t_total * 1000,
            avg_time_ * 1000,
            static_cast<int>(feats_undistort->size()),
            feats_down_size,
            voxel_map_.effct_feat_num_);
    }

    // ── 发布函数 ────────────────────────────────────────────────

    void publish_odometry(double timestamp) {
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(timestamp * 1e9));
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id  = "base_link";

        odom_msg.pose.pose.position.x = state_.pos_end.x();
        odom_msg.pose.pose.position.y = state_.pos_end.y();
        odom_msg.pose.pose.position.z = state_.pos_end.z();

        Eigen::Quaterniond q(state_.rot_end);
        odom_msg.pose.pose.orientation.w = q.w();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();

        odom_msg.twist.twist.linear.x = state_.vel_end.x();
        odom_msg.twist.twist.linear.y = state_.vel_end.y();
        odom_msg.twist.twist.linear.z = state_.vel_end.z();

        // 协方差（仅报告 pose 6-DOF）
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                odom_msg.pose.covariance[i * 6 + j] = state_.cov(i, j);
            }
        }
        pub_odom_->publish(odom_msg);
    }

    void publish_cloud_world(const PointCloudT::Ptr& cloud,
                              const builtin_interfaces::msg::Time& stamp) {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.stamp = stamp;
        cloud_msg.header.frame_id = "odom";
        pub_cloud_->publish(cloud_msg);
    }

    void publish_path(const builtin_interfaces::msg::Time& stamp) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = "odom";
        pose.pose.position.x = state_.pos_end.x();
        pose.pose.position.y = state_.pos_end.y();
        pose.pose.position.z = state_.pos_end.z();

        Eigen::Quaterniond q(state_.rot_end);
        pose.pose.orientation.w = q.w();
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();

        path_msg_.header.stamp = stamp;
        path_msg_.header.frame_id = "odom";
        path_msg_.poses.push_back(pose);
        pub_path_->publish(path_msg_);
    }

    void publish_tf(const builtin_interfaces::msg::Time& stamp) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = "odom";
        tf.child_frame_id = "base_link";

        tf.transform.translation.x = state_.pos_end.x();
        tf.transform.translation.y = state_.pos_end.y();
        tf.transform.translation.z = state_.pos_end.z();

        Eigen::Quaterniond q(state_.rot_end);
        tf.transform.rotation.w = q.w();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();

        tf_broadcaster_->sendTransform(tf);
    }

    void save_pcd() {
        if (pcd_save_en_ && !pcd_accumulated_.empty()) {
            RCLCPP_INFO(get_logger(),
                "Saving %ld points to %s",
                pcd_accumulated_.points.size(), map_save_path_.c_str());
            // HACK: 从 ASCII 换成 Binary——长时间建图累积到百万级点时，
            // ASCII 每点一行文本格式化 I/O 耗时和文件体积都数倍于 binary，
            // 手动触发保存场景下用户在等这个操作完成，不该让格式选择成为
            // 瓶颈。Foxglove/PCL/CloudCompare 等下游工具都原生支持读取
            // binary PCD，不存在兼容性代价。
            pcl::io::savePCDFileBinary(map_save_path_, pcd_accumulated_);
            RCLCPP_INFO(get_logger(), "PCD saved (%ld points, binary).",
                pcd_accumulated_.points.size());
        } else if (pcd_save_en_) {
            RCLCPP_WARN(get_logger(), "PCD save requested but no points accumulated yet.");
        }
    }

    // 检查 map_save_path_ 同目录下是否存在触发文件（.script/odin-map-save
    // 负责 touch 它），存在则立即保存当前累积点云并删除触发文件（避免
    // 下一轮定时器重复触发）。不停止建图，保存完继续累积。
    void check_save_trigger() {
        namespace fs = std::filesystem;
        const fs::path trigger_path = save_trigger_path_;
        std::error_code ec;
        if (!fs::exists(trigger_path, ec) || ec) { return; }
        RCLCPP_INFO(get_logger(), "Save-map trigger detected, saving current map...");
        save_pcd();
        fs::remove(trigger_path, ec);
        if (ec) {
            RCLCPP_WARN(get_logger(), "Failed to remove save-map trigger file: %s",
                ec.message().c_str());
        }
    }

    // ── 成员变量 ─────────────────────────────────────────────────
    std::string lidar_topic_, imu_topic_, img_topic_;
    int    slam_mode_ = SlamMode::LIVO;
    bool   img_en_    = true;
    double img_time_offset_ = 0.0;
    double imu_time_offset_ = 0.0;
    double img_scale_       = 0.5;
    double filter_size_surf_ = 0.1;
    bool   pcd_save_en_     = false;
    std::string map_save_path_;
    std::string save_trigger_path_ = "/tmp/fast_livo2_save_map";
    rclcpp::TimerBase::SharedPtr save_trigger_timer_;

    Preprocess  preprocess_;
    ImuProcess  imu_process_;

    // 缓冲区（各自有独立互斥锁）
    std::mutex imu_buf_mutex_;
    std::mutex lidar_buf_mutex_;
    std::mutex img_buf_mutex_;
    std::vector<ImuData, Eigen::aligned_allocator<ImuData>>  imu_buf_;
    std::deque<sensor_msgs::msg::PointCloud2::SharedPtr>    lidar_buf_;
    std::deque<sensor_msgs::msg::Image::SharedPtr>          img_buf_;

    // 体素地图管理器（替换原来的 ikd-Tree）
    VoxelMapManager  voxel_map_;
    VoxelMapConfig   voxel_config_;
    StatesGroup      state_;             // ESIKF 状态（持续跨帧更新）
    bool             lidar_map_inited_ = false;

    // 视觉直接法前端（仅 LIVO 模式启用）
    VIOManager       vio_manager_;

    // 发布者 & TF
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr         sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       sub_img_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          pub_odom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_cloud_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr              pub_path_;
    std::unique_ptr<tf2_ros::TransformBroadcaster>                 tf_broadcaster_;
    nav_msgs::msg::Path path_msg_;

    // PCD 累积
    PointCloudT         pcd_accumulated_;
    int                 frame_count_ = 0;
    double              avg_time_    = 0.0;
};

}  // namespace radar::fast_livo2

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<radar::fast_livo2::LivMapperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
