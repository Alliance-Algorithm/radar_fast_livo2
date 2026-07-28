// livmapper_node.cpp - FAST-LIVO2 ROS2 节点主入口
// 替换原 ROS1 的 LIVMapper.cpp + main()
//
// 订阅:
//   /odin1/cloud_raw  (sensor_msgs::msg::PointCloud2)  - Odin1 原始点云
//   /odin1/imu        (sensor_msgs::msg::Imu)           - 400Hz IMU
//
// 相机输入（仅 LIVO 模式）:
//   camera_input_mode: "shm"（默认） — 通过 raw hikcamera::imageSHM 从 POSIX
//     共享内存直接读取 HIK 相机帧（RGB → gray CV_8UC1 @ 目标分辨率）。
//     相机线程在 img_en_=true 时启动，持续等待 SHM 新帧并写入内部队列。
//   camera_input_mode: "ros_image"（MCAP 回放） — 订阅 ROS Image topic
//     （sensor_msgs/Image，BGR8），ROS 时间戳直接使用，不应用 img_time_offset。
//     不初始化 ShmCamera 或 SHM 捕获线程。
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
//   7. (LIVO) VIO 光度更新 + 发布 odom / cloud_world / path / TF

#include "radar_fast_livo2/camera_frame_queue.hpp"
#include "radar_fast_livo2/common_lib.hpp"
#include "radar_fast_livo2/esikf_state.hpp"
#include "radar_fast_livo2/imu_processing.hpp"
#include "radar_fast_livo2/preprocess.hpp"
#include "radar_fast_livo2/ros_image_camera.hpp"
#include "radar_fast_livo2/shm_camera.hpp"
#include "radar_fast_livo2/vio.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "radar_fast_livo2/lio_drift_diagnostics.hpp"
#include "radar_fast_livo2/voxel_map.hpp"

namespace radar::fast_livo2 {

class LivMapperNode : public rclcpp::Node {
public:
    explicit LivMapperNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("radar_fast_livo2_node", options) {
        declare_parameters();
        load_parameters();
        init_subscribers();
        init_publishers();
        start_camera(); // 仅在 LIVO 模式启动 SHM 相机线程
        processing_running_.store(true);
        processing_thread_ = std::thread([this]() { processing_loop(); });
        // HACK: 手动 SLAM 保存触发用轮询文件（约定同 Odin1 厂商驱动的
        // "echo 'set save_map 1' > /tmp/odin_command.txt"），而非 ROS2
        // Service/信号处理器——PCL 的 savePCDFileBinary 内部有动态内存
        // 分配和文件 IO，在 POSIX signal handler（如 SIGUSR1）里直接调用
        // 是未定义行为（分配器不是异步信号安全的），必须放到普通执行路径
        // 里跑。用 1Hz 定时器轮询触发文件比信号处理器更安全，也比新增
        // Service 接口更轻量（.script/odin-map-save 直接 touch 文件即可，
        // 不需要额外的 ROS2 client 依赖）。
        save_trigger_timer_ =
            create_wall_timer(std::chrono::seconds(1), [this]() { check_save_trigger(); });
        RCLCPP_INFO(get_logger(), "radar_fast_livo2 node started (mode=%d)", slam_mode_);
    }

    ~LivMapperNode() override {
        stop_camera(); // 先停相机线程
        processing_running_.store(false);
        process_signal_cv_.notify_all();
        if (processing_thread_.joinable()) processing_thread_.join();
        save_pcd();
    }

private:
    // ── 参数声明 ────────────────────────────────────────────────
    void declare_parameters() {
        // Topics
        declare_parameter("lidar_topic", "/odin1/cloud_raw");
        declare_parameter("imu_topic", "/odin1/imu");

        // Camera SHM（替换原 img_topic）
        declare_parameter("camera/shm_name", std::string("/hikcamera_shm"));
        // camera_input_mode: "shm"（默认，现有行为）或 "ros_image"（MCAP 回放）
        declare_parameter("camera_input_mode", std::string("shm"));
        // camera/image_topic: ROS Image topic，仅 ros_image 模式使用
        declare_parameter("camera/image_topic", std::string("/fast_livo2/camera/bgr8"));
        // shm_image_width/height: raw SHM RGB resolution written by the HIK driver.
        // Must match hikcamera.yaml width/height (5472×3648).  These are the
        // source dimensions for ShmCamera; separate from cam_width/cam_height
        // which describe the VIO grayscale working resolution after downsample.
        declare_parameter("camera/shm_image_width", 5472);
        declare_parameter("camera/shm_image_height", 3648);

        // Sensor config
        declare_parameter("lidar_type", static_cast<int>(LidarType::ODIN1));
        declare_parameter("point_filter_num", 4);
        declare_parameter("blind", 0.1);
        declare_parameter("max_range", 30.0);
        declare_parameter("confidence_threshold", 35);

        // SLAM mode: 1=ONLY_LIO, 2=LIVO
        declare_parameter("slam_mode", static_cast<int>(SlamMode::ONLY_LIO));
        declare_parameter("img_en", false);

        // IMU
        declare_parameter("imu_en", true);
        declare_parameter("init_imu_num", 400);
        // 陀螺仪/加速度计噪声协方差 (rad/s、m/s^2，单位 per sqrt(Hz))。
        // 默认值 0.1 是 imu_processing.hpp 里未针对 Odin1 400Hz IMU 调过的占位值，
        // 在大场景(有效特征数千+)、帧耗时变长(~150ms)时会让 ESIKF 先验协方差
        // 过大，几乎完全依赖 LiDAR 观测，一旦匹配点数振荡就失去正则化，
        // 引发静止状态下的发散（已用 Oracle 诊断确认）。
        declare_parameter("gyr_cov", 0.01);
        declare_parameter("acc_cov", 0.01);
        // bias 随机游走协方差（bias 变化速率）。
        // 对齐官方 FAST-LIVO2 源码 IMU_Processing.cpp 构造函数硬编码值 0.1：
        //   cov_bias_gyr = V3D(0.1, 0.1, 0.1);
        //   cov_bias_acc = V3D(0.1, 0.1, 0.1);
        // 0.1 允许 bias 在几帧内快速收敛（官方默认不从 init 数据赋初值）。
        declare_parameter("gyr_bias_cov", 0.1);
        declare_parameter("acc_bias_cov", 0.1);
        // 默认0：Odin1 在 use_host_ros_time=0 下各 topic 已共享统一时钟，
        // 详见 config/odin_livo2.yaml 内说明，通常不需要调整。
        declare_parameter("imu_time_offset", 0.0);
        declare_parameter("img_time_offset", 0.0);

        // Gravity alignment (SPARK-FAST-LIO): correct roll/pitch drift after
        // a stationary period. Threshold is |acc_norm - 9.81| in m/s².
        declare_parameter("gravity_alignment_en", true);
        declare_parameter("gravity_acc_thresh", 0.3);

        // Extrinsics: LiDAR w.r.t. IMU
        declare_parameter("extrinsic_T", std::vector<double> { 0.0, 0.0, 0.0 });
        declare_parameter(
            "extrinsic_R", std::vector<double> { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        // Extrinsics: Camera w.r.t. LiDAR (Rcl, Pcl)
        declare_parameter("Pcl", std::vector<double> { 0.0, 0.0, 0.0 });
        declare_parameter(
            "Rcl", std::vector<double> { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 });
        declare_parameter("extrinsic_est_en", true);

        // Camera intrinsics (pinhole)
        declare_parameter("cam_fx", 500.0);
        declare_parameter("cam_fy", 500.0);
        declare_parameter("cam_cx", 1368.0);
        declare_parameter("cam_cy", 912.0);
        declare_parameter("cam_width", 2736);
        declare_parameter("cam_height", 1824);
        declare_parameter("img_scale", 0.5);

        // VIO
        declare_parameter("patch_size", 8);
        declare_parameter("patch_pyramid_level", 4);
        // This is only a sensor-time association window. It does not gate or
        // reject any ESIKF state update.
        declare_parameter("camera/sync_tolerance_sec", 0.12);
        declare_parameter("camera/queue_size", 200);
        declare_parameter("camera/sync_diag_en", true);

        // Map (downsampling + voxel map config)
        declare_parameter("filter_size_surf", 0.1);
        declare_parameter("filter_size_map", 0.15);
        declare_parameter("cube_side_length", 1000.0);

        // Voxel Map config (从 FAST-LIVO2 voxel_map.cpp loadVoxelConfig 移植)
        declare_parameter("voxel_map/max_layer", 1);
        declare_parameter("voxel_map/voxel_size", 0.5);
        declare_parameter("voxel_map/min_eigen_value", 0.01);
        declare_parameter("voxel_map/sigma_num", 3.0);
        declare_parameter("voxel_map/beam_err", 0.02);
        declare_parameter("voxel_map/dept_err", 0.05);
        declare_parameter(
            "voxel_map/layer_init_num", std::vector<int64_t> { 5LL, 5LL, 5LL, 5LL, 5LL });
        declare_parameter("voxel_map/max_points_num", 50);
        declare_parameter("voxel_map/max_iterations", 5);
        declare_parameter("voxel_map/map_sliding_en", false);
        declare_parameter("voxel_map/half_map_size", 100);
        declare_parameter("voxel_map/sliding_thresh", 8.0);

        // PCD save
        declare_parameter("pcd_save_en", false);
        declare_parameter("pcd_save_interval", -1);
        declare_parameter("map_save_path", std::string("/tmp/fast_livo2_map.pcd"));
        declare_parameter("map_save_trigger", std::string("/tmp/fast_livo2_save_map"));
        // FIXME: 首帧 BuildVoxelMap 后，voxel map 里的平面约束还很稀疏，
        // 接下来几帧 ESIKF 还在收敛（实测 frame2→frame7 average residual
        // 从 0.0236 单调降到 0.0189，effective feature 从 2168 涨到 2673），
        // state_.pos_end/rot_end 在这几帧里逐帧微调。若从 frame2 就开始
        // 累积 pcd_accumulated_，同一块静止表面会用几个还没收敛、逐帧变化
        // 的位姿投影到世界系，叠加起来在导出地图里表现为从传感器原点发散
        // 出去的"射线"伪影（静止测试时最明显，因为设备没动，射线不会被
        // 运动带来的新视角覆盖/稀释掉）。跳过前 N 帧的地图累积，等 ESIKF
        // 收敛稳定后才开始记录永久地图，从根源上避免这类伪影而不是靠
        // 后处理（voxel 降采样对射线伪影效果有限——射线上的点在空间上本
        // 就稀疏分散，不会被同一个体素的去重命中；离群点剔除倒是能清掉
        // 大部分，但会连带损失一些正常的稀疏区域点，不如从源头避免）。
        declare_parameter("pcd_save_warmup_frames", 30);
    }

    void load_parameters() {
        lidar_topic_ = get_parameter("lidar_topic").as_string();
        imu_topic_   = get_parameter("imu_topic").as_string();
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
        const double gyr_cov      = get_parameter("gyr_cov").as_double();
        const double acc_cov      = get_parameter("acc_cov").as_double();
        imu_process_.set_gyr_cov(Eigen::Vector3d(gyr_cov, gyr_cov, gyr_cov));
        imu_process_.set_acc_cov(Eigen::Vector3d(acc_cov, acc_cov, acc_cov));
        const double gyr_bias_cov = get_parameter("gyr_bias_cov").as_double();
        const double acc_bias_cov = get_parameter("acc_bias_cov").as_double();
        imu_process_.set_gyr_bias_cov(Eigen::Vector3d(gyr_bias_cov, gyr_bias_cov, gyr_bias_cov));
        imu_process_.set_acc_bias_cov(Eigen::Vector3d(acc_bias_cov, acc_bias_cov, acc_bias_cov));

        // 外参加载
        auto ext_t = get_parameter("extrinsic_T").as_double_array();
        if (ext_t.size() != 3) {
            throw std::runtime_error(
                "extrinsic_T must have exactly 3 elements, got " + std::to_string(ext_t.size()));
        }
        auto ext_r = get_parameter("extrinsic_R").as_double_array();
        if (ext_r.size() != 9) {
            throw std::runtime_error(
                "extrinsic_R must have exactly 9 elements, got " + std::to_string(ext_r.size()));
        }
        Eigen::Vector3d t_li(ext_t[0], ext_t[1], ext_t[2]);
        Eigen::Matrix3d r_li;
        r_li << ext_r[0], ext_r[1], ext_r[2], ext_r[3], ext_r[4], ext_r[5], ext_r[6], ext_r[7],
            ext_r[8];
        imu_process_.set_extrinsic(t_li, r_li);

        img_time_offset_    = get_parameter("img_time_offset").as_double();
        imu_time_offset_    = get_parameter("imu_time_offset").as_double();
        img_scale_          = get_parameter("img_scale").as_double();
        gravity_acc_thresh_ = get_parameter("gravity_acc_thresh").as_double();
        if (!get_parameter("gravity_alignment_en").as_bool()) {
            imu_process_.disable_gravity_est();
        }

        camera_queue_size_ =
            static_cast<size_t>(std::max<int64_t>(1, get_parameter("camera/queue_size").as_int()));
        camera_sync_tolerance_sec_ = get_parameter("camera/sync_tolerance_sec").as_double();
        camera_sync_diag_en_       = get_parameter("camera/sync_diag_en").as_bool();
        camera_queue_.set_max_size(camera_queue_size_);

        // ── Camera 初始化（仅 LIVO 模式）──
        // 支持两种输入模式：shm（默认，现有行为）和 ros_image（MCAP 回放）
        if (img_en_) {
            camera_input_mode_ = get_parameter("camera_input_mode").as_string();
            if (camera_input_mode_ != "shm" && camera_input_mode_ != "ros_image") {
                throw std::runtime_error("camera_input_mode must be 'shm' or 'ros_image', got: '"
                    + camera_input_mode_ + "'");
            }
            camera_image_topic_ = get_parameter("camera/image_topic").as_string();
            RosImageCamera::validate_topic_not_empty(camera_input_mode_, camera_image_topic_);

            cam_width_  = get_parameter("cam_width").as_int();
            cam_height_ = get_parameter("cam_height").as_int();

            if (cam_width_ <= 0 || cam_height_ <= 0) {
                throw std::runtime_error("LIVO: invalid camera size (cam_width/height)");
            }

            if (camera_input_mode_ == "shm") {
                const std::string shm_name = get_parameter("camera/shm_name").as_string();
                int shm_image_w            = get_parameter("camera/shm_image_width").as_int();
                int shm_image_h            = get_parameter("camera/shm_image_height").as_int();
                if (shm_image_w <= 0 || shm_image_h <= 0) {
                    throw std::runtime_error("LIVO: invalid SHM image size "
                                             "(shm_image_width/height)");
                }
                camera_ = std::make_unique<ShmCamera>(
                    shm_name, shm_image_w, shm_image_h, cam_width_, cam_height_, img_time_offset_);
                RCLCPP_INFO(get_logger(),
                    "Camera SHM: '%s' source=%dx%d → target=%dx%d gray (RGB→GRAY+resize), "
                    "offset=%.3fs",
                    shm_name.c_str(), shm_image_w, shm_image_h, cam_width_, cam_height_,
                    img_time_offset_);
            } else {
                RCLCPP_INFO(get_logger(), "Camera ROS Image: topic='%s' -> %dx%d gray, queue=%zu",
                    camera_image_topic_.c_str(), cam_width_, cam_height_, camera_queue_size_);
            }
        }

        // ── VIOManager 初始化（仅 LIVO 模式）──
        if (img_en_) {
            auto pcl_ = get_parameter("Pcl").as_double_array();
            if (pcl_.size() != 3) {
                throw std::runtime_error(
                    "Pcl must have exactly 3 elements, got " + std::to_string(pcl_.size()));
            }
            auto rcl_ = get_parameter("Rcl").as_double_array();
            if (rcl_.size() != 9) {
                throw std::runtime_error(
                    "Rcl must have exactly 9 elements, got " + std::to_string(rcl_.size()));
            }
            V3D pcl(pcl_[0], pcl_[1], pcl_[2]);
            M3D rcl;
            rcl << rcl_[0], rcl_[1], rcl_[2], rcl_[3], rcl_[4], rcl_[5], rcl_[6], rcl_[7], rcl_[8];

            // cam_width/cam_height already loaded above
            const double cam_fx           = get_parameter("cam_fx").as_double();
            const double cam_fy           = get_parameter("cam_fy").as_double();
            const double cam_cx           = get_parameter("cam_cx").as_double();
            const double cam_cy           = get_parameter("cam_cy").as_double();
            const int patch_size          = get_parameter("patch_size").as_int();
            const int patch_pyramid_level = get_parameter("patch_pyramid_level").as_int();

            // 上游 setImuToLidarExtrinsic(extT, extR): Rli=R^T, Pli=-R^T*t
            // yaml extrinsic 与 LIO 相同: p_imu = R_il * p_lidar + t_il
            const M3D Rli = r_li.transpose();
            const V3D Pli = -r_li.transpose() * t_li;

            const bool identity_cam = rcl.isIdentity(1e-6) && pcl.norm() < 1e-6;
            if (identity_cam) {
                RCLCPP_WARN(get_logger(),
                    "LIVO: Rcl/Pcl still identity/zero — fill real LiDAR-Camera "
                    "extrinsics before trusting visual updates");
            }
            if (cam_width_ <= 0 || cam_height_ <= 0 || cam_fx <= 1.0 || cam_fy <= 1.0) {
                throw std::runtime_error("LIVO: invalid camera intrinsics/size "
                                         "(cam_fx/fy/width/height)");
            }

            vio_manager_.init(cam_fx, cam_fy, cam_cx, cam_cy, cam_width_, cam_height_, rcl, pcl,
                Rli, Pli, patch_size, patch_pyramid_level, /*grid_size=*/20,
                /*normal_en=*/true, /*ncc_en=*/true,
                /*img_point_cov=*/100.0, /*ncc_thre=*/0.6,
                /*max_iterations=*/5);
            RCLCPP_INFO(get_logger(),
                "VIO extrinsics: Rli=R_il^T applied; cam %dx%d fx=%.1f fy=%.1f", cam_width_,
                cam_height_, cam_fx, cam_fy);
        }

        // 降采样参数
        filter_size_surf_ = get_parameter("filter_size_surf").as_double();

        // 参数合法性校验
        if (preprocess_.point_filter_num <= 0) {
            throw std::runtime_error("point_filter_num must be > 0");
        }
        if (filter_size_surf_ <= 0.0) {
            throw std::runtime_error("filter_size_surf must be > 0");
        }

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
        for (auto v : layer_init_raw)
            voxel_config_.layer_init_num_.push_back(static_cast<int>(v));
        while (voxel_config_.layer_init_num_.size() < 5) {
            voxel_config_.layer_init_num_.push_back(5);
        }

        // 将 voxel_config 复制到 VoxelMapManager
        voxel_map_.config_setting_ = voxel_config_;

        // 设置 VoxelMapManager 外参
        voxel_map_.extR_ = r_li;
        voxel_map_.extT_ = t_li;

        // PCD
        pcd_save_en_            = get_parameter("pcd_save_en").as_bool();
        pcd_save_interval_      = get_parameter("pcd_save_interval").as_int();
        map_save_path_          = get_parameter("map_save_path").as_string();
        save_trigger_path_      = get_parameter("map_save_trigger").as_string();
        pcd_save_warmup_frames_ = get_parameter("pcd_save_warmup_frames").as_int();

        RCLCPP_INFO(get_logger(), "LiDAR topic: %s", lidar_topic_.c_str());
        RCLCPP_INFO(get_logger(), "IMU   topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(get_logger(),
            "VoxelMap: max_layer=%d, voxel_size=%.3f, max_iter=%d, plane_thresh=%.4f",
            voxel_config_.max_layer_, voxel_config_.max_voxel_size_, voxel_config_.max_iterations_,
            voxel_config_.planner_threshold_);
    }

    // ── 相机线程管理 ─────────────────────────────────────────────

    /// 仅在 img_en_ 为 true 时启动相机输入。
    /// shm 模式：打开 SHM 并启动采集线程，打开失败抛异常。
    /// ros_image 模式：创建 ROS Image 订阅，无需线程。
    void start_camera() {
        if (!img_en_) return;

        if (camera_input_mode_ == "shm") {
            if (!camera_) return;
            auto open_result = camera_->open();
            if (!open_result) {
                throw std::runtime_error(
                    "LIVO: failed to open camera SHM: " + open_result.error().message);
            }
            RCLCPP_INFO(get_logger(), "Camera SHM opened, starting capture thread");
            camera_running_.store(true);
            camera_thread_ = std::thread([this]() { camera_loop(); });
        } else if (camera_input_mode_ == "ros_image") {
            sub_image_ = create_subscription<sensor_msgs::msg::Image>(camera_image_topic_,
                rclcpp::SensorDataQoS(),
                [this](const sensor_msgs::msg::Image::SharedPtr msg) { on_image(msg); });
            RCLCPP_INFO(get_logger(), "Camera ROS Image subscription created on '%s'",
                camera_image_topic_.c_str());
        }
    }

    /// Set running=false, then join the camera thread unconditionally.
    /// wait_next uses 200ms timeout, so join is bounded by that + conversion.
    /// Also joins if the thread already exited itself (running flag already false).
    /// ros_image mode: no thread to join; subscription is RAII.
    void stop_camera() {
        if (camera_input_mode_ == "shm") {
            camera_running_.store(false);
            if (camera_thread_.joinable()) {
                camera_thread_.join();
            }
        }
    }

    /// 相机采集循环：阻塞等待新 SHM 帧，转灰度 → 入队
    void camera_loop() {
        while (camera_running_.load()) {
            auto result = camera_->wait_next(std::chrono::milliseconds(200));
            if (!result) {
                const auto& err = result.error();
<<<<<<< HEAD
                if (err.code == FrameReadErrorCode::Timeout
                    || err.code == FrameReadErrorCode::ShmError) {
                    continue;
=======
                if (err.code == ShmCameraErrorCode::Timeout) {
                    continue;           // timeout: normal, retry
>>>>>>> feat/lio-tuning-safety-gate
                }
                // Fatal reader error: log once, terminate worker
                RCLCPP_ERROR(get_logger(), "Camera SHM fatal read error: %s", err.message.c_str());
                camera_running_.store(false);
                break;
            }

            std::ignore = camera_queue_.push(std::move(*result));
            request_processing();
        }
    }

    /// ROS Image 回调（ros_image 模式）：验证 BGR8，转灰度 → 入队
    void on_image(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto frame = RosImageCamera::convert(*msg, cam_width_, cam_height_, ++ros_image_seq_);
        if (!frame.has_value()) return;
        std::ignore = camera_queue_.push(std::move(*frame));
        request_processing();
    }

    // ── 订阅 ────────────────────────────────────────────────────
    void init_subscribers() {
        auto qos = rclcpp::SensorDataQoS();

        sub_lidar_ = create_subscription<sensor_msgs::msg::PointCloud2>(lidar_topic_, qos,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { on_lidar(msg); });

        sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic_,
            rclcpp::QoS(rclcpp::KeepLast(4000)).reliable(),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) { on_imu(msg); });
    }

    void init_publishers() {
        pub_odom_  = create_publisher<nav_msgs::msg::Odometry>("/fast_livo2/odom", 10);
        pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("/fast_livo2/cloud_world", 10);
        pub_cloud_lidar_ =
            create_publisher<sensor_msgs::msg::PointCloud2>("/fast_livo2/cloud_lidar", 10);
        pub_path_       = create_publisher<nav_msgs::msg::Path>("/fast_livo2/path", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // ── 回调 ────────────────────────────────────────────────────
    void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(imu_buf_mutex_);
            ImuData d;
            d.timestamp = rclcpp::Time(msg->header.stamp).seconds() + imu_time_offset_;
            d.acc       = { msg->linear_acceleration.x, msg->linear_acceleration.y,
                msg->linear_acceleration.z };
            d.gyro = { msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z };
            imu_buf_.push_back(d);
        }
        request_processing();
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
        request_processing();
    }

    void request_processing() {
        {
            std::lock_guard<std::mutex> lock(process_signal_mutex_);
            process_requested_ = true;
        }
        process_signal_cv_.notify_one();
    }

    void trim_consumed_imu() {
        if (!imu_process_.imu_en) return;

        const double consumed_until = imu_process_.last_prop_end_time();
        if (consumed_until <= 0.0) return;

        std::lock_guard<std::mutex> lock(imu_buf_mutex_);
        const auto first_unconsumed =
            std::upper_bound(imu_buf_.begin(), imu_buf_.end(), consumed_until,
                [](double timestamp, const ImuData& imu) { return timestamp < imu.timestamp; });
        imu_buf_.erase(imu_buf_.begin(), first_unconsumed);
    }

    void processing_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(process_signal_mutex_);
            process_signal_cv_.wait(
                lock, [this]() { return process_requested_ || !processing_running_.load(); });
            if (!processing_running_.load()) return;
            process_requested_ = false;
            lock.unlock();
            process_frame();
        }
    }

    double camera_match_time(double frame_beg, double frame_end) const {
        (void)frame_end;
        return frame_beg;
    }

    // 从相机队列中选时间最接近的一帧并消费之。
    // 超过 camera/sync_tolerance_sec 容忍范围时返回 false，跳过本帧 VIO。
    bool get_nearest_image(double target_time, double frame_duration, cv::Mat& out_gray) {
        const auto nearest      = camera_queue_.nearest_timestamp(target_time);
        const auto oldest       = camera_queue_.oldest_timestamp();
        const size_t queue_size = camera_queue_.size();
        auto result = camera_queue_.take_nearest(target_time, camera_sync_tolerance_sec_);
        if (!result) {
            if (camera_sync_diag_en_) {
                const double nearest_dt = nearest ? (*nearest - target_time) : 0.0;
                const double oldest_age = oldest ? (target_time - *oldest) : 0.0;
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "VIO sync miss: target=%.3f tol=%.3fs frame_dt=%.1fms "
                    "queue=%zu nearest_dt=%s%.1fms oldest_age=%s%.1fms",
                    target_time, camera_sync_tolerance_sec_, frame_duration * 1000.0, queue_size,
                    nearest ? "" : "n/a ", nearest ? nearest_dt * 1000.0 : 0.0,
                    oldest ? "" : "n/a ", oldest ? oldest_age * 1000.0 : 0.0);
            }
            return false;
        }

        const double dt = result->timestamp_seconds - target_time;
        if (camera_sync_diag_en_) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                "VIO sync hit: target=%.3f image=%.3f dt=%.1fms frame_dt=%.1fms queue=%zu",
                target_time, result->timestamp_seconds, dt * 1000.0, frame_duration * 1000.0,
                queue_size);
        }
        out_gray = std::move(result->gray);
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
    //   7. (LIVO) VIO 光度更新
    //   8. 发布 odom / cloud_world / path / TF
    //   9. PCD 累积（可选）
    //
    // 简化（相对原版）:
    //   - 跳过曝光时间估计 (inv_expo_time)
    //   - 跳过平面可视化发布 (pubVoxelMap)
    //   - 跳过 pose_output txt 日志
    // ══════════════════════════════════════════════════════════════

    void process_frame() {
        using clock = std::chrono::high_resolution_clock;
        auto t0     = clock::now();

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
            lidar_msg = lidar_buf_.front(); // 窥视，暂不 pop
        }

        // ── 2. 预处理: ROS PointCloud2 → PointCloudT (raw) ──
        // IMU callbacks retry the same front LiDAR frame until IMU catches up.
        // Cache the preprocessed front frame so those retries do not repeatedly
        // parse tens of thousands of points and starve image callbacks.
        auto raw_cloud   = std::make_shared<PointCloudT>();
        double frame_beg = 0.0;
        double frame_end = 0.0;
        if (pending_lidar_msg_.get() == lidar_msg.get() && pending_raw_cloud_) {
            raw_cloud = pending_raw_cloud_;
            frame_beg = pending_frame_beg_;
            frame_end = pending_frame_end_;
        } else {
            preprocess_.process(lidar_msg, raw_cloud);
            frame_beg = rclcpp::Time(lidar_msg->header.stamp).seconds();
            frame_end = frame_beg;
            for (size_t i = 0; i < raw_cloud->points.size(); ++i) {
                double pt_time = frame_beg + raw_cloud->points[i].curvature / 1000.0;
                if (pt_time > frame_end) frame_end = pt_time;
            }
            if (frame_end <= frame_beg) frame_end = frame_beg + 0.005;

            pending_lidar_msg_ = lidar_msg;
            pending_raw_cloud_ = raw_cloud;
            pending_frame_beg_ = frame_beg;
            pending_frame_end_ = frame_end;
        }
        if (raw_cloud->empty()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000, "Empty point cloud after preprocessing");
            // 空点云无法恢复，此帧真正丢弃（非 IMU 未就绪的可重试场景）
            std::lock_guard<std::mutex> lock(lidar_buf_mutex_);
            if (!lidar_buf_.empty()) lidar_buf_.pop_front();
            pending_lidar_msg_.reset();
            pending_raw_cloud_.reset();
            return;
        }

        // ── 3. 收集 IMU 数据 ──
        const double frame_duration = frame_end - frame_beg;

        MeasureGroup meas;
        meas.lidar_beg_time = frame_beg;
        meas.lidar_end_time = frame_end;
        meas.lidar          = raw_cloud;

        if (imu_process_.imu_en) {
            std::lock_guard<std::mutex> lock(imu_buf_mutex_);
            // IMU 窗口取所有 <= frame_end 的样本（不设下界），由
            // undistort_pcl 内部的 prop_beg_time 门控自动跳过早于上帧末
            // （last_prop_end_time_）的部分，从而覆盖 [last_prop_end_time_,
            // frame_end] 的完整积分区间，不依赖 frame_beg 本身。
            meas.imu.reserve(512);
            for (size_t i = 0; i < imu_buf_.size(); ++i) {
                const auto& imu = imu_buf_[i];
                if (imu.timestamp <= frame_end) {
                    meas.imu.push_back(imu);
                }
            }
            // IMU 还没追上这一帧的结束时间：不丢帧，直接 return 重试
            // （帧还在 lidar_buf_.front()，下次任意回调触发时会重新窥视）。
            if (imu_buf_.empty() || imu_buf_.back().timestamp < frame_end) {
                return;
            }
            if (meas.imu.size() < 2) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Insufficient IMU data (%zu samples) for frame [%.3f, %.3f]", meas.imu.size(),
                    frame_beg, frame_end);
                return;
            }
        }

        // IMU 数据已集齐，正式消费这一帧
        {
            std::lock_guard<std::mutex> lock(lidar_buf_mutex_);
            if (!lidar_buf_.empty()) lidar_buf_.pop_front();
        }
        pending_lidar_msg_.reset();
        pending_raw_cloud_.reset();

        // ── 4. IMU 去畸变 ──
        auto feats_undistort = std::make_shared<PointCloudT>();
        bool imu_ok          = imu_process_.process(meas, state_, feats_undistort);
        trim_consumed_imu();
        if (!imu_ok) {
            return;
        }

        if (feats_undistort->empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Empty undistorted point cloud");
            return;
        }

        // Publish undistorted LiDAR-frame cloud for downstream consumers (KISS-Matcher etc.)
        {
            sensor_msgs::msg::PointCloud2 lidar_cloud_msg;
            pcl::toROSMsg(*feats_undistort, lidar_cloud_msg);
            lidar_cloud_msg.header.stamp =
                rclcpp::Time(static_cast<int64_t>(std::llround(frame_end * 1e9)));
            lidar_cloud_msg.header.frame_id = "odin1_base_link";
            pub_cloud_lidar_->publish(lidar_cloud_msg);
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
                int64_t ix     = static_cast<int64_t>(std::floor(pt.x * inv_leaf));
                int64_t iy     = static_cast<int64_t>(std::floor(pt.y * inv_leaf));
                int64_t iz     = static_cast<int64_t>(std::floor(pt.z * inv_leaf));
                uint64_t key   = (static_cast<uint64_t>(ix + 32768) << 42)
                    | (static_cast<uint64_t>(iy + 32768) << 21)
                    | (static_cast<uint64_t>(iz + 32768));
                if (voxel_map.emplace(key, feats_down_body->size()).second) {
                    feats_down_body->push_back(pt);
                }
            }
            feats_down_body->header   = feats_undistort->header;
            feats_down_body->is_dense = false;
        }

        int feats_down_size = static_cast<int>(feats_down_body->points.size());
        auto t_down         = clock::now();

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
                "First frame: built voxel map with %d points (%ld root voxels)", feats_down_size,
                voxel_map_.voxel_map_.size());
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

        // ── 8.5. 重力对齐恢复 (SPARK-FAST-LIO inspired) ──
        // When stationary, accumulate gravity direction in world frame.
        // On motion onset, compute rotation to re-align roll/pitch — useful
        // for recovery after occlusion-induced drift.
        if (imu_process_.gravity_align && lidar_map_inited_) {
            const double acc_norm =
                (imu_process_.mean_acc_norm() > 1e-6) ? imu_process_.mean_acc_norm() : 9.81;
            const double acc_std = std::abs(acc_norm - 9.81);

            if (acc_std < gravity_acc_thresh_ && feats_down_size > 500) {
                // Stationary: accumulate world-frame gravity direction
                const V3D grav_world = state_.gravity;
                if (!gravity_is_stationary_) {
                    gravity_stationary_acc_   = Eigen::Vector3d::Zero();
                    gravity_stationary_count_ = 0;
                }
                gravity_stationary_acc_ += grav_world;
                gravity_stationary_count_++;
                gravity_is_stationary_ = true;
            } else if (gravity_is_stationary_ && gravity_stationary_count_ > 0) {
                // Motion onset after stationary period:
                // compute rotation aligning estimated gravity → expected gravity
                const V3D avg_grav = gravity_stationary_acc_ / gravity_stationary_count_;
                const V3D expected_grav(0.0, 0.0, -9.81);
                const V3D axis         = avg_grav.cross(expected_grav);
                const double axis_norm = axis.norm();
                if (axis_norm > 1e-6) {
                    const double angle = std::acos(std::clamp(
                        avg_grav.dot(expected_grav) / (avg_grav.norm() * expected_grav.norm()),
                        -1.0, 1.0));
                    const V3D u        = axis / axis_norm;
                    const double ca = std::cos(angle), sa = std::sin(angle);
                    Eigen::Matrix3d R_align;
                    R_align << ca + u.x() * u.x() * (1 - ca), u.x() * u.y() * (1 - ca) - u.z() * sa,
                        u.x() * u.z() * (1 - ca) + u.y() * sa,
                        u.y() * u.x() * (1 - ca) + u.z() * sa, ca + u.y() * u.y() * (1 - ca),
                        u.y() * u.z() * (1 - ca) - u.x() * sa,
                        u.z() * u.x() * (1 - ca) - u.y() * sa,
                        u.z() * u.y() * (1 - ca) + u.x() * sa, ca + u.z() * u.z() * (1 - ca);
                    state_.rot_end = R_align * state_.rot_end;
                    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                        "Gravity alignment: %.1fdeg correction (%.0f stationary frames)",
                        angle * 180.0 / M_PI, static_cast<double>(gravity_stationary_count_));
                }
                gravity_is_stationary_ = false;
            }
            if (!gravity_is_stationary_) {
                gravity_is_stationary_ = false;
            }
        }

        auto t2 = clock::now();

        // ── 9. 增量更新体素地图 ──
        // 用更新后的状态重新变换点云，计算世界帧协方差
        auto world_lidar = std::make_shared<PointCloudT>();
        voxel_map_.TransformLidar(state_.rot_end, state_.pos_end, feats_down_body, world_lidar);

        for (size_t i = 0; i < static_cast<size_t>(feats_down_size); i++) {
            voxel_map_.pv_list_[i].point_w =
                V3D(world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z);

            M3D point_crossmat = voxel_map_.cross_mat_list_[i];
            M3D var            = voxel_map_.body_cov_list_[i];
            var                = (state_.rot_end * voxel_map_.extR_) * var
                                   * (state_.rot_end * voxel_map_.extR_).transpose()
                + (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose()
                + state_.cov.block<3, 3>(3, 3);
            voxel_map_.pv_list_[i].var = var;
        }

        // Guard: skip voxel map update when effective features are too low
        // to prevent a transient state divergence from permanently corrupting
        // the voxel map (e.g. during aggressive turns where LiDAR constraints
        // momentarily weaken).  State estimation itself still runs so the
        // ESIKF can recover on subsequent frames with stronger constraints.
        const double eff_ratio =
            static_cast<double>(voxel_map_.effct_feat_num_) / static_cast<double>(feats_down_size);
        const bool safe_to_update = (eff_ratio >= 0.15 && voxel_map_.effct_feat_num_ >= 600);
        if (safe_to_update) {
            voxel_map_.UpdateVoxelMap(voxel_map_.pv_list_);
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Skip UpdateVoxelMap: eff=%d/down=%d (%.1f%%) — state may be unreliable",
                voxel_map_.effct_feat_num_, feats_down_size, eff_ratio * 100.0);
        }

        auto t3 = clock::now();

        // ── 9.5. VIO 光度更新（仅 LIVO 模式，且找到时间匹配的图像帧）──
        if (img_en_) {
            cv::Mat gray;
            const double target_image_time = camera_match_time(frame_beg, frame_end);
            if (get_nearest_image(target_image_time, frame_duration, gray)) {
                vio_manager_.state_          = &state_;
                vio_manager_.state_propagat_ = &state_propagat;
                vio_manager_.processFrame(gray, voxel_map_.pv_list_, voxel_map_.voxel_map_);
            } else {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                    "VIO: no camera frame within %.0fms of target t=%.3f",
                    camera_sync_tolerance_sec_ * 1000.0, target_image_time);
            }
        }

        // VIO can update state_. Rebuild the published cloud from the final
        // state so cloud, odometry, path and TF describe the same pose.
        voxel_map_.TransformLidar(state_.rot_end, state_.pos_end, feats_down_body, world_lidar);

        // ── 10. 地图滑动 ──
        if (voxel_config_.map_sliding_en) {
            voxel_map_.mapSliding();
        }

        auto t4 = clock::now();

        // ── 11. 发布 odometry ──
        publish_odometry(frame_end);

        // ── 12. 发布世界系点云 ──
        const int64_t output_ns = static_cast<int64_t>(std::llround(frame_end * 1e9));
        builtin_interfaces::msg::Time output_stamp;
        output_stamp.sec     = static_cast<int32_t>(output_ns / 1000000000LL);
        output_stamp.nanosec = static_cast<uint32_t>(output_ns % 1000000000LL);
        publish_cloud_world(world_lidar, output_stamp);

        // ── 13. 发布 path ──
        publish_path(output_stamp);

        // ── 14. 发布 TF ──
        publish_tf(output_stamp);

        // ── 15. PCD 累积 + 定期刷盘 ──
        if (pcd_save_en_ && frame_count_ >= pcd_save_warmup_frames_) {
            for (const auto& pt : world_lidar->points) {
                pcd_accumulated_.points.push_back(pt);
            }
            // 定期刷盘：每 pcd_save_interval_ 帧保存一次，清空内存防 OOM
            if (pcd_save_interval_ > 0) {
                pcd_save_period_counter_++;
                if (pcd_save_period_counter_ >= pcd_save_interval_) {
                    pcd_save_period_counter_ = 0;
                    std::string path         = map_save_path_;
                    auto dot                 = path.rfind(".pcd");
                    if (dot != std::string::npos) {
                        path.insert(dot, "." + std::to_string(pcd_save_seq_++));
                    } else {
                        path += "." + std::to_string(pcd_save_seq_++);
                    }
                    RCLCPP_INFO(get_logger(), "Periodic PCD save: %s (%ld points)", path.c_str(),
                        pcd_accumulated_.points.size());
                    pcl::io::savePCDFileBinary(path, pcd_accumulated_);
                    pcd_accumulated_.clear();
                }
            }
        }

        // ── 16. 计时日志 ──
        frame_count_++;
        auto t_total = std::chrono::duration<double>(t4 - t0).count();
        avg_time_    = avg_time_ * (frame_count_ - 1) / frame_count_ + t_total / frame_count_;

        RCLCPP_INFO(get_logger(),
            "[ LIO ] frame %d | down: %.1fms | ICP: %.1fms | update: %.1fms"
            " | total: %.1fms (avg: %.1fms) | pts: %d/%d/%d",
            frame_count_, std::chrono::duration<double>(t_down - t0).count() * 1000,
            std::chrono::duration<double>(t2 - t1).count() * 1000,
            std::chrono::duration<double>(t3 - t2).count() * 1000, t_total * 1000, avg_time_ * 1000,
            static_cast<int>(feats_undistort->size()), feats_down_size, voxel_map_.effct_feat_num_);

        // ── 17. LIO drift diagnostics (ONLY_LIO, 1Hz throttled) ──
        if (lidar_map_inited_) {
            if (!drift_ref_captured_) {
                drift_ref_position_  = state_.pos_end;
                last_drift_position_ = state_.pos_end;
                drift_ref_captured_  = true;
            }
            total_path_ += (state_.pos_end - last_drift_position_).norm();
            last_drift_position_ = state_.pos_end;

            const auto diag = compute_lio_drift_metrics(
                frame_count_, state_, state_propagat, drift_ref_position_);

            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "[DRIFT] fr=%d | disp=%.3fm path=%.1fm | spd=%.2f | dP=%.3fm | dR=%.2fdeg"
                " | bg=%.5f | ba=%.5f | cov_r=%.2e cov_p=%.2e cov_v=%.2e",
                diag.frame_count, diag.displacement, total_path_, diag.speed_norm,
                diag.pos_correction, diag.ang_correction, diag.gyro_bias_norm, diag.accel_bias_norm,
                diag.cov_rot, diag.cov_pos, diag.cov_vel);
        }
    }

    // ── 发布函数 ────────────────────────────────────────────────

    void publish_odometry(double timestamp) {
        auto odom_msg            = nav_msgs::msg::Odometry();
        odom_msg.header.stamp    = rclcpp::Time(static_cast<int64_t>(timestamp * 1e9));
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

        // 协方差: state 内部布局 [rot(0-2), pos(3-5)]，
        // nav_msgs/Odometry 要求 [pos(0-2), rot(3-5)]，需重排。
        constexpr std::array<int, 6> pose_idx { 3, 4, 5, 0, 1, 2 };
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                odom_msg.pose.covariance[i * 6 + j] = state_.cov(pose_idx[i], pose_idx[j]);
            }
        }
        pub_odom_->publish(odom_msg);
    }

    void publish_cloud_world(
        const PointCloudT::Ptr& cloud, const builtin_interfaces::msg::Time& stamp) {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.stamp    = stamp;
        cloud_msg.header.frame_id = "odom";
        pub_cloud_->publish(cloud_msg);
    }

    void publish_path(const builtin_interfaces::msg::Time& stamp) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp    = stamp;
        pose.header.frame_id = "odom";
        pose.pose.position.x = state_.pos_end.x();
        pose.pose.position.y = state_.pos_end.y();
        pose.pose.position.z = state_.pos_end.z();

        Eigen::Quaterniond q(state_.rot_end);
        pose.pose.orientation.w = q.w();
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();

        path_msg_.header.stamp    = stamp;
        path_msg_.header.frame_id = "odom";
        path_msg_.poses.push_back(pose);
        pub_path_->publish(path_msg_);
    }

    void publish_tf(const builtin_interfaces::msg::Time& stamp) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp    = stamp;
        tf.header.frame_id = "odom";
        tf.child_frame_id  = "base_link";

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
            RCLCPP_INFO(get_logger(), "Saving %ld points to %s", pcd_accumulated_.points.size(),
                map_save_path_.c_str());
            // HACK: 从 ASCII 换成 Binary——长时间建图累积到百万级点时，
            // ASCII 每点一行文本格式化 I/O 耗时和文件体积都数倍于 binary，
            // 手动触发保存场景下用户在等这个操作完成，不该让格式选择成为
            // 瓶颈。Foxglove/PCL/CloudCompare 等下游工具都原生支持读取
            // binary PCD，不存在兼容性代价。
            pcl::io::savePCDFileBinary(map_save_path_, pcd_accumulated_);
            RCLCPP_INFO(
                get_logger(), "PCD saved (%ld points, binary).", pcd_accumulated_.points.size());
        } else if (pcd_save_en_) {
            RCLCPP_WARN(get_logger(), "PCD save requested but no points accumulated yet.");
        }
    }

    // 检查 map_save_path_ 同目录下是否存在触发文件（.script/odin-map-save
    // 负责 touch 它），存在则立即保存当前累积点云并删除触发文件（避免
    // 下一轮定时器重复触发）。不停止建图，保存完继续累积。
    void check_save_trigger() {
        namespace fs                = std::filesystem;
        const fs::path trigger_path = save_trigger_path_;
        std::error_code ec;
        if (!fs::exists(trigger_path, ec) || ec) {
            return;
        }
        RCLCPP_INFO(get_logger(), "Save-map trigger detected, saving current map...");
        save_pcd();
        fs::remove(trigger_path, ec);
        if (ec) {
            RCLCPP_WARN(
                get_logger(), "Failed to remove save-map trigger file: %s", ec.message().c_str());
        }
    }

    // ── 成员变量 ─────────────────────────────────────────────────
    std::string lidar_topic_, imu_topic_;
    int slam_mode_                    = SlamMode::ONLY_LIO;
    bool img_en_                      = false;
    double img_time_offset_           = 0.0;
    double imu_time_offset_           = 0.0;
    double img_scale_                 = 0.5;
    double camera_sync_tolerance_sec_ = 0.12;
    size_t camera_queue_size_         = 30;
    bool camera_sync_diag_en_         = true;
    double filter_size_surf_          = 0.1;
    bool pcd_save_en_                 = false;
    int pcd_save_interval_            = -1;
    int pcd_save_seq_                 = 0;
    int pcd_save_period_counter_      = 0;
    std::string map_save_path_;
    std::string save_trigger_path_ = "/tmp/fast_livo2_save_map";
    rclcpp::TimerBase::SharedPtr save_trigger_timer_;
    int pcd_save_warmup_frames_ = 30;

    Preprocess preprocess_;
    ImuProcess imu_process_;

    // ── SHM 相机（仅 LIVO 模式）──
<<<<<<< HEAD
    std::unique_ptr<ShmCamera> camera_; // hikcamera SHMRead adapter
=======
    std::unique_ptr<ShmCamera> camera_; // raw imageSHM adapter
>>>>>>> feat/lio-tuning-safety-gate
    std::thread camera_thread_;         // 采集线程
    std::atomic<bool> camera_running_ { false };
    CameraFrameQueue camera_queue_ { 5 }; // bounded ≤5, at-most-once per sequence

    // ── ROS Image 相机（仅 ros_image 模式）──
    std::string camera_input_mode_ { "shm" }; // "shm" or "ros_image"
    std::string camera_image_topic_;          // ROS Image topic name
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
    std::atomic<uint64_t> ros_image_seq_ { 0 }; // monotonically-increasing local sequence
    int cam_width_ { 0 };
    int cam_height_ { 0 };

    // 缓冲区（各自有独立互斥锁）
    std::mutex imu_buf_mutex_;
    std::mutex lidar_buf_mutex_;
    std::mutex process_signal_mutex_;
    std::condition_variable process_signal_cv_;
    std::atomic<bool> processing_running_ { false };
    std::thread processing_thread_;
    bool process_requested_ = false;
    std::vector<ImuData, Eigen::aligned_allocator<ImuData>> imu_buf_;
    std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> lidar_buf_;
    sensor_msgs::msg::PointCloud2::SharedPtr pending_lidar_msg_;
    PointCloudT::Ptr pending_raw_cloud_;
    double pending_frame_beg_ = 0.0;
    double pending_frame_end_ = 0.0;

    // 体素地图管理器（替换原来的 ikd-Tree）
    VoxelMapManager voxel_map_;
    VoxelMapConfig voxel_config_;
    StatesGroup state_; // ESIKF 状态（持续跨帧更新）
    bool lidar_map_inited_ = false;

    // LIO drift diagnostics reference (captured after first post-init posterior)
    V3D drift_ref_position_  = V3D::Zero();
    bool drift_ref_captured_ = false;
    double total_path_       = 0.0;
    V3D last_drift_position_ = V3D::Zero();

    // Gravity alignment
    double gravity_acc_thresh_  = 0.3;
    bool gravity_is_stationary_ = false;
    V3D gravity_stationary_acc_ { 0, 0, 0 };
    int gravity_stationary_count_ = 0;

    // 视觉直接法前端（仅 LIVO 模式启用）
    VIOManager vio_manager_;

    // 发布者 & TF
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_lidar_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    nav_msgs::msg::Path path_msg_;

    // PCD 累积
    PointCloudT pcd_accumulated_;
    int frame_count_ = 0;
    double avg_time_ = 0.0;
};

} // namespace radar::fast_livo2

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<radar::fast_livo2::LivMapperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
