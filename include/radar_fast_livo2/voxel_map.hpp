#pragma once
// voxel_map.hpp - 哈希体素八叉树 + 平面拟合 + ESIKF 地图管理
// 原始来源: hku-mars/FAST-LIVO2/include/voxel_map.h
// 移植: radar_fast_livo2 项目（ROS2 + 纯 Eigen，无 Sophus）
//
// 核心类:
//   - VoxelOctoTree: 八叉树体素节点，支持点集累积 → 平面拟合 → 递归分裂
//   - VoxelMapManager: 哈希表管理体素地图，ESIKF 迭代匹配 + 增量更新

#include "radar_fast_livo2/common_lib.hpp"
#include "radar_fast_livo2/esikf_state.hpp"

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace radar::fast_livo2 {

// ── 哈希参数 ──────────────────────────────────────────────────────
constexpr int64_t VOXELMAP_HASH_P = 116101;
constexpr int64_t VOXELMAP_MAX_N = 10000000000;

// ── VoxelMapConfig: 体素地图参数 ──────────────────────────────────
struct VoxelMapConfig {
    double      max_voxel_size_    = 0.5;   // 根体素大小 (m)
    int         max_layer_         = 1;     // 八叉树最大层数
    int         max_iterations_    = 5;     // ESIKF 最大迭代次数
    std::vector<int> layer_init_num_{5, 5, 5, 5, 5};  // 每层最小点数阈值
    int         max_points_num_    = 50;    // 体素最大点数（超出则停止更新）
    double      planner_threshold_ = 0.01;  // 平面性阈值（最小特征值）
    double      beam_err_          = 0.02;  // 角度误差 (deg)
    double      dept_err_          = 0.05;  // 深度误差 (m)
    double      sigma_num_         = 3.0;   // 残差 outlier 阈值（sigma 倍数）
    bool        is_pub_plane_map_  = false; // TODO: 是否发布平面可视化（ROS2 暂未实现）

    // 局部地图滑动窗口
    bool        map_sliding_en = false;
    double      sliding_thresh = 8.0;
    int         half_map_size  = 100;
};

// ── PointToPlane: 点到平面残差 ────────────────────────────────────
struct alignas(16) PointToPlane {
    Eigen::Vector3d     point_b_;       // LiDAR body frame 坐标
    Eigen::Vector3d     point_w_;       // world frame 坐标
    Eigen::Vector3d     normal_;        // 平面法向量 (world frame)
    Eigen::Vector3d     center_;        // 平面中心 (world frame)
    Eigen::Matrix<double, 6, 6> plane_var_;  // 平面不确定性 (6×6: center + normal)
    M3D                 body_cov_;      // body frame 下点协方差
    int                 layer_  = 0;    // 匹配所在八叉树层级
    double              d_      = 0.0;  // 平面方程: n·x + d = 0
    double              eigen_value_ = 0.0;
    bool                is_valid_ = false;
    float               dis_to_plane_ = 0.0f;  // 点到平面残差
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ── VoxelPlane: 体素内拟合的平面 ──────────────────────────────────
struct VoxelPlane {
    Eigen::Vector3d     center_;
    Eigen::Vector3d     normal_;
    Eigen::Vector3d     y_normal_;
    Eigen::Vector3d     x_normal_;
    Eigen::Matrix3d     covariance_;
    Eigen::Matrix<double, 6, 6> plane_var_;
    float               radius_           = 0.0f;
    float               min_eigen_value_  = 1.0f;
    float               mid_eigen_value_  = 1.0f;
    float               max_eigen_value_  = 1.0f;
    float               d_                = 0.0f;
    int                 points_size_      = 0;
    bool                is_plane_         = false;
    bool                is_init_          = false;
    int                 id_               = 0;
    bool                is_update_        = false;

    VoxelPlane() {
        plane_var_  = Eigen::Matrix<double, 6, 6>::Zero();
        covariance_ = Eigen::Matrix3d::Zero();
        center_     = Eigen::Vector3d::Zero();
        normal_     = Eigen::Vector3d::Zero();
    }
};

// ── VOXEL_LOCATION: 体素哈希键 ────────────────────────────────────
struct VOXEL_LOCATION {
    int64_t x, y, z;
    VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}
    bool operator==(const VOXEL_LOCATION& other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

}  // namespace radar::fast_livo2

// std::hash 特化（必须在 std 命名空间）
namespace std {
template <>
struct hash<radar::fast_livo2::VOXEL_LOCATION> {
    int64_t operator()(const radar::fast_livo2::VOXEL_LOCATION& s) const {
        using radar::fast_livo2::VOXELMAP_HASH_P;
        using radar::fast_livo2::VOXELMAP_MAX_N;
        return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
    }
};
}  // namespace std

namespace radar::fast_livo2 {

// ── 前向声明 ─────────────────────────────────────────────────────
void calcBodyCov(const Eigen::Vector3d& pb, float range_inc, float degree_inc, Eigen::Matrix3d& cov);

// ── VoxelOctoTree: 八叉树体素节点 ──────────────────────────────────
//
// 每个节点存储点集 temp_points_，当点数超过阈值时调用 init_plane()
// 做 PCA + 特征值分析判断平面性:
//   - 若最小特征值 < planer_threshold_ → is_plane_=true，停止分裂
//   - 否则 → octo_state_=1，调用 cut_octo_tree() 递归分裂 8 个子节点
//
// 递归深度受 max_layer_ 限制，达到最大层数后停止分裂。
class VoxelOctoTree {
public:
    VoxelOctoTree() = default;

    std::vector<pointWithVar> temp_points_;
    std::unique_ptr<VoxelPlane>           plane_ptr_;
    int   layer_            = 0;
    int   octo_state_       = 0;  // 0=叶节点, 1=继续分裂
    std::array<std::unique_ptr<VoxelOctoTree>, 8> leaves_{};
    std::array<double, 3>                  voxel_center_{0.0, 0.0, 0.0};
    std::vector<int> layer_init_num_;
    float quater_length_       = 0.0f;
    float planer_threshold_    = 0.01f;
    int   points_size_threshold_ = 5;
    int   update_size_threshold_ = 5;
    int   max_points_num_      = 50;
    int   max_layer_           = 1;
    int   new_points_          = 0;
    bool  init_octo_           = false;
    bool  update_enable_       = true;

    VoxelOctoTree(int max_layer, int layer, int points_size_threshold,
                  int max_points_num, float planer_threshold)
        : max_layer_(max_layer), layer_(layer),
          points_size_threshold_(points_size_threshold),
          max_points_num_(max_points_num),
          planer_threshold_(planer_threshold) {
        temp_points_.clear();
        octo_state_       = 0;
        new_points_       = 0;
        update_size_threshold_ = 5;
        init_octo_        = false;
        update_enable_    = true;
        plane_ptr_ = std::make_unique<VoxelPlane>();
    }

    VoxelOctoTree(const VoxelOctoTree&) = delete;
    VoxelOctoTree& operator=(const VoxelOctoTree&) = delete;
    VoxelOctoTree(VoxelOctoTree&&) = default;
    VoxelOctoTree& operator=(VoxelOctoTree&&) = default;
    ~VoxelOctoTree() = default;

    // 平面拟合: PCA → 特征值分析 → 判断平面性
    void init_plane(const std::vector<pointWithVar>& points, VoxelPlane* plane);

    // 初始化八叉树（首次建图调用）
    void init_octo_tree();

    // 递归分裂（非平面节点）
    void cut_octo_tree();

    // 增量更新（新帧点插入已有体素）
    void UpdateOctoTree(const pointWithVar& pv);

    // 查找到达叶节点（平面节点或最大层数节点）
    [[nodiscard]] VoxelOctoTree* find_correspond(Eigen::Vector3d pw);

    // 插入新点（建图时用）
    [[nodiscard]] VoxelOctoTree* Insert(const pointWithVar& pv);
};

// ── VoxelMapManager: 体素地图管理器 ───────────────────────────────
//
// 核心流程:
//   1. BuildVoxelMap()   — 首帧建图：点云→哈希体素分组→每体素 InitOctoTree
//   2. StateEstimation() — ESIKF 迭代：点到平面残差 + 雅可比 + 卡尔曼更新
//   3. UpdateVoxelMap()  — 增量更新地图：新点插入体素，重新拟合平面
//
// 注意: 不包含 ROS 发布功能（由 LivMapperNode 负责），保持纯算法逻辑。
class VoxelMapManager {
public:
    VoxelMapManager() = default;

    VoxelMapConfig config_setting_;
    int    current_frame_id_ = 0;
    std::unordered_map<VOXEL_LOCATION, std::unique_ptr<VoxelOctoTree>> voxel_map_;

    // 当前帧点云（由外部设置）
    PointCloudT::Ptr feats_undistort_;
    PointCloudT::Ptr feats_down_body_;
    PointCloudT::Ptr feats_down_world_;

    // 外参
    M3D extR_ = M3D::Identity();
    V3D extT_ = V3D::Zero();

    // 状态
    StatesGroup state_;
    V3D position_last_ = V3D::Zero();
    V3D last_slide_position = V3D::Zero();

    // 当前帧临时数据
    int feats_down_size_ = 0;
    int effct_feat_num_  = 0;
    std::vector<M3D>        cross_mat_list_;
    std::vector<M3D>        body_cov_list_;
    std::vector<pointWithVar> pv_list_;
    std::vector<PointToPlane> ptpl_list_;

    // ── 核心方法 ────────────────────────────────────────────────

    /// ESIKF 主循环: 迭代匹配 + 卡尔曼更新
    /// @param state_propagat  IMU 正向传播得到的状态先验
    void StateEstimation(StatesGroup& state_propagat);

    /// 坐标变换: LiDAR body frame → world frame
    void TransformLidar(const M3D& rot, const V3D& t,
                        const PointCloudT::Ptr& input_cloud,
                        PointCloudT::Ptr& trans_cloud);

    /// 首次建图: 将 feat_down_world 点插入哈希体素表，初始化八叉树
    void BuildVoxelMap();

    /// 体素着色（用于可视化，非核心算法）
    [[nodiscard]] Eigen::Vector3f RGBFromVoxel(const V3D& input_point);

    /// 增量地图更新: 将新扫描点插入体素
    void UpdateVoxelMap(const std::vector<pointWithVar>& input_points);

    /// OpenMP 并行构建点到平面残差列表（供 StateEstimation 调用）
    void BuildResidualListOMP(std::vector<pointWithVar>& pv_list,
                              std::vector<PointToPlane>& ptpl_list);

    /// 单个残差构建: 递归搜索体素树找到最近有效平面
    void build_single_residual(pointWithVar& pv, const VoxelOctoTree* current_octo,
                               int current_layer, bool& is_sucess, double& prob,
                               PointToPlane& single_ptpl);

    /// 局部地图滑动（内存管理，大规模场景用）
    void mapSliding();

    /// 清理超出边界的体素
    void clearMemOutOfMap(const int& x_max, const int& x_min,
                          const int& y_max, const int& y_min,
                          const int& z_max, const int& z_min);
};

}  // namespace radar::fast_livo2
