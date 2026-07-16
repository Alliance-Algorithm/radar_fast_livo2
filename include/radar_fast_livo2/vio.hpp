#pragma once
// vio.hpp - 视觉直接法前端（LIVO 模式）
// 原始来源: hku-mars/FAST-LIVO2/include/vio.h (VIOManager, SubSparseMap, Warp)
//           hku-mars/FAST-LIVO2/include/visual_point.h (VisualPoint)
//           hku-mars/FAST-LIVO2/include/feature.h (Feature)
// 移植: radar_fast_livo2 项目（ROS2 + 纯 Eigen，无 Sophus/vikit）
//
// 与原版差异:
//   - 无 Sophus::SE3 → 相机位姿直接用 (M3D rot, V3D pos) 表示
//   - 无 vikit::AbstractCamera → 相机投影内联为 pinhole 公式
//   - 18 维状态（无 inv_expo_time）→ 光度雅可比矩阵 H 为 N×6（无曝光列）
//   - LiDAR 地图用 VoxelMapManager（voxel_map.hpp），视觉地图另建
//     独立哈希表 feat_map_（VOXEL_LOCATION → VisualPoint 列表）
//
// 算法流程 (每帧):
//   1. retrieveFromVisualSparseMap: 已有 VisualPoint 投影匹配 → SubSparseMap
//   2. computeJacobianAndUpdateEKF: 金字塔粗到细，光度残差 ESIKF 迭代更新
//   3. generateVisualMapPoints: LiDAR 平面点生成新 VisualPoint
//   4. updateVisualMapPoints: 现有 VisualPoint 按视角变化添加新观测
//   5. updateReferencePatch: 重新评分 NCC，选最佳参考 patch

#include "radar_fast_livo2/common_lib.hpp"
#include "radar_fast_livo2/esikf_state.hpp"
#include "radar_fast_livo2/voxel_map.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace radar::fast_livo2 {

// ── Feature: 单次观测的 patch ──────────────────────────────────────
// 原始来源: FAST-LIVO2/include/feature.h
struct Feature {
    Eigen::Vector2d px_;              // 创建时的像素坐标 (level 0)
    Eigen::Vector3d f_;               // 单位承载向量 (归一化相机坐标)
    int             level_ = 0;       // 提取时所在金字塔层
    std::vector<float> patch_;        // patch_size×patch_size 参考图像亮度（供 NCC/误差参考用）
    // img_ 只存 px_ 周围的裁剪区域（非整帧 clone），供 warpAffine 重采样。
    // patch_origin_ 是裁剪区域左上角在原图坐标系中的偏移，warpAffine
    // 采样时需要用 px_ - patch_origin_ 换算成裁剪图内坐标；px_ 本身仍保持
    // 原图坐标，因为 getWarpMatrixAffine(Homography) 要用它算相机内参投影。
    cv::Mat         img_;
    Eigen::Vector2i patch_origin_ = Eigen::Vector2i::Zero();
    float           score_ = 0.0f;    // NCC 质量分（越高越好）
    M3D             T_f_w_rot_;       // 创建该观测时的相机位姿：world→camera 旋转
    V3D             T_f_w_pos_;       // world→camera 平移（即 Pcw）

    Feature(Eigen::Vector2d px, Eigen::Vector3d f, cv::Mat img, M3D rot, V3D pos, int level)
        : px_(std::move(px)), f_(std::move(f)), level_(level), img_(std::move(img)),
          T_f_w_rot_(std::move(rot)), T_f_w_pos_(std::move(pos)) {}
};

// ── VisualPoint: 视觉地图中的 3D 点 + 观测集合 ──────────────────────
// 原始来源: FAST-LIVO2/include/visual_point.h
class VisualPoint {
public:
    V3D  pos_;                        // world frame 3D 坐标
    V3D  normal_        = V3D::Zero();// 表面法向量（来自 LiDAR 平面）
    M3D  covariance_    = M3D::Zero();// 位置协方差（来自 pointWithVar.var）
    bool is_normal_initialized_ = false;

    std::list<std::unique_ptr<Feature>> obs_;   // 所有历史观测（patch）
    Feature* ref_patch_ = nullptr;              // 当前最佳参考观测（原始指针，指向 obs_ 中的元素）

    explicit VisualPoint(const V3D& pos) : pos_(pos) {}

    // 从 obs_ 中找到与当前相机位置视角最接近的观测
    // 对应 FAST-LIVO2 VisualPoint::getCloseViewObs
    [[nodiscard]] Feature* getCloseViewObs(const V3D& cam_pos_w) const {
        if (obs_.empty()) return nullptr;
        V3D dir_new = (pos_ - cam_pos_w).normalized();
        Feature* best = nullptr;
        double best_cos = -1.0;
        for (const auto& f : obs_) {
            V3D dir_obs = (pos_ - (-f->T_f_w_rot_.transpose() * f->T_f_w_pos_)).normalized();
            double cos_angle = dir_new.dot(dir_obs);
            if (cos_angle > best_cos) { best_cos = cos_angle; best = f.get(); }
        }
        return best;
    }
};

// ── VOXEL_POINTS_VIO: 视觉地图哈希桶（每个体素内的 VisualPoint 列表） ──
struct VOXEL_POINTS_VIO {
    std::vector<std::unique_ptr<VisualPoint>> points;
};

// ── SubSparseMap: 当前帧视觉工作集 ─────────────────────────────────
// 原始来源: FAST-LIVO2/include/vio.h SubSparseMap
struct SubSparseMap {
    std::vector<float>              errors;         // 光度误差（用于统计/调试）
    std::vector<std::vector<float>> warp_patch;      // 每点的 warp 后参考 patch（含全部金字塔层，扁平存储）
    std::vector<int>                search_levels;   // 每点选用的搜索金字塔层
    std::vector<VisualPoint*>       voxel_points;     // 匹配到的 VisualPoint 指针（非 owning）

    void clear() {
        errors.clear();
        warp_patch.clear();
        search_levels.clear();
        voxel_points.clear();
    }
};

// ── VIOManager: 视觉直接法前端主类 ──────────────────────────────────
// 原始来源: FAST-LIVO2/include/vio.h VIOManager
//
// 注意: 不包含 ROS 发布功能（由 LivMapperNode 负责），保持纯算法逻辑，
//       与 VoxelMapManager 的设计原则一致。
class VIOManager {
public:
    VIOManager() = default;

    // ── 相机内参 ────────────────────────────────────────────────
    double fx_ = 500.0, fy_ = 500.0, cx_ = 640.0, cy_ = 360.0;
    int    width_ = 1280, height_ = 720;

    // ── 外参: Camera-from-LiDAR (Rcl,Pcl) + LiDAR-from-IMU (Rli,Pli) ──
    // Rci = Rcl * Rli, Pci = Rcl * Pli + Pcl（Camera-from-IMU，构造时算出）
    M3D Rcl_ = M3D::Identity(), Rli_ = M3D::Identity();
    V3D Pcl_ = V3D::Zero(),     Pli_ = V3D::Zero();
    M3D Rci_ = M3D::Identity();
    V3D Pci_ = V3D::Zero();

    // 派生外参雅可比（对应 FAST-LIVO2 VIOManager::initializeVIO）
    M3D Jdphi_dR_ = M3D::Identity();  // ∂φ_ci / ∂R_wi
    M3D Jdp_dR_   = M3D::Zero();      // ∂p_ci  / ∂R_wi (via Pic 反对称)

    // ── 算法配置 ────────────────────────────────────────────────
    int    patch_size_          = 8;
    int    patch_pyramid_level_ = 4;
    int    patch_size_total_    = 64;   // patch_size^2
    int    patch_size_half_     = 4;
    int    grid_size_           = 20;   // 特征筛选网格大小 (px)
    int    max_iterations_      = 5;
    double img_point_cov_       = 100.0;
    double ncc_thre_            = 0.6;
    bool   normal_en_           = true;  // 用平面法向量做单应性 warp
    bool   ncc_en_               = true;  // NCC outlier 剔除
    double visual_voxel_size_   = 0.5;   // 视觉地图哈希体素大小 (m)，独立于 LiDAR voxel_map

    // ── 当前帧状态（由外部在调用 processFrame 前设置）──────────────
    StatesGroup* state_           = nullptr;  // 当前 ESIKF 状态（会被本类更新）
    StatesGroup* state_propagat_  = nullptr;  // IMU 正向传播先验（用于 ESIKF 残差项）

    // 当前帧相机位姿: world→camera（每次 processFrame 开始时由 state_ 计算）
    M3D Rcw_ = M3D::Identity();
    V3D Pcw_ = V3D::Zero();

    // ── 视觉地图（与 LiDAR 体素地图分离的独立哈希表）────────────────
    std::unordered_map<VOXEL_LOCATION, VOXEL_POINTS_VIO> feat_map_;

    // ── 每帧工作集 ──────────────────────────────────────────────
    SubSparseMap visual_submap_;
    int total_points_ = 0;

    // ── 初始化: 设置内参/外参/配置，计算派生雅可比 ──────────────────
    void init(double fx, double fy, double cx, double cy, int width, int height,
              const M3D& Rcl, const V3D& Pcl, const M3D& Rli, const V3D& Pli,
              int patch_size, int patch_pyramid_level, int grid_size,
              bool normal_en, bool ncc_en, double img_point_cov, double ncc_thre,
              int max_iterations);

    // ── 主入口: 处理一帧图像 ─────────────────────────────────────
    // @param img         灰度图 (CV_8UC1)
    // @param pg           当前帧 LiDAR 点（world frame + 平面法向量，来自 VoxelMapManager）
    // @param plane_map    LiDAR 体素地图（只读，用于深度一致性检查 + 新点生成）
    void processFrame(const cv::Mat& img,
                       std::vector<pointWithVar>& pg,
                       const std::unordered_map<VOXEL_LOCATION,
                           std::unique_ptr<VoxelOctoTree>>& plane_map);

private:
    // ── 相机投影（内联 pinhole，替代 vikit::AbstractCamera） ────────
    [[nodiscard]] Eigen::Vector2d world2cam(const Eigen::Vector3d& p_cam) const {
        return {fx_ * p_cam.x() / p_cam.z() + cx_, fy_ * p_cam.y() / p_cam.z() + cy_};
    }
    [[nodiscard]] Eigen::Vector3d cam2world(const Eigen::Vector2d& px) const {
        return {(px.x() - cx_) / fx_, (px.y() - cy_) / fy_, 1.0};
    }
    [[nodiscard]] bool isInFrame(const Eigen::Vector2d& px, int boundary) const {
        return px.x() >= boundary && px.x() < width_ - boundary
            && px.y() >= boundary && px.y() < height_ - boundary;
    }

    // ── 核心算法函数（对应 FAST-LIVO2 src/vio.cpp） ──────────────────
    // 返回 false 表示 p.z() 过小（点在相机后方或退化），J 已清零，
    // 调用方（updateState）应跳过该点本次迭代的贡献。
    [[nodiscard]] bool computeProjectionJacobian(const V3D& p, Eigen::Matrix<double, 2, 3>& J) const;

    void getImagePatch(const cv::Mat& img, const Eigen::Vector2d& pc,
                        float* patch_tmp, int level) const;

    static void getWarpMatrixAffine(
        const VIOManager& self, const Eigen::Vector2d& px_ref, const Eigen::Vector3d& f_ref,
        double depth_ref, const M3D& R_cur_ref, const V3D& t_cur_ref,
        int level_ref, int pyramid_level, int halfpatch_size, Eigen::Matrix2d& A_cur_ref);

    void getWarpMatrixAffineHomography(
        const Eigen::Vector2d& px_ref, const Eigen::Vector3d& xyz_ref,
        const Eigen::Vector3d& normal_ref, const M3D& R_cur_ref, const V3D& t_cur_ref,
        int level_ref, Eigen::Matrix2d& A_cur_ref) const;

    void warpAffine(const Eigen::Matrix2d& A_cur_ref, const cv::Mat& img_ref,
                     const Eigen::Vector2d& px_ref, int level_ref, int search_level,
                     int pyramid_level, int halfpatch_size, float* patch) const;

    [[nodiscard]] static int getBestSearchLevel(const Eigen::Matrix2d& A_cur_ref, int max_level);

    [[nodiscard]] static double calculateNCC(const float* ref_patch, const float* cur_patch,
                                              int patch_size);

    // 从 img 中裁出以 pc 为中心、半径 kFeatureCropRadius 的区域（deep copy），
    // 供 Feature::img_ 存储；避免每个观测 clone 整帧图（~1MB→~9KB/观测）。
    // 半径推导见 vio.cpp 注释：4级金字塔 + det≤16 warp 时的最大采样偏移。
    [[nodiscard]] cv::Mat cropAroundPixel(const cv::Mat& img, const Eigen::Vector2d& pc,
                                           Eigen::Vector2i& origin_out) const;

    // ── 每帧流水线阶段 ────────────────────────────────────────────
    void retrieveFromVisualSparseMap(
        const cv::Mat& img, const std::vector<pointWithVar>& pg,
        const std::unordered_map<VOXEL_LOCATION, std::unique_ptr<VoxelOctoTree>>& plane_map);

    void computeJacobianAndUpdateEKF(const cv::Mat& img);

    void updateState(const cv::Mat& img, int level);

    void generateVisualMapPoints(
        const cv::Mat& img, std::vector<pointWithVar>& pg);

    void updateVisualMapPoints(const cv::Mat& img);

    void updateReferencePatch(
        const std::unordered_map<VOXEL_LOCATION, std::unique_ptr<VoxelOctoTree>>& plane_map);

    // ── 网格筛选状态（每帧重置） ───────────────────────────────────
    int grid_n_width_ = 0, grid_n_height_ = 0;
    std::vector<VisualPoint*> grid_best_point_;
    std::vector<float>        grid_best_score_;

    void resetGrid();
};

}  // namespace radar::fast_livo2
