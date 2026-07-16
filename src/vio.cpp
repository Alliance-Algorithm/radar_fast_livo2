// vio.cpp - 视觉直接法前端实现（LIVO 模式）
// 原始来源: hku-mars/FAST-LIVO2/src/vio.cpp
// 移植: radar_fast_livo2 项目（ROS2 + 纯 Eigen，无 Sophus/vikit）

#include "radar_fast_livo2/vio.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace radar::fast_livo2 {

void VIOManager::init(double fx, double fy, double cx, double cy, int width, int height,
                       const M3D& Rcl, const V3D& Pcl, const M3D& Rli, const V3D& Pli,
                       int patch_size, int patch_pyramid_level, int grid_size,
                       bool normal_en, bool ncc_en, double img_point_cov, double ncc_thre,
                       int max_iterations) {
    fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
    width_ = width; height_ = height;
    Rcl_ = Rcl; Pcl_ = Pcl; Rli_ = Rli; Pli_ = Pli;

    patch_size_          = patch_size;
    patch_pyramid_level_ = patch_pyramid_level;
    patch_size_total_    = patch_size * patch_size;
    patch_size_half_     = patch_size / 2;
    grid_size_           = grid_size;
    normal_en_           = normal_en;
    ncc_en_              = ncc_en;
    img_point_cov_       = img_point_cov;
    ncc_thre_            = ncc_thre;
    max_iterations_      = max_iterations;

    // Camera-from-IMU 外参链: Rci = Rcl * Rli, Pci = Rcl * Pli + Pcl
    Rci_ = Rcl_ * Rli_;
    Pci_ = Rcl_ * Pli_ + Pcl_;

    // 派生雅可比（对应 FAST-LIVO2 VIOManager::initializeVIO）
    Jdphi_dR_ = Rci_;
    V3D Pic = -Rci_.transpose() * Pci_;
    Jdp_dR_  = -Rci_ * skewSym(Pic);

    grid_n_width_  = width_  / grid_size_ + 1;
    grid_n_height_ = height_ / grid_size_ + 1;
}

// ══════════════════════════════════════════════════════════════════
// computeProjectionJacobian — 投影雅可比 ∂π/∂P
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::computeProjectionJacobian
// π(P) = [fx*X/Z+cx, fy*Y/Z+cy]
// ══════════════════════════════════════════════════════════════════
bool VIOManager::computeProjectionJacobian(const V3D& p, Eigen::Matrix<double, 2, 3>& J) const {
    constexpr double kMinCamDepth = 1e-3;  // 1mm，低于此值视为退化/相机后方
    if (p.z() < kMinCamDepth) {
        J.setZero();
        return false;
    }
    const double z_inv   = 1.0 / p.z();
    const double z_inv_2 = z_inv * z_inv;
    J(0, 0) = fx_ * z_inv;
    J(0, 1) = 0.0;
    J(0, 2) = -fx_ * p.x() * z_inv_2;
    J(1, 0) = 0.0;
    J(1, 1) = fy_ * z_inv;
    J(1, 2) = -fy_ * p.y() * z_inv_2;
    return true;
}

// ══════════════════════════════════════════════════════════════════
// getImagePatch — 双线性插值提取 patch（任意金字塔层）
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::getImagePatch
// ══════════════════════════════════════════════════════════════════
void VIOManager::getImagePatch(const cv::Mat& img, const Eigen::Vector2d& pc,
                                float* patch_tmp, int level) const {
    const int scale = 1 << level;
    const int u_ref_i = static_cast<int>(std::floor(pc.x() / scale)) * scale;
    const int v_ref_i = static_cast<int>(std::floor(pc.y() / scale)) * scale;
    const float subpix_u = static_cast<float>(pc.x() - u_ref_i) / scale;
    const float subpix_v = static_cast<float>(pc.y() - v_ref_i) / scale;
    const float w_tl = (1.0f - subpix_u) * (1.0f - subpix_v);
    const float w_tr = subpix_u * (1.0f - subpix_v);
    const float w_bl = (1.0f - subpix_u) * subpix_v;
    const float w_br = subpix_u * subpix_v;

    for (int x = 0; x < patch_size_; x++) {
        const uint8_t* row = img.ptr<uint8_t>(v_ref_i - patch_size_half_ * scale + x * scale)
                            + (u_ref_i - patch_size_half_ * scale);
        for (int y = 0; y < patch_size_; y++, row += scale) {
            patch_tmp[patch_size_total_ * level + x * patch_size_ + y] =
                w_tl * row[0] + w_tr * row[scale]
              + w_bl * row[scale * img.cols] + w_br * row[scale * img.cols + scale];
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// getWarpMatrixAffine — 平面仿射 warp（无法向量时的退化方案）
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::getWarpMatrixAffine
//
// 取参考 patch 中心 + 两个半径偏移点，反投影到 3D（假设与中心同深度），
// 变换到当前帧，重新投影，差分得到 2×2 warp 矩阵。
// ══════════════════════════════════════════════════════════════════
void VIOManager::getWarpMatrixAffine(
    const VIOManager& self, const Eigen::Vector2d& px_ref, const Eigen::Vector3d& f_ref,
    double depth_ref, const M3D& R_cur_ref, const V3D& t_cur_ref,
    int level_ref, int pyramid_level, int halfpatch_size, Eigen::Matrix2d& A_cur_ref) {
    const V3D xyz_ref = f_ref * depth_ref;
    const double scale = static_cast<double>(1 << level_ref) * (1 << pyramid_level);

    V3D xyz_du_ref = self.cam2world(px_ref + Eigen::Vector2d(halfpatch_size, 0) * scale);
    V3D xyz_dv_ref = self.cam2world(px_ref + Eigen::Vector2d(0, halfpatch_size) * scale);
    xyz_du_ref *= xyz_ref.z() / xyz_du_ref.z();
    xyz_dv_ref *= xyz_ref.z() / xyz_dv_ref.z();

    const Eigen::Vector2d px_cur = self.world2cam(R_cur_ref * xyz_ref     + t_cur_ref);
    const Eigen::Vector2d px_du  = self.world2cam(R_cur_ref * xyz_du_ref  + t_cur_ref);
    const Eigen::Vector2d px_dv  = self.world2cam(R_cur_ref * xyz_dv_ref  + t_cur_ref);

    A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

// ══════════════════════════════════════════════════════════════════
// getWarpMatrixAffineHomography — 平面单应性 warp（已知法向量时）
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::getWarpMatrixAffineHomography
// H = R_cur_ref * (n·p * I - t * n^T)，t = 当前帧到参考帧的平移
// ══════════════════════════════════════════════════════════════════
void VIOManager::getWarpMatrixAffineHomography(
    const Eigen::Vector2d& px_ref, const Eigen::Vector3d& xyz_ref,
    const Eigen::Vector3d& normal_ref, const M3D& R_cur_ref, const V3D& t_cur_ref,
    int level_ref, Eigen::Matrix2d& A_cur_ref) const {
    // t_cur_ref 变换定义为 p_cur = R_cur_ref * p_ref + t_cur_ref，
    // 其逆变换的平移部分 = -R_cur_ref^T * t_cur_ref
    const V3D t = -R_cur_ref.transpose() * t_cur_ref;
    const M3D H_cur_ref = R_cur_ref
        * (normal_ref.dot(xyz_ref) * M3D::Identity() - t * normal_ref.transpose());

    constexpr int kHalfPatchSize = 4;
    const double scale = static_cast<double>(1 << level_ref);
    const V3D f_du_ref = cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * scale);
    const V3D f_dv_ref = cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * scale);

    const V3D f_cur    = H_cur_ref * xyz_ref;
    const V3D f_du_cur = H_cur_ref * f_du_ref;
    const V3D f_dv_cur = H_cur_ref * f_dv_ref;

    const Eigen::Vector2d px_cur    = world2cam(f_cur);
    const Eigen::Vector2d px_du_cur = world2cam(f_du_cur);
    const Eigen::Vector2d px_dv_cur = world2cam(f_dv_cur);

    A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
    A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

// ══════════════════════════════════════════════════════════════════
// warpAffine — 用仿射矩阵从参考图像重采样出当前帧视角下的 patch
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::warpAffine
// ══════════════════════════════════════════════════════════════════
void VIOManager::warpAffine(const Eigen::Matrix2d& A_cur_ref, const cv::Mat& img_ref,
                             const Eigen::Vector2d& px_ref, int level_ref, int search_level,
                             int pyramid_level, int halfpatch_size, float* patch) const {
    const int patch_size = halfpatch_size * 2;
    const Eigen::Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if (std::isnan(A_ref_cur(0, 0))) return;

    for (int y = 0; y < patch_size; ++y) {
        for (int x = 0; x < patch_size; ++x) {
            Eigen::Vector2f px_patch(static_cast<float>(x - halfpatch_size),
                                      static_cast<float>(y - halfpatch_size));
            px_patch *= static_cast<float>(1 << search_level);
            px_patch *= static_cast<float>(1 << pyramid_level);
            const Eigen::Vector2f px = A_ref_cur * px_patch + px_ref.cast<float>();

            float* dst = &patch[patch_size_total_ * pyramid_level + y * patch_size + x];
            if (px.x() < 0 || px.y() < 0
                || px.x() >= img_ref.cols - 1 || px.y() >= img_ref.rows - 1) {
                *dst = 0.0f;
                continue;
            }
            // 双线性插值（等价于原版 vikit::interpolateMat_8u）
            const int xi = static_cast<int>(px.x()), yi = static_cast<int>(px.y());
            const float ax = px.x() - xi, ay = px.y() - yi;
            const uint8_t* row0 = img_ref.ptr<uint8_t>(yi) + xi;
            const uint8_t* row1 = img_ref.ptr<uint8_t>(yi + 1) + xi;
            *dst = (1 - ax) * (1 - ay) * row0[0] + ax * (1 - ay) * row0[1]
                 + (1 - ax) * ay       * row1[0] + ax * ay       * row1[1];
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// getBestSearchLevel — 根据 warp 畸变程度选择金字塔搜索层
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::getBestSearchLevel
// ══════════════════════════════════════════════════════════════════
int VIOManager::getBestSearchLevel(const Eigen::Matrix2d& A_cur_ref, int max_level) {
    int search_level = 0;
    double D = A_cur_ref.determinant();
    while (D > 3.0 && search_level < max_level) {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

// ══════════════════════════════════════════════════════════════════
// calculateNCC — 归一化互相关（patch 质量评分）
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::calculateNCC
// ══════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════
// cropAroundPixel — 裁剪 Feature::img_ 存储区域（避免整帧 clone）
//
// 半径推导: warpAffine 在最粗金字塔层的最大采样偏移
//   r = halfpatch · 2^(search_level + pyramid_level_max) / σ_min(A_cur_ref)
// 取 halfpatch=4, pyramid_level_max=3, 各向同性 warp det≤16（对应
// getBestSearchLevel 收敛到 search_level=2）时 r=32px；
// 半径 48（96×96 裁剪）留出各向异性 warp（拉伸比 k≈2.25）的余量，
// 且远小于整帧 clone（~1MB → ~9KB/观测）。裁剪越界时 warpAffine 已有
// 边界检查会写 0（见 :168-172），越界只降质不崩溃。
// ══════════════════════════════════════════════════════════════════
cv::Mat VIOManager::cropAroundPixel(const cv::Mat& img, const Eigen::Vector2d& pc,
                                     Eigen::Vector2i& origin_out) const {
    constexpr int kFeatureCropRadius = 48;
    const int cx_px = static_cast<int>(std::lround(pc.x()));
    const int cy_px = static_cast<int>(std::lround(pc.y()));
    const int ox = std::clamp(cx_px - kFeatureCropRadius, 0, img.cols - 1);
    const int oy = std::clamp(cy_px - kFeatureCropRadius, 0, img.rows - 1);
    const int ex = std::clamp(cx_px + kFeatureCropRadius, 1, img.cols);
    const int ey = std::clamp(cy_px + kFeatureCropRadius, 1, img.rows);
    origin_out = Eigen::Vector2i(ox, oy);
    return img(cv::Rect(ox, oy, ex - ox, ey - oy)).clone();
}

double VIOManager::calculateNCC(const float* ref_patch, const float* cur_patch, int patch_size) {
    const double mean_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0) / patch_size;
    const double mean_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0) / patch_size;

    double numerator = 0.0, denom_ref = 0.0, denom_cur = 0.0;
    for (int i = 0; i < patch_size; i++) {
        const double dr = ref_patch[i] - mean_ref;
        const double dc = cur_patch[i] - mean_cur;
        numerator += dr * dc;
        denom_ref += dr * dr;
        denom_cur += dc * dc;
    }
    return numerator / std::sqrt(denom_ref * denom_cur + 1e-10);
}

// ══════════════════════════════════════════════════════════════════
// resetGrid — 每帧开始时清空网格筛选状态
// ══════════════════════════════════════════════════════════════════
void VIOManager::resetGrid() {
    const size_t n = static_cast<size_t>(grid_n_width_) * grid_n_height_;
    grid_best_point_.assign(n, nullptr);
    grid_best_score_.assign(n, 0.0f);
}

// ══════════════════════════════════════════════════════════════════
// retrieveFromVisualSparseMap — 已有 VisualPoint 投影匹配，构建 SubSparseMap
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::retrieveFromVisualSparseMap
//
// 流程: 遍历视觉地图哈希表 → 投影到当前帧 → 网格内选最优 → 计算仿射 warp →
//       多金字塔层提取 warp patch → 与当前帧 patch 比较（可选 NCC 剔除）
// ══════════════════════════════════════════════════════════════════
void VIOManager::retrieveFromVisualSparseMap(
    const cv::Mat& img, const std::vector<pointWithVar>& /*pg*/,
    const std::unordered_map<VOXEL_LOCATION, std::unique_ptr<VoxelOctoTree>>& /*plane_map*/) {
    resetGrid();
    const int boundary = patch_size_half_ * (1 << (patch_pyramid_level_ - 1)) + 1;

    // ── 1. 投影所有视觉地图点，网格内选 NCC/深度最优的一个 ──
    for (auto& [loc, bucket] : feat_map_) {
        for (auto& vp : bucket.points) {
            const V3D p_cam = Rcw_ * vp->pos_ + Pcw_;
            if (p_cam.z() <= 0.0) continue;
            const Eigen::Vector2d pc = world2cam(p_cam);
            if (!isInFrame(pc, boundary)) continue;

            const int gx = static_cast<int>(pc.x()) / grid_size_;
            const int gy = static_cast<int>(pc.y()) / grid_size_;
            const size_t idx = static_cast<size_t>(gy) * grid_n_width_ + gx;
            if (idx >= grid_best_score_.size()) continue;

            Feature* ref = vp->ref_patch_ ? vp->ref_patch_
                                          : vp->getCloseViewObs(-Rcw_.transpose() * Pcw_);
            if (ref == nullptr) continue;

            const float score = ref->score_;
            if (score > grid_best_score_[idx]) {
                grid_best_score_[idx] = score;
                grid_best_point_[idx] = vp.get();
            }
        }
    }

    // ── 2. 对每个网格winner: 计算仿射 warp + 提取多层 patch ──
    visual_submap_.clear();
    for (size_t idx = 0; idx < grid_best_point_.size(); ++idx) {
        VisualPoint* vp = grid_best_point_[idx];
        if (vp == nullptr) continue;

        Feature* ref = vp->ref_patch_ ? vp->ref_patch_
                                      : vp->getCloseViewObs(-Rcw_.transpose() * Pcw_);
        if (ref == nullptr) continue;

        // 参考帧位姿: p_cam_ref = ref->T_f_w_rot_ * p_world + ref->T_f_w_pos_
        // 当前-参考相对位姿: p_cam_cur = R_cur_ref * p_cam_ref + t_cur_ref
        const M3D R_cur_ref = Rcw_ * ref->T_f_w_rot_.transpose();
        const V3D t_cur_ref = Pcw_ - R_cur_ref * ref->T_f_w_pos_;

        Eigen::Matrix2d A_cur_ref;
        const double depth_ref = (ref->T_f_w_rot_ * vp->pos_ + ref->T_f_w_pos_).z();
        if (normal_en_ && vp->is_normal_initialized_) {
            const V3D xyz_ref = ref->T_f_w_rot_ * vp->pos_ + ref->T_f_w_pos_;
            const V3D normal_ref = ref->T_f_w_rot_ * vp->normal_;
            getWarpMatrixAffineHomography(ref->px_, xyz_ref, normal_ref,
                                           R_cur_ref, t_cur_ref, ref->level_, A_cur_ref);
        } else {
            getWarpMatrixAffine(*this, ref->px_, ref->f_, depth_ref,
                                R_cur_ref, t_cur_ref, ref->level_, 0, patch_size_half_, A_cur_ref);
        }

        const int search_level = getBestSearchLevel(A_cur_ref, patch_pyramid_level_ - 1);

        // ref->img_ 只存 ref->px_ 周围的裁剪区域，采样锚点需换算为裁剪图内坐标；
        // 仿射位移是平移不变量，重定基准点不影响 warp 本身的计算结果。
        const Eigen::Vector2d px_local = ref->px_ - ref->patch_origin_.cast<double>();
        std::vector<float> warp_patch(static_cast<size_t>(patch_size_total_) * patch_pyramid_level_, 0.0f);
        for (int lvl = 0; lvl < patch_pyramid_level_; ++lvl) {
            warpAffine(A_cur_ref, ref->img_, px_local, ref->level_,
                       search_level, lvl, patch_size_half_, warp_patch.data());
        }

        // 当前帧同一层 patch（用于 NCC 剔除判断）
        const V3D p_cam = Rcw_ * vp->pos_ + Pcw_;
        const Eigen::Vector2d pc = world2cam(p_cam);
        std::vector<float> cur_patch(static_cast<size_t>(patch_size_total_) * patch_pyramid_level_, 0.0f);
        getImagePatch(img, pc, cur_patch.data(), search_level);

        if (ncc_en_) {
            const double ncc = calculateNCC(
                &warp_patch[patch_size_total_ * search_level],
                &cur_patch[patch_size_total_ * search_level], patch_size_total_);
            if (ncc < ncc_thre_) continue;  // 视角变化过大/遮挡，剔除
        }

        visual_submap_.voxel_points.push_back(vp);
        visual_submap_.warp_patch.push_back(std::move(warp_patch));
        visual_submap_.search_levels.push_back(search_level);
        visual_submap_.errors.push_back(0.0f);
    }

    total_points_ = static_cast<int>(visual_submap_.voxel_points.size());
}

// ══════════════════════════════════════════════════════════════════
// updateState — 单个金字塔层的前向组合 ESIKF 光度更新
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::updateState
//
// 差异: 18 维状态无 inv_expo_time → H_sub 为 N×6（无曝光列），
//       曝光时间固定为 1.0（不做曝光估计）
// ══════════════════════════════════════════════════════════════════
void VIOManager::updateState(const cv::Mat& img, int level) {
    if (total_points_ == 0) return;

    const int H_DIM = total_points_ * patch_size_total_;
    Eigen::VectorXd z(H_DIM);
    Eigen::MatrixXd H_sub(H_DIM, 6);

    bool ekf_end = false;
    for (int iteration = 0; iteration < max_iterations_ && !ekf_end; ++iteration) {
        z.setZero();
        H_sub.setZero();

        const M3D Rwi = state_->rot_end;
        const V3D Pwi = state_->pos_end;
        Rcw_ = Rci_ * Rwi.transpose();
        Pcw_ = -Rci_ * Rwi.transpose() * Pwi + Pci_;
        const M3D Jdp_dt = Rci_ * Rwi.transpose();

        for (int i = 0; i < total_points_; ++i) {
            const int search_level = visual_submap_.search_levels[i];
            const int pyramid_level = level + search_level;
            const int scale = 1 << pyramid_level;

            VisualPoint* vp = visual_submap_.voxel_points[i];
            const V3D pf = Rcw_ * vp->pos_ + Pcw_;

            Eigen::Matrix<double, 2, 3> Jdpi;
            // ESIKF 迭代中 state_ 每轮都变，之前有效的点可能中途转到相机后方；
            // 跳过该点本轮贡献（z/H 对应行保持 setZero()，不参与本轮更新）。
            if (!computeProjectionJacobian(pf, Jdpi)) continue;
            const Eigen::Vector2d pc = world2cam(pf);
            const M3D p_hat = skewSym(pf);

            const int u_ref_i = static_cast<int>(std::floor(pc.x() / scale)) * scale;
            const int v_ref_i = static_cast<int>(std::floor(pc.y() / scale)) * scale;
            const float subpix_u = static_cast<float>(pc.x() - u_ref_i) / scale;
            const float subpix_v = static_cast<float>(pc.y() - v_ref_i) / scale;
            const float w_tl = (1 - subpix_u) * (1 - subpix_v);
            const float w_tr = subpix_u * (1 - subpix_v);
            const float w_bl = (1 - subpix_u) * subpix_v;
            const float w_br = subpix_u * subpix_v;

            const std::vector<float>& P = visual_submap_.warp_patch[i];

            for (int x = 0; x < patch_size_; x++) {
                const uint8_t* row = img.ptr<uint8_t>(v_ref_i + x * scale - patch_size_half_ * scale)
                                    + (u_ref_i - patch_size_half_ * scale);
                for (int y = 0; y < patch_size_; ++y, row += scale) {
                    const float du = 0.5f * (
                        (w_tl * row[scale] + w_tr * row[2 * scale]
                       + w_bl * row[scale * img.cols + scale] + w_br * row[scale * img.cols + 2 * scale])
                      - (w_tl * row[-scale] + w_tr * row[0]
                       + w_bl * row[scale * img.cols - scale] + w_br * row[scale * img.cols]));
                    const float dv = 0.5f * (
                        (w_tl * row[scale * img.cols] + w_tr * row[scale + scale * img.cols]
                       + w_bl * row[2 * scale * img.cols] + w_br * row[2 * scale * img.cols + scale])
                      - (w_tl * row[-scale * img.cols] + w_tr * row[-scale * img.cols + scale]
                       + w_bl * row[0] + w_br * row[scale]));

                    Eigen::Vector2d Jimg(du, dv);
                    Jimg /= scale;

                    const Eigen::Matrix<double, 1, 3> Jdphi = Jimg.transpose() * Jdpi * p_hat;
                    const Eigen::Matrix<double, 1, 3> Jdp   = -Jimg.transpose() * Jdpi;
                    const Eigen::Matrix<double, 1, 3> JdR   = Jdphi * Jdphi_dR_ + Jdp * Jdp_dR_;
                    const Eigen::Matrix<double, 1, 3> Jdt   = Jdp * Jdp_dt;

                    const double cur_value = w_tl * row[0] + w_tr * row[scale]
                                            + w_bl * row[scale * img.cols] + w_br * row[scale * img.cols + scale];
                    const double res = cur_value - P[patch_size_total_ * level + x * patch_size_ + y];

                    const int row_idx = i * patch_size_total_ + x * patch_size_ + y;
                    z(row_idx) = res;
                    H_sub.block<1, 3>(row_idx, 0) = JdR;
                    H_sub.block<1, 3>(row_idx, 3) = Jdt;
                }
            }
        }

        const Eigen::MatrixXd H_sub_T = H_sub.transpose();
        Eigen::Matrix<double, 6, 6> H_T_H = H_sub_T * H_sub;
        Eigen::Matrix<double, DIM_STATE, DIM_STATE> K_1 =
            Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
        K_1.block<6, 6>(0, 0) = (H_T_H + (state_->cov.block<6, 6>(0, 0) / img_point_cov_).inverse()).inverse();

        const Eigen::VectorXd HTz = H_sub_T * z;
        const VD<DIM_STATE> vec = (*state_propagat_) - (*state_);

        Eigen::Matrix<double, DIM_STATE, DIM_STATE> G =
            Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
        G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H;

        VD<DIM_STATE> solution = VD<DIM_STATE>::Zero();
        solution.head<6>() = -K_1.block<6, 6>(0, 0) * HTz;
        solution += vec - G * vec;

        (*state_) += solution;

        const V3D rot_add = solution.block<3, 1>(0, 0);
        const V3D t_add   = solution.block<3, 1>(3, 0);
        if (rot_add.norm() * 57.3 < 0.001 && t_add.norm() * 100.0 < 0.001) {
            ekf_end = true;
        }
        if (iteration == max_iterations_ - 1 || ekf_end) {
            state_->cov -= G * state_->cov;
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// computeJacobianAndUpdateEKF — 金字塔粗到细 ESIKF 光度更新总控
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::computeJacobianAndUpdateEKF
// ══════════════════════════════════════════════════════════════════
void VIOManager::computeJacobianAndUpdateEKF(const cv::Mat& img) {
    if (total_points_ == 0) return;
    for (int level = patch_pyramid_level_ - 1; level >= 0; --level) {
        updateState(img, level);
    }
}

// ══════════════════════════════════════════════════════════════════
// generateVisualMapPoints — 从 LiDAR 平面点生成新 VisualPoint
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::generateVisualMapPoints
//
// 对每个带法向量的 LiDAR 点: 投影到当前帧，Shi-Tomasi 角点分数筛选，
// 网格内选最优，插入视觉地图哈希表。
// ══════════════════════════════════════════════════════════════════
void VIOManager::generateVisualMapPoints(const cv::Mat& img, std::vector<pointWithVar>& pg) {
    const int boundary = patch_size_half_ * (1 << (patch_pyramid_level_ - 1)) + 1;

    std::vector<float> grid_score(grid_best_point_.size(), 0.0f);
    std::vector<int>   grid_pt_idx(grid_best_point_.size(), -1);

    for (size_t i = 0; i < pg.size(); ++i) {
        const pointWithVar& pv = pg[i];
        if (pv.normal.norm() < 1e-6) continue;  // 无有效法向量，跳过

        const V3D p_cam = Rcw_ * pv.point_w + Pcw_;
        if (p_cam.z() <= 0.0) continue;
        const Eigen::Vector2d pc = world2cam(p_cam);
        if (!isInFrame(pc, boundary)) continue;

        const int gx = static_cast<int>(pc.x()) / grid_size_;
        const int gy = static_cast<int>(pc.y()) / grid_size_;
        const size_t idx = static_cast<size_t>(gy) * grid_n_width_ + gx;
        if (idx >= grid_score.size() || grid_best_point_[idx] != nullptr) continue;  // 该格已有匹配点，不再新增

        // Shi-Tomasi 角点响应（简化: 用 3x3 Sobel 梯度方差近似）
        const int u = static_cast<int>(pc.x()), v = static_cast<int>(pc.y());
        if (u < 3 || v < 3 || u >= img.cols - 3 || v >= img.rows - 3) continue;
        double gxx = 0, gyy = 0, gxy = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const double ix = (img.at<uint8_t>(v + dy, u + dx + 1) - img.at<uint8_t>(v + dy, u + dx - 1)) * 0.5;
                const double iy = (img.at<uint8_t>(v + dy + 1, u + dx) - img.at<uint8_t>(v + dy - 1, u + dx)) * 0.5;
                gxx += ix * ix; gyy += iy * iy; gxy += ix * iy;
            }
        }
        const double trace = gxx + gyy;
        const double det = gxx * gyy - gxy * gxy;
        const double disc = std::max(0.0, trace * trace / 4.0 - det);
        const double min_eig = trace / 2.0 - std::sqrt(disc);
        const float score = static_cast<float>(min_eig);

        if (score > grid_score[idx]) {
            grid_score[idx] = score;
            grid_pt_idx[idx] = static_cast<int>(i);
        }
    }

    constexpr float kMinCornerScore = 20.0f;
    for (size_t idx = 0; idx < grid_pt_idx.size(); ++idx) {
        if (grid_pt_idx[idx] < 0 || grid_score[idx] < kMinCornerScore) continue;
        const pointWithVar& pv = pg[static_cast<size_t>(grid_pt_idx[idx])];

        const V3D p_cam = Rcw_ * pv.point_w + Pcw_;
        const Eigen::Vector2d pc = world2cam(p_cam);
        const V3D f_ref = cam2world(pc).normalized();

        auto vp = std::make_unique<VisualPoint>(pv.point_w);
        vp->normal_ = pv.normal;
        vp->covariance_ = pv.var;
        vp->is_normal_initialized_ = true;

        std::vector<float> patch(static_cast<size_t>(patch_size_total_) * patch_pyramid_level_, 0.0f);
        for (int lvl = 0; lvl < patch_pyramid_level_; ++lvl) {
            getImagePatch(img, pc, patch.data(), lvl);
        }
        Eigen::Vector2i origin;
        cv::Mat crop = cropAroundPixel(img, pc, origin);
        auto feat = std::make_unique<Feature>(pc, f_ref, std::move(crop), Rcw_, Pcw_, 0);
        feat->patch_origin_ = origin;
        feat->patch_ = std::move(patch);
        feat->score_ = grid_score[idx];
        vp->ref_patch_ = feat.get();
        vp->obs_.push_back(std::move(feat));

        const double voxel_size = visual_voxel_size_;
        const V3D loc = pv.point_w / voxel_size;
        VOXEL_LOCATION key(static_cast<int64_t>(std::floor(loc.x())),
                            static_cast<int64_t>(std::floor(loc.y())),
                            static_cast<int64_t>(std::floor(loc.z())));
        feat_map_[key].points.push_back(std::move(vp));
    }
}

// ══════════════════════════════════════════════════════════════════
// updateVisualMapPoints — 现有 VisualPoint 按视角变化添加新观测
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::updateVisualMapPoints
// ══════════════════════════════════════════════════════════════════
void VIOManager::updateVisualMapPoints(const cv::Mat& img) {
    const V3D cam_pos_w = -Rcw_.transpose() * Pcw_;

    for (int i = 0; i < total_points_; ++i) {
        VisualPoint* vp = visual_submap_.voxel_points[i];

        // 视角变化足够大才新增观测，避免冗余
        bool need_new_obs = true;
        if (vp->ref_patch_ != nullptr) {
            const V3D ref_cam_pos = -vp->ref_patch_->T_f_w_rot_.transpose() * vp->ref_patch_->T_f_w_pos_;
            const double baseline = (cam_pos_w - ref_cam_pos).norm();
            const double depth = (Rcw_ * vp->pos_ + Pcw_).z();
            need_new_obs = (baseline / std::max(depth, 1e-3)) > 0.1;  // 视差比阈值
        }
        if (!need_new_obs) continue;

        const V3D p_cam = Rcw_ * vp->pos_ + Pcw_;
        const Eigen::Vector2d pc = world2cam(p_cam);
        const V3D f_ref = cam2world(pc).normalized();

        std::vector<float> patch(static_cast<size_t>(patch_size_total_) * patch_pyramid_level_, 0.0f);
        for (int lvl = 0; lvl < patch_pyramid_level_; ++lvl) {
            getImagePatch(img, pc, patch.data(), lvl);
        }
        Eigen::Vector2i origin;
        cv::Mat crop = cropAroundPixel(img, pc, origin);
        auto feat = std::make_unique<Feature>(pc, f_ref, std::move(crop), Rcw_, Pcw_, 0);
        feat->patch_origin_ = origin;
        feat->patch_ = std::move(patch);
        feat->score_ = vp->ref_patch_ ? vp->ref_patch_->score_ : 0.0f;

        vp->obs_.push_back(std::move(feat));
        if (vp->obs_.size() > 10) {
            // WARN: 驱逐最旧观测，但绝不驱逐 ref_patch_ 指向的节点：ref_patch_ 是
            // 裸指针指向 obs_ 内部元素，删掉它会导致后续
            // retrieveFromVisualSparseMap/updateReferencePatch 里的 UAF。
            auto victim = vp->obs_.begin();
            if (victim->get() == vp->ref_patch_ && vp->obs_.size() > 1) {
                ++victim;
            }
            vp->obs_.erase(victim);
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// updateReferencePatch — 更新法向量，重选最佳参考 patch
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::updateReferencePatch
// ══════════════════════════════════════════════════════════════════
void VIOManager::updateReferencePatch(
    const std::unordered_map<VOXEL_LOCATION, std::unique_ptr<VoxelOctoTree>>& /*plane_map*/) {
    for (int i = 0; i < total_points_; ++i) {
        VisualPoint* vp = visual_submap_.voxel_points[i];
        if (vp->obs_.empty()) continue;

        // 用当前帧观测视角挑选与相机连线夹角最小的历史 patch 作为新参考
        Feature* best = vp->getCloseViewObs(-Rcw_.transpose() * Pcw_);
        if (best != nullptr) vp->ref_patch_ = best;
    }
}

// ══════════════════════════════════════════════════════════════════
// processFrame — VIOManager 主入口
// 原始来源: FAST-LIVO2/src/vio.cpp VIOManager::processFrame
// ══════════════════════════════════════════════════════════════════
void VIOManager::processFrame(const cv::Mat& img,
                               std::vector<pointWithVar>& pg,
                               const std::unordered_map<VOXEL_LOCATION,
                                   std::unique_ptr<VoxelOctoTree>>& plane_map) {
    const M3D Rwi = state_->rot_end;
    const V3D Pwi = state_->pos_end;
    Rcw_ = Rci_ * Rwi.transpose();
    Pcw_ = -Rci_ * Rwi.transpose() * Pwi + Pci_;

    retrieveFromVisualSparseMap(img, pg, plane_map);
    computeJacobianAndUpdateEKF(img);
    generateVisualMapPoints(img, pg);
    updateVisualMapPoints(img);
    updateReferencePatch(plane_map);
}

}  // namespace radar::fast_livo2
