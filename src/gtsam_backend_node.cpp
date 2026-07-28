// gtsam_backend_node.cpp - Standalone GTSAM ISAM2 loop closure backend
//
// Architecture (aligned with SPARK-FAST-LIO + KISS-Matcher):
//   LIO publishes /odom + /scan → this node subscribes → ISAM2 optimization
//   → publishes /pose_stamped + /path/corrected
//
// Completely decoupled from LIO. Crash in backend does not affect LIO.

#include "radar_fast_livo2/gtsam_backend.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>

namespace radar::fast_livo2 {

class GtsamBackendNode : public rclcpp::Node {
public:
    explicit GtsamBackendNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("gtsam_backend_node", options) {

        declare_parameter("odom_topic", "/fast_livo2/odom");
        declare_parameter("scan_topic", "/fast_livo2/cloud_lidar");
        declare_parameter("keyframe_dist", 0.5);
        declare_parameter("keyframe_ang", 15.0);
        declare_parameter("loop_radius", 1.0);
        declare_parameter("loop_min_skip", 20);
        declare_parameter("map_frame", "map");
        declare_parameter("odom_frame", "odom");
        declare_parameter("base_frame", "base_link");

        GtsamBackend::Config cfg;
        cfg.keyframe_dist = get_parameter("keyframe_dist").as_double();
        cfg.keyframe_ang  = get_parameter("keyframe_ang").as_double();
        cfg.loop_radius   = get_parameter("loop_radius").as_double();
        cfg.loop_min_skip = get_parameter("loop_min_skip").as_int();
        backend_ = std::make_unique<GtsamBackend>(cfg);

        // ── Synchronized subscribers ──
        auto qos = rclcpp::SensorDataQoS();
        sub_odom_.subscribe(this, get_parameter("odom_topic").as_string(), qos.get_rmw_qos_profile());
        sub_scan_.subscribe(this, get_parameter("scan_topic").as_string(), qos.get_rmw_qos_profile());

        sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(100), sub_odom_, sub_scan_);
        sync_->registerCallback(
            std::bind(&GtsamBackendNode::callback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // ── Publishers ──
        pub_pose_  = create_publisher<geometry_msgs::msg::PoseStamped>("pose_stamped", 10);
        pub_path_  = create_publisher<nav_msgs::msg::Path>("path/corrected", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        path_msg_.header.frame_id = get_parameter("map_frame").as_string();

        RCLCPP_INFO(get_logger(),
            "GTSAM backend node started (kf_dist=%.1fm, loop_radius=%.1fm)",
            cfg.keyframe_dist, cfg.loop_radius);
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;

    void callback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom,
                  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& scan) {
        const auto& p = odom->pose.pose.position;
        const auto& q = odom->pose.pose.orientation;
        gtsam::Pose3 pose(gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
                          gtsam::Point3(p.x, p.y, p.z));
        double ts = odom->header.stamp.sec + odom->header.stamp.nanosec * 1e-9;

        bool is_kf = backend_->add_odometry(pose, ts);
        if (!is_kf) return;

        backend_->try_loop_closure(pose, backend_->latest_idx());
        gtsam::Pose3 corrected = backend_->optimize();

        publish_pose(corrected, odom->header.stamp);

        if (backend_->num_loops() > 0 && backend_->latest_idx() % 50 == 0) {
            RCLCPP_INFO(get_logger(),
                "[GTSAM] kf=%d loops=%d edges=%d",
                backend_->latest_idx(), backend_->num_loops(), backend_->num_edges());
        }
    }

    void publish_pose(const gtsam::Pose3& pose, const builtin_interfaces::msg::Time& stamp) {
        auto map_frame = get_parameter("map_frame").as_string();

        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = stamp;
        ps.header.frame_id = map_frame;
        ps.pose.position.x = pose.translation().x();
        ps.pose.position.y = pose.translation().y();
        ps.pose.position.z = pose.translation().z();
        auto q = pose.rotation().toQuaternion();
        ps.pose.orientation.w = q.w();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        pub_pose_->publish(ps);

        path_msg_.header.stamp = stamp;
        path_msg_.poses.push_back(ps);
        pub_path_->publish(path_msg_);

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = map_frame;
        tf.child_frame_id = get_parameter("odom_frame").as_string();
        tf.transform.translation.x = pose.translation().x();
        tf.transform.translation.y = pose.translation().y();
        tf.transform.translation.z = pose.translation().z();
        tf.transform.rotation = ps.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }

    std::unique_ptr<GtsamBackend> backend_;

    message_filters::Subscriber<nav_msgs::msg::Odometry> sub_odom_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_scan_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    nav_msgs::msg::Path path_msg_;
};

} // namespace radar::fast_livo2

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<radar::fast_livo2::GtsamBackendNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
