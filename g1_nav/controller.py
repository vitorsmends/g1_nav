#!/usr/bin/env python3
"""
controller.py
=============
Pure-pursuit path following controller.

Reads the planned path and the robot pose, then publishes cmd_vel
(geometry_msgs/Twist) to drive the G1 along the path.

Subscribes:
  /g1nav/path         (nav_msgs/Path)
  /inorbit/odom_pose  (nav_msgs/Odometry)

Publishes:
  /cmd_vel            (geometry_msgs/Twist)

Parameters:
  lookahead_dist   (float, default 0.8)   -- pure pursuit lookahead [m]
  max_linear_vel   (float, default 0.3)   -- max forward speed [m/s]
  max_angular_vel  (float, default 0.8)   -- max rotation speed [rad/s]
  goal_tolerance   (float, default 0.4)   -- stop distance from final goal [m]
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist


class Controller(Node):

    def __init__(self):
        super().__init__("controller")

        self.declare_parameter("lookahead_dist",  0.8)
        self.declare_parameter("max_linear_vel",  0.3)
        self.declare_parameter("max_angular_vel", 0.8)
        self.declare_parameter("goal_tolerance",  0.4)

        self._lookahead  = self.get_parameter("lookahead_dist").value
        self._max_lin    = self.get_parameter("max_linear_vel").value
        self._max_ang    = self.get_parameter("max_angular_vel").value
        self._goal_tol   = self.get_parameter("goal_tolerance").value

        self._path = None
        self._robot_x = self._robot_y = self._robot_yaw = 0.0
        self._has_pose = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", qos)

        self.create_subscription(Path,     "/g1nav/path",        self._cb_path, qos)
        self.create_subscription(Odometry, "/inorbit/odom_pose", self._cb_odom, qos_be)

        # Control loop at 20 Hz
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info(
            f"controller: ready — "
            f"lookahead={self._lookahead}m "
            f"max_lin={self._max_lin}m/s "
            f"max_ang={self._max_ang}rad/s"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_path(self, msg: Path):
        self._path = msg
        self.get_logger().info(
            f"controller: new path received with {len(msg.poses)} poses"
        )

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)
        )
        self._has_pose = True

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        if self._path is None or not self._has_pose:
            return
        if len(self._path.poses) == 0:
            self._stop()
            return

        # Check if we reached the final goal
        final = self._path.poses[-1]
        dist_to_goal = math.hypot(
            final.pose.position.x - self._robot_x,
            final.pose.position.y - self._robot_y,
        )
        if dist_to_goal <= self._goal_tol:
            self._stop()
            self._path = None
            self.get_logger().info("controller: goal reached, stopping")
            return

        # Find lookahead point
        target = self._lookahead_point()
        if target is None:
            self._stop()
            return

        # Pure pursuit
        tx, ty = target
        dx = tx - self._robot_x
        dy = ty - self._robot_y

        # Angle to target in robot frame
        angle_to_target = math.atan2(dy, dx)
        heading_error   = math.atan2(
            math.sin(angle_to_target - self._robot_yaw),
            math.cos(angle_to_target - self._robot_yaw),
        )

        # Reduce linear speed when turning sharply
        linear  = self._max_lin * max(0.0, math.cos(heading_error))
        angular = max(-self._max_ang, min(self._max_ang, 1.5 * heading_error))

        cmd = Twist()
        cmd.linear.x  = linear
        cmd.angular.z = angular
        self._pub_cmd.publish(cmd)

    def _lookahead_point(self):
        """Find the first path point beyond lookahead distance."""
        poses = self._path.poses

        # Find closest point on path first
        closest_idx = 0
        min_dist = float('inf')
        for i, ps in enumerate(poses):
            d = math.hypot(
                ps.pose.position.x - self._robot_x,
                ps.pose.position.y - self._robot_y,
            )
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Search forward from closest point for lookahead
        for i in range(closest_idx, len(poses)):
            d = math.hypot(
                poses[i].pose.position.x - self._robot_x,
                poses[i].pose.position.y - self._robot_y,
            )
            if d >= self._lookahead:
                return poses[i].pose.position.x, poses[i].pose.position.y

        # If no point found beyond lookahead, use the last point
        last = poses[-1]
        return last.pose.position.x, last.pose.position.y

    def _stop(self):
        self._pub_cmd.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()