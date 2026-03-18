#!/usr/bin/env python3
"""
waypoint_manager.py
===================
Receives PoseStamped waypoints, queues them, and sends them one by one
to the internal planner once the robot is close enough to the current goal.

Subscribes:
  /g1nav/waypoint      (geometry_msgs/PoseStamped) -- add waypoint to queue
  /g1nav/cancel        (std_msgs/Empty)            -- cancel and clear queue
  /inorbit/odom_pose   (nav_msgs/Odometry)         -- robot pose in map frame

Publishes:
  /g1nav/current_goal  (geometry_msgs/PoseStamped) -- current goal for planner
  /g1nav/status        (std_msgs/String)           -- current status

Parameters:
  goal_tolerance   (float, default 0.5)  -- distance [m] to consider goal reached
  publish_rate_hz  (float, default 2.0)  -- how often to re-send the current goal
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String
from collections import deque


class WaypointManager(Node):

    def __init__(self):
        super().__init__("waypoint_manager")

        self.declare_parameter("goal_tolerance",  0.5)
        self.declare_parameter("publish_rate_hz", 2.0)

        self._tolerance = self.get_parameter("goal_tolerance").value
        self._rate_hz   = self.get_parameter("publish_rate_hz").value

        self._queue        = deque()
        self._current_goal = None
        self._robot_x      = 0.0
        self._robot_y      = 0.0
        self._has_pose     = False

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

        self._pub_goal   = self.create_publisher(PoseStamped, "/g1nav/current_goal", qos)
        self._pub_status = self.create_publisher(String,       "/g1nav/status",       qos)

        self.create_subscription(PoseStamped, "/g1nav/waypoint",    self._cb_waypoint, qos)
        self.create_subscription(Empty,       "/g1nav/cancel",      self._cb_cancel,   qos)
        self.create_subscription(Odometry,    "/inorbit/odom_pose", self._cb_odom,     qos_be)

        self.create_timer(1.0 / self._rate_hz, self._tick)

        self.get_logger().info(
            f"waypoint_manager: ready — "
            f"tolerance={self._tolerance}m rate={self._rate_hz}Hz"
        )
        self._publish_status("idle")

    def _cb_waypoint(self, msg):
        self._queue.append(msg)
        self.get_logger().info(
            f"waypoint_manager: waypoint queued "
            f"x={msg.pose.position.x:.2f} y={msg.pose.position.y:.2f} "
            f"| queue={len(self._queue)}"
        )
        self._publish_status(f"queued {len(self._queue)} waypoints")
        if self._current_goal is None:
            self._advance()

    def _cb_cancel(self, _):
        self._queue.clear()
        self._current_goal = None
        self.get_logger().info("waypoint_manager: cancelled")
        self._publish_status("idle")

    def _cb_odom(self, msg):
        self._robot_x  = msg.pose.pose.position.x
        self._robot_y  = msg.pose.pose.position.y
        self._has_pose = True

    def _tick(self):
        if self._current_goal is None:
            return
        if not self._has_pose:
            self._send_goal(self._current_goal)
            return

        dist = math.hypot(
            self._current_goal.pose.position.x - self._robot_x,
            self._current_goal.pose.position.y - self._robot_y,
        )

        if dist <= self._tolerance:
            self.get_logger().info(
                f"waypoint_manager: goal reached (dist={dist:.2f}m)"
            )
            self._advance()
        else:
            self._send_goal(self._current_goal)

    def _advance(self):
        if self._queue:
            self._current_goal = self._queue.popleft()
            gx = self._current_goal.pose.position.x
            gy = self._current_goal.pose.position.y
            remaining = len(self._queue)
            self.get_logger().info(
                f"waypoint_manager: navigating to "
                f"x={gx:.2f} y={gy:.2f} | {remaining} remaining"
            )
            self._publish_status(f"navigating | {remaining} remaining")
            self._send_goal(self._current_goal)
        else:
            self._current_goal = None
            self.get_logger().info("waypoint_manager: all waypoints reached")
            self._publish_status("idle — all waypoints reached")

    def _send_goal(self, goal):
        goal.header.stamp = self.get_clock().now().to_msg()
        self._pub_goal.publish(goal)

    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self._pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()