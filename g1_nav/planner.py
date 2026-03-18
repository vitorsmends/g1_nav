#!/usr/bin/env python3
"""
planner.py
==========
Dijkstra-based path planner on a 2D occupancy map.

Subscribes:
  /map                 (nav_msgs/OccupancyGrid)
  /g1nav/current_goal  (geometry_msgs/PoseStamped) -- set by waypoint_manager
  /inorbit/odom_pose   (nav_msgs/Odometry)

Publishes:
  /g1nav/path          (nav_msgs/Path)
"""

import math
import heapq
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


# ── Catmull-Rom smoothing ─────────────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _catmull_rom(points, samples=8):
    if len(points) < 2:
        return points[:]
    P = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i-1], P[i], P[i+1], P[i+2]
        t0 = 0.0
        t1 = t0 + math.sqrt(_dist(p0, p1))
        t2 = t1 + math.sqrt(_dist(p1, p2))
        t3 = t2 + math.sqrt(_dist(p2, p3))
        if t1 == t0 or t2 == t1 or t3 == t2:
            if not out or _dist(out[-1], p1) > 1e-6:
                out.append(p1)
            continue
        for s in range(samples):
            t = t1 + (t2 - t1) * s / samples
            A1 = (((t1-t)/(t1-t0))*p0[0]+((t-t0)/(t1-t0))*p1[0],
                  ((t1-t)/(t1-t0))*p0[1]+((t-t0)/(t1-t0))*p1[1])
            A2 = (((t2-t)/(t2-t1))*p1[0]+((t-t1)/(t2-t1))*p2[0],
                  ((t2-t)/(t2-t1))*p1[1]+((t-t1)/(t2-t1))*p2[1])
            A3 = (((t3-t)/(t3-t2))*p2[0]+((t-t2)/(t3-t2))*p3[0],
                  ((t3-t)/(t3-t2))*p2[1]+((t-t2)/(t3-t2))*p3[1])
            B1 = (((t2-t)/(t2-t0))*A1[0]+((t-t0)/(t2-t0))*A2[0],
                  ((t2-t)/(t2-t0))*A1[1]+((t-t0)/(t2-t0))*A2[1])
            B2 = (((t3-t)/(t3-t1))*A2[0]+((t-t1)/(t3-t1))*A3[0],
                  ((t3-t)/(t3-t1))*A2[1]+((t-t1)/(t3-t1))*A3[1])
            C  = (((t2-t)/(t2-t1))*B1[0]+((t-t1)/(t2-t1))*B2[0],
                  ((t2-t)/(t2-t1))*B1[1]+((t-t1)/(t2-t1))*B2[1])
            if not out or _dist(out[-1], C) > 1e-6:
                out.append(C)
    if not out or _dist(out[-1], points[-1]) > 1e-6:
        out.append(points[-1])
    return out


# ── Planner node ─────────────────────────────────────────────────────────────

class Planner(Node):

    def __init__(self):
        super().__init__("planner")

        self.declare_parameter("inflation_radius_m", 0.35)
        self.declare_parameter("occ_threshold",      50)

        self._inflation_m  = self.get_parameter("inflation_radius_m").value
        self._occ_th       = self.get_parameter("occ_threshold").value

        self._map: OccupancyGrid | None = None
        self._occ_inf = []
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

        self._pub_path = self.create_publisher(Path, "/g1nav/path", qos)

        self.create_subscription(OccupancyGrid, "/map",                self._cb_map,  qos)
        self.create_subscription(Odometry,      "/inorbit/odom_pose",  self._cb_odom, qos_be)
        self.create_subscription(PoseStamped,   "/g1nav/current_goal", self._cb_goal, qos)

        self.get_logger().info("planner: ready")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        self._map = msg
        r = int(math.ceil(self._inflation_m / msg.info.resolution)) if msg.info.resolution > 0 else 0
        self._occ_inf = self._inflate(list(msg.data), msg.info.width, msg.info.height, r)
        self.get_logger().info(
            f"planner: map received {msg.info.width}x{msg.info.height} "
            f"res={msg.info.resolution:.3f}m"
        )

    def _cb_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)
        )
        self._has_pose = True

    def _cb_goal(self, msg: PoseStamped):
        if not self._has_pose:
            self.get_logger().warn("planner: no pose yet, skipping plan")
            return
        if self._map is None:
            self.get_logger().warn("planner: no map yet, skipping plan")
            return

        gx = msg.pose.position.x
        gy = msg.pose.position.y

        path = self._plan(self._robot_x, self._robot_y, gx, gy)
        self._pub_path.publish(path)

    # ── Planning ──────────────────────────────────────────────────────────────

    def _plan(self, sx, sy, gx, gy) -> Path:
        m   = self._map
        res = m.info.resolution
        ox  = m.info.origin.position.x
        oy  = m.info.origin.position.y
        w   = m.info.width
        h   = m.info.height

        def w2g(x, y):
            return int(math.floor((x-ox)/res)), int(math.floor((y-oy)/res))

        def g2w(ix, iy):
            return ox + (ix+0.5)*res, oy + (iy+0.5)*res

        def in_bounds(ix, iy):
            return 0 <= ix < w and 0 <= iy < h

        def is_occ(ix, iy):
            v = self._occ_inf[iy*w + ix]
            return v >= self._occ_th and v != 255

        six, siy = w2g(sx, sy)
        gix, giy = w2g(gx, gy)

        # Fallback to straight line if out of bounds or occupied
        if not in_bounds(six, siy) or not in_bounds(gix, giy) \
                or is_occ(six, siy) or is_occ(gix, giy):
            pts = [(sx + (gx-sx)*i/20, sy + (gy-sy)*i/20) for i in range(21)]
            return self._make_path(pts)

        # Dijkstra
        dist  = {(six, siy): 0.0}
        prev  = {}
        pq    = [(0.0, six, siy, self._robot_yaw)]
        vis   = set()
        rt2   = math.sqrt(2)
        dirs  = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                 (-1,-1,rt2),(1,-1,rt2),(-1,1,rt2),(1,1,rt2)]

        while pq:
            d, x, y, yaw_prev = heapq.heappop(pq)
            if (x, y) in vis:
                continue
            vis.add((x, y))
            if (x, y) == (gix, giy):
                break
            for dx, dy, c in dirs:
                nx, ny = x+dx, y+dy
                if not in_bounds(nx, ny) or is_occ(nx, ny):
                    continue
                yaw_next = math.atan2(dy, dx)
                turn = abs(math.atan2(
                    math.sin(yaw_next - yaw_prev),
                    math.cos(yaw_next - yaw_prev)
                ))
                nd = d + c * (1.0 + 2.0 * turn)
                if nd < dist.get((nx, ny), float('inf')):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = (x, y, yaw_next)
                    heapq.heappush(pq, (nd, nx, ny, yaw_next))

        # Reconstruct path
        if (gix, giy) not in dist:
            self.get_logger().warn("planner: no path found, using straight line")
            pts = [(sx + (gx-sx)*i/20, sy + (gy-sy)*i/20) for i in range(21)]
            return self._make_path(pts)

        cells = []
        cur = (gix, giy)
        while cur in prev or cur == (six, siy):
            cells.append(cur)
            if cur == (six, siy):
                break
            cur = (prev[cur][0], prev[cur][1])
        cells.reverse()

        # Shortcut + smooth
        pts = [g2w(ix, iy) for ix, iy in cells]
        pts = self._shortcut(pts, lambda ix, iy: in_bounds(ix, iy) and not is_occ(ix, iy), w2g)
        pts = _catmull_rom(pts, samples=8)

        return self._make_path(pts)

    def _make_path(self, pts) -> Path:
        path = Path()
        path.header = Header()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        prev_yaw = self._robot_yaw
        for i, (x, y) in enumerate(pts):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            if i < len(pts) - 1:
                nx, ny = pts[i+1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = prev_yaw
            prev_yaw = yaw
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(ps)
        return path

    def _shortcut(self, pts, free_fn, w2g):
        if len(pts) <= 2:
            return pts
        grid = [w2g(x, y) for x, y in pts]

        def line_clear(a, b):
            x0, y0 = a
            x1, y1 = b
            dx, dy = abs(x1-x0), abs(y1-y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            x, y = x0, y0
            while True:
                if not free_fn(x, y):
                    return False
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy; x += sx
                if e2 < dx:
                    err += dx; y += sy
            return True

        out = [pts[0]]
        i = 0
        while i < len(grid) - 1:
            j = len(grid) - 1
            while j > i + 1 and not line_clear(grid[i], grid[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _inflate(self, occ, w, h, r):
        if r <= 0:
            return occ[:]
        inf = [0] * (w * h)
        cells = [(i % w, i // w) for i, v in enumerate(occ)
                 if v >= self._occ_th and v != 255]
        for ox, oy in cells:
            for y in range(max(0, oy-r), min(h-1, oy+r)+1):
                for x in range(max(0, ox-r), min(w-1, ox+r)+1):
                    if (x-ox)**2 + (y-oy)**2 <= r*r:
                        inf[y*w+x] = max(inf[y*w+x], 100)
        for i, v in enumerate(occ):
            if v == 255:
                inf[i] = 255
        return inf


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()