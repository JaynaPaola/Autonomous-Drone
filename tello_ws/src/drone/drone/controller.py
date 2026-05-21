import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Bool, String
import numpy as np
import time

# -----------------------------
# PARÁMETROS
# -----------------------------
RC_LIMIT    = 30
RC_POS_MAX  = 25
TOL_M       = 0.10
PAUSE_S     = 10.0

K_POS       = 0.5
K_YAW       = 15.0

# -----------------------------
# WAYPOINTS en coordenadas OptiTrack (x, y, z) metros
# -----------------------------
WAYPOINTS = [
    np.array([0.0, 0.0, 0.67]),
    np.array([0.0, 0.0, 0.0]),
    #np.array([1.09,  1.06, 0.67]),
    #np.array([1.09,  0.40, 0.67]),
    #np.array([0.40,  0.40, 0.67]),
    #np.array([0.40,  1.06, 0.67]),
    #np.array([1.09,  1.06, 0.67]),
]

WP_NAMES = [
    "Posición inicial",
    "Segunda posición",
    "Tercera posición",
    "Cuarta posición",
    "Regreso a posición inicial",
]


def angle_diff(target, current):
    diff = target - current
    return (diff + np.pi) % (2 * np.pi) - np.pi


def velocity_to_rc(u_pos, u_yaw):
    lr  = int(np.clip(np.round(u_pos[0] * RC_LIMIT), -RC_POS_MAX, RC_POS_MAX))
    fb  = int(np.clip(np.round(u_pos[1] * RC_LIMIT), -RC_POS_MAX, RC_POS_MAX))
    ud  = int(np.clip(np.round(u_pos[2] * RC_LIMIT), -RC_POS_MAX, RC_POS_MAX))
    yaw = int(np.clip(np.round(u_yaw),               -RC_LIMIT,   RC_LIMIT))
    return lr, fb, ud, yaw


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')

        self.rc_pub     = self.create_publisher(Int32MultiArray, '/rc_command',      10)
        self.land_pub   = self.create_publisher(Bool,            '/land_signal',     10)
        self.status_pub = self.create_publisher(String,          '/waypoint_status', 10)

        self.pose_sub = self.create_subscription(
            Float32MultiArray, '/estimated_pose', self.pose_callback, 10
        )

        self.q            = None
        self.yaw          = None
        self.yaw_ref      = None

        self.waypoints    = WAYPOINTS
        self.wp_index     = 0
        self.q_d          = self.waypoints[self.wp_index]
        self.iteration    = 0
        self.goal_reached = False
        self.pausing      = False

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('=' * 50)
        self.get_logger().info('TrajectoryNode iniciado. Esperando primera pose...')
        self.get_logger().info('=' * 50)

    def _publish_status(self, text):
        msg      = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _log_waypoint(self):
        msg = (
            f"⏳ Dirigiéndose a WP {self.wp_index + 1}/{len(self.waypoints)} "
            f"— {WP_NAMES[self.wp_index]} | "
            f"x={self.q_d[0]:.3f}  y={self.q_d[1]:.3f}  z={self.q_d[2]:.3f} [m]"
        )
        self.get_logger().info('=' * 50)
        self.get_logger().info(msg)
        self.get_logger().info('=' * 50)
        self._publish_status(msg)

    def pose_callback(self, msg):
        self.q   = np.array(msg.data[:3])
        self.yaw = float(msg.data[3])
        if self.yaw_ref is None:
            self.yaw_ref = self.yaw
            self.get_logger().info(
                f"Yaw de referencia capturado: {np.degrees(self.yaw_ref):.1f}°"
            )

    def _on_waypoint_reached(self, dist):
        self.pausing = True
        self._send_rc(0, 0, 0, 0)

        arrival = (
            f"✅ ¡LLEGÓ! WP {self.wp_index + 1}/{len(self.waypoints)} "
            f"— {WP_NAMES[self.wp_index]} "
            f"(error={dist:.3f} m) — DETENIDO {PAUSE_S:.0f}s"
        )
        self.get_logger().info('=' * 50)
        self.get_logger().info(arrival)
        self.get_logger().info('=' * 50)
        self._publish_status(arrival)

        time.sleep(PAUSE_S)

        self.wp_index += 1
        if self.wp_index < len(self.waypoints):
            self.q_d       = self.waypoints[self.wp_index]
            self.iteration = 0
            self.pausing   = False
            self._log_waypoint()
        else:
            done = '🏁 MISIÓN COMPLETA — Todos los waypoints visitados.'
            self.get_logger().info('=' * 50)
            self.get_logger().info(done)
            self.get_logger().info('=' * 50)
            self._publish_status(done)
            self._send_rc(0, 0, 0, 0)
            self._send_land_signal()
            self.goal_reached = True
            self.pausing      = False

    def control_loop(self):
        if self.goal_reached or self.pausing:
            return

        if self.q is None or self.yaw_ref is None:
            self.get_logger().warn('Sin pose estimada todavía, esperando...')
            return

        if self.iteration == 0 and self.wp_index == 0:
            self._log_waypoint()

        e_pos = self.q_d - self.q
        dist  = np.linalg.norm(e_pos)
        u_pos = K_POS * e_pos

        e_yaw = angle_diff(self.yaw_ref, self.yaw)
        u_yaw = K_YAW * e_yaw

        rc = velocity_to_rc(u_pos, u_yaw)

        self.get_logger().info(
            f"WP {self.wp_index + 1}/{len(self.waypoints)} | "
            f"x={self.q[0]:.3f} y={self.q[1]:.3f} z={self.q[2]:.3f} [m] | "
            f"err={dist:.3f} m | "
            f"yaw={np.degrees(self.yaw):.1f}° err_yaw={np.degrees(e_yaw):.1f}° | "
            f"rc={rc}"
        )

        if dist < TOL_M:
            self._on_waypoint_reached(dist)
            return

        if self.iteration > 10 and rc[0] == 0 and rc[1] == 0 and rc[2] == 0:
            self.get_logger().info(
                f'⚠️  Waypoint {self.wp_index + 1} alcanzado por comando mínimo.'
            )
            self._on_waypoint_reached(dist)
            return

        self._send_rc(*rc)
        self.iteration += 1

    def _send_rc(self, lr, fb, ud, yaw):
        msg      = Int32MultiArray()
        msg.data = [lr, fb, ud, yaw]
        self.rc_pub.publish(msg)

    def _send_land_signal(self):
        msg      = Bool()
        msg.data = True
        self.land_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por el usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()