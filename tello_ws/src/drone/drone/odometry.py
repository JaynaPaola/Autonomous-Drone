import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Bool
from geometry_msgs.msg import PoseStamped

from djitellopy import Tello
import numpy as np
import time


# -----------------------------
# PARÁMETROS
# -----------------------------
DT            = 0.1
RC_LIMIT      = 40
MAX_SPEED_M_S = 0.40
ALPHA         = 0.6

FUSION_ALPHA      = 0.85
OPTITRACK_TIMEOUT = 0.5

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qy + qx * qz)
    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.arctan2(siny_cosp, cosy_cosp)


def odometry_step(q, v_est, rc_cmd, dt):
    lr, fb, ud, _ = rc_cmd
    v_meas = np.array([
        lr / RC_LIMIT * MAX_SPEED_M_S,
        fb / RC_LIMIT * MAX_SPEED_M_S,
        ud / RC_LIMIT * MAX_SPEED_M_S,
    ])
    v_est  = ALPHA * v_est + (1 - ALPHA) * v_meas
    q_next = q + v_est * dt
    return q_next, v_est


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        self.pose_pub = self.create_publisher(
            Float32MultiArray, '/estimated_pose', 10
        )
        self.rc_sub = self.create_subscription(
            Int32MultiArray, '/rc_command', self.rc_callback, 10
        )
        self.optitrack_sub = self.create_subscription(
            PoseStamped, '/optitrack/rigid_body',
            self.optitrack_callback, SENSOR_QOS
        )
        self.land_sub = self.create_subscription(
            Bool, '/land_signal', self.land_callback, 10
        )

        self.q       = np.array([0.0,  0.0, 0.0])
        self.v_est   = np.array([0.0, 0.0, 0.0])
        self.yaw     = 0.0
        self.last_rc = [0, 0, 0, 0]
        self.landing = False

        self.optitrack_pos       = None
        self.optitrack_yaw       = None
        self.last_optitrack_time = None
        self.optitrack_available = False

        self.drone = Tello()
        try:
            self.drone.connect()
            self.get_logger().info(f"Batería: {self.drone.get_battery()}%")
        except Exception as e:
            self.get_logger().error(f"Error de conexión con el dron: {e}")
            raise

        self.drone.takeoff()
        time.sleep(10)
        self.get_logger().info('OdometryNode iniciado. Dron en vuelo.')
        self.timer = self.create_timer(DT, self.update_loop)

    def rc_callback(self, msg):
        if self.landing:
            return
        self.last_rc = list(msg.data)
        self.drone.send_rc_control(*self.last_rc)

    def optitrack_callback(self, msg: PoseStamped):
        self.optitrack_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.optitrack_yaw = quat_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        self.last_optitrack_time = self.get_clock().now()
        self.optitrack_available = True

    def land_callback(self, msg: Bool):
        if msg.data and not self.landing:
            self.landing = True
            self.get_logger().info('Señal de aterrizaje recibida. Aterrizando...')
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            try:
                self.drone.land()
                self.drone.end()
                self.get_logger().info('Dron aterrizado correctamente.')
            except Exception as e:
                self.get_logger().error(f"Error al aterrizar: {e}")

    def _check_optitrack_timeout(self):
        if self.last_optitrack_time is None:
            self.optitrack_available = False
            return
        elapsed = (
            self.get_clock().now() - self.last_optitrack_time
        ).nanoseconds / 1e9
        if elapsed > OPTITRACK_TIMEOUT:
            if self.optitrack_available:
                self.get_logger().warn('OptiTrack: señal perdida. Usando solo odometría.')
            self.optitrack_available = False

    def update_loop(self):
        if self.landing:
            return

        self._check_optitrack_timeout()
        self.q, self.v_est = odometry_step(self.q, self.v_est, self.last_rc, DT)

        if self.optitrack_available and self.optitrack_pos is not None:
            self.q   = FUSION_ALPHA * self.optitrack_pos + (1 - FUSION_ALPHA) * self.q
            self.yaw = self.optitrack_yaw
            error_corr  = self.optitrack_pos - self.q
            self.v_est += error_corr / DT * (1 - FUSION_ALPHA)
            modo = "FUSIÓN"
        else:
            modo = "ODOM"

        out      = Float32MultiArray()
        out.data = [self.q[0], self.q[1], self.q[2], self.yaw]
        self.pose_pub.publish(out)

        self.get_logger().debug(
            f"[{modo}] x={self.q[0]:.3f}  y={self.q[1]:.3f}  "
            f"z={self.q[2]:.3f}  yaw={np.degrees(self.yaw):.1f}°"
        )

    def destroy_node(self):
        if not self.landing:
            self.get_logger().info('Cierre inesperado: aterrizaje de emergencia...')
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            try:
                self.drone.land()
                self.drone.end()
            except Exception as e:
                self.get_logger().error(f"Error al aterrizar: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por el usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()