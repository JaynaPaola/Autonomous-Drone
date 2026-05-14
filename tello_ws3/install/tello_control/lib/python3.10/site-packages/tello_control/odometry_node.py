import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray

from djitellopy import Tello
import numpy as np
import time


# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
ALPHA = 0.6


# -----------------------------
# ODOMETRÍA
# -----------------------------
def odometry(q, v_est, rc_cmd, dt):
    lr, fb, ud, _ = rc_cmd

    v_meas = np.array([
        fb / RC_LIMIT * MAX_SPEED_CM_S,
        lr / RC_LIMIT * MAX_SPEED_CM_S,
        ud / RC_LIMIT * MAX_SPEED_CM_S
    ])

    v_est = ALPHA * v_est + (1 - ALPHA) * v_meas
    q_next = q + v_est * dt

    return q_next, v_est


# -----------------------------
# NODO
# -----------------------------
class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        # Publicador de posición estimada
        self.odom_pub = self.create_publisher(Float32MultiArray, '/odom_position', 10)

        # Suscriptor de comandos RC
        self.rc_sub = self.create_subscription(
            Int32MultiArray,
            '/rc_command',
            self.rc_callback,
            10
        )

        # Estado interno
        self.q = np.array([0.0, 0.0, 0.0])
        self.v_est = np.array([0.0, 0.0, 0.0])
        self.last_rc = [0, 0, 0, 0]

        # Conexión con el dron
        self.drone = Tello()
        try:
            self.drone.connect()
            self.get_logger().info(f"Batería: {self.drone.get_battery()}%")
        except Exception as e:
            self.get_logger().error(f"Error de conexión: {e}")
            raise

        self.drone.takeoff()
        time.sleep(2)
        self.get_logger().info('OdometryNode iniciado. Dron en vuelo.')

        # Timer de actualización a 10 Hz
        self.timer = self.create_timer(DT, self.update_loop)

    def rc_callback(self, msg):
        """Recibe comandos RC desde trajectory_node y los envía al dron."""
        rc = list(msg.data)
        self.last_rc = rc
        self.drone.send_rc_control(*rc)

    def update_loop(self):
        """Actualiza la odometría y publica la posición estimada."""
        self.q, self.v_est = odometry(self.q, self.v_est, self.last_rc, DT)

        out = Float32MultiArray()
        out.data = self.q.tolist()
        self.odom_pub.publish(out)

        self.get_logger().debug(
            f"Posición estimada [cm]: x={self.q[0]:.1f}, y={self.q[1]:.1f}, z={self.q[2]:.1f}"
        )

    def destroy_node(self):
        """Aterrizaje seguro al cerrar el nodo."""
        self.get_logger().info('Aterrizando...')
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        self.drone.land()
        self.drone.end()
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