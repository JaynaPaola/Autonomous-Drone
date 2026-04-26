import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int16MultiArray

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
ALPHA = 0.6


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        # Estado
        self.q = np.array([0.0, 0.0, 110.0])
        self.v_est = np.array([0.0, 0.0, 0.0])

        # Suscripción a RC
        self.rc_sub = self.create_subscription(
            Int16MultiArray,
            '/rc_cmd',
            self.rc_callback,
            10
        )

        # Publicación de estado
        self.state_pub = self.create_publisher(
            Float32MultiArray,
            '/state',
            10
        )

        self.rc_cmd = np.array([0, 0, 0, 0])

        self.timer = self.create_timer(DT, self.update)

        self.get_logger().info("Odometry node iniciado")

    # -----------------------------
    # RC CALLBACK
    # -----------------------------
    def rc_callback(self, msg):
        self.rc_cmd = np.array(msg.data)

    # -----------------------------
    # ODOMETRÍA (igual a tu función)
    # -----------------------------
    def odometry(self, q, v_est, rc_cmd):

        lr, fb, ud, _ = rc_cmd

        v_meas = np.array([
            fb / RC_LIMIT * MAX_SPEED_CM_S,
            lr / RC_LIMIT * MAX_SPEED_CM_S,
            ud / RC_LIMIT * MAX_SPEED_CM_S
        ])

        v_est = ALPHA * v_est + (1 - ALPHA) * v_meas
        q_next = q + v_est * DT

        return q_next, v_est, v_meas

    # -----------------------------
    # LOOP
    # -----------------------------
    def update(self):

        self.q, self.v_est, v_meas = self.odometry(self.q, self.v_est, self.rc_cmd)

        msg = Float32MultiArray()
        msg.data = [
            self.q[0], self.q[1], self.q[2],
            self.v_est[0], self.v_est[1], self.v_est[2]
        ]

        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()