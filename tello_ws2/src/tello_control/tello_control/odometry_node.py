import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray

# -----------------------------
# PARÁMETROS (IGUALES)
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
ALPHA = 0.6

# -----------------------------
# ODOMETRÍA (IGUAL)
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

    return q_next, v_est, v_meas


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        self.sub_rc = self.create_subscription(
            Int32MultiArray,
            '/rc_cmd',
            self.rc_callback,
            10
        )

        self.pub_q = self.create_publisher(
            Float32MultiArray,
            '/q',
            10
        )

        # estado interno (NO MODIFICADO)
        self.q = np.array([0.0, 0.0, 110.0])
        self.v_est = np.array([0.0, 0.0, 0.0])
        self.v_meas = np.array([0.0, 0.0, 0.0])

    def rc_callback(self, msg):

        rc = list(msg.data)

        self.q, self.v_est, self.v_meas = odometry(
            self.q,
            self.v_est,
            rc,
            DT
        )

        out = Float32MultiArray()
        out.data = [
            float(self.q[0]),
            float(self.q[1]),
            float(self.q[2])
        ]

        self.pub_q.publish(out)


def main():
    rclpy.init()
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()