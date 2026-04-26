import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32MultiArray

# -----------------------------
# PARÁMETROS (IGUALES A TU CÓDIGO)
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
ALPHA = 0.6

# -----------------------------
# ODOMETRÍA (MISMA FUNCIÓN)
# -----------------------------
def odometry(q, v_est, rc_cmd, dt):
    lr, fb, ud = rc_cmd

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
        super().__init__('tello_odometry_node')

        self.sub_rc = self.create_subscription(
            Float32MultiArray,
            '/rc_cmd',
            self.rc_callback,
            10
        )

        self.pub_q = self.create_publisher(Float32MultiArray, '/q_next', 10)
        self.pub_v_est = self.create_publisher(Float32MultiArray, '/v_est', 10)
        self.pub_v_meas = self.create_publisher(Float32MultiArray, '/v_meas', 10)

        self.q = np.array([0.0, 0.0, 110.0])
        self.v_est = np.array([0.0, 0.0, 0.0])
        self.rc_cmd = [0, 0, 0]

        self.timer = self.create_timer(DT, self.update)

    def rc_callback(self, msg):
        self.rc_cmd = msg.data

    def update(self):

        q_next, v_est, v_meas = odometry(self.q, self.v_est, self.rc_cmd, DT)

        self.q = q_next
        self.v_est = v_est

        q_msg = Float32MultiArray(data=self.q.tolist())
        v_est_msg = Float32MultiArray(data=v_est.tolist())
        v_meas_msg = Float32MultiArray(data=v_meas.tolist())

        self.pub_q.publish(q_msg)
        self.pub_v_est.publish(v_est_msg)
        self.pub_v_meas.publish(v_meas_msg)


def main():
    rclpy.init()
    node = OdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()