import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray

class OdomNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        self.q = np.array([0.0, 0.0, 110.0])
        self.v_est = np.zeros(3)

        self.dt = 0.1
        self.alpha = 0.6
        self.rc_limit = 40
        self.max_speed = 40

        self.sub = self.create_subscription(
            Float32MultiArray,
            '/cmd_rc',
            self.cb,
            10)

        self.pub = self.create_publisher(
            Float32MultiArray,
            '/q',
            10)

        self.rc = [0,0,0,0]
        self.timer = self.create_timer(self.dt, self.update)

    def cb(self, msg):
        self.rc = msg.data

    def update(self):
        lr, fb, ud, _ = self.rc

        v = np.array([
            fb / self.rc_limit * self.max_speed,
            lr / self.rc_limit * self.max_speed,
            ud / self.rc_limit * self.max_speed
        ])

        self.v_est = self.alpha * self.v_est + (1 - self.alpha) * v
        self.q = self.q + self.v_est * self.dt

        msg = Float32MultiArray()
        msg.data = self.q.tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = OdomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()