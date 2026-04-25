import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from djitellopy import Tello
import time

class TelloNode(Node):
    def __init__(self):
        super().__init__('tello_interface')

        self.drone = Tello()

        # 🔥 NO conectar en constructor directamente
        self.connected = False

        self.sub = self.create_subscription(
            Float32MultiArray,
            '/cmd_rc',
            self.callback,
            10)

        self.get_logger().info("Nodo Tello listo. Esperando conexión...")

        # conectar después de iniciar ROS
        self.timer = self.create_timer(2.0, self.init_drone)

        self.x, self.y, self.z = [], [], []
        self.i = 0

    def init_drone(self):
        if self.connected:
            return

        try:
            self.drone.connect()
            self.drone.streamoff()
            self.drone.streamon()

            self.drone.takeoff()
            self.connected = True

            self.get_logger().info("Drone conectado y en vuelo 🚁")

        except Exception as e:
            self.get_logger().error(f"Error conectando drone: {e}")

    def callback(self, msg):
        if not self.connected:
            return

        rc = list(map(int, msg.data))
        self.drone.send_rc_control(*rc)

        fb, lr, ud = rc[1], rc[0], rc[2]

        if self.i == 0:
            self.x.append(0)
            self.y.append(0)
            self.z.append(110)
        else:
            self.x.append(self.x[-1] + fb * 0.1)
            self.y.append(self.y[-1] + lr * 0.1)
            self.z.append(self.z[-1] + ud * 0.1)

        self.i += 1

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.x, label="x")
        plt.plot(self.y, label="y")
        plt.plot(self.z, label="z")
        plt.legend()
        plt.title("Trayectoria Tello")
        plt.grid()
        plt.show()


def main():
    rclpy.init()
    node = TelloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.plot()

    if node.connected:
        node.drone.send_rc_control(0, 0, 0, 0)
        node.drone.land()
        node.drone.end()

    node.destroy_node()
    rclpy.shutdown()