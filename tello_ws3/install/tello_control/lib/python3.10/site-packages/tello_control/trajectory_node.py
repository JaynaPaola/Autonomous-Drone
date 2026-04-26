import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from djitellopy import Tello
import time

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
TOL_CM = 5


# -----------------------------
# CONTROL (igual que original)
# -----------------------------
def control(q, q_d, K):
    e = q_d - q
    u = K @ e
    return u, e


# -----------------------------
# RC
# -----------------------------
def velocity_to_rc(u):
    lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
    fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
    ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
    return lr, fb, ud, 0


class TrajectoryNode(Node):

    def __init__(self):
        super().__init__('trajectory_node')

        # Tello
        self.drone = Tello()
        self.drone.connect()
        self.get_logger().info(f"Batería: {self.drone.get_battery()}%")

        self.drone.takeoff()
        time.sleep(2)

        # Objetivo y estado deseado
        self.q_d = np.array([50.0, 50.0, 160.0])

        self.q = np.array([0.0, 0.0, 0.0])

        self.K = np.diag([1.2, 1.2, 1.2])

        # Subscripción a estado
        self.state_sub = self.create_subscription(
            Float32MultiArray,
            '/state',
            self.state_callback,
            10
        )

        # Publicación RC
        self.rc_pub = self.create_publisher(
            Int16MultiArray,
            '/rc_cmd',
            10
        )

        self.timer = self.create_timer(DT, self.update)

        self.get_logger().info("Trajectory node iniciado")

    # -----------------------------
    # CALLBACK ESTADO
    # -----------------------------
    def state_callback(self, msg):
        data = msg.data
        self.q = np.array([data[0], data[1], data[2]])

    # -----------------------------
    # LOOP CONTROL
    # -----------------------------
    def update(self):

        u, e = control(self.q, self.q_d, self.K)
        dist = np.linalg.norm(e)

        rc = velocity_to_rc(u)

        # publicar RC
        msg = Int16MultiArray()
        msg.data = list(rc)
        self.rc_pub.publish(msg)

        # enviar al dron
        self.drone.send_rc_control(*rc)

        self.get_logger().info(f"q={self.q} error={dist:.2f} rc={rc}")

        # condición de paro
        if dist < TOL_CM:
            self.get_logger().info("Objetivo alcanzado")
            self.stop()

    # -----------------------------
    # STOP
    # -----------------------------
    def stop(self):
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        self.drone.land()
        self.drone.end()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()