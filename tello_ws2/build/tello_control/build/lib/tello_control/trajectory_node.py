import numpy as np
import time
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from djitellopy import Tello

# -----------------------------
# PARÁMETROS (IGUALES)
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2
ALPHA = 0.6

# -----------------------------
# CONTROL (IGUAL)
# -----------------------------
def control(q, q_d, K):
    e = q_d - q
    u = K @ e
    return u, e

# -----------------------------
# RC (IGUAL)
# -----------------------------
def velocity_to_rc(u):
    lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
    fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
    ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
    return lr, fb, ud, 0


class TrajectoryNode(Node):

    def __init__(self):
        super().__init__('trajectory_node')

        self.sub_q = self.create_subscription(
            Float32MultiArray,
            '/q',
            self.q_callback,
            10
        )

        self.pub_rc = self.create_publisher(
            Int32MultiArray,
            '/rc_cmd',
            10
        )

        # -----------------------------
        # TELLO
        # -----------------------------
        self.drone = Tello()
        self.drone.connect()
        print("Batería:", self.drone.get_battery(), "%")

        self.drone.takeoff()
        time.sleep(2)

        # -----------------------------
        # ESTADOS (IGUAL)
        # -----------------------------
        self.q = np.array([0.0, 0.0, 110.0])
        self.q_d = np.array([-50.0, -50.0, 160.0])
        self.K = np.diag([1.2, 1.2, 1.2])

        # -----------------------------
        # LOGS (IGUAL)
        # -----------------------------
        self.t_hist = []
        self.q_hist = []
        self.qd_hist = []
        self.e_xyz_hist = []
        self.e_norm_hist = []
        self.u_hist = []
        self.rc_hist = []

        self.i = 0

        self.timer = self.create_timer(DT, self.loop)

    def q_callback(self, msg):
        self.q = np.array(msg.data)

    def loop(self):

        u, e = control(self.q, self.q_d, self.K)
        dist = np.linalg.norm(e)

        rc = velocity_to_rc(u)

        # publicar RC
        msg = Int32MultiArray()
        msg.data = list(rc)
        self.pub_rc.publish(msg)

        # -------------------------
        # LOG (IGUAL)
        # -------------------------
        self.t_hist.append(self.i * DT)
        self.q_hist.append(self.q.copy())
        self.qd_hist.append(self.q_d.copy())
        self.e_xyz_hist.append(e.copy())
        self.e_norm_hist.append(dist)
        self.u_hist.append(u.copy())
        self.rc_hist.append(rc)

        print(f"q={self.q} | error={dist:.2f} | rc={rc[:3]}")

        self.i += 1

        if dist < TOL_CM or self.i > 200:
            self.shutdown()

    def shutdown(self):

        print("Aterrizando...")
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        self.drone.land()
        self.drone.end()

        self.plot_results()
        rclpy.shutdown()

    # -----------------------------
    # GRÁFICAS (IGUAL)
    # -----------------------------
    def plot_results(self):

        q = np.array(self.q_hist)
        qd = np.array(self.qd_hist)
        e_xyz = np.array(self.e_xyz_hist)
        u = np.array(self.u_hist)
        rc = np.array(self.rc_hist)

        plt.figure()
        plt.plot(self.t_hist, q[:,0], label="x")
        plt.plot(self.t_hist, q[:,1], label="y")
        plt.plot(self.t_hist, q[:,2], label="z")
        plt.legend(); plt.grid()

        plt.figure()
        plt.plot(q[:,0], q[:,1])
        plt.scatter(qd[0,0], qd[0,1])
        plt.scatter(qd[-1,0], qd[-1,1])
        plt.grid()

        plt.figure()
        plt.plot(self.t_hist, e_xyz[:,0])
        plt.plot(self.t_hist, e_xyz[:,1])
        plt.plot(self.t_hist, e_xyz[:,2])
        plt.grid()

        plt.show()


def main():
    rclpy.init()
    node = TrajectoryNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()