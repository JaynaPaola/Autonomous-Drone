import rclpy
from rclpy.node import Node
import numpy as np
import time
import matplotlib.pyplot as plt

from std_msgs.msg import Float32MultiArray
from djitellopy import Tello

# -----------------------------
# PARÁMETROS (IGUAL)
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
        super().__init__('tello_trajectory_node')

        # -----------------------------
        # TELLO
        # -----------------------------
        self.drone = Tello()
        self.drone.connect()
        print("Batería:", self.drone.get_battery(), "%")

        self.drone.takeoff()
        time.sleep(2)

        # -----------------------------
        # ESTADO
        # -----------------------------
        self.q = np.array([0.0, 0.0, 110.0])
        self.K = np.diag([1.2, 1.2, 1.2])
        self.v_est = np.array([0.0, 0.0, 0.0])

        # -----------------------------
        # TRAYECTORIA (IGUAL)
        # -----------------------------
        self.q_d_list = [
            np.array([50.0, 0.0, 110.0]),
            np.array([50.0, 50.0, 110.0]),
            np.array([50.0, 50.0, 150.0])
        ]

        self.step = 0
        self.q_d = self.q_d_list[self.step]

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
        self.v_est_hist = []
        self.v_meas_hist = []
        self.phase_hist = []

        # -----------------------------
        # SUBS (ODOMETRÍA)
        # -----------------------------
        self.sub_q = self.create_subscription(
            Float32MultiArray,
            '/q_next',
            self.q_callback,
            10
        )

        self.timer = self.create_timer(DT, self.loop)

    # -----------------------------
    # RECIBE ODOMETRÍA
    # -----------------------------
    def q_callback(self, msg):
        self.q = np.array(msg.data)

    # -----------------------------
    # LOOP PRINCIPAL (IGUAL A TU MAIN)
    # -----------------------------
    def loop(self):

        u, e = control(self.q, self.q_d, self.K)
        dist = np.linalg.norm(e)

        rc = velocity_to_rc(u)

        # -----------------------------
        # LOGS (IGUAL)
        # -----------------------------
        self.t_hist.append(len(self.t_hist) * DT)
        self.q_hist.append(self.q.copy())
        self.qd_hist.append(self.q_d.copy())
        self.e_xyz_hist.append(e.copy())
        self.e_norm_hist.append(dist)
        self.u_hist.append(u.copy())
        self.rc_hist.append(rc)
        self.phase_hist.append(self.step)

        print(f"q={self.q} | qd={self.q_d} | error={dist:.2f} | rc={rc[:3]}")

        # -----------------------------
        # CONTROL A TELLO
        # -----------------------------
        self.drone.send_rc_control(*rc)

        # -----------------------------
        # CAMBIO DE FASE (IGUAL)
        # -----------------------------
        if dist < TOL_CM:
            self.step += 1

            if self.step >= len(self.q_d_list):
                print("Trayectoria completa")
                self.finish()
                return

            self.q_d = self.q_d_list[self.step]
            print(f"\n➡ Cambio de fase {self.step}: {self.q_d}\n")
            time.sleep(1)

    # -----------------------------
    # GRÁFICAS (IGUAL)
    # -----------------------------
    def finish(self):

        print("Aterrizando...")
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        self.drone.land()

        # (TODO TU PLOT ORIGINAL SIN CAMBIOS)
        plt.figure()
        q = np.array(self.q_hist)
        plt.plot(q[:,0], q[:,1])
        plt.title("Trayectoria XY")
        plt.grid()

        plt.figure()
        plt.plot(self.t_hist, self.e_norm_hist)
        plt.title("Error total")
        plt.grid()

        plt.show()

        self.drone.end()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = TrajectoryNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()