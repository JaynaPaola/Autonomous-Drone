import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray

import numpy as np


# -----------------------------
# PARÁMETROS
# -----------------------------
RC_LIMIT = 40
TOL_CM = 2


# -----------------------------
# CONTROL
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


# -----------------------------
# NODO
# -----------------------------
class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')

        # --- Publicadores ---
        self.rc_pub = self.create_publisher(Int32MultiArray, '/rc_command', 10)

        # --- Suscriptores ---
        # Se suscribe a la posición fusionada publicada por odometry_node
        self.pose_sub = self.create_subscription(
            Float32MultiArray,
            '/estimated_pose',
            self.pose_callback,
            10
        )

        # --- Estado ---
        self.q = np.array([0.0, 0.0, 0.0])          # posición estimada actual [cm]
        self.q_d = np.array([50.0, 0.0, 150.0])      # objetivo [cm]
        self.K = np.diag([1.2, 1.2, 1.2])
        self.iteration = 0
        self.goal_reached = False

        # --- Timer de control a 10 Hz ---
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('TrajectoryNode iniciado.')
        self.get_logger().info(
            f"Objetivo: x={self.q_d[0]:.1f}, y={self.q_d[1]:.1f}, z={self.q_d[2]:.1f} cm"
        )

    # --------------------------------------------------
    def pose_callback(self, msg):
        """
        Actualiza la posición estimada.
        Recibe la posición fusionada (OptiTrack + odometría) desde odometry_node.
        """
        self.q = np.array(msg.data[:3])

    # --------------------------------------------------
    def control_loop(self):
        if self.goal_reached:
            return

        # --- Ley de control proporcional ---
        u, e = control(self.q, self.q_d, self.K)
        dist = np.linalg.norm(e)

        # --- Conversión a comandos RC ---
        rc = velocity_to_rc(u)

        self.get_logger().info(
            f"q_est [cm]: x={self.q[0]:.1f}, y={self.q[1]:.1f}, z={self.q[2]:.1f} | "
            f"error={dist:.2f} cm | rc={rc[:3]}"
        )

        # --- Condición de parada: precisión ---
        if dist < TOL_CM:
            self.get_logger().info('¡Objetivo alcanzado por precisión!')
            self._send_rc(0, 0, 0, 0)
            self.goal_reached = True
            return

        # --- Condición de parada: estancamiento (comando mínimo) ---
        if self.iteration > 10 and rc[0] == 0 and rc[1] == 0 and rc[2] == 0:
            self.get_logger().info(
                '¡Objetivo alcanzado por comando mínimo! Finalizando...'
            )
            self._send_rc(0, 0, 0, 0)
            self.goal_reached = True
            return

        # --- Publicar comando RC ---
        self._send_rc(*rc)
        self.iteration += 1

    # --------------------------------------------------
    def _send_rc(self, lr, fb, ud, yaw):
        msg = Int32MultiArray()
        msg.data = [lr, fb, ud, yaw]
        self.rc_pub.publish(msg)


# -----------------------------
# MAIN
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por el usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()