#!/usr/bin/env python3
"""
Tello Controller Node — ROS2 Humble
Migrado desde el script Python original con djitellopy.

Match con optitrack_client.cpp:
  - Se suscribe a 'optitrack/rigid_body' (geometry_msgs/PoseStamped)
    con SensorDataQoS (BEST_EFFORT) igual al publisher del nodo C++.
  - La posición llega en METROS (sistema NatNet) → se convierte a cm.
  - msg.header.frame_id contiene el nombre del rigid body (de Motive).

Lógica de control, parámetros y variables idénticos al script original.
"""

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32MultiArray

from djitellopy import Tello

# -----------------------------
# PARÁMETROS (idénticos al original)
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2

ALPHA = 0.6


class TelloControllerNode(Node):
    """
    Nodo ROS2 que:
      1. Se conecta al Tello vía djitellopy.
      2. Se suscribe a 'optitrack/rigid_body' del nodo optitrack_client.cpp
         para obtener la posición real (metros → cm).
      3. Ejecuta el loop de control proporcional en un timer a DT segundos.
      4. Publica estado estimado, error y señal de objetivo para monitoreo.
    """

    def __init__(self):
        super().__init__('tello_controller')

        # ---- Parámetros ROS2 declarables desde CLI / launch file ----------
        self.declare_parameter('target_x_cm', 50.0)
        self.declare_parameter('target_y_cm',  0.0)
        self.declare_parameter('target_z_cm', 150.0)
        self.declare_parameter('k_gain', 1.2)
        self.declare_parameter('max_iterations', 200)

        q_d_x = self.get_parameter('target_x_cm').value
        q_d_y = self.get_parameter('target_y_cm').value
        q_d_z = self.get_parameter('target_z_cm').value
        k_gain = self.get_parameter('k_gain').value
        self.max_iterations = int(self.get_parameter('max_iterations').value)

        # ---- Estado del controlador (idéntico al original) -----------------
        self.q     = np.array([0.0, 0.0, 0.0])          # posición estimada [cm]
        self.q_d   = np.array([q_d_x, q_d_y, q_d_z])   # objetivo [cm]
        self.K     = np.diag([k_gain, k_gain, k_gain])
        self.v_est = np.array([0.0, 0.0, 0.0])

        self.iteration        = 0
        self.goal_reached     = False
        self.optitrack_active = False   # True cuando llega el primer dato real

        # ---- Conexión Tello ------------------------------------------------
        self.drone = Tello()
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f'Error de conexión con Tello: {e}')
            raise

        self.get_logger().info(f"Batería: {self.drone.get_battery()} %")
        self.drone.takeoff()
        time.sleep(2)   # espera de estabilización (igual al original)

        # ---- QoS que hace match con SensorDataQoS del optitrack_client -----
        # El nodo C++ usa rclcpp::SensorDataQoS() → BEST_EFFORT, KEEP_LAST(10)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---- Suscriptor: posición real desde optitrack_client.cpp ----------
        # Topic : 'optitrack/rigid_body'
        # Tipo  : geometry_msgs/msg/PoseStamped
        # Campos usados:
        #   msg.pose.position.x/y/z  → posición en METROS (NatNet)
        #   msg.header.frame_id      → nombre del rigid body (Motive)
        self._optitrack_sub = self.create_subscription(
            PoseStamped,
            'optitrack/rigid_body',
            self._optitrack_callback,
            sensor_qos
        )

        # ---- Publicadores de monitoreo -------------------------------------
        # tello/estimated_state : [x, y, z, dist_to_goal]  (cm)
        # tello/control_error   : [ex, ey, ez]              (cm)
        # tello/goal_reached    : Bool
        self._state_pub = self.create_publisher(Float32MultiArray,
                                                'tello/estimated_state', 10)
        self._error_pub = self.create_publisher(Float32MultiArray,
                                                'tello/control_error', 10)
        self._goal_pub  = self.create_publisher(Bool,
                                                'tello/goal_reached', 10)

        # ---- Timer principal de control — DT = 0.1 s ----------------------
        self._control_timer = self.create_timer(DT, self._control_loop)

        self.get_logger().info(
            f"Controlador iniciado | objetivo [cm]: "
            f"x={self.q_d[0]}, y={self.q_d[1]}, z={self.q_d[2]}"
        )

    # =========================================================================
    # CALLBACK OptiTrack
    # =========================================================================
    def _optitrack_callback(self, msg: PoseStamped):
        """
        Recibe geometry_msgs/PoseStamped publicado por optitrack_client.cpp.

        optitrack_client.cpp asigna:
            msg.pose.position.x = data->RigidBodies[i].x   (metros, NatNet)
            msg.pose.position.y = data->RigidBodies[i].y
            msg.pose.position.z = data->RigidBodies[i].z
            msg.header.frame_id = nombre del rigid body en Motive

        Conversión a centímetros para que el controlador trabaje en la misma
        escala que el script original.
        """
        x_cm = msg.pose.position.x * 100.0
        y_cm = msg.pose.position.y * 100.0
        z_cm = msg.pose.position.z * 100.0

        self.q = np.array([x_cm, y_cm, z_cm])
        self.optitrack_active = True

    # =========================================================================
    # CONTROL  — idéntico al original
    # =========================================================================
    @staticmethod
    def _control(q, q_d, K):
        e = q_d - q
        u = K @ e
        return u, e

    # =========================================================================
    # CONVERSIÓN RC  — idéntica al original
    # =========================================================================
    @staticmethod
    def _velocity_to_rc(u):
        lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
        fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
        ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
        return lr, fb, ud, 0

    # =========================================================================
    # ODOMETRÍA  — idéntica al original
    # Se usa como fallback cuando OptiTrack aún no entrega datos.
    # =========================================================================
    def _odometry(self, q, v_est, rc_cmd, dt):
        lr, fb, ud, _ = rc_cmd

        v_meas = np.array([
            fb / RC_LIMIT * MAX_SPEED_CM_S,
            lr / RC_LIMIT * MAX_SPEED_CM_S,
            ud / RC_LIMIT * MAX_SPEED_CM_S
        ])

        v_est  = ALPHA * v_est + (1 - ALPHA) * v_meas
        q_next = q + v_est * dt

        return q_next, v_est

    # =========================================================================
    # LOOP DE CONTROL — ejecutado por el timer cada DT segundos
    # =========================================================================
    def _control_loop(self):
        if self.goal_reached:
            return

        if self.iteration >= self.max_iterations:
            self.get_logger().warn("Máximo de iteraciones alcanzado. Aterrizando.")
            self._shutdown_drone()
            return

        # ------ CONTROL (igual al original) ----------------------------------
        u, e = self._control(self.q, self.q_d, self.K)
        dist = float(np.linalg.norm(e))

        # ------ Conversión a comandos RC (igual al original) -----------------
        rc = self._velocity_to_rc(u)

        self.get_logger().info(
            f"[{self.iteration:03d}] "
            f"q_est [cm]: x={self.q[0]:.1f}, y={self.q[1]:.1f}, z={self.q[2]:.1f} | "
            f"error={dist:.2f} | rc={rc[:3]} | "
            f"fuente={'optitrack' if self.optitrack_active else 'odometría'}"
        )

        # ------ CONDICIONES DE PARADA (idénticas al original) ----------------
        if dist < TOL_CM:
            self.get_logger().info("¡Objetivo alcanzado por precisión!")
            self._publish_goal(True)
            self._shutdown_drone()
            return

        if self.iteration > 10 and rc[0] == 0 and rc[1] == 0 and rc[2] == 0:
            self.get_logger().info(
                "¡Objetivo alcanzado por comando mínimo (estancamiento)! Finalizando..."
            )
            self._publish_goal(True)
            self._shutdown_drone()
            return

        # ------ Enviar comandos RC al dron -----------------------------------
        self.drone.send_rc_control(*rc)

        # ------ Actualizar posición ------------------------------------------
        # Si OptiTrack ya está activo, self.q se actualiza en el callback.
        # Si aún no hay datos, se usa la odometría estimada (igual al original).
        if not self.optitrack_active:
            self.q, self.v_est = self._odometry(self.q, self.v_est, rc, DT)

        # ------ Publicar estado para monitoreo --------------------------------
        self._publish_state(dist)

        self.iteration += 1

    # =========================================================================
    # PUBLICADORES DE MONITOREO
    # =========================================================================
    def _publish_state(self, dist: float):
        state_msg = Float32MultiArray()
        state_msg.data = [float(self.q[0]), float(self.q[1]),
                          float(self.q[2]), dist]
        self._state_pub.publish(state_msg)

        error_msg = Float32MultiArray()
        e = self.q_d - self.q
        error_msg.data = [float(e[0]), float(e[1]), float(e[2])]
        self._error_pub.publish(error_msg)

    def _publish_goal(self, reached: bool):
        msg = Bool()
        msg.data = reached
        self._goal_pub.publish(msg)

    # =========================================================================
    # APAGADO ORDENADO
    # =========================================================================
    def _shutdown_drone(self):
        self.goal_reached = True
        self._control_timer.cancel()
        self.get_logger().info("Aterrizando...")
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        self.drone.land()
        self.drone.end()
        rclpy.shutdown()

    def destroy_node(self):
        """Limpieza segura ante Ctrl+C o destrucción externa del nodo."""
        if not self.goal_reached:
            try:
                self.drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                self.drone.land()
                self.drone.end()
            except Exception:
                pass
        super().destroy_node()


# =============================================================================
# MAIN
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TelloControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("\nInterrumpido por el usuario")
    except Exception as e:
        print(f"Error fatal: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
