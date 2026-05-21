#!/usr/bin/env python3
"""
Tello Controller Node — ROS2 Humble  (v2)
==========================================
Fuente de posición : SOLO OptiTrack  (optitrack/rigid_body)
Target             : hardcodeado en la sección PARÁMETROS
Gráfica            : matplotlib en tiempo real (mismo nodo, hilo separado)

Flujo de vuelo
--------------
1. INIT        — conectar y despegar
2. WAIT_FIX    — esperar primer mensaje de OptiTrack → registrar posición inicial
3. FLY         — control proporcional hacia el target
4. HOLD        — mantener posición 10 segundos
5. LAND        — aterrizar y cerrar
"""

import threading
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # backend con ventana propia; cambia a 'Qt5Agg' si prefieres
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registro del projection='3d')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, String

from djitellopy import Tello

# =============================================================================
# PARÁMETROS  — edita aquí antes de volar
# =============================================================================
DT           = 0.1      # [s]   período del loop de control
RC_LIMIT     = 40       # [-]   saturación máxima de comandos RC
TOL_CM       = 5.0      # [cm]  radio de la zona de llegada
HOLD_SECS    = 10.0     # [s]   tiempo de espera en el target antes de aterrizar
K_P          = 1.2      # [-]   ganancia proporcional (igual en los 3 ejes)

# Target en centímetros  (sistema de coordenadas NatNet × 100)
TARGET_X_CM  =  50.0
TARGET_Y_CM  =   0.0
TARGET_Z_CM  = 150.0


# =============================================================================
# ESTADOS DEL FLUJO
# =============================================================================
class State:
    INIT      = 'INIT'       # conectando y despegando
    WAIT_FIX  = 'WAIT_FIX'  # esperando primer dato de OptiTrack
    FLY       = 'FLY'       # volando hacia el target
    HOLD      = 'HOLD'      # manteniendo posición en el target
    LAND      = 'LAND'      # aterrizando


# =============================================================================
# NODO PRINCIPAL
# =============================================================================
class TelloControllerNode(Node):

    def __init__(self):
        super().__init__('tello_controller')

        # ---- Estado interno ------------------------------------------------
        self.state       = State.INIT
        self.q           = None          # posición actual [cm]  — None hasta OptiTrack
        self.q_initial   = None          # posición de despegue [cm]
        self.q_d         = np.array([TARGET_X_CM, TARGET_Y_CM, TARGET_Z_CM])
        self.hold_start  = None          # timestamp cuando se llega al target

        # ---- Trayectoria para la gráfica -----------------------------------
        self.traj_x = []
        self.traj_y = []
        self.traj_z = []
        self._traj_lock = threading.Lock()

        # ---- Conexión Tello ------------------------------------------------
        self.drone = Tello()
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f'Error de conexión con Tello: {e}')
            raise
        self.get_logger().info(f'Batería: {self.drone.get_battery()} %')

        self.drone.takeoff()
        time.sleep(2)                    # estabilización post-despegue
        self.get_logger().info('Dron en el aire. Esperando posición OptiTrack...')
        self.state = State.WAIT_FIX

        # ---- QoS: debe coincidir con SensorDataQoS del optitrack_client ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---- Suscriptor OptiTrack -----------------------------------------
        # optitrack_client.cpp publica:
        #   msg.pose.position.{x,y,z}  en METROS  (NatNet)
        #   msg.header.frame_id         nombre del rigid body en Motive
        self._optitrack_sub = self.create_subscription(
            PoseStamped,
            'optitrack/rigid_body',
            self._optitrack_cb,
            sensor_qos
        )

        # ---- Publicadores de monitoreo ------------------------------------
        self._state_pub  = self.create_publisher(String,           'tello/state',           10)
        self._pos_pub    = self.create_publisher(Float32MultiArray, 'tello/position',        10)
        self._error_pub  = self.create_publisher(Float32MultiArray, 'tello/control_error',   10)

        # ---- Timer de control ---------------------------------------------
        self._control_timer = self.create_timer(DT, self._control_loop)

        # ---- Hilo de gráfica (no bloquea el spin de ROS2) -----------------
        self._plot_thread = threading.Thread(target=self._run_plot, daemon=True)
        self._plot_thread.start()

        self.get_logger().info(
            f'Target [cm]: x={TARGET_X_CM}, y={TARGET_Y_CM}, z={TARGET_Z_CM}'
        )

    # =========================================================================
    # CALLBACK OPTITRACK
    # =========================================================================
    def _optitrack_cb(self, msg: PoseStamped):
        """
        Recibe geometry_msgs/PoseStamped desde optitrack_client.cpp.
        Convierte metros → centímetros y actualiza self.q.

        optitrack_client.cpp asigna:
            msg.pose.position.x = RigidBodies[i].x   (metros, NatNet)
            msg.pose.position.y = RigidBodies[i].y
            msg.pose.position.z = RigidBodies[i].z
        """
        x_cm = msg.pose.position.x * 100.0
        y_cm = msg.pose.position.y * 100.0
        z_cm = msg.pose.position.z * 100.0
        self.q = np.array([x_cm, y_cm, z_cm])

        # Primer dato: registrar posición inicial y transicionar a FLY
        if self.state == State.WAIT_FIX:
            self.q_initial = self.q.copy()
            self.get_logger().info(
                f'Posición inicial detectada [cm]: '
                f'x={x_cm:.1f}, y={y_cm:.1f}, z={z_cm:.1f}'
            )
            self.state = State.FLY

        # Registrar trayectoria para la gráfica
        with self._traj_lock:
            self.traj_x.append(x_cm)
            self.traj_y.append(y_cm)
            self.traj_z.append(z_cm)

    # =========================================================================
    # CONTROL PROPORCIONAL
    # =========================================================================
    @staticmethod
    def _control_p(q, q_d, k_p):
        """
        Controlador P puro.
        e = q_d - q         vector de error [cm]
        u = k_p * e         señal de control (proporcional al error)
        """
        e = q_d - q
        u = k_p * e
        return u, e

    @staticmethod
    def _to_rc(u):
        """
        Convierte señal de control a comandos RC enteros saturados.
        Mapeo de ejes:
            u[0] → fb (forward/backward)  eje X del controlador
            u[1] → lr (left/right)        eje Y del controlador
            u[2] → ud (up/down)           eje Z del controlador
        """
        fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
        lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
        ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
        return lr, fb, ud, 0    # (lr, fb, ud, yaw)

    # =========================================================================
    # LOOP DE CONTROL — timer DT
    # =========================================================================
    def _control_loop(self):

        # ---- WAIT_FIX: aún no hay dato de OptiTrack -----------------------
        if self.state == State.WAIT_FIX:
            # Hovering suave mientras esperamos la primera posición
            self.drone.send_rc_control(0, 0, 0, 0)
            return

        # ---- INIT no debería llegar aquí, pero por seguridad -------------
        if self.state == State.INIT:
            return

        # ---- FLY: control proporcional hacia el target --------------------
        if self.state == State.FLY:
            u, e = self._control_p(self.q, self.q_d, K_P)
            dist = float(np.linalg.norm(e))
            rc   = self._to_rc(u)

            self.get_logger().info(
                f'[FLY] pos=[{self.q[0]:.1f}, {self.q[1]:.1f}, {self.q[2]:.1f}] cm | '
                f'dist={dist:.1f} cm | rc={rc[:3]}'
            )

            self._publish_state()
            self._publish_error(e)

            if dist < TOL_CM:
                self.get_logger().info(
                    f'¡Target alcanzado! dist={dist:.1f} cm — '
                    f'manteniendo posición por {HOLD_SECS:.0f} s'
                )
                self.drone.send_rc_control(0, 0, 0, 0)
                self.hold_start = time.time()
                self.state = State.HOLD
                return

            self.drone.send_rc_control(*rc)
            return

        # ---- HOLD: mantener posición con P en el target -------------------
        if self.state == State.HOLD:
            u, e = self._control_p(self.q, self.q_d, K_P)
            rc   = self._to_rc(u)
            self.drone.send_rc_control(*rc)

            elapsed = time.time() - self.hold_start
            self.get_logger().info(
                f'[HOLD] {elapsed:.1f}/{HOLD_SECS:.0f} s | '
                f'pos=[{self.q[0]:.1f}, {self.q[1]:.1f}, {self.q[2]:.1f}] cm'
            )

            if elapsed >= HOLD_SECS:
                self.get_logger().info('HOLD completado — iniciando aterrizaje')
                self.state = State.LAND
            return

        # ---- LAND ---------------------------------------------------------
        if self.state == State.LAND:
            self._control_timer.cancel()
            self._shutdown_drone()

    # =========================================================================
    # PUBLICADORES DE MONITOREO
    # =========================================================================
    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self._state_pub.publish(msg)

        pos_msg = Float32MultiArray()
        pos_msg.data = [float(self.q[0]), float(self.q[1]), float(self.q[2])]
        self._pos_pub.publish(pos_msg)

    def _publish_error(self, e: np.ndarray):
        msg = Float32MultiArray()
        msg.data = [float(e[0]), float(e[1]), float(e[2])]
        self._error_pub.publish(msg)

    # =========================================================================
    # GRÁFICA EN TIEMPO REAL  (hilo separado — no bloquea ROS2)
    # =========================================================================
    def _run_plot(self):
        """
        Dibuja en tiempo real:
          🔵  punto inicial  (q_initial)
          🔴  punto target   (q_d)
          ─── trayectoria seguida por el dron (actualización cada 0.5 s)

        El hilo corre indefinidamente hasta que el nodo se destruye.
        """
        plt.ion()
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        fig.suptitle('Tello — Trayectoria en tiempo real', fontsize=13, fontweight='bold')

        while rclpy.ok():
            ax.cla()

            # ---- Ejes y etiquetas -----------------------------------------
            ax.set_xlabel('X [cm]')
            ax.set_ylabel('Y [cm]')
            ax.set_zlabel('Z [cm]')
            ax.set_title(f'Estado: {self.state}', fontsize=10)

            # ---- Punto target (siempre visible) ---------------------------
            ax.scatter(
                TARGET_X_CM, TARGET_Y_CM, TARGET_Z_CM,
                c='red', s=120, marker='*', zorder=5, label='Target'
            )
            ax.text(
                TARGET_X_CM, TARGET_Y_CM, TARGET_Z_CM + 5,
                f'  Target\n  ({TARGET_X_CM:.0f}, {TARGET_Y_CM:.0f}, {TARGET_Z_CM:.0f})',
                color='red', fontsize=8
            )

            # ---- Punto inicial (aparece cuando OptiTrack lo da) -----------
            if self.q_initial is not None:
                qi = self.q_initial
                ax.scatter(
                    qi[0], qi[1], qi[2],
                    c='blue', s=100, marker='o', zorder=5, label='Inicio'
                )
                ax.text(
                    qi[0], qi[1], qi[2] + 5,
                    f'  Inicio\n  ({qi[0]:.0f}, {qi[1]:.0f}, {qi[2]:.0f})',
                    color='blue', fontsize=8
                )

            # ---- Trayectoria seguida en tiempo real -----------------------
            with self._traj_lock:
                tx = list(self.traj_x)
                ty = list(self.traj_y)
                tz = list(self.traj_z)

            if len(tx) >= 2:
                ax.plot(tx, ty, tz, c='green', linewidth=1.5,
                        alpha=0.8, label='Trayectoria')

            # Posición actual
            if len(tx) >= 1:
                ax.scatter(
                    tx[-1], ty[-1], tz[-1],
                    c='green', s=80, marker='^', zorder=6, label='Posición actual'
                )

            # ---- Línea punteada inicio → target ---------------------------
            if self.q_initial is not None:
                qi = self.q_initial
                ax.plot(
                    [qi[0], TARGET_X_CM],
                    [qi[1], TARGET_Y_CM],
                    [qi[2], TARGET_Z_CM],
                    'k--', linewidth=0.8, alpha=0.4, label='Trayectoria ideal'
                )

            # ---- Autoescala con margen ------------------------------------
            all_x = [TARGET_X_CM] + tx + ([self.q_initial[0]] if self.q_initial is not None else [])
            all_y = [TARGET_Y_CM] + ty + ([self.q_initial[1]] if self.q_initial is not None else [])
            all_z = [TARGET_Z_CM] + tz + ([self.q_initial[2]] if self.q_initial is not None else [])

            margin = 20
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            ax.set_zlim(max(0, min(all_z) - margin), max(all_z) + margin)

            ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.pause(0.5)   # refresca cada 0.5 s

        plt.ioff()
        plt.close(fig)

    # =========================================================================
    # APAGADO ORDENADO
    # =========================================================================
    def _shutdown_drone(self):
        self.get_logger().info('Aterrizando...')
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            self.drone.land()
            self.drone.end()
        except Exception as e:
            self.get_logger().warn(f'Error durante aterrizaje: {e}')
        self.get_logger().info('Aterrizaje completado.')
        rclpy.shutdown()

    def destroy_node(self):
        """Aterrizaje de emergencia ante Ctrl+C o destrucción externa."""
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.3)
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
            node.get_logger().info('Interrumpido por el usuario (Ctrl+C)')
    except Exception as e:
        print(f'Error fatal: {e}')
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
