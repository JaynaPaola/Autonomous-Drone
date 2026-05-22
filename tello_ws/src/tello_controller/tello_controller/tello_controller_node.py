#!/usr/bin/env python3
"""
Tello Controller Node — ROS2 Humble  (v6)
==========================================
Flujo de vuelo
--------------
1. TAKEOFF   — despega, espera 5 s estabilizando
2. WAIT_FIX  — acumula 20 muestras OptiTrack → posición inicial
3. FLY       — control P hacia el target
4. BRAKE     — freno activo al llegar: envía RC contrario a la velocidad
5. HOLD      — banda muerta: solo corrige si el error supera DEADBAND_M
6. LAND      — aterriza y cierra

Cambios v6
----------
- DT reducido a 0.05 s (20 Hz) para reacción más rápida
- Estado BRAKE nuevo: detecta velocidad y envía RC opuesto hasta frenar
- HOLD con banda muerta: no envía comando si dist < DEADBAND_M (evita jitter)
- Ctrl+C aterriza limpiamente (signal handler)
- ExternalShutdownException capturada en hilo ROS2
- Logs reducidos: solo imprime en FLY cada 5 ciclos para no saturar
"""

import signal
import threading
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

from djitellopy import Tello

# =============================================================================
# PARÁMETROS
# =============================================================================
DT                   = 0.05   # [s]   20 Hz — reacción más rápida
RC_LIMIT             = 20     # [-]   saturación RC
TOL_M                = 0.08   # [m]   radio para entrar a BRAKE
DEADBAND_M           = 0.10   # [m]   banda muerta en HOLD (no corrige si dist < esto)
BRAKE_SECS           = 0.6    # [s]   duración del freno activo
HOLD_SECS            = 10.0   # [s]   tiempo en HOLD antes de aterrizar
K_P                  = 1.0    # [-]   ganancia proporcional en FLY

TAKEOFF_WAIT_S       = 5.0    # [s]   espera post-despegue
INIT_SAMPLES         = 20     # [-]   muestras para posición inicial
MAX_JUMP_M           = 0.30   # [m]   salto máximo válido entre frames
OPTITRACK_TIMEOUT_S  = 1.5    # [s]   sin dato → aterriza
STUCK_TIMEOUT_S      = 8.0    # [s]   sin progreso → aterriza
STUCK_THRESHOLD_M    = 0.02   # [m]   mejora mínima para resetear stuck
MAX_DISPLACEMENT_M   = 2.5    # [m]   límite de seguridad desde inicio

# Target — desplazamiento relativo al punto inicial [fb, lr, ud]
TARGET_FB  = -0.50
TARGET_LR  =  0.00
TARGET_UD  =  0.00


# =============================================================================
# ESTADOS
# =============================================================================
class State:
    TAKEOFF  = 'TAKEOFF'
    WAIT_FIX = 'WAIT_FIX'
    FLY      = 'FLY'
    BRAKE    = 'BRAKE'
    HOLD     = 'HOLD'
    LAND     = 'LAND'


# =============================================================================
# NODO
# =============================================================================
class TelloControllerNode(Node):

    def __init__(self):
        super().__init__('tello_controller')

        self.state         = State.TAKEOFF
        self.q             = None        # posición actual [fb, lr, ud]
        self.q_prev        = None        # posición frame anterior
        self.q_initial     = None        # posición inicial estable
        self.q_d           = None        # target absoluto
        self._init_samples = []
        self._takeoff_time = None
        self.hold_start    = None
        self.brake_start   = None
        self._fly_log_cnt  = 0           # contador para reducir logs en FLY

        # Watchdogs
        self._last_optitrack_time = None
        self._stuck_start         = None
        self._best_dist           = np.inf

        # Trayectoria gráfica
        self.traj_x = []
        self.traj_y = []
        self._traj_lock = threading.Lock()

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1          # solo el dato más reciente
        )

        self._optitrack_sub = self.create_subscription(
            PoseStamped, 'optitrack/rigid_body', self._optitrack_cb, sensor_qos
        )

        # Tello
        self.drone = Tello()
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f'Error de conexión: {e}')
            raise
        self.get_logger().info(f'Batería: {self.drone.get_battery()} %')
        self.drone.takeoff()
        self._takeoff_time = time.time()
        self.get_logger().info(f'En el aire. Estabilizando {TAKEOFF_WAIT_S:.0f} s...')

        # Publicadores
        self._pos_pub   = self.create_publisher(Float32MultiArray, 'tello/position',      1)
        self._error_pub = self.create_publisher(Float32MultiArray, 'tello/control_error', 1)

        # Timer de control a 20 Hz
        self._control_timer = self.create_timer(DT, self._control_loop)

        # ROS2 spin en hilo secundario
        self._ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._ros_thread.start()

        # Señal Ctrl+C → aterrizar limpiamente
        signal.signal(signal.SIGINT,  self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

    # -------------------------------------------------------------------------
    def _handle_sigint(self, sig, frame):
        self.get_logger().info('Señal recibida — aterrizando...')
        self.state = State.LAND

    def _ros_spin(self):
        try:
            rclpy.spin(self)
        except Exception:
            pass   # ExternalShutdownException al Ctrl+C — ignorar

    # =========================================================================
    # TRADUCTOR DE EJES
    # =========================================================================
    @staticmethod
    def _optitrack_to_tello(q_ot: np.ndarray) -> np.ndarray:
        # OptiTrack: X=adelante, Y=arriba, Z=lateral
        # Tello RC:  fb=adelante, ud=arriba, lr=lateral
        return np.array([q_ot[0], q_ot[2], q_ot[1]])   # [fb, lr, ud]

    # =========================================================================
    # VALIDACIÓN
    # =========================================================================
    def _is_valid_jump(self, new_q: np.ndarray) -> bool:
        if self.q is None:
            return True
        jump = float(np.linalg.norm(new_q - self.q))
        if jump > MAX_JUMP_M:
            self.get_logger().warn(f'[JUMP] {jump:.3f} m descartado')
            return False
        return True

    # =========================================================================
    # CALLBACK OPTITRACK
    # =========================================================================
    def _optitrack_cb(self, msg: PoseStamped):
        raw     = np.array([msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z])
        q_tello = self._optitrack_to_tello(raw)

        if self.state == State.TAKEOFF:
            if time.time() - self._takeoff_time < TAKEOFF_WAIT_S:
                return
            self.state = State.WAIT_FIX
            self.get_logger().info(f'Acumulando {INIT_SAMPLES} muestras...')

        if self.state == State.WAIT_FIX:
            self._init_samples.append(q_tello)
            self.get_logger().info(
                f'[WAIT_FIX] {len(self._init_samples)}/{INIT_SAMPLES} '
                f'fb={q_tello[0]:.3f} lr={q_tello[1]:.3f} ud={q_tello[2]:.3f}'
            )
            if len(self._init_samples) >= INIT_SAMPLES:
                self.q_initial = np.mean(self._init_samples, axis=0)
                self.q         = self.q_initial.copy()
                self.q_prev    = self.q_initial.copy()
                self.q_d       = self.q_initial + np.array([TARGET_FB, TARGET_LR, TARGET_UD])
                self._last_optitrack_time = time.time()
                self.get_logger().info(
                    f'Inicio: fb={self.q_initial[0]:.3f} lr={self.q_initial[1]:.3f} ud={self.q_initial[2]:.3f}'
                )
                self.get_logger().info(
                    f'Target: fb={self.q_d[0]:.3f} lr={self.q_d[1]:.3f} ud={self.q_d[2]:.3f}'
                )
                self.state = State.FLY
            return

        if not self._is_valid_jump(q_tello):
            return

        self.q_prev = self.q.copy()
        self.q      = q_tello
        self._last_optitrack_time = time.time()

        with self._traj_lock:
            self.traj_x.append(float(q_tello[0]))
            self.traj_y.append(float(q_tello[1]))

    # =========================================================================
    # WATCHDOGS
    # =========================================================================
    def _check_optitrack_watchdog(self) -> bool:
        if self._last_optitrack_time is None:
            return True
        elapsed = time.time() - self._last_optitrack_time
        if elapsed > OPTITRACK_TIMEOUT_S:
            self.get_logger().error(f'[WATCHDOG] Sin OptiTrack {elapsed:.2f} s')
            return False
        return True

    def _check_progress(self, dist: float) -> bool:
        if dist < self._best_dist - STUCK_THRESHOLD_M:
            self._best_dist   = dist
            self._stuck_start = time.time()
            return True
        if self._stuck_start is None:
            self._stuck_start = time.time()
            return True
        if time.time() - self._stuck_start > STUCK_TIMEOUT_S:
            self.get_logger().error(f'[WATCHDOG] Sin progreso {STUCK_TIMEOUT_S:.0f} s')
            return False
        return True

    def _check_displacement(self) -> bool:
        if self.q_initial is None or self.q is None:
            return True
        dist = float(np.linalg.norm(self.q - self.q_initial))
        if dist > MAX_DISPLACEMENT_M:
            self.get_logger().error(f'[WATCHDOG] Desplazamiento {dist:.2f} m > límite')
            return False
        return True

    # =========================================================================
    # CONTROL P
    # =========================================================================
    @staticmethod
    def _control_p(q, q_d):
        e = q_d - q
        u = K_P * e
        return u, e

    @staticmethod
    def _to_rc(u):
        fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
        lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
        ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
        return lr, fb, ud, 0

    # =========================================================================
    # VELOCIDAD ESTIMADA
    # =========================================================================
    def _velocity(self) -> np.ndarray:
        """Estima velocidad [m/s] con diferencia finita."""
        if self.q_prev is None or self.q is None:
            return np.zeros(3)
        return (self.q - self.q_prev) / DT

    # =========================================================================
    # LOOP DE CONTROL
    # =========================================================================
    def _control_loop(self):

        # TAKEOFF
        if self.state == State.TAKEOFF:
            self.drone.send_rc_control(0, 0, 0, 0)
            remaining = TAKEOFF_WAIT_S - (time.time() - self._takeoff_time)
            if remaining > 0:
                self.get_logger().info(f'[TAKEOFF] {remaining:.1f} s')
            return

        # WAIT_FIX
        if self.state == State.WAIT_FIX:
            self.drone.send_rc_control(0, 0, 0, 0)
            return

        # Watchdogs globales
        if not self._check_optitrack_watchdog() or not self._check_displacement():
            self.state = State.LAND
            return

        # FLY
        if self.state == State.FLY:
            u, e = self._control_p(self.q, self.q_d)
            dist = float(np.linalg.norm(e))
            rc   = self._to_rc(u)

            if not self._check_progress(dist):
                self.state = State.LAND
                return

            self._fly_log_cnt += 1
            if self._fly_log_cnt % 5 == 0:   # log cada 5 ciclos (0.25 s)
                self.get_logger().info(
                    f'[FLY] fb={self.q[0]:.3f} lr={self.q[1]:.3f} | '
                    f'dist={dist:.3f} m | rc={rc[:3]}'
                )

            self._publish_pos()
            self._publish_error(e)

            if dist < TOL_M:
                self.get_logger().info(f'¡Target! dist={dist:.3f} m — FRENO')
                self.drone.send_rc_control(0, 0, 0, 0)
                self.brake_start = time.time()
                self.state = State.BRAKE
                return

            self.drone.send_rc_control(*rc)
            return

        # BRAKE — freno activo: RC opuesto a la velocidad actual
        if self.state == State.BRAKE:
            vel = self._velocity()
            speed = float(np.linalg.norm(vel))

            # RC opuesto a la velocidad para frenar
            brake_u = -vel * 15.0   # ganancia de freno
            lr  = int(np.clip(np.round(brake_u[1]), -RC_LIMIT, RC_LIMIT))
            fb  = int(np.clip(np.round(brake_u[0]), -RC_LIMIT, RC_LIMIT))
            ud  = int(np.clip(np.round(brake_u[2]), -RC_LIMIT, RC_LIMIT))
            self.drone.send_rc_control(lr, fb, ud, 0)

            elapsed = time.time() - self.brake_start
            self.get_logger().info(
                f'[BRAKE] {elapsed:.2f}/{BRAKE_SECS:.1f} s | vel={speed:.3f} m/s'
            )

            if elapsed >= BRAKE_SECS or speed < 0.02:
                self.get_logger().info('[BRAKE] Completo — HOLD')
                self.drone.send_rc_control(0, 0, 0, 0)
                self.hold_start = time.time()
                self.state = State.HOLD
            return

        # HOLD — banda muerta: solo corrige si el error supera DEADBAND_M
        if self.state == State.HOLD:
            e    = self.q_d - self.q
            dist = float(np.linalg.norm(e))

            if dist > DEADBAND_M:
                # Corrección suave con ganancia reducida
                u  = 0.6 * K_P * e
                rc = self._to_rc(u)
                self.drone.send_rc_control(*rc)
            else:
                self.drone.send_rc_control(0, 0, 0, 0)

            elapsed = time.time() - self.hold_start
            self.get_logger().info(
                f'[HOLD] {elapsed:.1f}/{HOLD_SECS:.0f} s | '
                f'fb={self.q[0]:.3f} lr={self.q[1]:.3f} | dist={dist:.3f} m'
            )
            self._publish_pos()
            self._publish_error(e)

            if elapsed >= HOLD_SECS:
                self.get_logger().info('HOLD completo — aterrizando')
                self.state = State.LAND
            return

        # LAND
        if self.state == State.LAND:
            self._control_timer.cancel()
            self._shutdown_drone()

    # =========================================================================
    # PUBLICADORES
    # =========================================================================
    def _publish_pos(self):
        msg = Float32MultiArray()
        msg.data = [float(self.q[0]), float(self.q[1]), float(self.q[2])]
        self._pos_pub.publish(msg)

    def _publish_error(self, e: np.ndarray):
        msg = Float32MultiArray()
        msg.data = [float(e[0]), float(e[1]), float(e[2])]
        self._error_pub.publish(msg)

    # =========================================================================
    # GRÁFICA 2D — hilo principal
    # =========================================================================
    def run_plot(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.suptitle('Tello — Trayectoria (planta)', fontsize=13, fontweight='bold')

        def update(_frame):
            if self.state == State.LAND:
                plt.close(fig)
                return
            ax.cla()
            ax.set_xlabel('fb [m]')
            ax.set_ylabel('lr [m]')
            ax.set_title(f'Estado: {self.state}', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)

            if self.q_d is not None:
                ax.scatter(self.q_d[0], self.q_d[1],
                           c='red', s=120, marker='*', zorder=5, label='Target')
            if self.q_initial is not None:
                ax.scatter(self.q_initial[0], self.q_initial[1],
                           c='blue', s=100, marker='o', zorder=5, label='Inicio')

            with self._traj_lock:
                tx = list(self.traj_x)
                ty = list(self.traj_y)

            if len(tx) >= 2:
                ax.plot(tx, ty, c='green', linewidth=1.5, alpha=0.8, label='Trayectoria')
            if len(tx) >= 1:
                ax.scatter(tx[-1], ty[-1], c='green', s=80,
                           marker='^', zorder=6, label='Actual')
                ax.annotate(f'({tx[-1]:.2f}, {ty[-1]:.2f})',
                            xy=(tx[-1], ty[-1]),
                            xytext=(tx[-1] + 0.03, ty[-1] + 0.03),
                            color='green', fontsize=7)

            pts_x = ([self.q_d[0]] if self.q_d is not None else []) + tx
            pts_y = ([self.q_d[1]] if self.q_d is not None else []) + ty
            if pts_x and pts_y:
                margin = 0.30
                ax.set_xlim(min(pts_x) - margin, max(pts_x) + margin)
                ax.set_ylim(min(pts_y) - margin, max(pts_y) + margin)

            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()

        ani = animation.FuncAnimation(   # noqa: F841
            fig, update, interval=500, cache_frame_data=False
        )
        plt.show()

        if self.state not in (State.LAND,):
            self.get_logger().warn('Ventana cerrada — aterrizaje de emergencia')
            self.state = State.LAND

    # =========================================================================
    # APAGADO
    # =========================================================================
    def _shutdown_drone(self):
        self.get_logger().info('Aterrizando...')
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.4)
            self.drone.land()
            self.drone.end()
        except Exception as e:
            self.get_logger().warn(f'Error aterrizaje: {e}')
        self.get_logger().info('Aterrizaje completado.')
        rclpy.shutdown()

    def destroy_node(self):
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
        node.run_plot()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error fatal: {e}')
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()