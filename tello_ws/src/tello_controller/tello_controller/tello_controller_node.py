#!/usr/bin/env python3
"""
Tello Controller Node — ROS2 Humble  (v9)
==========================================

ARQUITECTURA DE LAZO CERRADO CON TRAYECTORIA FIJA
--------------------------------------------------

Una vez planeada la trayectoria, el dron la sigue punto a punto
sin nunca desviarse de ella. Si el dron se aleja de la línea,
el control lo jala de regreso a la trayectoria — NO al goal final.

El sistema sigue este orden de inicio obligatorio:
  1. Conectar a OptiTrack y esperar señal válida
  2. Conectar al dron y despegar
  3. Abrir gráfica en tiempo real
  4. Identificar posición inicial (promedio de muestras OptiTrack)
  5. Planear trayectoria al waypoint (línea recta, N puntos fijos)
  6. Ejecutar movimiento en lazo cerrado siguiendo la trayectoria
  7. Mantener posición 5 s
  8. Aterrizar

LAZO CERRADO - SEGUIMIENTO DE TRAYECTORIA FIJA (20 Hz)
-------------------------------------------------------
La trayectoria se divide en PLAN_STEPS puntos equidistantes.
Hay un índice `_wp_idx` que avanza cuando el dron llega cerca
del punto actual. En cada ciclo:

  1. Leo posición actual de OptiTrack  → q_now
  2. Leo el punto objetivo actual       → tray.waypoints[_wp_idx]
  3. Calculo error al punto actual      → e = punto - q_now
  4. Calculo corrección lateral         → jalón de vuelta a la línea
  5. Señal total = K_P * e + K_LAT * lateral
  6. Envío RC
  7. Si dist al punto < TOL_WP → avanzo al siguiente punto
  8. Si _wp_idx llegó al final → BRAKE

El dron nunca "salta" al goal. Sigue cada sub-waypoint en orden.
Si se desvía, el error lateral lo devuelve a la línea.

ESTADOS
-------
  WAIT_OPTITRACK → esperando señal válida de OptiTrack
  WAIT_FIX       → acumulando muestras para posición inicial
  PLAN           → calculando trayectoria al waypoint (instante)
  FLY            → lazo cerrado siguiendo trayectoria fija
  BRAKE          → freno activo al llegar al WP final
  HOLD           → mantener posición 5 s
  LAND           → aterrizar
"""

import signal
import threading
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

from djitellopy import Tello


# =============================================================================
# PARÁMETROS
# =============================================================================

DT                  = 0.05    # [s]  Periodo de control (20 Hz)
RC_LIMIT            = 20      # [-]  Saturación RC (Tello: -100 a 100)
TOL_M               = 0.08    # [m]  Radio de llegada al waypoint
DEADBAND_M          = 0.10    # [m]  Banda muerta en HOLD
BRAKE_SECS          = 0.6     # [s]  Duración del freno activo
HOLD_SECS           = 5.0     # [s]  Tiempo de mantenimiento en waypoint
K_P                 = 1.0     # [-]  Ganancia proporcional
TAKEOFF_WAIT_S      = 5.0     # [s]  Espera post-despegue para estabilización
INIT_SAMPLES        = 20      # [-]  Muestras para calcular posición inicial
MAX_JUMP_M          = 0.30    # [m]  Salto máximo permitido entre frames
OPTITRACK_TIMEOUT_S = 1.0     # [s]  Timeout de señal OptiTrack → emergencia
STUCK_TIMEOUT_S     = 10.0    # [s]  Sin progreso → emergencia
STUCK_THRESHOLD_M   = 0.02    # [m]  Mejora mínima para resetear watchdog
MAX_DISPLACEMENT_M  = 2.5     # [m]  Distancia máxima desde origen → emergencia
OPTITRACK_VALID_MIN = 5       # [-]  Muestras mínimas para considerar señal válida

# Waypoint destino — coordenadas absolutas en el sistema OptiTrack [m]
TARGET_X =  1.09
TARGET_Y = -0.05
TARGET_Z =  0.67

K_LAT               = 1.5     # [-]  Ganancia de corrección lateral
                              #      Jala al dron de vuelta a la línea si se desvía.
                              #      Más alto = más agresivo al volver a la trayectoria.

TOL_WP              = 0.12    # [m]  Radio de llegada a cada sub-waypoint intermedio.
                              #      Al entrar en este radio, se avanza al siguiente punto.
                              #      Más pequeño = sigue más fielmente la línea.
                              #      Más grande = avanza más rápido pero con menos precisión.

# Resolución de trayectoria planeada (número de sub-waypoints)
# Más puntos = el dron sigue la línea con más fidelidad pero avanza más despacio.
# Para una línea recta, 20–50 es suficiente.
PLAN_STEPS = 50


# =============================================================================
# MÁQUINA DE ESTADOS
# =============================================================================
class State:
    WAIT_OPTITRACK = 'WAIT_OPTITRACK'  # esperando señal
    WAIT_FIX       = 'WAIT_FIX'        # acumulando muestras iniciales
    PLAN           = 'PLAN'            # calculando trayectoria
    FLY            = 'FLY'             # lazo cerrado hacia waypoint
    BRAKE          = 'BRAKE'           # freno activo
    HOLD           = 'HOLD'            # mantener posición
    LAND           = 'LAND'            # aterrizar


# =============================================================================
# TRAYECTORIA PLANEADA — seguimiento punto a punto
# =============================================================================
class Trajectory:
    """
    Línea recta dividida en PLAN_STEPS sub-waypoints equidistantes.

    El dron sigue esta trayectoria en orden estricto:
      - Siempre apunta al sub-waypoint actual (waypoints[idx])
      - Cuando llega a TOL_WP metros del sub-waypoint, avanza al siguiente
      - Nunca salta al goal directamente
      - Si se desvía lateralmente, el control lo regresa a la línea

    Métodos clave
    -------------
    current_target()   → coordenadas del sub-waypoint actual
    advance()          → avanza al siguiente sub-waypoint; True si hay más
    lateral_error(q)   → vector desde q hasta la línea (para corrección)
    progress()         → fracción [0-1] del trayecto completado
    """

    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray,
                 steps: int = PLAN_STEPS):
        self.q_start   = q_start.copy()
        self.q_goal    = q_goal.copy()
        # Sub-waypoints equidistantes incluyendo start y goal
        self.waypoints = np.linspace(q_start, q_goal, steps)
        self._idx      = 0          # índice del sub-waypoint actual
        self._steps    = steps

    # ── Sub-waypoint actual ───────────────────────────────────────────────────
    def current_target(self) -> np.ndarray:
        """Devuelve las coordenadas del sub-waypoint que el dron debe seguir ahora."""
        return self.waypoints[self._idx].copy()

    @property
    def idx(self) -> int:
        return self._idx

    # ── Avanzar al siguiente punto ────────────────────────────────────────────
    def advance(self) -> bool:
        """
        Avanza al siguiente sub-waypoint.
        Devuelve True si aún hay puntos por visitar, False si se llegó al final.
        """
        self._idx += 1
        return self._idx < self._steps

    def is_finished(self) -> bool:
        return self._idx >= self._steps

    # ── Progreso [0–1] ────────────────────────────────────────────────────────
    def progress(self) -> float:
        return self._idx / max(self._steps - 1, 1)

    # ── Error lateral — jalón de vuelta a la línea ────────────────────────────
    def lateral_error(self, q: np.ndarray) -> np.ndarray:
        """
        Proyecta q sobre el segmento completo q_start→q_goal.
        Devuelve el vector que apunta desde q hasta el punto más cercano
        de la línea planeada.

        Este vector se usa para corregir desviaciones laterales:
        si el dron se fue a la derecha, lateral_error apunta a la izquierda
        (de vuelta a la línea).

        NO apunta al goal — apunta a la trayectoria.
        """
        ab        = self.q_goal - self.q_start
        ap        = q - self.q_start
        ab_norm_sq = np.dot(ab, ab)
        if ab_norm_sq < 1e-12:
            return self.q_start - q
        t  = np.clip(np.dot(ap, ab) / ab_norm_sq, 0.0, 1.0)
        cp = self.q_start + t * ab
        return cp - q   # vector: dron → línea planeada


# =============================================================================
# NODO PRINCIPAL
# =============================================================================
class TelloControllerNode(Node):

    def __init__(self):
        super().__init__('tello_controller')

        # ── Estado ────────────────────────────────────────────────────────────
        self.state         = State.WAIT_OPTITRACK
        self.q             = None          # posición actual [X, Y, Z] OptiTrack
        self.q_prev        = None          # posición anterior (para velocidad)
        self.q_initial     = None          # posición inicial (post-despegue)
        self.q_d           = None          # waypoint destino absoluto
        self.trajectory    = None          # trayectoria planeada

        self._init_samples  = []
        self._takeoff_time  = None
        self.hold_start     = None
        self.brake_start    = None
        self._fly_log_cnt   = 0

        # ── Watchdogs ─────────────────────────────────────────────────────────
        self._last_optitrack_time = None
        self._stuck_start         = None
        self._best_dist           = np.inf
        self._optitrack_count     = 0     # total de frames recibidos

        # ── Log de replanificación (para la UI) ───────────────────────────────
        # Cada ciclo de control escribe aquí un dict con el estado del ciclo
        self._cycle_log   = deque(maxlen=6)   # últimos N ciclos visibles en UI
        self._cycle_lock  = threading.Lock()

        # ── Trayectoria ejecutada (para gráfica) ──────────────────────────────
        self.traj_x = []
        self.traj_y = []
        self.traj_z = []
        self._traj_lock = threading.Lock()

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._optitrack_sub = self.create_subscription(
            PoseStamped,
            'optitrack/rigid_body',
            self._optitrack_cb,
            sensor_qos
        )

        # ── Publicadores de monitoreo ──────────────────────────────────────────
        self._pos_pub   = self.create_publisher(Float32MultiArray, 'tello/position',      1)
        self._error_pub = self.create_publisher(Float32MultiArray, 'tello/control_error', 1)

        # ── Señales ───────────────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

        self.get_logger().info('Nodo iniciado. Esperando señal de OptiTrack...')

    # =========================================================================
    # FASE 1: ESPERAR OPTITRACK ANTES DE CONECTAR EL DRON
    # =========================================================================
    def wait_for_optitrack(self):
        """
        Bloquea hasta recibir al menos OPTITRACK_VALID_MIN frames de OptiTrack.
        Se llama desde main() ANTES de conectar el dron.
        """
        self.get_logger().info(
            f'Esperando {OPTITRACK_VALID_MIN} frames de OptiTrack...'
        )
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._optitrack_count >= OPTITRACK_VALID_MIN:
                self.get_logger().info(
                    f'OptiTrack OK — {self._optitrack_count} frames recibidos. '
                    f'Posición: X={self.q[0]:.3f} Y={self.q[1]:.3f} Z={self.q[2]:.3f}'
                )
                return
            self.get_logger().info(
                f'  OptiTrack: {self._optitrack_count}/{OPTITRACK_VALID_MIN} frames...'
            )

    # =========================================================================
    # FASE 2: CONECTAR DRON Y DESPEGAR
    # =========================================================================
    def connect_and_takeoff(self):
        """
        Conecta al Tello, muestra batería, despega.
        Se llama desde main() DESPUÉS de wait_for_optitrack().
        """
        self.drone = Tello()
        try:
            self.drone.connect()
        except Exception as e:
            self.get_logger().error(f'Error de conexión al dron: {e}')
            raise
        bat = self.drone.get_battery()
        self.get_logger().info(f'Dron conectado — Batería: {bat} %')
        if bat < 20:
            raise RuntimeError(f'Batería insuficiente: {bat}%')

        self.drone.takeoff()
        self._takeoff_time = time.time()
        self.state = State.WAIT_FIX  # empieza a acumular muestras post-despegue
        self.get_logger().info(
            f'Despegue exitoso. Estabilizando {TAKEOFF_WAIT_S:.0f} s...'
        )

        # Ahora que el dron está en el aire, arrancar el timer de control
        self._control_timer = self.create_timer(DT, self._control_loop)

        # ROS2 spin en hilo secundario (plt.show() bloquea el principal)
        self._ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._ros_thread.start()

    # =========================================================================
    # SEÑALES
    # =========================================================================
    def _handle_sigint(self, sig, frame):
        self.get_logger().info('Señal recibida — aterrizando...')
        self.state = State.LAND

    def _ros_spin(self):
        try:
            rclpy.spin(self)
        except Exception:
            pass

    # =========================================================================
    # CALLBACK OPTITRACK
    # =========================================================================
    def _optitrack_cb(self, msg: PoseStamped):
        q_raw = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Fase de espera inicial: solo contar frames y guardar posición
        if self.state == State.WAIT_OPTITRACK:
            self.q = q_raw
            self._optitrack_count += 1
            self._last_optitrack_time = time.time()
            return

        # WAIT_FIX: esperar TAKEOFF_WAIT_S antes de acumular
        if self.state == State.WAIT_FIX:
            self._last_optitrack_time = time.time()
            elapsed = time.time() - self._takeoff_time
            if elapsed < TAKEOFF_WAIT_S:
                # Todavía estabilizando — actualizar q pero no acumular
                self.q = q_raw
                return
            # Ya pasó el tiempo de estabilización → acumular
            self._init_samples.append(q_raw)
            self.get_logger().info(
                f'[WAIT_FIX] {len(self._init_samples)}/{INIT_SAMPLES} '
                f'X={q_raw[0]:.3f} Y={q_raw[1]:.3f} Z={q_raw[2]:.3f}'
            )
            if len(self._init_samples) >= INIT_SAMPLES:
                self.q_initial = np.mean(self._init_samples, axis=0)
                self.q         = self.q_initial.copy()
                self.q_prev    = self.q_initial.copy()
                self.get_logger().info(
                    f'Posición inicial confirmada: '
                    f'X={self.q_initial[0]:.3f} '
                    f'Y={self.q_initial[1]:.3f} '
                    f'Z={self.q_initial[2]:.3f}'
                )
                self.state = State.PLAN
            return

        # FLY / BRAKE / HOLD: validar y actualizar
        if not self._is_valid_jump(q_raw):
            return

        self.q_prev = self.q.copy() if self.q is not None else q_raw.copy()
        self.q      = q_raw
        self._last_optitrack_time = time.time()
        self._optitrack_count    += 1

        with self._traj_lock:
            self.traj_x.append(float(q_raw[0]))
            self.traj_y.append(float(q_raw[1]))
            self.traj_z.append(float(q_raw[2]))

    # =========================================================================
    # VALIDACIÓN DE SALTO
    # =========================================================================
    def _is_valid_jump(self, new_q: np.ndarray) -> bool:
        if self.q is None:
            return True
        jump = float(np.linalg.norm(new_q - self.q))
        if jump > MAX_JUMP_M:
            self.get_logger().warn(f'[JUMP] Descartado — salto de {jump:.3f} m')
            return False
        return True

    # =========================================================================
    # WATCHDOGS
    # =========================================================================
    def _check_optitrack_watchdog(self) -> bool:
        if self._last_optitrack_time is None:
            return True
        elapsed = time.time() - self._last_optitrack_time
        if elapsed > OPTITRACK_TIMEOUT_S:
            self.get_logger().error(
                f'[WATCHDOG] Sin OptiTrack {elapsed:.2f} s → emergencia'
            )
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
            self.get_logger().error(
                f'[WATCHDOG] Sin progreso {STUCK_TIMEOUT_S:.0f} s → emergencia'
            )
            return False
        return True

    def _check_displacement(self) -> bool:
        if self.q_initial is None or self.q is None:
            return True
        dist = float(np.linalg.norm(self.q - self.q_initial))
        if dist > MAX_DISPLACEMENT_M:
            self.get_logger().error(
                f'[WATCHDOG] Desplazamiento {dist:.2f} m > {MAX_DISPLACEMENT_M} m → emergencia'
            )
            return False
        return True

    # =========================================================================
    # CONTROL PROPORCIONAL
    # =========================================================================
    def _control_p(self, q, q_d):
        e = q_d - q
        u = K_P * e
        return u, e

    def _velocity(self) -> np.ndarray:
        if self.q_prev is None:
            return np.zeros(3)
        return (self.q - self.q_prev) / DT

    @staticmethod
    def _to_rc(u):
        """
        Mapeo de señal de control a comandos RC.
        Ajustar una vez identificado qué eje de OptiTrack
        corresponde a qué movimiento físico del dron.

        send_rc_control(lr, fb, ud, yaw)
        """
        fb = int(np.clip(np.round(u[0] * 100), -RC_LIMIT, RC_LIMIT))  # X → adelante/atrás
        ud = int(np.clip(np.round(u[1] * 100), -RC_LIMIT, RC_LIMIT))  # Y → arriba/abajo
        lr = int(np.clip(np.round(u[2] * 100), -RC_LIMIT, RC_LIMIT))  # Z → izquierda/derecha
        return lr, fb, ud, 0

    # =========================================================================
    # LOG DE CICLO (para la UI)
    # =========================================================================
    def _log_cycle(self, q, q_d, e, dist, rc, lateral_err=None):
        """
        Registra el estado del ciclo de replanificación para mostrarlo en la UI.
        """
        entry = {
            'time':    time.time(),
            'q':       q.copy(),
            'q_d':     q_d.copy(),
            'e':       e.copy(),
            'dist':    dist,
            'rc':      rc,
            'lat_err': float(np.linalg.norm(lateral_err)) if lateral_err is not None else 0.0,
        }
        with self._cycle_lock:
            self._cycle_log.append(entry)

    # =========================================================================
    # LOOP DE CONTROL — 20 Hz
    # =========================================================================
    def _control_loop(self):
        """
        Ciclo de replanificación continua:
          1. Leo posición actual de OptiTrack (self.q)
          2. Calculo error al waypoint (self.q_d)
          3. Calculo vector de control (P)
          4. Envío RC al dron
          5. En el siguiente ciclo, repito con posición real actualizada
        """

        # WAIT_FIX: hover quieto
        if self.state in (State.WAIT_FIX, State.WAIT_OPTITRACK):
            self.drone.send_rc_control(0, 0, 0, 0)
            return

        # PLAN: definir waypoint y trayectoria
        if self.state == State.PLAN:
            self.drone.send_rc_control(0, 0, 0, 0)
            self.q_d        = np.array([TARGET_X, TARGET_Y, TARGET_Z])
            self.trajectory = Trajectory(self.q_initial, self.q_d)
            dist_total      = float(np.linalg.norm(self.q_d - self.q_initial))
            self.get_logger().info('=' * 55)
            self.get_logger().info('TRAYECTORIA PLANEADA')
            self.get_logger().info(
                f'  Origen : X={self.q_initial[0]:.3f} '
                f'Y={self.q_initial[1]:.3f} Z={self.q_initial[2]:.3f}'
            )
            self.get_logger().info(
                f'  Destino: X={self.q_d[0]:.3f} '
                f'Y={self.q_d[1]:.3f} Z={self.q_d[2]:.3f}'
            )
            self.get_logger().info(f'  Distancia total: {dist_total:.3f} m')
            self.get_logger().info('=' * 55)
            self.state = State.FLY
            return

        # Watchdogs globales
        if not self._check_optitrack_watchdog() or not self._check_displacement():
            self.state = State.LAND
            return

        # ── FLY: lazo cerrado siguiendo trayectoria fija punto a punto ──────────
        if self.state == State.FLY:
            q_now = self.q.copy()
            traj  = self.trajectory

            # ── Avanzar índice si el dron ya está cerca del punto actual ──────
            # Hacemos esto PRIMERO para que en cuanto llegue a un punto,
            # inmediatamente apunte al siguiente en este mismo ciclo.
            while not traj.is_finished():
                pt   = traj.current_target()
                d_pt = float(np.linalg.norm(pt - q_now))
                if d_pt < TOL_WP:
                    has_more = traj.advance()
                    if not has_more:
                        # Llegamos al último punto → BRAKE
                        self.get_logger().info(
                            f'[FLY] ¡Trayectoria completa! '
                            f'dist al WP final = {float(np.linalg.norm(traj.q_goal - q_now)):.3f} m'
                            f' → BRAKE'
                        )
                        self.drone.send_rc_control(0, 0, 0, 0)
                        self.brake_start = time.time()
                        self.state = State.BRAKE
                        return
                else:
                    break   # aún no llegó al punto actual → salir del while

            if traj.is_finished():
                # Salvaguarda: si el índice ya está al final, ir a BRAKE
                self.drone.send_rc_control(0, 0, 0, 0)
                self.brake_start = time.time()
                self.state = State.BRAKE
                return

            # ── Punto objetivo actual (sub-waypoint en la trayectoria) ────────
            pt_target = traj.current_target()

            # ── Error al sub-waypoint actual (NO al goal final) ───────────────
            e_wp   = pt_target - q_now
            dist_wp = float(np.linalg.norm(e_wp))

            # ── Corrección lateral: jalar al dron de vuelta a la línea ────────
            # lateral_error() proyecta q sobre la línea completa start→goal
            # y devuelve el vector que apunta desde q hasta la línea.
            lat_err = traj.lateral_error(q_now)

            # ── Señal de control total ─────────────────────────────────────────
            # Componente principal:  K_P  * error al punto actual de la trayectoria
            # Componente lateral:    K_LAT * jalón de vuelta a la línea
            # El dron sigue los sub-waypoints Y permanece en la línea planeada.
            u_total = K_P * e_wp + K_LAT * lat_err
            rc      = self._to_rc(u_total)

            # Distancia al goal final (para el watchdog de progreso)
            dist_to_goal = float(np.linalg.norm(traj.q_goal - q_now))
            if not self._check_progress(dist_to_goal):
                self.state = State.LAND
                return

            # Log del ciclo (para UI y terminal)
            self._log_cycle(q_now, pt_target, e_wp, dist_wp, rc, lat_err)

            self._fly_log_cnt += 1
            if self._fly_log_cnt % 5 == 0:
                self.get_logger().info(
                    f'[FLY] '
                    f'Estoy: ({q_now[0]:.3f}, {q_now[1]:.3f}, {q_now[2]:.3f}) | '
                    f'Punto [{traj.idx}/{traj._steps-1}]: '
                    f'({pt_target[0]:.3f}, {pt_target[1]:.3f}, {pt_target[2]:.3f}) | '
                    f'dist_punto: {dist_wp:.3f} m | '
                    f'dist_goal: {dist_to_goal:.3f} m | '
                    f'lat: {np.linalg.norm(lat_err):.3f} m | '
                    f'RC: {rc[:3]} | '
                    f'progreso: {traj.progress()*100:.0f}%'
                )

            self._publish_pos()
            self._publish_error(e_wp)
            self.drone.send_rc_control(*rc)
            return


        # ── BRAKE ─────────────────────────────────────────────────────────────
        if self.state == State.BRAKE:
            vel   = self._velocity()
            speed = float(np.linalg.norm(vel))
            b_u   = -vel * 15.0
            fb    = int(np.clip(np.round(b_u[0]), -RC_LIMIT, RC_LIMIT))
            ud    = int(np.clip(np.round(b_u[1]), -RC_LIMIT, RC_LIMIT))
            lr    = int(np.clip(np.round(b_u[2]), -RC_LIMIT, RC_LIMIT))
            self.drone.send_rc_control(lr, fb, ud, 0)
            elapsed = time.time() - self.brake_start
            self.get_logger().info(
                f'[BRAKE] {elapsed:.2f}/{BRAKE_SECS:.1f} s | vel={speed:.3f} m/s'
            )
            if elapsed >= BRAKE_SECS or speed < 0.02:
                self.drone.send_rc_control(0, 0, 0, 0)
                self.hold_start = time.time()
                self.state = State.HOLD
            return

        # ── HOLD ──────────────────────────────────────────────────────────────
        if self.state == State.HOLD:
            e    = self.q_d - self.q
            dist = float(np.linalg.norm(e))

            if dist > DEADBAND_M:
                u  = 0.6 * K_P * e
                rc = self._to_rc(u)
                self.drone.send_rc_control(*rc)
            else:
                self.drone.send_rc_control(0, 0, 0, 0)

            elapsed = time.time() - self.hold_start
            self.get_logger().info(
                f'[HOLD] {elapsed:.1f}/{HOLD_SECS:.0f} s | '
                f'pos=({self.q[0]:.3f}, {self.q[1]:.3f}, {self.q[2]:.3f}) | '
                f'dist={dist:.3f} m'
            )
            self._publish_pos()
            self._publish_error(e)

            if elapsed >= HOLD_SECS:
                self.state = State.LAND
            return

        # ── LAND ──────────────────────────────────────────────────────────────
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
    # GRÁFICA EN TIEMPO REAL
    # =========================================================================
    def run_plot(self):
        """
        Panel de monitoreo con 3 secciones:
          ┌──────────────────┬──────────────────┐
          │  Planta X–Y      │  Altura Z(t)     │
          │  (vista arriba)  │                  │
          ├──────────────────┴──────────────────┤
          │  Log de ciclos de replanificación   │
          └─────────────────────────────────────┘
        """
        # Paleta oscura tipo HUD
        BG      = '#0a0e1a'
        GRID_C  = '#1a2035'
        ACCENT  = '#00d4ff'    # cyan
        TARGET  = '#ff4444'    # rojo target
        TRAJ    = '#00ff88'    # verde trayectoria ejecutada
        PLAN_C  = '#ffaa00'    # naranja trayectoria planeada
        INIT_C  = '#8888ff'    # azul posición inicial
        TEXT_C  = '#cce0ff'
        LOG_BG  = '#060912'

        plt.rcParams.update({
            'figure.facecolor':  BG,
            'axes.facecolor':    BG,
            'axes.edgecolor':    GRID_C,
            'axes.labelcolor':   TEXT_C,
            'xtick.color':       TEXT_C,
            'ytick.color':       TEXT_C,
            'text.color':        TEXT_C,
            'grid.color':        GRID_C,
            'legend.facecolor':  '#111827',
            'legend.edgecolor':  GRID_C,
        })

        fig = plt.figure(figsize=(15, 9), facecolor=BG)
        fig.suptitle(
            'TELLO — LAZO CERRADO CON REPLANIFICACIÓN CONTINUA',
            fontsize=13, fontweight='bold', color=ACCENT,
            fontfamily='monospace', y=0.98
        )

        gs = GridSpec(
            2, 2,
            figure=fig,
            height_ratios=[3, 1.4],
            hspace=0.35, wspace=0.28,
            left=0.06, right=0.97, top=0.94, bottom=0.04
        )

        ax_xy  = fig.add_subplot(gs[0, 0])   # planta X–Y
        ax_z   = fig.add_subplot(gs[0, 1])   # altura Z(t)
        ax_log = fig.add_subplot(gs[1, :])   # log de ciclos

        # ── Configuración inicial de ejes ─────────────────────────────────────
        for ax in (ax_xy, ax_z, ax_log):
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT_C, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(GRID_C)

        ax_xy.set_xlabel('X [m]', fontsize=8)
        ax_xy.set_ylabel('Y [m]', fontsize=8)
        ax_xy.set_aspect('equal')
        ax_xy.grid(True, color=GRID_C, linewidth=0.6)

        ax_z.set_xlabel('Tiempo [s]', fontsize=8)
        ax_z.set_ylabel('Z [m]', fontsize=8)
        ax_z.grid(True, color=GRID_C, linewidth=0.6)

        ax_log.axis('off')

        def update(_frame):
            if self.state == State.LAND and len(self.traj_x) > 0:
                # Dibujar el estado final y cerrar en 3 s
                plt.pause(3.0)
                plt.close(fig)
                return

            # Leer trayectoria
            with self._traj_lock:
                tx = list(self.traj_x)
                ty = list(self.traj_y)
                tz = list(self.traj_z)

            t_vec = [i * DT for i in range(len(tx))]

            # ── Planta X–Y ────────────────────────────────────────────────────
            ax_xy.cla()
            ax_xy.set_facecolor(BG)
            ax_xy.set_xlabel('X [m]', fontsize=8)
            ax_xy.set_ylabel('Y [m]', fontsize=8)
            ax_xy.set_aspect('equal')
            ax_xy.grid(True, color=GRID_C, linewidth=0.6)
            ax_xy.set_title(
                f'Planta X–Y  ·  Estado: {self.state}',
                fontsize=9, color=ACCENT, pad=6, fontfamily='monospace'
            )

            # Trayectoria planeada (punteada naranja)
            if self.trajectory is not None:
                wp = self.trajectory.waypoints
                ax_xy.plot(wp[:, 0], wp[:, 1],
                           color=PLAN_C, linewidth=1.0,
                           linestyle='--', alpha=0.6, label='Tray. planeada')
                # Sub-waypoints visitados (más opacos)
                idx_now = self.trajectory.idx
                if idx_now > 0:
                    ax_xy.scatter(
                        wp[:idx_now, 0], wp[:idx_now, 1],
                        c=PLAN_C, s=12, alpha=0.35, zorder=3
                    )
                # Sub-waypoint actual (diamante naranja brillante)
                if not self.trajectory.is_finished():
                    pt_c = self.trajectory.current_target()
                    ax_xy.scatter(pt_c[0], pt_c[1],
                                  c=PLAN_C, s=80, marker='D',
                                  zorder=7, label=f'Punto actual [{idx_now}]')

            # Posición inicial
            if self.q_initial is not None:
                ax_xy.scatter(self.q_initial[0], self.q_initial[1],
                              c=INIT_C, s=90, marker='o', zorder=6, label='Inicio')
                ax_xy.annotate(
                    f'  I ({self.q_initial[0]:.2f}, {self.q_initial[1]:.2f})',
                    (self.q_initial[0], self.q_initial[1]),
                    color=INIT_C, fontsize=7
                )

            # Target
            if self.q_d is not None:
                # Círculo de tolerancia
                circle = plt.Circle(
                    (self.q_d[0], self.q_d[1]), TOL_M,
                    color=TARGET, fill=False, linewidth=0.8,
                    linestyle=':', alpha=0.5
                )
                ax_xy.add_patch(circle)
                ax_xy.scatter(self.q_d[0], self.q_d[1],
                              c=TARGET, s=140, marker='*', zorder=7, label='Target')
                ax_xy.annotate(
                    f'  T ({self.q_d[0]:.2f}, {self.q_d[1]:.2f})',
                    (self.q_d[0], self.q_d[1]),
                    color=TARGET, fontsize=7
                )

            # Trayectoria ejecutada
            if len(tx) >= 2:
                ax_xy.plot(tx, ty, color=TRAJ, linewidth=1.8,
                           alpha=0.9, label='Ejecutada')

            # Posición actual + flecha hacia target
            if len(tx) >= 1 and self.q_d is not None:
                ax_xy.scatter(tx[-1], ty[-1],
                              c=ACCENT, s=80, marker='^', zorder=8, label='Dron')
                ax_xy.annotate(
                    f'  ({tx[-1]:.2f}, {ty[-1]:.2f})',
                    (tx[-1], ty[-1]),
                    color=ACCENT, fontsize=7
                )
                # Flecha de dirección al target
                dx = self.q_d[0] - tx[-1]
                dy = self.q_d[1] - ty[-1]
                dn = max(np.hypot(dx, dy), 1e-6)
                scale = min(dn, 0.20)
                ax_xy.annotate(
                    '', xy=(tx[-1] + dx/dn*scale, ty[-1] + dy/dn*scale),
                    xytext=(tx[-1], ty[-1]),
                    arrowprops=dict(arrowstyle='->', color=ACCENT,
                                   lw=1.5, mutation_scale=12)
                )

            # Autoescala con margen
            pts_x = ([self.q_d[0]] if self.q_d is not None else []) + tx + (
                [self.q_initial[0]] if self.q_initial is not None else [])
            pts_y = ([self.q_d[1]] if self.q_d is not None else []) + ty + (
                [self.q_initial[1]] if self.q_initial is not None else [])
            if pts_x and pts_y:
                m = 0.35
                ax_xy.set_xlim(min(pts_x) - m, max(pts_x) + m)
                ax_xy.set_ylim(min(pts_y) - m, max(pts_y) + m)

            leg = ax_xy.legend(loc='upper left', fontsize=7, framealpha=0.6)
            for t in leg.get_texts():
                t.set_color(TEXT_C)

            # ── Altura Z(t) ───────────────────────────────────────────────────
            ax_z.cla()
            ax_z.set_facecolor(BG)
            ax_z.set_xlabel('Tiempo [s]', fontsize=8)
            ax_z.set_ylabel('Z [m]', fontsize=8)
            ax_z.grid(True, color=GRID_C, linewidth=0.6)
            ax_z.set_title('Altura Z vs tiempo', fontsize=9,
                           color=ACCENT, pad=6, fontfamily='monospace')

            if len(tz) >= 2:
                ax_z.plot(t_vec, tz, color='#cc88ff', linewidth=1.8,
                          alpha=0.9, label='Z actual')
            if len(tz) >= 1:
                ax_z.scatter(t_vec[-1], tz[-1],
                             c='#cc88ff', s=60, marker='^', zorder=6)
                ax_z.annotate(
                    f'  Z={tz[-1]:.2f}',
                    (t_vec[-1], tz[-1]),
                    color='#cc88ff', fontsize=7
                )
            if self.q_d is not None:
                ax_z.axhline(self.q_d[2], color=TARGET, linestyle='--',
                             linewidth=1.0, label=f'Target Z={self.q_d[2]:.2f}')
            if tz:
                m = 0.25
                ax_z.set_ylim(min(tz) - m, max(tz) + m)

            leg2 = ax_z.legend(loc='upper left', fontsize=7, framealpha=0.6)
            for t in leg2.get_texts():
                t.set_color(TEXT_C)

            # ── Log de ciclos ─────────────────────────────────────────────────
            ax_log.cla()
            ax_log.axis('off')
            ax_log.set_facecolor(LOG_BG)

            # Barra de progreso de la trayectoria
            prog = self.trajectory.progress() if self.trajectory else 0.0
            n_filled = int(prog * 40)
            bar = '█' * n_filled + '░' * (40 - n_filled)
            prog_label = (
                f'TRAYECTORIA  [{bar}]  {prog*100:.0f}%  '
                f'punto {self.trajectory.idx if self.trajectory else 0}'
                f'/{PLAN_STEPS-1}'
            ) if self.trajectory else 'Esperando trayectoria...'

            ax_log.set_title(
                f'LOG DE SEGUIMIENTO  ·  {prog_label}',
                fontsize=7.5, color=ACCENT, pad=4,
                loc='left', fontfamily='monospace'
            )

            # Punto actual de la trayectoria (sub-waypoint)
            pt_now = (
                self.trajectory.current_target()
                if self.trajectory and not self.trajectory.is_finished()
                else (self.q_d if self.q_d is not None else None)
            )

            with self._cycle_lock:
                logs = list(self._cycle_log)

            if not logs:
                ax_log.text(
                    0.02, 0.5,
                    'Esperando primer ciclo de vuelo...',
                    transform=ax_log.transAxes,
                    fontsize=9, color=TEXT_C, fontfamily='monospace',
                    va='center'
                )
            else:
                for idx, entry in enumerate(reversed(logs)):
                    y_pos = 0.88 - idx * 0.17
                    q   = entry['q']
                    q_d = entry['q_d']   # sub-waypoint del ciclo
                    e   = entry['e']
                    d   = entry['dist']
                    rc  = entry['rc']
                    le  = entry['lat_err']

                    alpha = 1.0 if idx == 0 else max(0.35, 0.85 - idx * 0.15)
                    col   = ACCENT if idx == 0 else TEXT_C

                    line = (
                        f'▶  Estoy: ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})  '
                        f'│  Punto tray: ({q_d[0]:.3f}, {q_d[1]:.3f}, {q_d[2]:.3f})  '
                        f'│  dist_punto: {d:.3f} m  '
                        f'│  err_lat: {le:.3f} m  '
                        f'│  RC: (lr={rc[0]:+3d}, fb={rc[1]:+3d}, ud={rc[2]:+3d})'
                    )
                    ax_log.text(
                        0.01, y_pos, line,
                        transform=ax_log.transAxes,
                        fontsize=7.5, color=col,
                        fontfamily='monospace', alpha=alpha,
                        va='top'
                    )

            plt.draw()

        ani = animation.FuncAnimation(
            fig, update, interval=250, cache_frame_data=False
        )

        plt.show()

        if self.state not in (State.LAND,):
            self.get_logger().warn('Ventana cerrada — aterrizaje de emergencia')
            self.state = State.LAND

    # =========================================================================
    # APAGADO
    # =========================================================================
    def _shutdown_drone(self):
        self.get_logger().info('Iniciando aterrizaje...')
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.4)
            self.drone.land()
            self.drone.end()
        except Exception as e:
            self.get_logger().warn(f'Error en aterrizaje: {e}')
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
# MAIN — orden de arranque estricto
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # 1. Crear nodo (se suscribe a OptiTrack pero NO conecta dron todavía)
        node = TelloControllerNode()

        # 2. Esperar señal válida de OptiTrack antes de cualquier otra cosa
        #    (rclpy.spin_once en bucle — bloquea hasta tener señal)
        node.wait_for_optitrack()

        # 3. Conectar dron y despegar
        #    (también arranca el timer de control y el hilo de ROS2 spin)
        node.connect_and_takeoff()

        # 4. Abrir gráfica (bloquea el hilo principal con plt.show())
        #    El control y el spin corren en hilos secundarios.
        node.run_plot()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error fatal: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()