#!/usr/bin/env python3
"""
Tello — Rutina cuadrado con verificación OptiTrack (ROS2)
==========================================================

El dron ejecuta los move_*() normalmente (lazo abierto del Tello),
pero entre cada segmento OptiTrack verifica si llegó a donde debía.
Si el error supera CORRECTION_THRESHOLD_M, se aplica una corrección
con RC antes de continuar al siguiente punto.

Flujo por segmento
------------------
  1. Calcular coordenada esperada tras el movimiento
  2. Ejecutar drone.move_*()          ← Tello maneja el movimiento
  3. Esperar SETTLE_S para que el dron se estabilice
  4. Leer self.q de OptiTrack
  5. Calcular error = q_esperada − self.q
  6. Si |error| > umbral → corrección RC
  7. Continuar al siguiente segmento

Requisitos
----------
  - ROS2 Humble en ejecución
  - Topic: optitrack/rigid_body (geometry_msgs/PoseStamped)
  - djitellopy instalado
  - Tello encendido y conectado por WiFi
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped

from djitellopy import Tello


# =============================================================================
# PARÁMETROS
# =============================================================================

DISTANCIA_M  = 0.5          # [m]  Distancia de cada segmento del cuadrado
DISTANCIA_CM = int(DISTANCIA_M * 100)

SETTLE_S     = 2.0          # [s]  Espera post-movimiento para que el dron
                             #      se estabilice antes de leer OptiTrack

TAKEOFF_WAIT_S = 3.0        # [s]  Espera post-despegue

CORRECTION_THRESHOLD_M = 0.15  # [m]  Si el error de posición supera este
                                #      valor, se aplica corrección RC

CORRECTION_GAIN  = 30       # [-]  Ganancia de la corrección RC.
                             #      RC = gain * error_m (saturado a ±50)
CORRECTION_TIME  = 0.4      # [s]  Duración de cada pulso de corrección

MAX_CORRECTIONS  = 3        # [-]  Intentos máximos de corrección por segmento

OPTITRACK_TIMEOUT_S = 3.0   # [s]  Timeout esperando señal de OptiTrack


# =============================================================================
# NODO OPTITRACK — solo escucha, no controla
# =============================================================================
class OptiTrackListener(Node):
    """
    Suscriptor mínimo a OptiTrack.
    Expone self.q (numpy array [X, Y, Z]) con la posición más reciente.
    Se crea ANTES de conectar el dron para asegurarse de tener señal.
    """

    def __init__(self):
        super().__init__('optitrack_listener')

        self.q     = None          # posición actual, None hasta el primer frame
        self._lock = threading.Lock()
        self._count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(PoseStamped, 'optitrack/rigid_body',
                                 self._cb, qos)
        self.get_logger().info('Esperando señal de OptiTrack...')

    def _cb(self, msg: PoseStamped):
        with self._lock:
            self.q = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])
            self._count += 1

    def get_position(self) -> np.ndarray | None:
        """Devuelve una copia segura de la posición actual."""
        with self._lock:
            return self.q.copy() if self.q is not None else None

    def wait_for_signal(self, min_frames: int = 10, timeout: float = 10.0) -> bool:
        """
        Bloquea hasta recibir min_frames frames de OptiTrack.
        Devuelve True si los recibió, False si agotó el timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._lock:
                if self._count >= min_frames:
                    self.get_logger().info(
                        f'OptiTrack OK — {self._count} frames. '
                        f'Pos: {self.q}'
                    )
                    return True
        return False


# =============================================================================
# CORRECTOR DE POSICIÓN
# =============================================================================
class PositionCorrector:
    """
    Aplica correcciones RC al Tello basándose en el error de posición
    reportado por OptiTrack.

    El mapeo de ejes es provisional — ajustar según observación real:
        error[0] (X OptiTrack) → adelante/atrás (fb)
        error[1] (Y OptiTrack) → arriba/abajo   (ud)
        error[2] (Z OptiTrack) → izquierda/der  (lr)
    """

    def __init__(self, drone: Tello, listener: OptiTrackListener):
        self.drone    = drone
        self.listener = listener

    def correct(self, q_expected: np.ndarray, label: str = '') -> float:
        """
        Lee la posición actual, calcula el error respecto a q_expected,
        y aplica correcciones RC hasta MAX_CORRECTIONS veces o hasta que
        el error esté dentro del umbral.

        Devuelve el error final en metros.
        """
        for attempt in range(MAX_CORRECTIONS):
            # Dar tiempo a OptiTrack para estabilizarse
            time.sleep(SETTLE_S)

            # Leer posición real
            q_now = self._wait_optitrack()
            if q_now is None:
                print(f'  [WARN] Sin señal OptiTrack — saltando corrección')
                return 0.0

            error = q_expected - q_now
            dist  = float(np.linalg.norm(error))

            print(
                f'  [{label}] Intento {attempt+1}/{MAX_CORRECTIONS} | '
                f'Esperada: ({q_expected[0]:.3f}, {q_expected[1]:.3f}, {q_expected[2]:.3f}) | '
                f'Real:     ({q_now[0]:.3f}, {q_now[1]:.3f}, {q_now[2]:.3f}) | '
                f'Error:    {dist:.3f} m'
            )

            if dist <= CORRECTION_THRESHOLD_M:
                print(f'  [{label}] Posición OK (error {dist:.3f} m < {CORRECTION_THRESHOLD_M} m)')
                return dist

            # Calcular RC proporcional al error (saturado a ±50)
            def rc(v): return int(np.clip(v * CORRECTION_GAIN, -50, 50))

            fb = rc(error[0])   # X → adelante/atrás
            ud = rc(error[1])   # Y → arriba/abajo
            lr = rc(error[2])   # Z → izquierda/derecha

            print(f'  [{label}] Corrigiendo → RC(fb={fb}, ud={ud}, lr={lr}) por {CORRECTION_TIME} s')
            self.drone.send_rc_control(lr, fb, ud, 0)
            time.sleep(CORRECTION_TIME)
            self.drone.send_rc_control(0, 0, 0, 0)

        # Leer error final tras todos los intentos
        q_final = self._wait_optitrack()
        if q_final is not None:
            final_err = float(np.linalg.norm(q_expected - q_final))
            print(f'  [{label}] Error final: {final_err:.3f} m')
            return final_err
        return 0.0

    def _wait_optitrack(self, timeout: float = OPTITRACK_TIMEOUT_S) -> np.ndarray | None:
        """Devuelve la posición más reciente, esperando hasta timeout segundos."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self.listener, timeout_sec=0.05)
            q = self.listener.get_position()
            if q is not None:
                return q
        return None


# =============================================================================
# RUTINA PRINCIPAL
# =============================================================================
def main():
    rclpy.init()

    # ── 1. Conectar a OptiTrack primero ──────────────────────────────────────
    listener = OptiTrackListener()
    print('Conectando a OptiTrack...')
    if not listener.wait_for_signal(min_frames=10, timeout=10.0):
        print('ERROR: No se recibió señal de OptiTrack. Abortando.')
        listener.destroy_node()
        rclpy.shutdown()
        return

    # ── 2. Conectar y preparar el dron ───────────────────────────────────────
    drone = Tello()
    drone.connect()
    print(f'Batería: {drone.get_battery()} %')
    drone.streamon()

    # ── 3. Despegar ──────────────────────────────────────────────────────────
    print('Despegando...')
    drone.takeoff()
    time.sleep(TAKEOFF_WAIT_S)

    # ── 4. Registrar posición inicial ────────────────────────────────────────
    corrector = PositionCorrector(drone, listener)

    # Spin varias veces para tener la posición estabilizada post-despegue
    for _ in range(20):
        rclpy.spin_once(listener, timeout_sec=0.05)
    q0 = listener.get_position()

    if q0 is None:
        print('ERROR: No se pudo leer posición inicial. Aterrizando.')
        drone.land()
        drone.streamoff()
        listener.destroy_node()
        rclpy.shutdown()
        return

    print(f'Posición inicial: X={q0[0]:.3f} Y={q0[1]:.3f} Z={q0[2]:.3f}')

    # ── 5. Definir los 8 puntos del cuadrado ─────────────────────────────────
    #
    # El cuadrado se define en coordenadas de OptiTrack relativas a q0.
    # El mapeo de ejes es provisional — ajustar según tu setup real.
    #
    # Convención usada aquí (ajustar si OptiTrack tiene otro sistema):
    #   move_right   →  +X OptiTrack
    #   move_left    →  -X OptiTrack
    #   move_forward →  +Z OptiTrack   (frente del dron)
    #   move_back    →  -Z OptiTrack
    #   Y OptiTrack  →  altura (no cambia en este vuelo horizontal)
    #
    # Punto 1 = q0 (posición de despegue)
    #
    #   1 --→-- 2 --→-- 3
    #   ↑               ↓
    #   8               4
    #   ↑               ↓
    #   7 --←-- 6 --←-- 5
    #
    D = DISTANCIA_M
    waypoints = [
        ('Punto 1→2', 'move_right',   q0 + np.array([ D,  0,  0])),
        ('Punto 2→3', 'move_right',   q0 + np.array([2*D, 0,  0])),
        ('Punto 3→4', 'move_back',    q0 + np.array([2*D, 0, -D])),
        ('Punto 4→5', 'move_back',    q0 + np.array([2*D, 0,-2*D])),
        ('Punto 5→6', 'move_left',    q0 + np.array([ D,  0,-2*D])),
        ('Punto 6→7', 'move_left',    q0 + np.array([ 0,  0,-2*D])),
        ('Punto 7→8', 'move_forward', q0 + np.array([ 0,  0, -D])),
        ('Punto 8→1', 'move_forward', q0 + np.array([ 0,  0,  0])),
    ]

    # Comandos de movimiento del Tello
    move_cmds = {
        'move_right':   lambda: drone.move_right(DISTANCIA_CM),
        'move_left':    lambda: drone.move_left(DISTANCIA_CM),
        'move_forward': lambda: drone.move_forward(DISTANCIA_CM),
        'move_back':    lambda: drone.move_back(DISTANCIA_CM),
    }

    # ── 6. Ejecutar rutina con verificación OptiTrack ─────────────────────────
    print('\n' + '=' * 55)
    print('INICIANDO RUTINA — cuadrado 8 puntos')
    print('=' * 55)

    errors = []

    for label, cmd_key, q_expected in waypoints:
        print(f'\n▶ {label}')

        # Ejecutar el movimiento del Tello (bloqueante hasta que termina)
        move_cmds[cmd_key]()

        # Verificar posición con OptiTrack y corregir si hay desvío
        err = corrector.correct(q_expected, label=label)
        errors.append((label, err))

    # ── 7. Resumen ────────────────────────────────────────────────────────────
    print('\n' + '=' * 55)
    print('RESUMEN DE ERRORES POR SEGMENTO')
    print('=' * 55)
    for lbl, err in errors:
        mark = '✓' if err <= CORRECTION_THRESHOLD_M else '!'
        print(f'  {mark} {lbl:20s}  error final: {err:.3f} m')

    # ── 8. Aterrizar ──────────────────────────────────────────────────────────
    print('\nAterrizando...')
    drone.land()
    drone.streamoff()
    print('Rutina finalizada.')

    listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()