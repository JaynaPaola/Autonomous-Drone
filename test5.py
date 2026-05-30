import numpy as np
import time
from djitellopy import Tello

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 1 / 15          # 15 Hz
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2

# -----------------------------
# CONTROL
# Diagrama: e = q - q_d
#           u = -K·e + q_dot_d
# Para punto fijo: q_dot_d = 0  →  u = -K·e
# -----------------------------
def control(q, q_d, K, q_dot_d=None):
    if q_dot_d is None:
        q_dot_d = np.zeros(3)

    e = q - q_d               # error como en el diagrama: e = q - q_d
    u = -K @ e + q_dot_d      # ley de control del diagrama: u = -K·e + q_dot_d
    return u, e

# -----------------------------
# RC
# El control u representa velocidades deseadas
# Se mapean a comandos RC del dron
# -----------------------------
def velocity_to_rc(u):
    lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
    fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
    ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
    return lr, fb, ud, 0

# -----------------------------
# ODOMETRÍA
# Sin OptiTrack: se estima q integrando la velocidad comandada
# Cuando OptiTrack esté disponible, q vendrá directamente de él
# y esta función no será necesaria
# -----------------------------
def odometry(q, rc_cmd, dt):
    lr, fb, ud, _ = rc_cmd

    # Velocidad directa del comando RC actual (sin memoria del pasado)
    v = np.array([
        fb / RC_LIMIT * MAX_SPEED_CM_S,
        lr / RC_LIMIT * MAX_SPEED_CM_S,
        ud / RC_LIMIT * MAX_SPEED_CM_S
    ])

    # Integración: q_dot = u  →  q_next = q + v·dt
    return q + v * dt

# -----------------------------
# MAIN
# -----------------------------
def run_tello():
    drone = Tello()
    try:
        drone.connect()
    except Exception as e:
        print(f"Error de conexión: {e}")
        return

    print("Batería:", drone.get_battery(), "%")
    drone.takeoff()
    time.sleep(2)

    # Estado inicial estimado
    q = np.array([0.0, 0.0, 0.0])    # posición actual (sin OptiTrack: odometría)

    # ---------------------------------------------------
    # MODO PUNTO FIJO: q_d constante, q_dot_d = 0
    # MODO TRAYECTORIA: actualizar q_d y q_dot_d en el loop
    # ---------------------------------------------------
    q_d     = np.array([-100.0, 0.0, 0.0])   # posición deseada
    q_dot_d = np.array([0.0,  0.0,  0.0])   # velocidad deseada (punto fijo = 0)

    K = np.diag([1.2, 1.2, 1.2])    # ganancia K > 0  (garantiza estabilidad)

    try:
        for i in range(200):
            # --- LEY DE CONTROL DEL DIAGRAMA ---
            u, e = control(q, q_d, K, q_dot_d)
            dist = np.linalg.norm(e)

            # Conversión: u (velocidad) → comandos RC del dron
            rc = velocity_to_rc(u)

            print(
                f"[{i:03d}] q=[{q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f}] | "
                f"e=[{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] | "
                f"|e|={dist:.2f} cm | rc={rc[:3]}"
            )

            # Condición de parada: error dentro de tolerancia
            if dist < TOL_CM:
                print("¡Punto fijo alcanzado!")
                break

            # Punto muerto por comando mínimo
            if i > 10 and rc[0] == 0 and rc[1] == 0 and rc[2] == 0:
                print("Comando nulo — objetivo alcanzado por saturación mínima.")
                break

            # Enviar velocidad al dron y actualizar estimación de posición
            drone.send_rc_control(*rc)
            q = odometry(q, rc, DT)

            time.sleep(DT)   # 15 Hz

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")

    finally:
        print("Aterrizando...")
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        drone.land()
        drone.end()


if __name__ == "__main__":
    run_tello()