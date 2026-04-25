import numpy as np
import time
from djitellopy import Tello

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2  # Aumentado ligeramente para mayor estabilidad física

ALPHA = 0.6  # Un poco más bajo para que la odometría reaccione más rápido

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
    # Usamos round para evitar que valores menores a 1 se conviertan en 0 bruscamente
    lr = int(np.clip(np.round(u[1]), -RC_LIMIT, RC_LIMIT))
    fb = int(np.clip(np.round(u[0]), -RC_LIMIT, RC_LIMIT))
    ud = int(np.clip(np.round(u[2]), -RC_LIMIT, RC_LIMIT))
    return lr, fb, ud, 0

# -----------------------------
# ODOMETRÍA
# -----------------------------
def odometry(q, v_est, rc_cmd, dt):
    lr, fb, ud, _ = rc_cmd

    # 1. "medición" aproximada del movimiento
    v_meas = np.array([
        fb / RC_LIMIT * MAX_SPEED_CM_S,
        lr / RC_LIMIT * MAX_SPEED_CM_S,
        ud / RC_LIMIT * MAX_SPEED_CM_S
    ])

    # 2. filtro tipo IMU / sensor suave
    v_est = ALPHA * v_est + (1 - ALPHA) * v_meas

    # 3. integración física
    q_next = q + v_est * dt

    return q_next, v_est

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

    # Posición inicial estimada y Objetivo
    q = np.array([0.0, 0.0, 110.0])
    q_d = np.array([0.0, 0.0, 150.0])

    # K un poco más alto para asegurar que venza la inercia al final
    K = np.diag([1.2, 1.2, 1.2])

    v_est = np.array([0.0, 0.0, 0.0]) 

    try:
        for i in range(200):
            # CONTROL
            u, e = control(q, q_d, K)
            dist = np.linalg.norm(e)

            # Conversión a comandos RC
            rc = velocity_to_rc(u)
            
            print(
                f"q_est [cm]: x={q[0]:.1f}, y={q[1]:.1f}, z={q[2]:.1f} | "
                f"error={dist:.2f} | rc={rc[:3]}"
            )

            # --- CONDICIÓN DE PARADA MEJORADA ---
            # Si el error es menor a la tolerancia O si el comando RC ya es 0 (punto muerto)
            if dist < TOL_CM:
                print("¡Objetivo alcanzado por precisión!")
                break
            
            if i > 10 and rc[0] == 0 and rc[1] == 0 and rc[2] == 0:
                print("¡Objetivo alcanzado por comando mínimo (estancamiento)! Finalizando...")
                break

            # Enviar comandos y actualizar odometría
            drone.send_rc_control(*rc)
            q, v_est = odometry(q, v_est, rc, DT)

            time.sleep(DT)

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