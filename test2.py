import numpy as np
import time
from djitellopy import Tello

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2  # tolerancia

ALPHA = 0.6

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
# ODOMETRÍA
# -----------------------------
def odometry(q, v_est, rc_cmd, dt):
    lr, fb, ud, _ = rc_cmd

    v_meas = np.array([
        fb / RC_LIMIT * MAX_SPEED_CM_S,
        lr / RC_LIMIT * MAX_SPEED_CM_S,
        ud / RC_LIMIT * MAX_SPEED_CM_S
    ])

    v_est = ALPHA * v_est + (1 - ALPHA) * v_meas
    q_next = q + v_est * dt

    return q_next, v_est

# -----------------------------
# MAIN
# -----------------------------
def run_tello():
    drone = Tello()

    drone.connect()
    print("Batería:", drone.get_battery(), "%")

    drone.takeoff()
    time.sleep(2)

    # -----------------------------
    # ESTADO INICIAL
    # -----------------------------
    q = np.array([0.0, 0.0, 110.0])

    K = np.diag([1.2, 1.2, 1.2])
    v_est = np.array([0.0, 0.0, 0.0])

    # -----------------------------
    # TRAYECTORIA X → Y → Z
    # -----------------------------
    q_d_list = [
        np.array([50.0, 0.0, 110.0]),   # X
        np.array([50.0, 50.0, 110.0]),  # Y
        np.array([50.0, 50.0, 150.0])   # Z
    ]

    step = 0
    q_d = q_d_list[step]

    try:
        for i in range(300):

            u, e = control(q, q_d, K)
            dist = np.linalg.norm(e)

            rc = velocity_to_rc(u)

            print(
                f"q: {q} | q_d: {q_d} | error={dist:.2f} | rc={rc[:3]}"
            )

            # -----------------------------
            # CAMBIO DE FASE X → Y → Z
            # -----------------------------
            if dist < TOL_CM:
                step += 1

                if step >= len(q_d_list):
                    print("Trayectoria completa")
                    break

                q_d = q_d_list[step]
                print(f"\n➡ Cambiando a objetivo {step}: {q_d}\n")
                time.sleep(1)
                continue

            drone.send_rc_control(*rc)
            q, v_est = odometry(q, v_est, rc, DT)

            time.sleep(DT)

    finally:
        print("Aterrizando...")
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        drone.land()
        drone.end()


if __name__ == "__main__":
    run_tello()