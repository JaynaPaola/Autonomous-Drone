import numpy as np
import time
from djitellopy import Tello

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2

ALPHA = 0.7  # filtro (suavizado de odometría)

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
    lr = int(np.clip(u[1], -RC_LIMIT, RC_LIMIT))
    fb = int(np.clip(u[0], -RC_LIMIT, RC_LIMIT))
    ud = int(np.clip(u[2], -RC_LIMIT, RC_LIMIT))
    return lr, fb, ud, 0

# -----------------------------
# ODOMETRÍA
# -----------------------------
def odometry(q, v_est, rc_cmd, dt):

    lr, fb, ud, _ = rc_cmd

    # 1. "medición" aproximada del movimiento (no perfecta)
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
    drone.connect()
    print("Batería:", drone.get_battery(), "%")

    drone.takeoff()
    time.sleep(2)

    q = np.array([0.0, 0.0, 110.0])
    q_d = np.array([12.0, 12.0, 98.0])

    K = np.diag([0.8, 0.8, 0.8])

    v_est = np.array([0.0, 0.0, 0.0])  # odometría inicial

    try:
        for _ in range(200):

            # CONTROL
            u, e = control(q, q_d, K)

            rc = velocity_to_rc(u)
            drone.send_rc_control(*rc)

            # ODOMETRÍA MEJORADA
            q, v_est = odometry(q, v_est, rc, DT)

            dist = np.linalg.norm(e)

            print(
                f"q_est [cm]: x={q[0]:.1f}, y={q[1]:.1f}, z={q[2]:.1f} | "
                f"error={dist:.2f}"
            )

            if dist < TOL_CM:
                print("Objetivo alcanzado")
                break

            time.sleep(DT)

    finally:
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        drone.land()

if __name__ == "__main__":
    run_tello()