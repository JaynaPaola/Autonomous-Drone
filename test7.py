import numpy as np
import time
import cv2
from djitellopy import Tello

# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 1 / 15
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
TOL_CM = 2

# -----------------------------
# CONTROL (INTACTO)
# -----------------------------
def control(q, q_d, K, q_dot_d=None):
    if q_dot_d is None:
        q_dot_d = np.zeros(3)

    e = q - q_d
    u = -K @ e + q_dot_d
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
# ODOMETRÍA (INTACTA)
# -----------------------------
def odometry(q, rc_cmd, dt):
    lr, fb, ud, _ = rc_cmd

    v = np.array([
        fb / RC_LIMIT * MAX_SPEED_CM_S,
        lr / RC_LIMIT * MAX_SPEED_CM_S,
        ud / RC_LIMIT * MAX_SPEED_CM_S
    ])

    return q + v * dt

# -----------------------------
# FOTO
# -----------------------------
def take_photo(frame, idx):
    filename = f"wp_{idx}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Foto guardada: {filename}")

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

    # 🔴 IMPORTANTE: activar cámara UNA SOLA VEZ
    drone.streamon()
    frame_read = drone.get_frame_read()

    q = np.array([0.0, 0.0, 0.0])

    q_dot_d = np.array([0.0, 0.0, 0.0])
    K = np.diag([1.2, 1.2, 1.2])

    dx = 80.0
    dy = 80.0
    dz = 400.0

    waypoints = [
        #np.array([0.0, 0.0, 450.0]),
        #np.array([dx, 0.0, 450.0]),
        #np.array([2 * dx, 0.0, 450.0]),
        #np.array([2 * dx, -dy, 450.0]),
        #np.array([2 * dx, -2 * dy, 450.0]),
        #np.array([dx, -2 * dy, 450.0]),
        #np.array([0.0, -2 * dy, 450.0]),
        #np.array([0.0, -dy, 450.0]),
    
        np.array([0.0,      0.0,      dz]),   # 1
        np.array([0.0,     -dy,       dz]),   # 6
        np.array([0.0,     -2*dy,     dz]),   # 7

        np.array([dx,      -2*dy,     dz]),   # 8
        np.array([dx,      -dy,       dz]),   # 5
        np.array([dx,       0.0,      dz]),   # 2
        
        np.array([2*dx,     0.0,      dz]),   # 3
        np.array([2*dx,    -dy,       dz]),   # 4
        np.array([2*dx,    -2*dy,     dz]),   # 9
    ]

    try:

        for wp_idx, q_d in enumerate(waypoints):

            print("\n===================================")
            print(f"WAYPOINT {wp_idx + 1}")
            print("===================================")

            for i in range(200):

                u, e = control(q, q_d, K, q_dot_d)
                dist = np.linalg.norm(e)

                rc = velocity_to_rc(u)

                print(
                    f"[{i:03d}] q={q} | q_d={q_d} | |e|={dist:.2f}"
                )

                if dist < TOL_CM:

                    print("¡Waypoint alcanzado!")

                    drone.send_rc_control(0, 0, 0, 0)

                    q = q_d.copy()

                    # 📸 FOTO INMEDIATA (FRAME ACTUAL)
                    frame = frame_read.frame
                    take_photo(frame, wp_idx + 1)

                    break

                drone.send_rc_control(*rc)
                q = odometry(q, rc, DT)

                time.sleep(DT)

            print("Esperando 3 segundos...")
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")

    finally:

        print("Aterrizando...")

        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)

        drone.land()

        # 🔴 cerrar cámara correctamente
        drone.streamoff()
        drone.end()


if __name__ == "__main__":
    run_tello()