# -*- coding: utf-8 -*-
import sys
import io
# Forzar UTF-8 en consolas Windows (CP1252 no soporta todos los caracteres)
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from djitellopy import Tello

# -----------------------------
# PARÁMETROS
# -----------------------------
STEP_CM      = 20     # Distancia entre sub-waypoints intermedios (cm)
SPEED        = 30     # Velocidad de vuelo en go_xyz_speed (cm/s), rango: 10–100
WAIT_MAIN_WP = 1.5   # Pausa en segundos al llegar a un WP principal
WAIT_SUB_WP  = 0.0   # Pausa entre sub-waypoints (0 = fluido)

# -----------------------------
# ESTADO COMPARTIDO DE TRAYECTORIA (hilo de vuelo ↔ hilo de gráfica)
# -----------------------------
traj_lock = threading.Lock()
traj_x    = [0.0]
traj_y    = [0.0]
traj_z    = [0.0]
flight_done = False   # señal para cerrar la animación

# -----------------------------
# WAYPOINTS PRINCIPALES
# Cuadrado de 100 cm de lado en el plano XY (z constante = 0)
# Coordenadas relativas a la posición de despegue.
#   x → adelante/atrás  (forward)
#   y → izquierda/derecha (left)
#   z → arriba/abajo    (up)
# -----------------------------
MAIN_WAYPOINTS = [
    np.array([  0,   0,  0], dtype=float),   # WP1 – origen (despegue)
    np.array([100,   0,  0], dtype=float),   # WP2 – avanza 100 cm al frente
    np.array([100, 100,  0], dtype=float),   # WP3 – 100 cm a la derecha
    np.array([  0, 100,  0], dtype=float),   # WP4 – regresa 100 cm atrás
    np.array([  0,   0,  0], dtype=float),   # WP1 – cierra el cuadrado
]

# -----------------------------
# PATH PLANNING
# Interpola sub-waypoints entre dos puntos a pasos de STEP_CM cm
# -----------------------------
def interpolate_segment(p_start: np.ndarray, p_end: np.ndarray, step_cm: float) -> list:
    """
    Genera una lista de puntos equiespaciados (cada step_cm cm) entre
    p_start y p_end, incluyendo p_end pero NO p_start (ya se visitó).
    Si la distancia es menor que step_cm, devuelve solo [p_end].
    """
    direction = p_end - p_start
    total_dist = np.linalg.norm(direction)

    if total_dist < 1e-6:
        return []                        # Mismo punto, nada que hacer

    n_steps = int(np.floor(total_dist / step_cm))
    unit    = direction / total_dist

    sub_wps = []
    for i in range(1, n_steps + 1):
        sub_wps.append(p_start + unit * step_cm * i)

    # Asegurar que el punto final exacto siempre se incluye
    if n_steps == 0 or np.linalg.norm(sub_wps[-1] - p_end) > 1e-3:
        sub_wps.append(p_end.copy())

    return sub_wps


def build_full_path(main_waypoints: list, step_cm: float) -> list:
    """
    Construye el path completo interpolando entre cada par de WPs principales.
    Devuelve lista de (punto, es_waypoint_principal).
    """
    full_path = [(main_waypoints[0].copy(), True)]   # Punto de partida

    for i in range(len(main_waypoints) - 1):
        p_start = main_waypoints[i]
        p_end   = main_waypoints[i + 1]
        segment = interpolate_segment(p_start, p_end, step_cm)

        is_last_in_segment = [False] * len(segment)
        if segment:
            is_last_in_segment[-1] = True   # El último punto del segmento es WP principal

        for j, pt in enumerate(segment):
            full_path.append((pt, is_last_in_segment[j]))

    return full_path


# -----------------------------
# REGISTRO DE POSICIÓN (llamado desde el hilo de vuelo)
# -----------------------------
def record_position(pos: np.ndarray):
    global traj_x, traj_y, traj_z
    with traj_lock:
        traj_x.append(float(pos[0]))
        traj_y.append(float(pos[1]))
        traj_z.append(float(pos[2]))


# -----------------------------
# GRÁFICA EN TIEMPO REAL
# 3 paneles 2D: XY (vista superior), XZ (lateral), YZ (frontal)
# -----------------------------
def launch_realtime_plot(main_waypoints: list, full_path: list):
    """
    Lanza la ventana de matplotlib con animación en tiempo real.
    Se ejecuta en el hilo principal (matplotlib lo requiere en muchos OS).
    """
    # Path completo como referencia visual (línea gris punteada)
    ref_x = [p[0] for p, _ in full_path]
    ref_y = [p[1] for p, _ in full_path]
    ref_z = [p[2] for p, _ in full_path]

    # WPs principales para marcarlos
    mwp_x = [p[0] for p in main_waypoints]
    mwp_y = [p[1] for p in main_waypoints]
    mwp_z = [p[2] for p in main_waypoints]

    fig = plt.figure(figsize=(13, 4.5), facecolor="#0f0f0f")
    fig.suptitle("Trayectoria Tello – Tiempo Real", color="white",
                 fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    panels = []
    configs = [
        ("Vista Superior  X-Y", ref_x, ref_y, mwp_x, mwp_y, "X (cm)", "Y (cm)"),
        ("Vista Lateral   X-Z", ref_x, ref_z, mwp_x, mwp_z, "X (cm)", "Z (cm)"),
        ("Vista Frontal   Y-Z", ref_y, ref_z, mwp_y, mwp_z, "Y (cm)", "Z (cm)"),
    ]

    for i, (title, rx, ry, mx, my, xlabel, ylabel) in enumerate(configs):
        ax = fig.add_subplot(gs[i], facecolor="#1a1a2e")
        ax.set_title(title, color="#a0c4ff", fontsize=9, pad=6)
        ax.set_xlabel(xlabel, color="#888", fontsize=8)
        ax.set_ylabel(ylabel, color="#888", fontsize=8)
        ax.tick_params(colors="#555", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        # Path de referencia
        ax.plot(rx, ry, color="#2a2a4a", linewidth=1.5,
                linestyle="--", zorder=1, label="path planificado")

        # WPs principales
        ax.scatter(mx, my, color="#f72585", s=60, zorder=5,
                   label="WP principal", marker="D")
        for j, (wx, wy) in enumerate(zip(mx, my)):
            ax.annotate(f"WP{j+1}", (wx, wy), textcoords="offset points",
                        xytext=(5, 5), fontsize=6, color="#f72585")

        # Línea de trayectoria ejecutada (se actualiza)
        line, = ax.plot([], [], color="#00f5d4", linewidth=1.8,
                        zorder=3, label="recorrido real")
        # Punto actual del dron
        dot, = ax.plot([], [], "o", color="#ffe600", markersize=7,
                       zorder=6, label="dron")

        ax.legend(fontsize=6, facecolor="#111", edgecolor="#333",
                  labelcolor="white", loc="upper left")
        panels.append((ax, line, dot, i))

    def update(_):
        with traj_lock:
            xs = list(traj_x)
            ys = list(traj_y)
            zs = list(traj_z)

        data_pairs = [(xs, ys), (xs, zs), (ys, zs)]

        for ax, line, dot, i in panels:
            px, py = data_pairs[i]
            line.set_data(px, py)
            if px and py:
                dot.set_data([px[-1]], [py[-1]])
            ax.relim()
            ax.autoscale_view()

        if flight_done:
            ani.event_source.stop()

        return [item for _, line, dot, _ in panels for item in (line, dot)]

    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


# -----------------------------
# CONVERSIÓN A COORDENADAS RELATIVAS
# go_xyz_speed recibe desplazamiento relativo a la posición actual del dron
# -----------------------------
def absolute_to_relative(current: np.ndarray, target: np.ndarray):
    """
    Convierte coordenadas absolutas a desplazamiento relativo.
    go_xyz_speed(x, y, z, speed):
        x → adelante (+) / atrás  (-)
        y → izquierda (+) / derecha (-)   ← OJO: ejes SDK Tello
        z → arriba (+) / abajo (-)
    """
    delta = target - current
    x = int(np.round(delta[0]))   # forward
    y = int(np.round(delta[1]))   # left
    z = int(np.round(delta[2]))   # up
    return x, y, z


# -----------------------------
# MAIN
# -----------------------------
def run_tello():
    global flight_done

    drone = Tello()
    try:
        drone.connect()
    except Exception as e:
        print(f"Error de conexión: {e}")
        flight_done = True
        return

    print(f"Batería: {drone.get_battery()} %")

    # --- Construir path completo antes de despegar ---
    full_path = build_full_path(MAIN_WAYPOINTS, STEP_CM)

    print(f"\n{'='*55}")
    print(f"  PATH PLANNING - Cuadrado 100x100 cm")
    print(f"  WPs principales : {len(MAIN_WAYPOINTS)}")
    print(f"  Sub-waypoints   : {len(full_path) - len(MAIN_WAYPOINTS)}")
    print(f"  Puntos totales  : {len(full_path)}")
    print(f"  Paso de interp. : {STEP_CM} cm")
    print(f"{'='*55}\n")

    for idx, (pt, is_main) in enumerate(full_path):
        label = "[WP PRINCIPAL]" if is_main else "[ sub-wp    ]"
        print(f"  [{idx:3d}] {label}  x={pt[0]:6.1f}  y={pt[1]:6.1f}  z={pt[2]:6.1f}")

    print()

    drone.takeoff()
    time.sleep(2)

    current_pos = MAIN_WAYPOINTS[0].copy()
    record_position(current_pos)   # registra origen

    try:
        for idx, (target_pos, is_main) in enumerate(full_path[1:], start=1):
            x, y, z = absolute_to_relative(current_pos, target_pos)
            label = "[WP PRINCIPAL]" if is_main else "[ sub-wp    ]"

            print(
                f"[{idx:3d}/{len(full_path)-1}] {label:15s} | "
                f"pos=({current_pos[0]:.0f},{current_pos[1]:.0f},{current_pos[2]:.0f}) → "
                f"({target_pos[0]:.0f},{target_pos[1]:.0f},{target_pos[2]:.0f}) | "
                f"Δ=({x:+d},{y:+d},{z:+d})"
            )

            if abs(x) < 20 and abs(y) < 20 and abs(z) < 20:
                print("        ↳ Desplazamiento menor a 20 cm, omitiendo comando.")
                current_pos = target_pos.copy()
                record_position(current_pos)
                continue

            drone.go_xyz_speed(x, y, z, SPEED)

            current_pos = target_pos.copy()
            record_position(current_pos)   # ← actualiza la gráfica

            if is_main:
                print(f"        ↳ WP principal alcanzado. Esperando {WAIT_MAIN_WP}s...")
                time.sleep(WAIT_MAIN_WP)
            elif WAIT_SUB_WP > 0:
                time.sleep(WAIT_SUB_WP)

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        print("\nAterrizando...")
        drone.land()
        drone.end()
        flight_done = True   # señal para cerrar la animación
        print("Fin del vuelo.")


# -----------------------------
# ENTRY POINT
# La gráfica corre en el hilo principal (requerido por matplotlib).
# El vuelo corre en un hilo secundario para no bloquear la animación.
# -----------------------------
if __name__ == "__main__":
    full_path_preview = build_full_path(MAIN_WAYPOINTS, STEP_CM)

    # Hilo de vuelo
    flight_thread = threading.Thread(target=run_tello, daemon=True)
    flight_thread.start()

    # Gráfica en hilo principal (bloqueante hasta cerrar ventana)
    launch_realtime_plot(MAIN_WAYPOINTS, full_path_preview)

    flight_thread.join()