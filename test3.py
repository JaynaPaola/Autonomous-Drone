"""
Tello — Cuadrado 9 puntos con ascenso inicial (SIN OptiTrack)
"""

import time
from djitellopy import Tello


# =============================================================================
# PARÁMETROS
# =============================================================================

DISTANCIA_M = 0.6
DISTANCIA_CM = int(DISTANCIA_M * 100)

ALTURA_INICIAL_M = 0.5
ALTURA_CM = int(ALTURA_INICIAL_M * 100)

TAKEOFF_WAIT_S = 3.0


# =============================================================================
# RUTINA PRINCIPAL
# =============================================================================
def main():

    drone = Tello()
    drone.connect()

    print(f'Batería: {drone.get_battery()} %')

    # ── DESPEGUE ────────────────────────────────────────────────────────────
    print('Despegando...')
    drone.takeoff()

    # ⚠️ FIX CRÍTICO: estabilización IMU
    time.sleep(5)

    # Warm-up de sensores (evita "No valid imu")
    for _ in range(10):
        _ = drone.get_battery()
        _ = drone.get_height()
        time.sleep(0.3)

    # ── ASCENSO INICIAL ─────────────────────────────────────────────────────
    ALTURA_TOTAL_OBJETIVO_M = 2.5
    ALTURA_CM = int((ALTURA_TOTAL_OBJETIVO_M - 0.8) * 100)

    print(f'Subiendo a altura objetivo (~{ALTURA_TOTAL_OBJETIVO_M} m)...')
    drone.move_up(ALTURA_CM)

    # =========================================================
    # CUADRADO 8 MOVIMIENTOS (9 PUNTOS INCLUYENDO INICIO)
    #
    #   1 --→-- 2 --→-- 3
    #   ↑               ↓
    #   8               4
    #   ↑               ↓
    #   7 --←-- 6 --←-- 5
    # =========================================================

    print('Iniciando trayectoria en cuadrado...')

    # 1 → 2 → 3
    drone.move_right(DISTANCIA_CM)
    drone.move_right(DISTANCIA_CM)

    # 3 → 4 → 5
    drone.move_back(DISTANCIA_CM)
    drone.move_back(DISTANCIA_CM)

    # 5 → 6 → 7
    drone.move_left(DISTANCIA_CM)
    drone.move_left(DISTANCIA_CM)

    # 7 → 8 → 1 (cierre del cuadrado)
    drone.move_forward(DISTANCIA_CM)
    drone.move_forward(DISTANCIA_CM)

    # ── ATERRIZAJE ─────────────────────────────────────────────────────────
    print('Aterrizando...')
    drone.land()

    print('Listo.')


if __name__ == '__main__':
    main()