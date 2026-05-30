"""
Tello — Path Planning 4 puntos
"""

import time
import math
from djitellopy import Tello


# =============================================================================
# PARÁMETROS
# =============================================================================

ALTURA_OBJETIVO_M = 2.5
ALTURA_CM = int((ALTURA_OBJETIVO_M - 0.8) * 100)

DISTANCIA_MIN_CM = 20

TAKEOFF_WAIT_S = 5


# =============================================================================
# FUNCIONES
# =============================================================================

def dividir_segmento(p1, p2, paso_min=20):
    """
    Divide un segmento en la MENOR cantidad posible
    de waypoints respetando un desplazamiento mínimo.
    """

    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    distancia = math.sqrt(dx**2 + dy**2)

    # cantidad mínima de segmentos
    n = max(1, math.ceil(distancia / paso_min))

    waypoints = []

    for i in range(1, n + 1):

        xn = x1 + dx * i / n
        yn = y1 + dy * i / n

        waypoints.append((round(xn), round(yn)))

    return waypoints


def mover_relativo(drone, dx, dy):
    """
    Movimiento relativo usando comandos Tello.
    """

    # eje X
    if dx > 0:
        drone.move_right(abs(dx))
    elif dx < 0:
        drone.move_left(abs(dx))

    # eje Y
    if dy > 0:
        drone.move_forward(abs(dy))
    elif dy < 0:
        drone.move_back(abs(dy))


# =============================================================================
# RUTINA PRINCIPAL
# =============================================================================

def main():

    drone = Tello()
    drone.connect()

    print(f'Batería: {drone.get_battery()} %')

    # ── DESPEGUE ──────────────────────────────────────────────
    print('Despegando...')
    drone.takeoff()

    time.sleep(TAKEOFF_WAIT_S)

    # ── ASCENSO ───────────────────────────────────────────────
    print('Subiendo...')
    drone.move_up(ALTURA_CM)

    # =============================================================================
    # DEFINICIÓN DE 4 PUNTOS (cm)
    #
    #  P1 -------- P2
    #   |           |
    #   |           |
    #  P4 -------- P3
    # =============================================================================

    puntos = [
        (0, 0),        # P1
        (80, 0),      # P2
        (80, 80),    # P3
        (0, 80),      # P4
        (0, 0)         # regreso al inicio
    ]

    # =============================================================================
    # PATH PLANNING
    # =============================================================================

    trayectoria = []

    for i in range(len(puntos) - 1):

        p_actual = puntos[i]
        p_siguiente = puntos[i + 1]

        segmentos = dividir_segmento(
            p_actual,
            p_siguiente,
            DISTANCIA_MIN_CM
        )

        trayectoria.extend(segmentos)

    print('\nWaypoints generados:')
    for wp in trayectoria:
        print(wp)

    # =============================================================================
    # EJECUCIÓN
    # =============================================================================

    posicion_actual = puntos[0]

    for wp in trayectoria:

        dx = wp[0] - posicion_actual[0]
        dy = wp[1] - posicion_actual[1]

        print(f'Moviendo: dx={dx}, dy={dy}')

        mover_relativo(drone, dx, dy)

        posicion_actual = wp

        time.sleep(1)

    # ── LAND ──────────────────────────────────────────────────
    print('Aterrizando...')
    drone.land()

    print('Listo.')


if __name__ == '__main__':
    main()