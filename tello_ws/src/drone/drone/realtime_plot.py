"""
realtime_plot.py — Nodo ROS2 de visualización en tiempo real.

Abre DOS ventanas simultáneas:
  - Ventana 1: Trayectoria XY actualizada en tiempo real
  - Ventana 2: Altura Z vs Tiempo actualizada en tiempo real

Suscribe a:
  /estimated_pose  (Float32MultiArray) → [x, y, z, yaw]
"""

import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# -----------------------------
# WAYPOINTS
# -----------------------------
WAYPOINTS = [
    np.array([1.09,  1.06, 1.14]),
    np.array([1.09,  0.40, 1.14]),
    np.array([0.40,  0.40, 1.14]),
    np.array([0.40,  1.06, 1.14]),
    np.array([1.09,  1.06, 1.14]),
]
WP_COLORS  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
TOL_M      = 0.10
MAX_POINTS = 2000   # máximo de puntos en memoria


# -----------------------------
# NODO
# -----------------------------
class RealtimePlotNode(Node):
    def __init__(self):
        super().__init__('realtime_plot_node')

        self.pose_sub = self.create_subscription(
            Float32MultiArray, '/estimated_pose', self.pose_callback, 10
        )

        # Buffers compartidos (hilo ROS ↔ hilo matplotlib)
        self.lock  = threading.Lock()
        self.t0    = None
        self.times = []
        self.xs    = []
        self.ys    = []
        self.zs    = []

        # Waypoint activo
        self.current_wp = 0
        self.wp_reached = []   # índices de WP ya alcanzados

        self.get_logger().info('RealtimePlotNode iniciado.')

    # --------------------------------------------------
    def pose_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        with self.lock:
            if self.t0 is None:
                self.t0 = now
            t = now - self.t0
            x, y, z = msg.data[0], msg.data[1], msg.data[2]

            self.times.append(t)
            self.xs.append(x)
            self.ys.append(y)
            self.zs.append(z)

            # Recortar si hay demasiados puntos
            if len(self.times) > MAX_POINTS:
                self.times = self.times[-MAX_POINTS:]
                self.xs    = self.xs[-MAX_POINTS:]
                self.ys    = self.ys[-MAX_POINTS:]
                self.zs    = self.zs[-MAX_POINTS:]

            # Detectar llegada al waypoint activo
            if self.current_wp < len(WAYPOINTS):
                wp   = WAYPOINTS[self.current_wp]
                dist = np.sqrt(
                    (wp[0] - x)**2 + (wp[1] - y)**2 + (wp[2] - z)**2
                )
                if dist < TOL_M:
                    self.wp_reached.append(self.current_wp)
                    self.current_wp += 1


# -----------------------------
# VENTANA 1 — TRAYECTORIA XY
# -----------------------------
def make_xy_window(node):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#2D2D2D')
    fig.canvas.manager.set_window_title('Trayectoria XY — Tiempo Real')

    # Waypoints destino (estáticos)
    for i, wp in enumerate(WAYPOINTS):
        ax.scatter(wp[0], wp[1],
                   color=WP_COLORS[i], s=180, zorder=5,
                   edgecolors='white', linewidth=0.8)
        ax.annotate(f'WP{i+1}', (wp[0], wp[1]),
                    xytext=(6, 5), textcoords='offset points',
                    fontsize=9, color='white', fontweight='bold')

    # Elementos dinámicos
    trail,    = ax.plot([], [], color='#00E5FF', linewidth=1.2,
                        alpha=0.6, zorder=3)
    dot,      = ax.plot([], [], 'o', color='#FFD740', markersize=10,
                        zorder=6, label='Posición actual')
    start_dot,= ax.plot([], [], '^', color='#69F0AE', markersize=12,
                        zorder=7, label='Inicio')

    ax.set_xlabel('X [m]', color='white')
    ax.set_ylabel('Y [m]', color='white')
    ax.set_title('Trayectoria XY', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#555555')
    ax.grid(True, color='#444444', alpha=0.5)
    ax.legend(facecolor='#2D2D2D', labelcolor='white', fontsize=8)

    # Márgenes fijos con algo de rango visible desde el inicio
    all_x = [wp[0] for wp in WAYPOINTS]
    all_y = [wp[1] for wp in WAYPOINTS]
    margin = 0.3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Texto de info
    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        fontsize=9, color='white', va='top',
        bbox=dict(boxstyle='round', facecolor='#1E1E1E', alpha=0.7)
    )

    # Marcadores de llegada (se añaden dinámicamente)
    reached_scatter = ax.scatter([], [], marker='*', s=300,
                                  color='gold', zorder=8,
                                  edgecolors='black', linewidth=0.5)

    def update(_):
        with node.lock:
            if len(node.xs) < 1:
                return trail, dot, start_dot, info_text, reached_scatter

            x  = list(node.xs)
            y  = list(node.ys)
            t  = list(node.times)
            wp_reached = list(node.wp_reached)
            cur_wp     = node.current_wp

        trail.set_data(x, y)
        dot.set_data([x[-1]], [y[-1]])

        if len(x) >= 1:
            start_dot.set_data([x[0]], [y[0]])

        # Marcadores de llegada
        if wp_reached:
            rx = [WAYPOINTS[i][0] for i in wp_reached]
            ry = [WAYPOINTS[i][1] for i in wp_reached]
            reached_scatter.set_offsets(np.c_[rx, ry])

        # Info
        dur   = t[-1] if t else 0
        n_wp  = len(wp_reached)
        sig   = f"→ WP{cur_wp + 1}" if cur_wp < len(WAYPOINTS) else "Misión completa"
        info_text.set_text(
            f"t: {dur:.1f}s\n"
            f"x: {x[-1]:.3f} m\n"
            f"y: {y[-1]:.3f} m\n"
            f"WP: {n_wp}/{len(WAYPOINTS)}\n"
            f"{sig}"
        )

        return trail, dot, start_dot, info_text, reached_scatter

    ani = animation.FuncAnimation(
        fig, update, interval=100, blit=False, cache_frame_data=False
    )
    return fig, ani


# -----------------------------
# VENTANA 2 — ALTURA Z vs TIEMPO
# -----------------------------
def make_z_window(node):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#2D2D2D')
    fig.canvas.manager.set_window_title('Altura Z — Tiempo Real')

    # Línea objetivo
    z_target = WAYPOINTS[0][2]
    ax.axhline(z_target, color='#69F0AE', linestyle='--',
               linewidth=1.2, label=f'Objetivo {z_target} m', zorder=2)

    # Banda de tolerancia
    ax.axhspan(z_target - TOL_M, z_target + TOL_M,
               color='#69F0AE', alpha=0.08, zorder=1)

    # Elementos dinámicos
    line,  = ax.plot([], [], color='#00E5FF', linewidth=1.8,
                     zorder=3, label='Altura Z')
    dot_z, = ax.plot([], [], 'o', color='#FFD740', markersize=9,
                     zorder=5)

    ax.set_xlabel('Tiempo [s]', color='white')
    ax.set_ylabel('Z [m]', color='white')
    ax.set_title('Altura Z vs Tiempo', color='white',
                 fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#555555')
    ax.grid(True, color='#444444', alpha=0.5)
    ax.set_ylim(-0.1, 1.6)
    ax.legend(facecolor='#2D2D2D', labelcolor='white', fontsize=8)

    # Texto de info
    info_text = ax.text(
        0.01, 0.97, '', transform=ax.transAxes,
        fontsize=9, color='white', va='top',
        bbox=dict(boxstyle='round', facecolor='#1E1E1E', alpha=0.7)
    )

    # Líneas verticales de llegada (se añaden dinámicamente)
    vlines = []

    def update(_):
        nonlocal vlines
        with node.lock:
            if len(node.zs) < 1:
                return line, dot_z, info_text

            t          = list(node.times)
            z          = list(node.zs)
            wp_reached = list(node.wp_reached)

        line.set_data(t, z)
        dot_z.set_data([t[-1]], [z[-1]])

        # Ajustar eje X para mostrar ventana deslizante de 60s
        t_now = t[-1]
        if t_now > 60:
            ax.set_xlim(t_now - 60, t_now + 2)
        else:
            ax.set_xlim(0, max(60, t_now + 2))

        # Añadir líneas verticales de llegada nuevas
        while len(vlines) < len(wp_reached):
            i   = wp_reached[len(vlines)]
            # encontrar tiempo aproximado de llegada
            vl  = ax.axvline(t[-1], color=WP_COLORS[i % len(WP_COLORS)],
                             linestyle=':', linewidth=1.5, alpha=0.8,
                             label=f'WP{i+1} ✅')
            vlines.append(vl)

        # Info
        z_now  = z[-1]
        e_z    = z_target - z_now
        estado = "✅ En zona" if abs(e_z) < TOL_M else f"err: {e_z:+.3f} m"
        info_text.set_text(
            f"t: {t_now:.1f}s\n"
            f"Z: {z_now:.3f} m\n"
            f"{estado}"
        )

        return line, dot_z, info_text

    ani = animation.FuncAnimation(
        fig, update, interval=100, blit=False, cache_frame_data=False
    )
    return fig, ani


# -----------------------------
# MAIN
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RealtimePlotNode()

    # Hilo de ROS2 (spin) separado del hilo principal (matplotlib)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Crear las dos ventanas
    fig_xy, ani_xy = make_xy_window(node)
    fig_z,  ani_z  = make_z_window(node)

    # Posicionar ventanas (TkAgg)
    try:
        fig_xy.canvas.manager.window.geometry('750x750+0+0')
        fig_z.canvas.manager.window.geometry('950x450+760+0')
    except Exception:
        pass   # si el gestor de ventanas no lo soporta, no pasa nada

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('RealtimePlotNode cerrando...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()