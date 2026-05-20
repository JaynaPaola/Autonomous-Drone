"""
plotter.py — Nodo ROS2 de monitoreo y graficación.

Suscribe:
  /estimated_pose   (Float32MultiArray) → [x, y, z, yaw]
  /rc_command       (Int32MultiArray)   → [lr, fb, ud, yaw_rc]
  /waypoint_status  (String)            → eventos del controller

Al terminar (misión completa o Ctrl+C) muestra UNA ventana con 9 subgráficas
y una conclusión automática en cada una.
También guarda el resultado como PNG en ~/drone_plots/.
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # ventana interactiva; cambia a 'Agg' si no hay pantalla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String

# -----------------------------
# WAYPOINTS (igual que controller.py)
# -----------------------------
WAYPOINTS = [
    np.array([1.09,  1.06, 1.14]),
    np.array([1.09,  0.40, 1.14]),
    np.array([0.40,  0.40, 1.14]),
    np.array([0.40,  1.06, 1.14]),
    np.array([1.09,  1.06, 1.14]),
]
WP_NAMES  = ["WP1 Inicial", "WP2", "WP3", "WP4", "WP5 Regreso"]
WP_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
TOL_M     = 0.10

OUTPUT_DIR = os.path.expanduser("~/drone_plots")


# -----------------------------
# NODO
# -----------------------------
class PlotterNode(Node):
    def __init__(self):
        super().__init__('plotter_node')

        self.pose_sub = self.create_subscription(
            Float32MultiArray, '/estimated_pose', self.pose_callback, 10)
        self.rc_sub = self.create_subscription(
            Int32MultiArray, '/rc_command', self.rc_callback, 10)
        self.status_sub = self.create_subscription(
            String, '/waypoint_status', self.status_callback, 10)

        # --- Buffers ---
        self.t0       = None
        self.times    = []       # tiempo [s]
        self.xs       = []       # posición x
        self.ys       = []       # posición y
        self.zs       = []       # posición z (altura)
        self.yaws     = []       # yaw [grados]
        self.dists    = []       # distancia al WP activo
        self.err_x    = []       # error x
        self.err_y    = []       # error y
        self.err_z    = []       # error z

        self.rc_times = []       # tiempo de cada comando RC
        self.rc_lr    = []       # left/right
        self.rc_fb    = []       # forward/backward
        self.rc_ud    = []       # up/down
        self.rc_yaw   = []       # yaw RC

        self.current_wp   = 0
        self.wp_events    = []   # (t, wp_index, dist) — llegadas confirmadas
        self.end_reason   = "Interrumpido por el usuario ⚠️"
        self.mission_done = False

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.get_logger().info(f'PlotterNode listo. Gráficas → {OUTPUT_DIR}')

    # --------------------------------------------------
    def pose_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        x, y, z, yaw = msg.data[0], msg.data[1], msg.data[2], msg.data[3]

        self.times.append(t)
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.yaws.append(np.degrees(yaw))

        if self.current_wp < len(WAYPOINTS):
            wp   = WAYPOINTS[self.current_wp]
            ex   = wp[0] - x
            ey   = wp[1] - y
            ez   = wp[2] - z
            dist = np.sqrt(ex**2 + ey**2 + ez**2)
            self.dists.append(dist)
            self.err_x.append(ex)
            self.err_y.append(ey)
            self.err_z.append(ez)

            if dist < TOL_M and not self.mission_done:
                self.wp_events.append((t, self.current_wp, dist))
                self.current_wp += 1
                if self.current_wp >= len(WAYPOINTS):
                    self.mission_done = True
                    self.end_reason   = "Misión completa ✅"
        else:
            self.dists.append(0.0)
            self.err_x.append(0.0)
            self.err_y.append(0.0)
            self.err_z.append(0.0)

    # --------------------------------------------------
    def rc_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.t0 is None:
            return
        self.rc_times.append(now - self.t0)
        self.rc_lr.append(msg.data[0])
        self.rc_fb.append(msg.data[1])
        self.rc_ud.append(msg.data[2])
        self.rc_yaw.append(msg.data[3])

    # --------------------------------------------------
    def status_callback(self, msg):
        self.get_logger().info(f"[Status] {msg.data}")

    # --------------------------------------------------
    def _wp_vlines(self, ax):
        """Dibuja líneas verticales en cada llegada a waypoint."""
        for te, wi, _ in self.wp_events:
            ax.axvline(te, color=WP_COLORS[wi % len(WP_COLORS)],
                       linestyle='--', linewidth=1, alpha=0.6,
                       label=f'{WP_NAMES[wi]}')

    def _conclude(self, ax, text, color='#E3F2FD'):
        """Agrega cuadro de conclusión debajo del título de cada subgráfica."""
        ax.annotate(
            text,
            xy=(0.5, -0.22), xycoords='axes fraction',
            ha='center', va='top', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                      edgecolor='gray', alpha=0.85)
        )

    # --------------------------------------------------
    def generate_plots(self):
        if len(self.times) < 2:
            self.get_logger().warn('Pocos datos para graficar.')
            return

        t   = np.array(self.times)
        x   = np.array(self.xs)
        y   = np.array(self.ys)
        z   = np.array(self.zs)
        yaw = np.array(self.yaws)
        d   = np.array(self.dists)
        ex  = np.array(self.err_x)
        ey  = np.array(self.err_y)
        ez  = np.array(self.err_z)

        has_rc = len(self.rc_times) > 1
        if has_rc:
            tr  = np.array(self.rc_times)
            lr  = np.array(self.rc_lr)
            fb  = np.array(self.rc_fb)
            ud  = np.array(self.rc_ud)
            yrc = np.array(self.rc_yaw)

        duration = t[-1]
        n_wp     = len(self.wp_events)

        # =============================================
        # FIGURA ÚNICA  3 filas × 3 columnas + fila de conclusión global
        # =============================================
        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor('#F5F5F5')

        title_color = '#1A237E'
        fig.suptitle(
            f"Reporte de vuelo autónomo — {self.end_reason}\n"
            f"Duración: {duration:.1f} s   |   "
            f"Waypoints alcanzados: {n_wp}/{len(WAYPOINTS)}   |   "
            f"Muestras: {len(t)}",
            fontsize=14, fontweight='bold', color=title_color, y=0.98
        )

        gs = gridspec.GridSpec(
            3, 3, figure=fig,
            hspace=0.65, wspace=0.35,
            top=0.91, bottom=0.10
        )

        # ---- colores base ----
        C = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
             '#00BCD4', '#FF5722', '#607D8B', '#795548']

        # =============================================
        # 1. TRAYECTORIA XY
        # =============================================
        ax1 = fig.add_subplot(gs[0, 0])
        for i, wp in enumerate(WAYPOINTS):
            mask = np.array(self.current_wp if self.current_wp < len(WAYPOINTS)
                            else len(WAYPOINTS) - 1)
            # trazar segmento por WP
        # trazar trayectoria completa con gradiente de tiempo
        sc = ax1.scatter(x, y, c=t, cmap='plasma', s=8, zorder=3)
        plt.colorbar(sc, ax=ax1, label='t [s]', pad=0.02)

        for i, wp in enumerate(WAYPOINTS):
            ax1.scatter(wp[0], wp[1], color=WP_COLORS[i], s=120,
                        zorder=5, edgecolors='black', linewidth=0.7)
            ax1.annotate(f"WP{i+1}", (wp[0], wp[1]),
                         xytext=(5, 4), textcoords='offset points', fontsize=8)

        ax1.scatter(x[0],  y[0],  marker='^', s=160, color='green',
                    zorder=6, label='Inicio')
        ax1.scatter(x[-1], y[-1], marker='s', s=160, color='red',
                    zorder=6, label='Fin')
        for te, wi, _ in self.wp_events:
            idx = np.argmin(np.abs(t - te))
            ax1.scatter(x[idx], y[idx], marker='*', s=220,
                        color='gold', zorder=7, edgecolors='black', linewidth=0.5)

        ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]')
        ax1.set_title('Trayectoria XY', fontweight='bold')
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='datalim')

        # Conclusión
        desvio = np.mean(np.sqrt((x - np.mean([wp[0] for wp in WAYPOINTS]))**2 +
                                  (y - np.mean([wp[1] for wp in WAYPOINTS]))**2))
        concl1 = (f"Desviación media de ruta: {desvio:.3f} m\n"
                  f"Rango X: [{x.min():.2f}, {x.max():.2f}]  "
                  f"Rango Y: [{y.min():.2f}, {y.max():.2f}]")
        self._conclude(ax1, concl1)

        # =============================================
        # 2. ALTURA Z vs TIEMPO
        # =============================================
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, z, color=C[0], linewidth=1.5)
        ax2.axhline(WAYPOINTS[0][2], color='gray', linestyle='--',
                    linewidth=1, label=f'Objetivo {WAYPOINTS[0][2]} m')
        self._wp_vlines(ax2)
        ax2.set_xlabel('Tiempo [s]'); ax2.set_ylabel('Z [m]')
        ax2.set_title('Altura vs Tiempo', fontweight='bold')
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        z_std = np.std(z)
        z_mean = np.mean(z)
        concl2 = (f"Media: {z_mean:.3f} m   Desv. est.: {z_std:.3f} m\n"
                  f"{'✅ Altura estable' if z_std < 0.05 else '⚠️ Altura inestable'}")
        self._conclude(ax2, concl2,
                       color='#C8E6C9' if z_std < 0.05 else '#FFCDD2')

        # =============================================
        # 3. YAW vs TIEMPO
        # =============================================
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, yaw, color=C[3], linewidth=1.5)
        ax3.axhline(yaw[0], color='gray', linestyle='--',
                    linewidth=1, label=f'Ref {yaw[0]:.1f}°')
        self._wp_vlines(ax3)
        ax3.set_xlabel('Tiempo [s]'); ax3.set_ylabel('Yaw [°]')
        ax3.set_title('Yaw vs Tiempo', fontweight='bold')
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

        yaw_drift = np.max(np.abs(yaw - yaw[0]))
        concl3 = (f"Deriva máx. de yaw: {yaw_drift:.1f}°\n"
                  f"{'✅ Yaw estable' if yaw_drift < 10 else '⚠️ Deriva de yaw significativa'}")
        self._conclude(ax3, concl3,
                       color='#C8E6C9' if yaw_drift < 10 else '#FFCDD2')

        # =============================================
        # 4. DISTANCIA AL WAYPOINT ACTIVO
        # =============================================
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, d, color=C[4], linewidth=1.5)
        ax4.axhline(TOL_M, color='green', linestyle='--',
                    linewidth=1.2, label=f'Tolerancia {TOL_M} m')
        self._wp_vlines(ax4)
        ax4.set_xlabel('Tiempo [s]'); ax4.set_ylabel('Distancia [m]')
        ax4.set_title('Distancia al waypoint activo', fontweight='bold')
        ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

        concl4 = (f"WP alcanzados: {n_wp}/{len(WAYPOINTS)}\n"
                  f"Dist. media: {np.mean(d):.3f} m   Dist. mín: {np.min(d):.3f} m")
        self._conclude(ax4, concl4,
                       color='#C8E6C9' if n_wp == len(WAYPOINTS) else '#FFF9C4')

        # =============================================
        # 5. ERRORES POR EJE
        # =============================================
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(t, ex, color=C[0], linewidth=1.2, label='err X')
        ax5.plot(t, ey, color=C[1], linewidth=1.2, label='err Y')
        ax5.plot(t, ez, color=C[2], linewidth=1.2, label='err Z')
        ax5.axhline(0, color='black', linewidth=0.6, linestyle='-')
        ax5.axhline( TOL_M, color='gray', linestyle=':', linewidth=0.8)
        ax5.axhline(-TOL_M, color='gray', linestyle=':', linewidth=0.8)
        self._wp_vlines(ax5)
        ax5.set_xlabel('Tiempo [s]'); ax5.set_ylabel('Error [m]')
        ax5.set_title('Error por eje (objetivo − posición)', fontweight='bold')
        ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

        concl5 = (f"RMSE X:{np.sqrt(np.mean(ex**2)):.3f}  "
                  f"Y:{np.sqrt(np.mean(ey**2)):.3f}  "
                  f"Z:{np.sqrt(np.mean(ez**2)):.3f} [m]")
        self._conclude(ax5, concl5)

        # =============================================
        # 6. COMANDOS RC — POSICIÓN (lr, fb, ud)
        # =============================================
        ax6 = fig.add_subplot(gs[1, 2])
        if has_rc:
            ax6.plot(tr, lr,  color=C[0], linewidth=1.2, label='LR (X)')
            ax6.plot(tr, fb,  color=C[1], linewidth=1.2, label='FB (Y)')
            ax6.plot(tr, ud,  color=C[2], linewidth=1.2, label='UD (Z)')
            ax6.axhline(0, color='black', linewidth=0.6)
            self._wp_vlines(ax6)
            sat_pct = np.mean(
                (np.abs(lr) >= 25) | (np.abs(fb) >= 25) | (np.abs(ud) >= 25)
            ) * 100
            concl6 = (f"Saturación RC posición: {sat_pct:.1f}% del tiempo\n"
                      f"{'✅ Control suave' if sat_pct < 20 else '⚠️ Frecuente saturación'}")
        else:
            ax6.text(0.5, 0.5, 'Sin datos RC', ha='center', va='center',
                     transform=ax6.transAxes, fontsize=11, color='gray')
            concl6 = "No se recibieron comandos RC"

        ax6.set_xlabel('Tiempo [s]'); ax6.set_ylabel('RC [-40, 40]')
        ax6.set_title('Comandos RC — Posición', fontweight='bold')
        ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)
        self._conclude(ax6, concl6,
                       color='#C8E6C9' if has_rc and sat_pct < 20 else '#FFCDD2')

        # =============================================
        # 7. COMANDO RC — YAW
        # =============================================
        ax7 = fig.add_subplot(gs[2, 0])
        if has_rc:
            ax7.plot(tr, yrc, color=C[3], linewidth=1.2, label='Yaw RC')
            ax7.axhline(0, color='black', linewidth=0.6)
            self._wp_vlines(ax7)
            yaw_sat = np.mean(np.abs(yrc) >= 35) * 100
            concl7 = (f"Saturación RC yaw: {yaw_sat:.1f}% del tiempo\n"
                      f"Media |yaw_rc|: {np.mean(np.abs(yrc)):.1f}")
        else:
            ax7.text(0.5, 0.5, 'Sin datos RC', ha='center', va='center',
                     transform=ax7.transAxes, fontsize=11, color='gray')
            concl7 = "No se recibieron comandos RC"

        ax7.set_xlabel('Tiempo [s]'); ax7.set_ylabel('RC Yaw')
        ax7.set_title('Comando RC — Yaw', fontweight='bold')
        ax7.legend(fontsize=7); ax7.grid(True, alpha=0.3)
        self._conclude(ax7, concl7)

        # =============================================
        # 8. X e Y vs TIEMPO
        # =============================================
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(t, x, color=C[0], linewidth=1.3, label='X')
        ax8.plot(t, y, color=C[1], linewidth=1.3, label='Y')
        for i, wp in enumerate(WAYPOINTS):
            ax8.axhline(wp[0], color=WP_COLORS[i], linestyle=':',
                        linewidth=0.7, alpha=0.5)
            ax8.axhline(wp[1], color=WP_COLORS[i], linestyle='-.',
                        linewidth=0.7, alpha=0.5)
        self._wp_vlines(ax8)
        ax8.set_xlabel('Tiempo [s]'); ax8.set_ylabel('[m]')
        ax8.set_title('Posición X e Y vs Tiempo', fontweight='bold')
        ax8.legend(fontsize=7); ax8.grid(True, alpha=0.3)

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        concl8 = (f"Rango X recorrido: {x_range:.3f} m   "
                  f"Rango Y recorrido: {y_range:.3f} m")
        self._conclude(ax8, concl8)

        # =============================================
        # 9. TABLA RESUMEN DE WAYPOINTS
        # =============================================
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        rows = []
        for i, wp in enumerate(WAYPOINTS):
            ev = next((e for e in self.wp_events if e[1] == i), None)
            if ev:
                rows.append([f"WP{i+1}", f"{wp[0]:.2f}",
                              f"{wp[1]:.2f}", f"{wp[2]:.2f}",
                              f"{ev[0]:.1f}s", f"{ev[2]:.3f}m", "✅"])
            else:
                rows.append([f"WP{i+1}", f"{wp[0]:.2f}",
                              f"{wp[1]:.2f}", f"{wp[2]:.2f}",
                              "—", "—", "❌"])

        cols = ['WP', 'X', 'Y', 'Z', 'Tiempo', 'Error', 'Estado']
        tbl  = ax9.table(cellText=rows, colLabels=cols,
                         loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)

        for i, row in enumerate(rows):
            color = '#C8E6C9' if '✅' in row[-1] else '#FFCDD2'
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor(color)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor('#BBDEFB')

        ax9.set_title('Resumen de waypoints', fontweight='bold', pad=10)

        # =============================================
        # CONCLUSIÓN GLOBAL al pie de la figura
        # =============================================
        rmse_total = np.sqrt(np.mean(d**2))
        if n_wp == len(WAYPOINTS):
            estado_mision = "MISIÓN EXITOSA ✅"
            bg = '#C8E6C9'
        elif n_wp > 0:
            estado_mision = f"MISIÓN PARCIAL ⚠️  ({n_wp}/{len(WAYPOINTS)} WP)"
            bg = '#FFF9C4'
        else:
            estado_mision = "MISIÓN NO COMPLETADA ❌"
            bg = '#FFCDD2'

        conclusion = (
            f"{estado_mision}   |   Duración: {duration:.1f} s   |   "
            f"RMSE posición: {rmse_total:.3f} m   |   "
            f"Deriva yaw máx: {yaw_drift:.1f}°   |   {self.end_reason}"
        )
        fig.text(
            0.5, 0.02, conclusion,
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=bg,
                      edgecolor='#90A4AE', alpha=0.95)
        )

        # =============================================
        # GUARDAR Y MOSTRAR
        # =============================================
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(OUTPUT_DIR, f"vuelo_{timestamp}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        self.get_logger().info(f'✅ Gráfica guardada en: {fname}')
        print(f'\n📊 Imagen guardada: {fname}\n')

        plt.show()   # ventana interactiva — se cierra manualmente


# -----------------------------
# MAIN
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PlotterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido — generando reporte...')
        node.end_reason = "Interrumpido por el usuario ⚠️"
    finally:
        node.generate_plots()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()