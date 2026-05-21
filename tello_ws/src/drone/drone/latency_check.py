"""
latency_check.py — Nodo de diagnóstico de latencia.

Mide el tiempo que tarda cada etapa del pipeline:

  1. OptiTrack → odometry_node     (¿tarda en llegar la pose?)
  2. odometry_node → controller    (¿tarda en publicar /estimated_pose?)
  3. controller → rc_command       (¿tarda en calcular y publicar RC?)
  4. rc_command → dron             (no medible directamente, se estima)

Imprime un reporte cada 5 segundos con estadísticas de latencia.
Al cerrar genera una gráfica con el historial completo.
"""

import threading
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from geometry_msgs.msg import PoseStamped

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class LatencyNode(Node):
    def __init__(self):
        super().__init__('latency_check_node')

        # ── Suscripciones ───────────────────────────
        # 1. OptiTrack crudo
        self.optitrack_sub = self.create_subscription(
            PoseStamped, '/optitrack/rigid_body',
            self.optitrack_callback, SENSOR_QOS
        )
        # 2. Pose fusionada publicada por odometry_node
        self.pose_sub = self.create_subscription(
            Float32MultiArray, '/estimated_pose',
            self.pose_callback, 10
        )
        # 3. Comandos RC publicados por controller
        self.rc_sub = self.create_subscription(
            Int32MultiArray, '/rc_command',
            self.rc_callback, 10
        )

        # ── Buffers ─────────────────────────────────
        self.lock = threading.Lock()

        # Timestamps de cada etapa (segundos, tiempo local)
        self.t_optitrack = []   # cuando llega dato de OptiTrack
        self.t_pose      = []   # cuando llega /estimated_pose
        self.t_rc        = []   # cuando llega /rc_command

        # Latencias calculadas
        self.lat_opti_to_pose = []   # OptiTrack → pose estimada
        self.lat_pose_to_rc   = []   # pose estimada → RC command

        # Frecuencias
        self.freq_optitrack = []
        self.freq_pose      = []
        self.freq_rc        = []

        # Último timestamp de cada topic para calcular frecuencia
        self._last_opti = None
        self._last_pose = None
        self._last_rc   = None

        # Timer de reporte cada 5 segundos
        self.create_timer(5.0, self.print_report)

        self.t_start = time.time()
        self.get_logger().info('=' * 55)
        self.get_logger().info('LatencyNode iniciado. Midiendo latencias...')
        self.get_logger().info('=' * 55)

    # ────────────────────────────────────────────────
    def optitrack_callback(self, msg: PoseStamped):
        now = time.time()
        with self.lock:
            # Frecuencia de OptiTrack
            if self._last_opti is not None:
                dt = now - self._last_opti
                if dt > 0:
                    self.freq_optitrack.append(1.0 / dt)
            self._last_opti = now
            self.t_optitrack.append(now)

    # ────────────────────────────────────────────────
    def pose_callback(self, msg):
        now = time.time()
        with self.lock:
            # Frecuencia de /estimated_pose
            if self._last_pose is not None:
                dt = now - self._last_pose
                if dt > 0:
                    self.freq_pose.append(1.0 / dt)
            self._last_pose = now
            self.t_pose.append(now)

            # Latencia OptiTrack → pose estimada
            # (diferencia entre el último OptiTrack y este pose)
            if self.t_optitrack:
                lat = (now - self.t_optitrack[-1]) * 1000   # ms
                if 0 <= lat < 500:   # filtrar outliers
                    self.lat_opti_to_pose.append(lat)

    # ────────────────────────────────────────────────
    def rc_callback(self, msg):
        now = time.time()
        with self.lock:
            # Frecuencia de /rc_command
            if self._last_rc is not None:
                dt = now - self._last_rc
                if dt > 0:
                    self.freq_rc.append(1.0 / dt)
            self._last_rc = now
            self.t_rc.append(now)

            # Latencia pose → RC
            if self.t_pose:
                lat = (now - self.t_pose[-1]) * 1000   # ms
                if 0 <= lat < 500:
                    self.lat_pose_to_rc.append(lat)

    # ────────────────────────────────────────────────
    def print_report(self):
        with self.lock:
            l1 = list(self.lat_opti_to_pose)
            l2 = list(self.lat_pose_to_rc)
            f1 = list(self.freq_optitrack)
            f2 = list(self.freq_pose)
            f3 = list(self.freq_rc)

        elapsed = time.time() - self.t_start

        def stats(data, unit):
            if not data:
                return "sin datos"
            return (f"media={np.mean(data):.1f}{unit}  "
                    f"max={np.max(data):.1f}{unit}  "
                    f"p95={np.percentile(data, 95):.1f}{unit}")

        sep = '─' * 55
        self.get_logger().info(sep)
        self.get_logger().info(f"DIAGNÓSTICO DE LATENCIA  (t={elapsed:.0f}s)")
        self.get_logger().info(sep)
        self.get_logger().info(
            f"  Frecuencia OptiTrack :  {stats(f1, ' Hz')}")
        self.get_logger().info(
            f"  Frecuencia /est_pose :  {stats(f2, ' Hz')}")
        self.get_logger().info(
            f"  Frecuencia /rc_cmd   :  {stats(f3, ' Hz')}")
        self.get_logger().info(sep)
        self.get_logger().info(
            f"  Latencia OptiTrack→pose:  {stats(l1, ' ms')}")
        self.get_logger().info(
            f"  Latencia pose→RC cmd  :  {stats(l2, ' ms')}")
        if l1 and l2:
            total = np.mean(l1) + np.mean(l2)
            self.get_logger().info(
                f"  Latencia TOTAL estimada:  {total:.1f} ms")
        self.get_logger().info(sep)

        # Diagnóstico automático
        warns = []
        if f1 and np.mean(f1) < 50:
            warns.append(f"⚠️  OptiTrack lento ({np.mean(f1):.0f} Hz, esperado >100 Hz)")
        if f2 and np.mean(f2) < 8:
            warns.append(f"⚠️  /estimated_pose lento ({np.mean(f2):.0f} Hz, esperado ~10 Hz)")
        if f3 and np.mean(f3) < 8:
            warns.append(f"⚠️  /rc_command lento ({np.mean(f3):.0f} Hz, esperado ~10 Hz)")
        if l1 and np.mean(l1) > 50:
            warns.append(f"⚠️  Latencia OptiTrack→pose alta ({np.mean(l1):.0f} ms)")
        if l2 and np.mean(l2) > 50:
            warns.append(f"⚠️  Latencia pose→RC alta ({np.mean(l2):.0f} ms)")

        if warns:
            for w in warns:
                self.get_logger().warn(w)
        else:
            self.get_logger().info("  ✅ Latencias dentro de rango normal")
        self.get_logger().info(sep)

    # ────────────────────────────────────────────────
    def generate_plots(self):
        with self.lock:
            l1 = list(self.lat_opti_to_pose)
            l2 = list(self.lat_pose_to_rc)
            f1 = list(self.freq_optitrack)
            f2 = list(self.freq_pose)
            f3 = list(self.freq_rc)
            n_opti = len(self.t_optitrack)
            n_pose = len(self.t_pose)
            n_rc   = len(self.t_rc)

        if not l1 and not l2:
            self.get_logger().warn('Sin datos suficientes para graficar.')
            return

        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor('#F5F5F5')
        fig.suptitle(
            'Diagnóstico de latencia del pipeline',
            fontsize=14, fontweight='bold', color='#1A237E'
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                               top=0.90, bottom=0.10)

        # ── 1. Histograma latencia OptiTrack → pose ──
        ax1 = fig.add_subplot(gs[0, 0])
        if l1:
            ax1.hist(l1, bins=40, color='#2196F3', edgecolor='white', linewidth=0.5)
            ax1.axvline(np.mean(l1), color='red', linestyle='--',
                        linewidth=1.5, label=f'Media {np.mean(l1):.1f} ms')
            ax1.axvline(np.percentile(l1, 95), color='orange', linestyle=':',
                        linewidth=1.5, label=f'P95 {np.percentile(l1, 95):.1f} ms')
            ax1.legend(fontsize=7)
        ax1.set_title('Latencia OptiTrack → /estimated_pose', fontweight='bold')
        ax1.set_xlabel('Latencia [ms]')
        ax1.set_ylabel('Frecuencia')
        ax1.grid(True, alpha=0.3)
        self._conclude(ax1, l1, 'ms', 50)

        # ── 2. Histograma latencia pose → RC ─────────
        ax2 = fig.add_subplot(gs[0, 1])
        if l2:
            ax2.hist(l2, bins=40, color='#4CAF50', edgecolor='white', linewidth=0.5)
            ax2.axvline(np.mean(l2), color='red', linestyle='--',
                        linewidth=1.5, label=f'Media {np.mean(l2):.1f} ms')
            ax2.axvline(np.percentile(l2, 95), color='orange', linestyle=':',
                        linewidth=1.5, label=f'P95 {np.percentile(l2, 95):.1f} ms')
            ax2.legend(fontsize=7)
        ax2.set_title('Latencia /estimated_pose → /rc_command', fontweight='bold')
        ax2.set_xlabel('Latencia [ms]')
        ax2.set_ylabel('Frecuencia')
        ax2.grid(True, alpha=0.3)
        self._conclude(ax2, l2, 'ms', 50)

        # ── 3. Latencia total estimada ────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        if l1 and l2:
            n     = min(len(l1), len(l2))
            total = [l1[i] + l2[i] for i in range(n)]
            ax3.hist(total, bins=40, color='#FF9800', edgecolor='white', linewidth=0.5)
            ax3.axvline(np.mean(total), color='red', linestyle='--',
                        linewidth=1.5, label=f'Media {np.mean(total):.1f} ms')
            ax3.axvline(np.percentile(total, 95), color='orange', linestyle=':',
                        linewidth=1.5, label=f'P95 {np.percentile(total, 95):.1f} ms')
            ax3.legend(fontsize=7)
            self._conclude(ax3, total, 'ms', 100)
        ax3.set_title('Latencia TOTAL (OptiTrack → RC)', fontweight='bold')
        ax3.set_xlabel('Latencia [ms]')
        ax3.set_ylabel('Frecuencia')
        ax3.grid(True, alpha=0.3)

        # ── 4. Frecuencia OptiTrack ───────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        if f1:
            ax4.plot(f1, color='#2196F3', linewidth=0.8, alpha=0.7)
            ax4.axhline(np.mean(f1), color='red', linestyle='--',
                        linewidth=1.2, label=f'Media {np.mean(f1):.0f} Hz')
            ax4.axhline(100, color='green', linestyle=':',
                        linewidth=1, label='Objetivo 100 Hz')
            ax4.legend(fontsize=7)
        ax4.set_title('Frecuencia OptiTrack', fontweight='bold')
        ax4.set_xlabel('Muestra')
        ax4.set_ylabel('Hz')
        ax4.grid(True, alpha=0.3)

        # ── 5. Frecuencia /estimated_pose y /rc_command
        ax5 = fig.add_subplot(gs[1, 1])
        if f2:
            ax5.plot(f2, color='#4CAF50', linewidth=0.8,
                     alpha=0.7, label='/estimated_pose')
        if f3:
            ax5.plot(f3, color='#FF5722', linewidth=0.8,
                     alpha=0.7, label='/rc_command')
        ax5.axhline(10, color='gray', linestyle=':', linewidth=1,
                    label='Objetivo 10 Hz')
        ax5.legend(fontsize=7)
        ax5.set_title('Frecuencia /estimated_pose y /rc_command', fontweight='bold')
        ax5.set_xlabel('Muestra')
        ax5.set_ylabel('Hz')
        ax5.grid(True, alpha=0.3)

        # ── 6. Tabla resumen ──────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        def r(data, unit='ms'):
            if not data:
                return ['—', '—', '—', '—']
            return [
                f"{np.mean(data):.1f} {unit}",
                f"{np.min(data):.1f} {unit}",
                f"{np.max(data):.1f} {unit}",
                f"{np.percentile(data, 95):.1f} {unit}",
            ]

        rows = [
            ['OptiTrack→pose'] + r(l1),
            ['pose→RC cmd']    + r(l2),
            ['Total pipeline'] + (r([l1[i]+l2[i] for i in range(min(len(l1),len(l2)))]) if l1 and l2 else ['—']*4),
            ['Frec. OptiTrack'] + r(f1, 'Hz'),
            ['Frec. pose']      + r(f2, 'Hz'),
            ['Frec. RC cmd']    + r(f3, 'Hz'),
            ['Msgs OptiTrack', str(n_opti), '', '', ''],
            ['Msgs pose',       str(n_pose), '', '', ''],
            ['Msgs RC',         str(n_rc),   '', '', ''],
        ]
        cols = ['Métrica', 'Media', 'Mín', 'Máx', 'P95']
        tbl  = ax6.table(cellText=rows, colLabels=cols,
                         loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor('#BBDEFB')

        ax6.set_title('Resumen estadístico', fontweight='bold', pad=10)

        # ── Conclusión global ─────────────────────────
        problemas = []
        if l1 and np.mean(l1) > 50:
            problemas.append(f"latencia OptiTrack→pose alta ({np.mean(l1):.0f} ms)")
        if l2 and np.mean(l2) > 50:
            problemas.append(f"latencia pose→RC alta ({np.mean(l2):.0f} ms)")
        if f1 and np.mean(f1) < 50:
            problemas.append(f"OptiTrack lento ({np.mean(f1):.0f} Hz)")
        if f2 and np.mean(f2) < 8:
            problemas.append(f"/estimated_pose lento ({np.mean(f2):.0f} Hz)")

        if problemas:
            concl = "⚠️ Problemas: " + "  |  ".join(problemas)
            bg    = '#FFCDD2'
        else:
            concl = "✅ Pipeline dentro de rangos normales"
            bg    = '#C8E6C9'

        fig.text(0.5, 0.01, concl, ha='center', va='bottom',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=bg,
                           edgecolor='#90A4AE', alpha=0.95))

        plt.savefig(
            f'/tmp/latency_report_{int(time.time())}.png',
            dpi=130, bbox_inches='tight'
        )
        print('\n📊 Reporte de latencia guardado en /tmp/\n')
        plt.show()

    # ────────────────────────────────────────────────
    def _conclude(self, ax, data, unit, threshold):
        if not data:
            return
        mean = np.mean(data)
        ok   = mean < threshold
        ax.annotate(
            f"{'✅ OK' if ok else '⚠️ ALTO'}  media={mean:.1f} {unit}",
            xy=(0.5, -0.18), xycoords='axes fraction',
            ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#C8E6C9' if ok else '#FFCDD2',
                      edgecolor='gray', alpha=0.9)
        )


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LatencyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Generando reporte de latencia...')
    finally:
        node.generate_plots()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()