import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from geometry_msgs.msg import PointStamped
 
from djitellopy import Tello
import numpy as np
import time
 
 
# -----------------------------
# PARÁMETROS
# -----------------------------
DT = 0.1
RC_LIMIT = 40
MAX_SPEED_CM_S = 40.0
ALPHA = 0.6  # filtro de velocidad estimada
 
# Cuánto confiar en OptiTrack vs odometría (0=solo odom, 1=solo OptiTrack)
FUSION_ALPHA = 0.85
 
# Tiempo (segundos) sin señal OptiTrack para considerar que se perdió
OPTITRACK_TIMEOUT = 0.5
 
 
# -----------------------------
# ODOMETRÍA PURA
# -----------------------------
def odometry_step(q, v_est, rc_cmd, dt):
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
# NODO
# -----------------------------
class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
 
        # --- Publicadores ---
        self.pose_pub = self.create_publisher(
            Float32MultiArray, '/estimated_pose', 10
        )
 
        # --- Suscriptores ---
        self.rc_sub = self.create_subscription(
            Int32MultiArray,
            '/rc_command',
            self.rc_callback,
            10
        )
        self.optitrack_sub = self.create_subscription(
            PointStamped,
            '/optitrack/rigid_body',
            self.optitrack_callback,
            10
        )
 
        # --- Estado interno ---
        self.q = np.array([0.0, 0.0, 0.0])       # posición estimada fusionada [cm]
        self.v_est = np.array([0.0, 0.0, 0.0])    # velocidad filtrada
        self.last_rc = [0, 0, 0, 0]
 
        # OptiTrack
        self.optitrack_pos = None                  # última posición recibida [cm]
        self.last_optitrack_time = None            # timestamp de la última lectura
        self.optitrack_available = False
 
        # --- Conexión con el dron ---
        self.drone = Tello()
        try:
            self.drone.connect()
            self.get_logger().info(f"Batería: {self.drone.get_battery()}%")
        except Exception as e:
            self.get_logger().error(f"Error de conexión con el dron: {e}")
            raise
 
        self.drone.takeoff()
        time.sleep(2)
        self.get_logger().info('OdometryNode iniciado. Dron en vuelo.')
 
        # --- Timer de actualización a 10 Hz ---
        self.timer = self.create_timer(DT, self.update_loop)
 
    # --------------------------------------------------
    def rc_callback(self, msg):
        """Recibe comandos RC desde trajectory_node y los envía al dron."""
        self.last_rc = list(msg.data)
        self.drone.send_rc_control(*self.last_rc)
 
    # --------------------------------------------------
    def optitrack_callback(self, msg: PointStamped):
        """
        Recibe posición real desde OptiTrack.
        Convierte de metros a cm y guarda timestamp.
        """
        self.optitrack_pos = np.array([
            msg.point.x * 100.0,
            msg.point.y * 100.0,
            msg.point.z * 100.0
        ])
        self.last_optitrack_time = self.get_clock().now()
        self.optitrack_available = True
 
    # --------------------------------------------------
    def _check_optitrack_timeout(self):
        """Marca OptiTrack como no disponible si lleva más de OPTITRACK_TIMEOUT sin datos."""
        if self.last_optitrack_time is None:
            self.optitrack_available = False
            return
 
        elapsed = (
            self.get_clock().now() - self.last_optitrack_time
        ).nanoseconds / 1e9
 
        if elapsed > OPTITRACK_TIMEOUT:
            if self.optitrack_available:
                self.get_logger().warn(
                    'OptiTrack: señal perdida. Usando solo odometría.'
                )
            self.optitrack_available = False
 
    # --------------------------------------------------
    def update_loop(self):
        """
        Cada DT segundos:
        1. Avanza la odometría con los últimos RC.
        2. Si OptiTrack está disponible, fusiona su posición con la odom.
        3. Publica la posición estimada fusionada.
        """
        self._check_optitrack_timeout()
 
        # --- Paso de odometría ---
        self.q, self.v_est = odometry_step(
            self.q, self.v_est, self.last_rc, DT
        )
 
        # --- Fusión con OptiTrack ---
        if self.optitrack_available and self.optitrack_pos is not None:
            # Combinación convexa: confiamos más en OptiTrack que en la odom
            self.q = FUSION_ALPHA * self.optitrack_pos + (1 - FUSION_ALPHA) * self.q
 
            # Corregimos también v_est para que sea coherente con el salto
            # (evita que la inercia arrastre la estimación hacia atrás)
            error_corr = self.optitrack_pos - self.q
            self.v_est += error_corr / DT * (1 - FUSION_ALPHA)
 
            modo = "FUSIÓN"
        else:
            modo = "ODOM"
 
        # --- Publicar posición estimada ---
        out = Float32MultiArray()
        out.data = self.q.tolist()
        self.pose_pub.publish(out)
 
        self.get_logger().debug(
            f"[{modo}] Pos [cm]: x={self.q[0]:.1f}, "
            f"y={self.q[1]:.1f}, z={self.q[2]:.1f}"
        )
 
    # --------------------------------------------------
    def destroy_node(self):
        """Aterrizaje seguro al cerrar el nodo."""
        self.get_logger().info('Aterrizando...')
        self.drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        self.drone.land()
        self.drone.end()
        super().destroy_node()
 
 
# -----------------------------
# MAIN
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por el usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()