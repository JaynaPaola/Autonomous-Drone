import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point

class SimpleOdometry(Node):
    def __init__(self):
        super().__init__('odometry')

        # Suscriptor al comando de velocidad
        self.sub_vel = self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)
        # Publicador de la posición estimada
        self.pub_odom = self.create_publisher(Point, '/tello/estimated_pose', 10)

        # Bucle principal de integración (ej. 20 Hz / dt = 0.05)
        self.timer = self.create_timer(0.05, self.publish_odom)

        # Estado inicial (x, y, z)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # Velocidades actuales
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.last_time = self.get_clock().now()


    def vel_callback(self, msg: Twist):
        # Actualizamos las velocidades con lo que se le mandó al dron
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.vz = msg.linear.z


    def publish_odom(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Integración de Euler [x(k+1) = x(k) + dt * x'(k)]
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Publicar la odometría
        pose_msg = Point()
        pose_msg.x = self.x
        pose_msg.y = self.y
        pose_msg.z = self.z

        self.pub_odom.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()