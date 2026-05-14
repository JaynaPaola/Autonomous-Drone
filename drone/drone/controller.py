import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, PointStamped
from djitellopy import Tello

class TelloController(Node):
    def __init__(self):
        super().__init__('controller')

        # Subscriptions
        self.sub_optitrack = self.create_subscription(PointStamped, '/optitrack/rigid_body', self.optitrack_callback, 10)

        # Publishers
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.err_x_pub = self.create_publisher(Float32, '/err_x', 10)
        self.err_y_pub = self.create_publisher(Float32, '/err_y', 10)
        self.err_z_pub = self.create_publisher(Float32, '/err_z', 10)

        # Loop control timer
        self.timer = self.create_timer(0.0069, self.control_loop)

        self.current_pose = PointStamped()
        self.target_pose = PointStamped()
        self.err_x = Float32()
        self.err_y = Float32()
        self.err_z = Float32()
        self.target_pose.point.x = 0.5
        self.target_pose.point.y = 0.5
        self.target_pose.point.z = 0.5

        self.target_received = False
        self.is_landing = False

        # Gains
        self.kx, self.ky, self.kz = 0.75, 0.75, 0.75

        # Initialize drone
        self.tello = Tello()
        self.tello.connect()
        self.get_logger().info(f"Tello battery: {self.tello.get_battery()}%")
        self.tello.takeoff()


    def optitrack_callback(self, msg: PointStamped):
        self.current_pose.point.x = msg.point.x
        self.current_pose.point.y = msg.point.y
        self.current_pose.point.z = msg.point.z


    def target_callback(self, msg: PointStamped):
        self.target_pose.point.x = msg.point.x
        self.target_pose.point.y = msg.point.y
        self.target_pose.point.z = msg.point.z
        self.target_received = True
        self.get_logger().info(f"Target received: \
                                X={msg.point.x:.2f}, \
                                Y={msg.point.y:.2f}, \
                                Z={msg.point.z:.2f}")


    def control_loop(self):
        if self.is_landing:
            return

        # 1. Position error
        ex = self.target_pose.point.x - self.current_pose.point.x
        ey = self.target_pose.point.y - self.current_pose.point.y
        ez = self.target_pose.point.z - self.current_pose.point.z

        self.err_x.data = ex
        self.err_y.data = ey
        self.err_z.data = ez

        self.err_x_pub.publish(self.err_x)
        self.err_y_pub.publish(self.err_y)
        self.err_z_pub.publish(self.err_z)

        if abs(ex) < 0.05 and abs(ey) < 0.05 and abs(ez) < 0.05:
            self.get_logger().info("Target reached! Landing...")
            self.execute_landing()
            return

        # 2. Control Law -> u = K * e
        ux = self.kx * ex
        uy = self.ky * ey
        uz = self.kz * ez

        # 3. Map to commands for Tello
        scale = 80
        rc_fb = int(max(min(ux * scale, 100), -100))    # Forward/Backward (X)
        rc_lr = int(max(min(-uy * scale, 100), -100))   # Left/Right (Y) invertido
        rc_ud = int(max(min(uz * scale, 100), -100))    # Up/Down (Z)

        # 4. Sent to the drone
        self.tello.send_rc_control(rc_lr, rc_fb, rc_ud, 0)

        # 5. Publish Wtsit message
        t_msg = Twist()
        t_msg.linear.x, t_msg.linear.y, t_msg.linear.z = float(ux), float(uy), float(uz)
        self.pub_vel.publish(t_msg)


    def execute_landing(self):
        # Stop drone and start landing sequence
        self.is_landing = True
        self.target_received = False
        self.tello.send_rc_control(0, 0, 0, 0)

        zero_vel = Twist()
        self.pub_vel.publish(zero_vel)

        self.tello.land()
        self.get_logger().info("Drone in land.")


    def stop_all(self):
        # Shutdown the drone
        if self.tello:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tello.land()
            self.tello.end()


def main(args=None):
    rclpy.init(args=args)
    node = TelloController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupt.")
    finally:
        node.stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()