import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from djitellopy import Tello
import cv2

class TelloCameraNode(Node):
    def __init__(self):
        super().__init__('tello_camera_node')

        self.publisher_ = self.create_publisher(Image, '/tello/image_raw', 10)
        self.bridge = CvBridge()

        self.tello = Tello()
        self.tello.connect()
        self.get_logger().info(f'Batería: {self.tello.get_battery()}%')

        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()

        self.timer = self.create_timer(0.05, self.timer_callback)  # ~20 FPS

    def timer_callback(self):
        frame = self.frame_read.frame

        if frame is not None:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TelloCameraNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
