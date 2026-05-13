import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')

        self.subscription = self.create_subscription(
            Image,
            '/tello/image_raw',
            self.listener_callback,
            10
        )

        self.publisher_ = self.create_publisher(Image, '/tello/map', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 1. Escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Blur (reduce ruido)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # 3. Bordes
        edges = cv2.Canny(blur, 50, 150)

        # 4. Detectar líneas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=50, maxLineGap=10)

        # 5. Crear "mapa"
        map_img = np.zeros_like(frame)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(map_img, (x1,y1), (x2,y2), (255,255,255), 2)

        # Mostrar
        cv2.imshow("Mapa tipo laberinto", map_img)
        cv2.waitKey(1)

        # Publicar
        map_msg = self.bridge.cv2_to_imgmsg(map_img, encoding='bgr8')
        self.publisher_.publish(map_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
