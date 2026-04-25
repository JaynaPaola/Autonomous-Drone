import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/paolaramos/Documents/Autonomous-Drone/tello_ws/install/tello_odometry'
