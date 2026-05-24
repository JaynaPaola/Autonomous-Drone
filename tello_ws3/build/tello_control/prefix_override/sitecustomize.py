import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/paolaramos/Github/testROS/Autonomous-Drone/tello_ws3/install/tello_control'
