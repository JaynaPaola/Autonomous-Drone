"""
tello_with_optitrack.launch.py
Lanza optitrack_client (C++) + tello_controller (Python) juntos.

Uso:
    ros2 launch tello_controller tello_with_optitrack.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    optitrack_node = Node(
        package='optitrack_client',
        executable='optitrack_client',
        name='optitrack_client',
        output='screen',
    )

    tello_node = Node(
        package='tello_control',
        executable='tello_control_node',
        name='tello_control',
        output='screen',
    )

    return LaunchDescription([
        optitrack_node,
        tello_node,
    ])
