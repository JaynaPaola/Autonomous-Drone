import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from datetime import datetime

def generate_launch_description():
    # Nombre del archivo de rosbag basado en la fecha/hora actual
    bag_name = 'tello_test_' + datetime.now().strftime('%Y_%m%d_%H%M%S')

    # 1. Nodo de Odometría (Integrador simple)
    odom_node = Node(
        package='drone', # Cambia por el nombre de tu paquete
        executable='odometry',
        name='odometry'
    )

    # 2. Nodo del Controlador
    controller_node = Node(
        package='drone', # Cambia por el nombre de tu paquete
        executable='controller',
        name='controller',
    )

    # 3. Proceso para grabar el Rosbag
    # Grabamos solo los tópicos necesarios para PlotJuggler
    rosbag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_name,
             '/tello/estimated_pose',
             '/target_pose',
             '/cmd_vel'],
        output='screen'
    )

    return LaunchDescription([
        odom_node,
        controller_node,
        rosbag_record
    ])