"""
Launch file: tello_with_optitrack.launch.py
Arranca el nodo optitrack_client (C++) y el tello_controller (Python) juntos.

Uso:
    ros2 launch tello_controller tello_with_optitrack.launch.py
    ros2 launch tello_controller tello_with_optitrack.launch.py target_x_cm:=100.0 target_z_cm:=120.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Argumentos configurables desde CLI --------------------------------
    target_x_arg = DeclareLaunchArgument(
        'target_x_cm', default_value='50.0',
        description='Objetivo X en centímetros'
    )
    target_y_arg = DeclareLaunchArgument(
        'target_y_cm', default_value='0.0',
        description='Objetivo Y en centímetros'
    )
    target_z_arg = DeclareLaunchArgument(
        'target_z_cm', default_value='150.0',
        description='Objetivo Z en centímetros'
    )
    k_gain_arg = DeclareLaunchArgument(
        'k_gain', default_value='1.2',
        description='Ganancia proporcional K (escalar, se aplica a los 3 ejes)'
    )
    max_iter_arg = DeclareLaunchArgument(
        'max_iterations', default_value='200',
        description='Máximo de iteraciones del loop de control'
    )

    # ---- Nodo optitrack_client (C++, paquete separado) --------------------
    # Publica:
    #   optitrack/rigid_body  → geometry_msgs/PoseStamped  (SensorDataQoS)
    #   optitrack/marker      → geometry_msgs/PointStamped (SensorDataQoS)
    optitrack_node = Node(
        package='optitrack_client',
        executable='optitrack_client',
        name='optitrack_client',
        output='screen',
    )

    # ---- Nodo tello_controller (Python) ------------------------------------
    # Se suscribe a optitrack/rigid_body y envía comandos RC al Tello.
    tello_node = Node(
        package='tello_controller',
        executable='tello_controller_node',
        name='tello_controller',
        output='screen',
        parameters=[{
            'target_x_cm':    LaunchConfiguration('target_x_cm'),
            'target_y_cm':    LaunchConfiguration('target_y_cm'),
            'target_z_cm':    LaunchConfiguration('target_z_cm'),
            'k_gain':         LaunchConfiguration('k_gain'),
            'max_iterations': LaunchConfiguration('max_iterations'),
        }]
    )

    return LaunchDescription([
        target_x_arg,
        target_y_arg,
        target_z_arg,
        k_gain_arg,
        max_iter_arg,
        optitrack_node,
        tello_node,
    ])
