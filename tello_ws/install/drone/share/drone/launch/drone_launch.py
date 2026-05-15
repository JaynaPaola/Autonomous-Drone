from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():

    optitrack_node = Node(
        package='mocap_optitrack',
        executable='mocap_node',
        name='optitrack_node',
        output='screen',
    )

    odometry_node = Node(
        package='drone',
        executable='odometry',
        name='odometry_node',
        output='screen',
    )

    # controller arranca 5 segundos después para dar tiempo al takeoff
    controller_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='drone',
                executable='controller',
                name='controller_node',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        optitrack_node,
        odometry_node,
        controller_node,
    ])