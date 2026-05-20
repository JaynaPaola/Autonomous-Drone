from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():

    optitrack_node = Node(
        package='optitrack_client',
        executable='optitrack_client',
        name='optitrack_node',
        output='screen',
    )

    odometry_node = Node(
        package='drone',
        executable='odometry',
        name='odometry_node',
        output='screen',
    )

    plotter_node = Node(
        package='drone',
        executable='plotter',
        name='plotter_node',
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
        plotter_node,
        controller_node,
    ])