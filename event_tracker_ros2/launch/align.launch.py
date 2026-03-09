"""
Launch file: Alignment tool (interactive GUI)

Usage:
  ros2 launch event_tracker align.launch.py h5_path:=/path/to/data.h5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('h5_path', description='Path to M3ED .h5 file'),
        DeclareLaunchArgument('side', default_value='left',
                              description='Event camera side: left or right'),
        DeclareLaunchArgument('time_sec', default_value='10.0',
                              description='Starting timestamp (seconds)'),

        Node(
            package='event_tracker',
            executable='align_node',
            name='align_node',
            output='screen',
            parameters=[{
                'h5_path':  LaunchConfiguration('h5_path'),
                'side':     LaunchConfiguration('side'),
                'time_sec': LaunchConfiguration('time_sec'),
            }],
        ),
    ])
