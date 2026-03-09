"""
Launch file: Calibration node (standalone)

Usage:
  ros2 launch event_tracker calibrate.launch.py h5_path:=/path/to/data.h5

This runs the calibration immediately on startup and saves the result.
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
        DeclareLaunchArgument('num_frames', default_value='20',
                              description='Number of frames to sample'),
        DeclareLaunchArgument('wide', default_value='false',
                              description='Wide search (unknown cameras)'),
        DeclareLaunchArgument('output_path', default_value='calibration.yaml',
                              description='Output YAML path'),
        DeclareLaunchArgument('preview', default_value='false',
                              description='Show visual preview'),

        Node(
            package='event_tracker',
            executable='calibrate_node',
            name='calibrate_node',
            output='screen',
            parameters=[{
                'h5_path':      LaunchConfiguration('h5_path'),
                'side':         LaunchConfiguration('side'),
                'num_frames':   LaunchConfiguration('num_frames'),
                'wide':         LaunchConfiguration('wide'),
                'output_path':  LaunchConfiguration('output_path'),
                'preview':      LaunchConfiguration('preview'),
                'run_on_start': True,
            }],
        ),
    ])
