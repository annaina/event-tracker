"""
Launch file: H5 event publisher + tracker node

Usage:
  ros2 launch event_tracker tracker.launch.py h5_path:=/path/to/data.h5

Parameters can be overridden on the command line, e.g.:
  ros2 launch event_tracker tracker.launch.py \
      h5_path:=/data/falcon.h5 side:=left gui:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ── Arguments ────────────────────────────────────────────
        DeclareLaunchArgument('h5_path', description='Path to M3ED .h5 file'),
        DeclareLaunchArgument('side', default_value='left',
                              description='Event camera side: left or right'),
        DeclareLaunchArgument('start_sec', default_value='0.0',
                              description='Start time in seconds'),
        DeclareLaunchArgument('duration_sec', default_value='0.0',
                              description='Duration (0 = full recording)'),
        DeclareLaunchArgument('event_window_ms', default_value='10.0',
                              description='Event accumulation window (ms)'),
        DeclareLaunchArgument('calib_path', default_value='',
                              description='Path to calibration YAML'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show OpenCV GUI'),
        DeclareLaunchArgument('show_overlay', default_value='true',
                              description='Show event overlay on image'),
        DeclareLaunchArgument('show_trail', default_value='true',
                              description='Show tracking trail'),

        # ── H5 Event Publisher ───────────────────────────────────
        Node(
            package='event_tracker',
            executable='h5_event_publisher',
            name='h5_event_publisher',
            output='screen',
            parameters=[{
                'h5_path':       LaunchConfiguration('h5_path'),
                'side':          LaunchConfiguration('side'),
                'start_sec':     LaunchConfiguration('start_sec'),
                'duration_sec':  LaunchConfiguration('duration_sec'),
                'event_window_ms': LaunchConfiguration('event_window_ms'),
            }],
        ),

        # ── Tracker Node ────────────────────────────────────────
        Node(
            package='event_tracker',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[{
                'calib_path':      LaunchConfiguration('calib_path'),
                'event_window_ms': LaunchConfiguration('event_window_ms'),
                'show_overlay':    LaunchConfiguration('show_overlay'),
                'show_trail':      LaunchConfiguration('show_trail'),
                'gui':             LaunchConfiguration('gui'),
            }],
            remappings=[
                ('~/image_raw', '/h5_event_publisher/image_raw'),
                ('~/events',    '/h5_event_publisher/events'),
            ],
        ),
    ])
