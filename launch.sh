#!/bin/bash

tmux new-session -d -s "percept" "ros2 launch depthai_ros_driver camera.launch.py"
tmux split-window -h "ros2 launch livox_ros_driver2 rviz_MID360_launch.py"
tmux new-window -t percept:1 "ros2 run perception run_detection"
tmux split-window -h "ros2 run perception lidar_camera_projection"

tmux attach-session -t "percept"