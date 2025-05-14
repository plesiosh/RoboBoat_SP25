#!/bin/bash

tmux new-session -d -s "perception" "ros2 launch depthai_ros_driver camera.launch.py"
tmux split-window -v "ros2 launch livox_ros_driver2 rviz_MID360_launch.py"
tmux new-window -t "perception:1" -n livox "ros2 run perception yolov8_inference_node"
tmux split-window -v "ros2 run perception lidar_camera_projection"

tmux attach-session -t "perception"