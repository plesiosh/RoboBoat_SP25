#!/bin/bash

tmux new-session -d -s "percept" "bash -c 'ros2 launch depthai_ros_driver camera.launch.py; exec bash'"
tmux split-window -h "bash -c 'cd src/sensors/launch && ros2 launch msg_MID360_launch.py; exec bash'"
tmux new-window -t percept:1 "bash -c 'ros2 run perception run_detection; exec bash'"
tmux split-window -h "bash -c 'ros2 run perception lidar_camera_projection; exec bash'"
tmux new-window -t percept:2 "bash -c 'python3 src/navigation/fuse_navigation.py; exec bash'"
tmux split-window -h "bash -c 'ros2 bag record /annotated_image /steering_command'"

tmux attach-session -t "percept"