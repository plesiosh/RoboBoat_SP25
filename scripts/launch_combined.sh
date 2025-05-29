#!/bin/bash

tmux new-session -d -s "percept" "bash -c 'cd src/sensors/launch && ros2 launch msg_MID360_launch.py; exec bash'"
tmux split-window -h "bash -c 'ros2 run perception sick_fusion; exec bash'"
tmux new-window -t percept:1 "bash -c 'python3 src/navigation/fuse_navigation.py; exec bash'"
tmux split-window -h "bash -c 'ros2 bag record /camera/fused_img /steering_command'"

tmux attach-session -t "percept"