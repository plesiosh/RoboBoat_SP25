#!/bin/bash

tmux new-session -d -s "nav" "bash -c 'python3 src/navigation/buoy_navigation.py; exec bash'"
tmux split-window -h "bash -c 'python3 src/navigation/motortest.py; exec bash'"
tmux split-window -h "bash -c 'python3 src/navigation/auto_nav.py; exec bash'"
tmux split-window -h "bash -c 'ros2 bag record /annotated_image /steering_command'"

tmux attach-session -t "nav"