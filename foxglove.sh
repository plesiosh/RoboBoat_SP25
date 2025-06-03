tmux new-session -d -s "foxglove" "bash -c 'ros2 launch foxglove_bridge foxglove_bridge_launch.xml; exec bash'"
