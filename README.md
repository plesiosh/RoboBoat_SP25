# RoboBoat SP25


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/plesiosh/RoboBoat_SP25.git
cd RoboBoat_SP25
````

### 2. Build the Docker Image

```bash
docker build -t roboboat-fusion .
```

> **Note:** This image uses `humble-gpu-ubuntu22.04:latest` as its base. (To be updated)

* The Dockerfile includes setup for:

  * Livox SDK
  * Livox ROS 2 Drivers

- Modify the Livox configuration JSON files in the `config/MID360_config.json` directory as needed.

### 3. Run the Docker Container & Install custom package

```bash
sh run_docker.sh
colcon build
source install/setup.bash
```

This script sets up and enters the development container.

### 4. Launch the Sensor Fusion Pipeline

```bash
sh launch.sh
```

This script launches tmux terminals including ROS2 nodes below:

* Camera node
* Livox LiDAR node
* YOLOv8 object detection node
* LiDAR-camera fusion node

