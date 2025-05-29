# RoboBoat SP25


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/plesiosh/RoboBoat_SP25.git
cd RoboBoat_SP25
````

### ~~2. Build the Docker Image (Skip this for now)~~

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
- Make sure you set your container_name within `run_docker.sh`


### 4. Launch the Navigation

```bash
sh scripts/launch_script.sh
sh scripts/launch_motors.sh
```

- For debugging purposes, do not run `scripts/launch_motors.sh`
- Example `launch_script.sh` are:
   
| Command              | Published Topics                                                                 | Fusion             |Description                                                                                     |
|----------------------|----------------------------------------------------------------------------------|--------------------|-------------------------------------------------------------------------------------------------|
| `launch_base.sh`     | `/annotated_image`                                                               |                    |  Pipeline based on previous system (RoboBoat 2024) with .pt instead of blob file. |
| `launch_old.sh`      | `/oak/rgb/projected`<br>`/centroids`                                             | ✅                 |  Initial Prototype. Uses ROS2 Image for processing frames.                                       |
| `launch_combined.sh` | `/camera/fused_img`<br>`/centroids`                                              | ✅                 |  Pipeline based on sick-lidar. Camera/YOLO/Fusion runs in single node.                          |
| `launch_modular.sh`  | `/dai_node/annotated_image`<br>`/oak/rgb/projected`<br>`/centroids`              | ✅                 |  Pipeline with DepthAI-inference node. Camera/YOLO and Fusion node are separated.               |

- Note. If `fusion.visualize` is set to False in `config/general_configuration.yaml`, visualization topics such as `/oak/rgb/projected` or `camera/fused_img` may not be published.
