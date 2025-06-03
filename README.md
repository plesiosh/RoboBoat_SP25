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
sh scripts/<script>
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


## Model Training

Note: Follow these instructions on the device you want to train the model on.

Note: Our model is available in this repository in two formats:
- PyTorch (`.pt`): `src/perception/YOLOv8_model/buoy_detection.pt`
- DepthAI Blob (`.blob`): `src/perception/YOLOv8_model/buoy_detection.blob`
  - `buoy_detection.json` is necessary to run the blob on the camera (stored in the same directory)
Follow the rest of the instructions if you want to retrain the model.

### 1. Clone the MHSeals buoy-model repository

[https://github.com/MHSeals/buoy-model](https://github.com/MHSeals/buoy-model)

```bash
git clone https://github.com/MHSeals/buoy-model.git
cd buoy-model
```

### 2. Get the buoy dataset API key

Copy the API key from here: [https://universe.roboflow.com/mhseals/buoys-4naae/model/16/documentation](https://universe.roboflow.com/mhseals/buoys-4naae/model/16/documentation)

Create a new file inside your local `buoy-model` directory called `secret.py`. Insert the following line:

```python
roboflow_api_key = <API_KEY> # replace with the RoboFlow API key of the dataset
```

### 3. Train the model

Open `train.py`. Install any dependencies as necessary (`ultralytics`, `roboflow`).

To train a YOLOv8 model, change this line:
```python
model = YOLO("./runs/detect/v13/weights/last.pt") # refers to a past set of weights that doesn't exist for us
```
To this:
```python
model = YOLO("yolov8n.pt") # this file doesn't need to be on your system
```

Finally, in `model.train()`, change the parameter `resume=True` to `resume=False`. This will start from a general YOLOv8 model rather than trying to use pre-existing weights that don't currently exist on your system.

Start training with `python3 train.py`. This can take several hours, even on a powerful PC. Results will be saved at a location that looks similar to `./runs/detect/v13/weights/best.pt`.
