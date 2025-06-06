import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import time
import cv2
import depthai as dai
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray
from src.read_yaml import extract_configuration
import json

class DaiNode(Node):
    def __init__(self):
        super().__init__('dai_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        self.img_size = [config_file['camera']['image_size']['height'], config_file['camera']['image_size']['width']]

        # Load YOLOv8 model
        self.model_path = config_file['yolo']['dai_blob_path']
        self.json_path = config_file['yolo']['dai_json_path']

        with open(self.json_path) as json_file:
            data = json.load(json_file)
            self.nn_params = data["nn_config"]["NN_specific_metadata"]
            self.n_classes = self.nn_params["classes"]
            self.anchors = self.nn_params["anchors"]
            self.anchor_masks = self.nn_params["anchor_masks"]
            self.labels = data["mappings"]["labels"]

        # Publisher for annotated image
        self.image_pub = self.create_publisher(Image,'dai_node/annotated_image', 10)


        self.bbox_publisher = self.create_publisher(
            Float32MultiArray,
            '/oak/rgb/bounding_boxes',
            10
        )

        # Bridge for CV <-> ROS2 image conversion
        self.bridge = CvBridge()

        # Setup DepthAI pipeline
        self.pipeline = dai.Pipeline()
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(768, 480)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setInterleaved(False)

        nn = self.pipeline.createYoloDetectionNetwork()
        nn.setBlobPath(self.model_path)
        nn.setConfidenceThreshold(0.5)
        nn.setNumClasses(self.n_classes)
        nn.setCoordinateSize(4)
        nn.setIouThreshold(0.5)
        nn.setNumInferenceThreads(2)
        nn.setAnchors(self.anchors)
        nn.setAnchorMasks(self.anchor_masks)
        nn.input.setBlocking(False)

        cam_rgb.preview.link(nn.input)
        cam_rgb.setFps(30)

        xout_video = self.pipeline.createXLinkOut()
        xout_video.setStreamName("video")
        nn.passthrough.link(xout_video.input)

        self.device = dai.Device(self.pipeline)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=1, blocking=True)
        self.detection_queue = self.device.getOutputQueue(name="detections", maxSize=1, blocking=False)
        self.device.setTimesync(True)

        # Create a timer to repeatedly call the inference function
        self.timer = self.create_timer(0.03, self.process_frame)  # ~30 FPS

    def process_frame(self):
        in_video = self.video_queue.get()
        while self.video_queue.has():
            in_video = self.video_queue.get()
        frame = in_video.getCvFrame()

        latency_ms = (dai.Clock.now() - in_video.getTimestamp()).total_seconds() * 1000
        
        in_det = self.detection_queue.get()
        while self.detection_queue.has():
            in_det = self.detection_queue.get()
        detections = in_det.detections

        self.get_logger().info(f"Camera + inference to ROS latency: {latency_ms:.2f} ms")

        # Class name mapping from the model
        allowed_names = ['red_buoy', 'green_buoy', 'yellow_buoy', 'black_buoy']
        # target_classes = [i for i, name in self.model.names.items() if name in allowed_names]

        bbox_array = Float32MultiArray()
        bbox_data = []

        for det in detections:
            label = self.labels[det.label] if det.label < len(self.labels) else f"Class_{det.label}"
            if label not in allowed_names:
                continue

            x1 = float(det.xmin * frame.shape[1])
            y1 = float(det.ymin * frame.shape[0])
            x2 = float(det.xmax * frame.shape[1])
            y2 = float(det.ymax * frame.shape[0])
            conf = float(det.confidence)
            class_id = float(det.label)

            bbox_data.extend([x1, y1, x2, y2, conf, class_id])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {det.confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        bbox_array.data = bbox_data
        # Publish bounding boxes
        self.bbox_publisher.publish(bbox_array)

        # Annotate and publish
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DaiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
