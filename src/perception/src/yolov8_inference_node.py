import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloV8InferenceNode(Node):
    def __init__(self):
        super().__init__('yolov8_inference_node')
        
        # Subscribe to raw RGB camera frames
        self.subscription = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publisher for annotated frames
        self.publisher = self.create_publisher(
            Image,
            '/oak/rgb/image_annotated',
            10
        )

        # Bounding box publisher
        self.bbox_publisher = self.create_publisher(
            Float32MultiArray,
            '/oak/rgb/bounding_boxes',
            10
        )

        # Convert between ROS Image messages and OpenCV
        self.bridge = CvBridge()

        # Load default YOLOv8 model (nano variant)
        self.get_logger().info('Loading YOLOv8n model...')
        self.model = YOLO('/workspace/oak_ws/src/yolov8_inference/yolov8_inference/buoy_detection.pt')  # or 'yolov8s.pt', etc.
        self.model.to('cuda') # need jetson-compatible pytorch

    def image_callback(self, msg):
        try:
            # Convert ROS image (rgb8) to OpenCV BGR format
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # Run inference
            results = self.model(bgr_image)[0]

            # Draw predictions on the frame
            annotated_frame = results.plot()

            bbox_array = Float32MultiArray()
            bbox_data = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist()) 
                conf = float(box.conf[0])
                class_id = float(box.cls[0])  
                bbox_data.extend([x1, y1, x2, y2, conf, class_id])

            bbox_array.data = bbox_data
            # Publish bounding boxes
            self.bbox_publisher.publish(bbox_array)

            # Convert back to ROS image (bgr8) and publish
            output_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            output_msg.header = msg.header
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

