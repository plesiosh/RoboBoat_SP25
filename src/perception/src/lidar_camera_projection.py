#!/usr/bin/env python3

# Code Adapted from:
# https://github.com/CDonosoK/ros2_camera_lidar_fusion

import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import struct
import math

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer, Cache
from tf2_ros import TransformBroadcaster

from src.read_yaml import extract_configuration, load_extrinsic_matrix, load_camera_calibration
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox3D, BoundingBox3DArray, Detection3D, Detection3DArray

def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> np.ndarray:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)

    field_names = [f.name for f in cloud_msg.fields]
    if not all(k in field_names for k in ('x','y','z')):
        return np.zeros((0,3), dtype=np.float32)

    x_field = next(f for f in cloud_msg.fields if f.name=='x')
    y_field = next(f for f in cloud_msg.fields if f.name=='y')
    z_field = next(f for f in cloud_msg.fields if f.name=='z')

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])

    raw_data = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points[:,0] = raw_data['x']
    points[:,1] = raw_data['y']
    points[:,2] = raw_data['z']

    if skip_rate > 1:
        points = points[::skip_rate]

    return points

class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = config_file['general']['camera_extrinsic_calibration']
        extrinsic_yaml = os.path.join(config_folder, extrinsic_yaml)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        self.T_cam_to_lidar = np.linalg.inv(self.T_lidar_to_cam.T)
        self.img_size = [config_file['camera']['image_size']['height'], config_file['camera']['image_size']['width']]
        
        camera_yaml = config_file['general']['camera_intrinsic_calibration']
        camera_yaml = os.path.join(config_folder, camera_yaml)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")

        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)
        # visualization False turns off camera 
        self.bounding_boxes = []  # will hold latest list of boxes
        self.visualize = config_file['fusion']['visualize']

        if self.visualize:
            self.get_logger().info(f"Subscribing to image topic: {image_topic}")
            self.image_sub = Subscriber(self, Image, image_topic)
            self.ts = ApproximateTimeSynchronizer(
                [self.image_sub, self.lidar_sub],
                queue_size=5,
                slop=0.07
            )
            self.ts.registerCallback(self.sync_callback)
            self.create_subscription(
                Detection2DArray,
                '/oak/rgb/bounding_boxes',
                self.bbox_callback,
                10
            )
        else:
            self.get_logger().info(f"Subscribing to bounding box topic: /oak/rgb/bounding_boxes")
            self.boundingbox_sub = Subscriber(self, Detection2DArray, '/oak/rgb/bounding_boxes')
            self.ts = ApproximateTimeSynchronizer(
                [self.boundingbox_sub, self.lidar_sub],
                queue_size=10,
                slop=0.07
                , allow_headerless=True
            )
            self.ts.registerCallback(self.sync_callback_noVis)
        self.br_tf = TransformBroadcaster(self)

        projected_topic = config_file['camera']['projected_topic']
        if self.visualize:
            self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.pub_centr = self.create_publisher(Float32MultiArray, '/centroids', 1)
        self.bridge = CvBridge()

        self.skip_rate = 1

        
        self.class_names = {11: "Green Buoy", 16: "Red Buoy", 23: "Yellow Buoy"}

    def bbox_callback(self, msg: Detection2DArray):
        self.bounding_boxes = []
        for det in msg.detections:
            # x1, y1, x2, y2, conf, cls
            x1 = det.bbox.center.position.x - det.bbox.size_x / 2
            x2 = det.bbox.center.position.x + det.bbox.size_x / 2
            y1 = det.bbox.center.position.y - det.bbox.size_y / 2
            y2 = det.bbox.center.position.y + det.bbox.size_y / 2

            conf = det.results[0].hypothesis.score
            cls = int(det.results[0].hypothesis.class_id)

            box = [x1, y1, x2, y2, conf, cls]
            self.bounding_boxes.append(box)

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        n_points = xyz_lidar.shape[0]
        if n_points == 0:
            self.get_logger().warn("Empty cloud. Nothing to project.")
            if self.visualize:
                out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                out_msg.header = image_msg.header
                self.pub_image.publish(out_msg)
            return

        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))

        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]

        depth_vals = np.linalg.norm(xyz_lidar, axis=1)

        mask_in_front = (xyz_cam[:, 2] > 0.0)
        xyz_cam_front = xyz_cam[mask_in_front]
        depth_front = depth_vals[mask_in_front]

        n_front = xyz_cam_front.shape[0]
        if n_front == 0:
            self.get_logger().info("No points in front of camera (z>0).")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return
        
        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            xyz_cam_front,
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        image_points = image_points.reshape(-1, 2).astype(np.int64)
        
        lb = np.array([0, 0])
        ub = np.array(self.img_size)
        
        xs, ys = image_points[:, 0], image_points[:, 1]
        points = np.stack((ys, xs), axis=1)
        
        filtered_idx = np.all(np.logical_and(lb <= points, points < ub), axis=1)
        filtered_points = image_points[filtered_idx]
        filtered_depths = depth_front[filtered_idx]

        h, w = self.img_size[:2]
        z_vals = xyz_cam_front[:, 2]
        z_norm = cv2.normalize(z_vals, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        z_norm = 255 - z_norm
        
        z_norm_uint8 = z_norm.astype(np.uint8)
        colors = cv2.applyColorMap(z_norm_uint8, cv2.COLORMAP_JET)
        filtered_colors = colors[filtered_idx]
    
        # for (pt, color) in zip(filtered_points, filtered_colors):
        for (pt, color) in zip(filtered_points, filtered_colors):
            u, v = pt
            u, v = int(u), int(v)
            b, g, r = color.flatten().tolist()
            cv2.circle(cv_image, (u, v), 2, (b, g, r), -1)        

        ys, xs = filtered_points[:, 1], filtered_points[:, 0]
        coords = np.stack((ys, xs))
        
        abs_coords = np.ravel_multi_index(coords, self.img_size)
        depth_lookup = np.bincount(abs_coords, weights=filtered_depths, minlength=self.img_size[0]*self.img_size[1])
        depth_lookup = depth_lookup.reshape(self.img_size)
        depth_lookup[depth_lookup == 0.0] = np.inf

        centroid_array = Float32MultiArray()
        centroid_data = []
        closest_red = 40.0
        closest_green = 40.0
        xydepth = Float32MultiArray()
        xydepth = []        
        closest_red_xydepth = [-1.0, -1.0, -1.0]
        closest_green_xydepth = [-1.0, -1.0, -1.0]


        # Draw bounding boxes with average depth label
        for box in self.bounding_boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            if cls not in self.class_names.keys():
                continue
            
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w-1), min(y2, h-1)

            if len(depth_vals) > 0:
                object_depth = np.min(depth_lookup[y1:y2, x1:x2])
                class_name = self.class_names[cls]
                label = f"{class_name} ({object_depth:.2f}) m"
                
                if class_name == 'Green Buoy':
                    closest_green = min(closest_green, object_depth)
                    if closest_green_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                        closest_green_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                elif class_name == 'Red Buoy':
                    closest_red = min(closest_red, object_depth)
                    if closest_red_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                        closest_red_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
               
                # Return all detections
                # xydepth.extend(p=[(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth, cls])     
                    
            else:
                label = "N/A"

            (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_origin = (x1, y1 - 10 if y1 - 10 > 0 else y1 + font_h + 2)

            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Draw black rectangle background
            cv2.rectangle(cv_image,
                        (text_origin[0], text_origin[1] - font_h - baseline),
                        (text_origin[0] + font_w, text_origin[1] + baseline),
                        (0, 0, 0), thickness=-1)

            # Draw green text
            cv2.putText(cv_image, label, text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            
        # Take only the closest buoy for each color        
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)
        
        centroid_data.extend([closest_red, closest_green])
        # centroid_array.data = centroid_data
        xydepth.extend(closest_red_xydepth)
        xydepth.extend(closest_green_xydepth)
        centroid_array.data = xydepth

        self.pub_centr.publish(centroid_array)


    def sync_callback_noVis(self, boundingbox_msg: Detection2DArray, lidar_msg: PointCloud2):
        # self.get_logger().info("in noVis")
        # cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        bounding_b = ()
        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        n_points = xyz_lidar.shape[0]
        if n_points == 0:
            self.get_logger().warn("Empty cloud. Nothing to project.")
            return


        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))

        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]

        depth_vals = np.linalg.norm(xyz_lidar, axis=1)

        # degree = 45
        # threshold = math.tan(degree/180*math.pi)
        mask_in_front = (xyz_cam[:, 2] > 0.0)
        # mask_in_front = ((xyz_cam[:, 2] > xyz_cam[:, 0]/threshold) & (xyz_cam[:, 2] > -xyz_cam[:, 0]/threshold))
        xyz_cam_front = xyz_cam[mask_in_front]
        depth_front = depth_vals[mask_in_front]

        n_front = xyz_cam_front.shape[0]
        if n_front == 0:
            self.get_logger().info("No points in front of camera (z>0).")
            return
        
        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            xyz_cam_front,
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        image_points = image_points.reshape(-1, 2).astype(np.int64)
        
        lb = np.array([0, 0])
        ub = np.array(self.img_size)
        
        xs, ys = image_points[:, 0], image_points[:, 1]
        points = np.stack((ys, xs), axis=1)
        
        filtered_idx = np.all(np.logical_and(lb <= points, points < ub), axis=1)
        filtered_points = image_points[filtered_idx]
        filtered_depths = depth_front[filtered_idx]

        h, w = self.img_size[:2]
        z_vals = xyz_cam_front[:, 2]
        z_norm = cv2.normalize(z_vals, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        z_norm = 255 - z_norm
        
        z_norm_uint8 = z_norm.astype(np.uint8)
   
        ys, xs = filtered_points[:, 1], filtered_points[:, 0]
        coords = np.stack((ys, xs))
        
        abs_coords = np.ravel_multi_index(coords, self.img_size)
        depth_lookup = np.bincount(abs_coords, weights=filtered_depths, minlength=self.img_size[0]*self.img_size[1])
        depth_lookup = depth_lookup.reshape(self.img_size)
        depth_lookup[depth_lookup == 0.0] = np.inf

        centroid_array = Float32MultiArray()
        centroid_data = []
        closest_red = -1
        closest_green = -1

        xydepth = Float32MultiArray()
        xydepth = []        
        closest_red_xydepth = [-1.0, -1.0, -1.0]
        closest_green_xydepth = [-1.0, -1.0, -1.0]

        for det in boundingbox_msg.detections:
            x1 = int(det.bbox.center.position.x - det.bbox.size_x / 2)
            x2 = int(det.bbox.center.position.x + det.bbox.size_x / 2)
            y1 = int(det.bbox.center.position.y - det.bbox.size_y / 2)
            y2 = int(det.bbox.center.position.y + det.bbox.size_y / 2)
            conf = det.results[0].hypothesis.score
            cls = int(det.results[0].hypothesis.class_id)
            
            if cls not in self.class_names.keys():
                return# continue
            
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w-1), min(y2, h-1)

            if len(depth_vals) > 0:
                object_depth = np.min(depth_lookup[y1:y2, x1:x2])
                class_name = self.class_names[cls]
                label = f"{class_name} ({object_depth:.2f}) m"
                
                if class_name == 'Green Buoy':
                    if closest_green == -1:
                        closest_green = object_depth
                    else:
                        closest_green = min(closest_green, object_depth)
                    if closest_green_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                        closest_green_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                elif class_name == 'Red Buoy':
                    if closest_red == -1:
                        closest_red = object_depth
                    else:
                        closest_red = min(closest_red, object_depth)
                    if closest_red_xydepth[2] < 0 or closest_green_xydepth[2] > object_depth:
                        closest_red_xydepth = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, object_depth]
                    
            else:
                label = "N/A"
        
        centroid_data.extend([closest_red, closest_green])
        # centroid_array.data = centroid_data


        # x, y, depth for closest bouys
        xydepth.extend(closest_red_xydepth)
        xydepth.extend(closest_green_xydepth)
        centroid_array.data = xydepth
        self.pub_centr.publish(centroid_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()