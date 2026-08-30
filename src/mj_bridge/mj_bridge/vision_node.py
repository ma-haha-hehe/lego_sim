#!/usr/bin/env python3
"""ROS service that turns the latest fixed-camera RGB-D frame into part poses."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .vision_pipeline import Detection, make_backend


class LegoVisionNode(Node):
    """Run perception on demand after the arm reaches its observation pose."""

    def __init__(self):
        super().__init__("lego_vision")
        self.declare_parameter("backend", "foundationpose")
        self.declare_parameter("foundationpose_root", os.environ.get("FOUNDATIONPOSE_DIR", ""))
        self.declare_parameter("detector_model", "IDEA-Research/grounding-dino-tiny")
        self.declare_parameter("sam_model", "facebook/sam-vit-huge")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("detection_threshold", 0.20)
        self.declare_parameter("register_iterations", 5)
        self.declare_parameter("camera_position", [0.49, -0.15, 1.2])
        self.declare_parameter("source_x", [0.30, 0.68])
        self.declare_parameter("source_y", [-0.42, 0.12])
        self.declare_parameter("source_z", [0.035, 0.075])
        self.declare_parameter("capture_timeout_s", 30.0)
        self.declare_parameter("debug_dir", "/tmp/lego_bench/vision")

        backend_name = self.get_parameter("backend").value
        self.backend = make_backend(
            backend_name,
            foundationpose_root=self.get_parameter("foundationpose_root").value,
            detector_model=self.get_parameter("detector_model").value,
            sam_model=self.get_parameter("sam_model").value,
            device=self.get_parameter("device").value,
            detection_threshold=self.get_parameter("detection_threshold").value,
            register_iterations=self.get_parameter("register_iterations").value,
        )
        self.camera_position = np.asarray(
            self.get_parameter("camera_position").value, dtype=float
        )
        self.source_x = tuple(float(value) for value in self.get_parameter("source_x").value)
        self.source_y = tuple(float(value) for value in self.get_parameter("source_y").value)
        self.source_z = tuple(float(value) for value in self.get_parameter("source_z").value)
        self.capture_timeout_s = float(self.get_parameter("capture_timeout_s").value)
        self.debug_dir = Path(self.get_parameter("debug_dir").value)
        self.lock = threading.Lock()
        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.intrinsics: Optional[np.ndarray] = None
        self.rgb_stamp_ns = 0
        self.depth_stamp_ns = 0
        self.goal: Optional[dict] = None
        self.callback_group = ReentrantCallbackGroup()

        self.create_subscription(
            Image, "/camera/color/image_raw", self.on_rgb, 5,
            callback_group=self.callback_group
        )
        self.create_subscription(
            Image, "/camera/depth/image_raw", self.on_depth, 5,
            callback_group=self.callback_group
        )
        self.create_subscription(
            CameraInfo, "/camera/camera_info", self.on_info, 5,
            callback_group=self.callback_group
        )
        self.create_subscription(
            String, "/lego_bench/goal", self.on_goal, 5,
            callback_group=self.callback_group
        )
        self.publisher = self.create_publisher(String, "/lego_bench/vision_detections", 10)
        self.capture_client = self.create_client(
            Trigger, "/mj_bridge/capture_rgbd", callback_group=self.callback_group
        )
        self.service = self.create_service(
            Trigger, "/lego_vision/detect", self.on_detect,
            callback_group=self.callback_group
        )
        self.get_logger().info(
            f"Vision service ready: backend={self.backend.name}, camera=fixed overhead RGB-D"
        )

    def on_rgb(self, message: Image):
        if message.encoding not in {"rgb8", "bgr8"}:
            return
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.width, 3
        )
        if message.encoding == "bgr8":
            image = image[:, :, ::-1]
        with self.lock:
            self.rgb = image.copy()
            self.rgb_stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )

    def on_depth(self, message: Image):
        if message.encoding != "32FC1":
            return
        image = np.frombuffer(message.data, dtype=np.float32).reshape(
            message.height, message.width
        )
        with self.lock:
            self.depth = image.copy()
            self.depth_stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )

    def on_info(self, message: CameraInfo):
        with self.lock:
            self.intrinsics = np.asarray(message.k, dtype=float).reshape(3, 3)

    def on_goal(self, message: String):
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(f"Invalid benchmark goal JSON: {error}")
            return
        with self.lock:
            self.goal = value

    def snapshot(self):
        with self.lock:
            if any(value is None for value in (self.rgb, self.depth, self.intrinsics, self.goal)):
                return None
            return (
                self.rgb.copy(),
                self.depth.copy(),
                self.intrinsics.copy(),
                json.loads(json.dumps(self.goal)),
                min(self.rgb_stamp_ns, self.depth_stamp_ns),
            )

    def source_detections(self, detections: list[Detection]) -> list[Detection]:
        return [
            item
            for item in detections
            if self.source_x[0] <= item.position[0] <= self.source_x[1]
            and self.source_y[0] <= item.position[1] <= self.source_y[1]
            and self.source_z[0] <= item.position[2] <= self.source_z[1]
        ]

    def save_debug(self, rgb: np.ndarray, detections: list[Detection], stamp_ns: int) -> str:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for item in detections:
            x1, y1, x2, y2 = item.box_xyxy
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = (
                f"{item.color} {item.part_type} "
                f"({item.position[0]:.3f},{item.position[1]:.3f})"
            )
            cv2.putText(
                image,
                text,
                (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        path = self.debug_dir / f"detection-{stamp_ns}.png"
        cv2.imwrite(str(path), image)
        latest = self.debug_dir / "latest.png"
        cv2.imwrite(str(latest), image)
        return str(path)

    def save_payload(self, payload: dict, stamp_ns: int):
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        (self.debug_dir / f"detection-{stamp_ns}.json").write_text(text, encoding="utf-8")
        (self.debug_dir / "latest.json").write_text(text, encoding="utf-8")

    def on_detect(self, request, response):
        del request
        deadline = time.monotonic() + self.capture_timeout_s
        requested_stamp_ns = self.get_clock().now().nanoseconds
        if not self.capture_client.wait_for_service(timeout_sec=2.0):
            response.success = False
            response.message = json.dumps({"error": "RGB-D capture service is unavailable"})
            return response
        future = self.capture_client.call_async(Trigger.Request())
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done() or not future.result().success:
            response.success = False
            message = "RGB-D capture request timed out"
            if future.done():
                message = future.result().message
            response.message = json.dumps({"error": message})
            return response

        snapshot = self.snapshot()
        while (snapshot is None or snapshot[-1] <= requested_stamp_ns) and \
                time.monotonic() < deadline:
            time.sleep(0.05)
            snapshot = self.snapshot()
        if snapshot is None or snapshot[-1] <= requested_stamp_ns:
            response.success = False
            response.message = json.dumps(
                {"error": "A fresh RGB-D frame did not arrive after the capture request"}
            )
            return response
        rgb, depth, intrinsics, goal, stamp_ns = snapshot
        requested = goal.get("target_blocks", [])
        started = time.perf_counter()
        try:
            detections = self.backend.infer(
                rgb, depth, intrinsics, self.camera_position, requested
            )
            detections = self.source_detections(detections)
        except Exception as error:
            self.get_logger().error(f"Vision inference failed: {error}")
            response.success = False
            response.message = json.dumps(
                {"error": str(error), "backend": self.backend.name}, ensure_ascii=False
            )
            return response
        elapsed = time.perf_counter() - started
        debug_image = self.save_debug(rgb, detections, stamp_ns)
        payload = {
            "episode_id": goal.get("episode_id"),
            "frame_stamp_ns": stamp_ns,
            "backend": self.backend.name,
            "camera": "fixed_overhead_rgbd",
            "elapsed_s": round(elapsed, 4),
            "debug_image": debug_image,
            "detections": [item.as_dict(index) for index, item in enumerate(detections)],
        }
        self.save_payload(payload, stamp_ns)
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.publisher.publish(message)
        response.success = bool(detections)
        response.message = message.data
        self.get_logger().info(
            f"Perception completed in {elapsed:.3f}s: {len(detections)} source part(s), "
            f"backend={self.backend.name}"
        )
        return response


def main():
    rclpy.init()
    node = LegoVisionNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
