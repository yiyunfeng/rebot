"""Gazebo RGB-D tabletop HSV detector for the wrist-mounted DaBai camera.

腕部 DaBai RGB-D 相机的 Gazebo 桌面目标检测节点。

仿真部分只使用 HSV 阈值检测 Gazebo 里的绿色方块，并发布
/dabai_camera/target_pose。真机视觉和 AI 模型流程保持在 rebot_grasp /
hardware 相关脚本中，不在本节点里加载。

单独运行：
    cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # 通常由 gazebo_camera.launch.py mode:=vision/full/grasp 自动启动。
    # 若只调试检测节点，可在已有相机话题时单独运行：
    ros2 run rebotarm_gazebo camera_object_detector

配置：
    修改 rebotarm_gazebo/config/camera_object_detector.yaml 的 HSV 阈值和深度范围。
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from rebotarm_gazebo.camera_detection_common import ObjectDetection
from rebotarm_gazebo.camera_detection_hsv import HsvColorDetector


class CameraObjectDetector(Node):
    """Detect a single colored tabletop object from RGB-D frames.

    从 RGB-D 帧中检测单个彩色桌面物体，并发布目标中心位姿。

    输入：
        - color image: Gazebo/真实相机发布的 BGR/RGB 彩色图
        - depth image: 与彩色图近似对齐的深度图，单位会在回调中统一为米
        - camera info: 相机内参，优先用于像素反投影
    输出：
        - /dabai_camera/target_pose: 相机光学坐标系下的目标中心 PoseStamped
        - /dabai_camera/debug_image: 叠加检测轮廓后的调试图
    """

    def __init__(self) -> None:
        super().__init__("camera_object_detector")

        # Topic 参数保持可配置，便于同一节点在 Gazebo 仿真和真实 Orbbec 驱动之间复用。
        self.declare_parameter("color_image_topic", "/dabai_camera/image")
        self.declare_parameter("depth_image_topic", "/dabai_camera/depth_image")
        self.declare_parameter("camera_info_topic", "/dabai_camera/camera_info")
        self.declare_parameter("target_pose_topic", "/dabai_camera/target_pose")
        self.declare_parameter("debug_image_topic", "/dabai_camera/debug_image")
        # 目标位姿发布坐标系。Gazebo RGB-D 图像的 header.frame_id 可能是
        # rebotarm/gripper_link/dabai_camera 这种 Gazebo scoped sensor 名称，
        # 它不是 robot_state_publisher 发布的 TF frame。默认使用这里配置的
        # camera_frame，保证 camera_grasp_pipeline 可以查到 base_link <- camera TF。
        self.declare_parameter("camera_frame", "dabai_camera_optical_frame")
        # 只有当图像 header.frame_id 已经存在于 TF 树时才建议打开。
        # 默认 false，避免仿真图像 frame 名称污染抓取位姿坐标系。
        self.declare_parameter("use_image_frame_id", False)
        self.declare_parameter("horizontal_fov", 1.047)  # 没有 CameraInfo 时的 FOV 兜底值
        self.declare_parameter("min_area", 500.0)        # 过滤小噪声轮廓
        self.declare_parameter("min_depth", 0.05)        # 深度有效范围，避免近裁剪噪声
        self.declare_parameter("max_depth", 2.0)         # 过滤远处背景或无效深度
        self.declare_parameter("hsv_lower", [35, 40, 40])   # 默认检测绿色物体
        self.declare_parameter("hsv_upper", [90, 255, 255])  # 需要抓其它颜色时优先调这两个 HSV 参数
        # 仿真检测固定为 HSV。保留 detector_backend 参数只是为了 YAML 和日志可读，
        # 不再加载其它模型后端，避免 AI 依赖和 ROS Python 环境相互污染。
        self.declare_parameter("detector_backend", "hsv")
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("inference_stride", 1)

        self._bridge = CvBridge()
        # 深度图和 CameraInfo 由独立 topic 到达，先缓存最新值，等 RGB 帧触发检测。
        self._latest_depth: np.ndarray | None = None
        self._latest_info: CameraInfo | None = None
        self._frame_count = 0
        self._detector = HsvColorDetector(self._param_value)

        self._pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("target_pose_topic").value),
            10,
        )
        self._debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            10,
        )

        # 图像类 topic 使用 sensor_data QoS，匹配相机/仿真桥接的高频、允许丢帧语义。
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_image_topic").value),
            self._depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("color_image_topic").value),
            self._color_cb,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "HSV detector topics: "
            f"color={self.get_parameter('color_image_topic').value}, "
            f"depth={self.get_parameter('depth_image_topic').value}, "
            f"info={self.get_parameter('camera_info_topic').value}, "
            f"pose={self.get_parameter('target_pose_topic').value}, "
            f"debug={self.get_parameter('debug_image_topic').value}"
        )

    def _depth_cb(self, msg: Image) -> None:
        """缓存最新深度图，并统一转换为米单位 float32。"""
        try:
            # passthrough 保留原始编码，避免 cv_bridge 自动缩放深度值。
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"depth conversion failed: {exc}")
            return
        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            # 常见真实深度相机使用 uint16 毫米；Gazebo 通常直接给 float 米。
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        self._latest_depth = depth

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """缓存最新相机内参，供像素反投影使用。"""
        self._latest_info = msg

    def _color_cb(self, msg: Image) -> None:
        """RGB 图像到达时执行一次检测，并发布目标位姿/调试图。"""
        self._frame_count += 1
        stride = max(1, int(self.get_parameter("inference_stride").value))
        if self._frame_count % stride != 0:
            return
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError:
            # Gazebo bridge 的编码偶尔不是 bgr8，这里保留一个 RGB fallback。
            raw = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.ndim == 3 else raw

        if self._latest_depth is None:
            # 没有深度时先发布 RGB 调试图，方便确认真机相机链路已经通了。
            self._publish_debug(msg, bgr, None)
            return

        depth = self._latest_depth
        if depth.shape[:2] != bgr.shape[:2]:
            # 真机上 color/depth 分辨率可能不同；这里先用最近邻缩放到 color 尺寸。
            # 前提是驱动或 Gazebo 已完成 D2C 对齐；若未对齐，需要先在相机驱动侧打开 align。
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        detection = self._detector.detect(bgr, depth)
        if detection is None:
            # 即使没检测到目标，也发布原图，RViz 里能确认相机链路是否正常。
            self._publish_debug(msg, bgr, None)
            return

        # HSV 后端只提供像素中心和深度，需要反投影成相机坐标系 3D 点。
        if detection.position_xyz is not None:
            x, y, z = detection.position_xyz
        else:
            x, y, z = self._pixel_to_camera_point(
                detection.u,
                detection.v,
                detection.depth,
                bgr.shape[1],
                bgr.shape[0],
            )
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        # 检测结果的 x/y/z 是按当前相机内参反投影得到的相机坐标系点。
        # 发布时必须使用 TF 树里真实存在的相机 frame；否则下游 lookup_transform
        # 会报 source_frame does not exist，抓取流程无法把目标转换到 base_link。
        configured_frame = str(self.get_parameter("camera_frame").value)
        if bool(self.get_parameter("use_image_frame_id").value) and msg.header.frame_id:
            pose.header.frame_id = msg.header.frame_id
        else:
            pose.header.frame_id = configured_frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        # 仿真 HSV 只发布目标点位置，姿态由抓取节点按“末端朝下”策略生成。
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)
        self._publish_debug(msg, bgr, detection)

    def _param_value(self, name: str):
        """读取 ROS 参数值，传给算法模块使用。"""
        return self.get_parameter(name).value

    def _pixel_to_camera_point(
        self,
        u: int,
        v: int,
        depth: float,
        width: int,
        height: int,
    ) -> tuple[float, float, float]:
        """针孔模型反投影：像素坐标 + 深度 → 相机坐标系 3D 点。"""
        if self._latest_info is not None and self._latest_info.k[0] > 0.0:
            # 优先使用相机发布的内参，仿真和真机都能统一处理。
            fx = self._latest_info.k[0]
            fy = self._latest_info.k[4]
            cx = self._latest_info.k[2]
            cy = self._latest_info.k[5]
        else:
            # CameraInfo 不可用时，用水平 FOV 估计 fx/fy，保证仿真初期能先跑通。
            fov = float(self.get_parameter("horizontal_fov").value)
            fx = width / (2.0 * math.tan(fov / 2.0))
            fy = fx
            cx = width / 2.0
            cy = height / 2.0
        # ROS 光学坐标系约定：x 向右、y 向下、z 向前；这里输出的点遵循该约定。
        x = (float(u) - cx) * depth / fx
        y = (float(v) - cy) * depth / fy
        return x, y, depth

    def _publish_debug(
        self,
        source: Image,
        bgr: np.ndarray,
        detection: ObjectDetection | None,
    ) -> None:
        """发布带轮廓框的调试图，方便在 RViz Image 面板里看检测效果。"""
        debug = bgr.copy()
        if detection is not None:
            cv2.drawContours(debug, [detection.contour], -1, (0, 255, 0), 2)
            cv2.circle(debug, (detection.u, detection.v), 4, (0, 0, 255), -1)
            text = (
                f"{detection.backend}:{detection.label} "
                f"{detection.confidence:.2f} z={detection.depth:.3f} "
                f"a={detection.angle_deg:.1f}"
            )
            cv2.putText(
                debug,
                text,
                (max(0, detection.u - 80), max(20, detection.v - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        # 保留原图 header，让调试图和原始图像在时间戳/坐标系上对齐。
        msg.header = source.header
        self._debug_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraObjectDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
