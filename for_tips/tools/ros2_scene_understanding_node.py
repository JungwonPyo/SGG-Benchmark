from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SGG_ROOT = ROOT.parent
if str(SGG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGG_ROOT))

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

import torch
from torch_geometric.loader import DataLoader

from demo.demo_model import SGG_Model
from demo.onnx_model import SGG_ONNX_Model
from for_tips.situation_gnn.situation_gnn.dataset import build_graphs
from for_tips.situation_gnn.situation_gnn.model import SceneSituationGNN
from scene_understanding_msgs.msg import (
    BoundingBox3D,
    CameraModel,
    DetectedObject3D,
    SceneContext,
    SceneRelation,
    SituationHypothesis,
)

import threading


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class SceneUnderstandingNode(Node):
    def __init__(self, args):
        super().__init__("scene_understanding_node")
        self.args = args
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        self.last_pub_time = 0.0

        self.get_logger().info(f"Using device: {self.device}")

        self.is_onnx = args.sgg_weights.lower().endswith(".onnx")
        if self.is_onnx:
            provider = "CPUExecutionProvider" if args.cpu else args.onnx_provider
            self.sgg_model = SGG_ONNX_Model(
                args.sgg_config,
                args.sgg_weights,
                provider=provider,
                dcs=args.detections_per_img,
                tracking=args.enable_tracking,
                rel_conf=args.rel_conf,
                box_conf=args.box_conf,
                show_fps=False,
            )
        else:
            self.sgg_model = SGG_Model(
                args.sgg_config,
                args.sgg_weights,
                dcs=args.detections_per_img,
                tracking=args.enable_tracking,
                rel_conf=args.rel_conf,
                box_conf=args.box_conf,
                show_fps=False,
            )

        self.get_logger().info(
            f"SGG model loaded with config: {args.sgg_config} and weights: {args.sgg_weights}"
        )

        self.gnn_ckpt = torch.load(args.gnn_checkpoint, map_location=self.device, weights_only=False)
        self.maps = self.gnn_ckpt["maps"]
        self.task = self.maps["task"]
        self.idx2sit = {i: s for i, s in enumerate(self.maps["sit_list"])}

        self.gnn_model = SceneSituationGNN(
            num_obj_classes=len(self.maps["obj_list"]),
            num_rel_classes=len(self.maps["rel_list"]),
            num_situation_classes=len(self.maps["sit_list"]),
            node_num_dim=7,
            edge_num_dim=8,
            hidden_dim=self.gnn_ckpt["args"]["hidden_dim"],
            num_layers=self.gnn_ckpt["args"]["num_layers"],
            dropout=self.gnn_ckpt["args"]["dropout"],
        ).to(self.device)
        self.gnn_model.load_state_dict(self.gnn_ckpt["model_state"])
        self.gnn_model.eval()

        self.viz_pub = self.create_publisher(Image, args.viz_topic, 10)
        self.result_pub = self.create_publisher(String, args.result_topic, 10)
        self.planner_pub = self.create_publisher(SceneContext, args.planner_topic, 10)

        self.rgb_sub = Subscriber(self, Image, args.rgb_topic)
        self.depth_sub = Subscriber(self, Image, args.depth_topic)
        self.info_sub = Subscriber(self, CameraInfo, args.camera_info_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=args.sync_queue_size,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(
            f"Typed planner topic enabled: {args.planner_topic}, depth bbox mode: {args.depth_bbox_mode}"
        )
        
        self.latest_bundle = None
        self.bundle_lock = threading.Lock()
        self.worker_busy = False
        self.stop_worker = False
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

    def synced_callback(self, rgb_msg, depth_msg, info_msg):
        with self.bundle_lock:
            self.latest_bundle = (rgb_msg, depth_msg, info_msg)

    def worker_loop(self):
        period = 1.0 / self.args.max_publish_hz if self.args.max_publish_hz > 0 else 0.0
        last_run = 0.0

        while rclpy.ok() and not self.stop_worker:
            now = time.time()
            if period > 0 and (now - last_run) < period:
                time.sleep(0.001)
                continue

            with self.bundle_lock:
                bundle = self.latest_bundle
                self.latest_bundle = None

            if bundle is None:
                time.sleep(0.001)
                continue

            rgb_msg, depth_msg, info_msg = bundle

            try:
                rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                depth = self.bridge.imgmsg_to_cv2(depth_msg)

                intr = self.camera_info_to_intrinsics(info_msg)
                depth_m = self.depth_to_meters(depth, depth_msg.encoding)
                sgg = self.predict_graph(rgb_bgr)
                record, planner_objects = self.sgg_arrays_to_record_and_3d(
                    sgg["bboxes"], sgg["rels"], rgb_msg.header.stamp, depth_m, intr
                )
                pred = self.predict_situation(record)
                planner_msg = self.build_scene_context_msg(
                    record=record,
                    planner_objects=planner_objects,
                    pred=pred,
                    rgb_msg=rgb_msg,
                    info_msg=info_msg,
                )

                cur_label = "S0"
                if self.args.situation_thres < pred["probs"][pred["pred_label"]]:
                    cur_label = pred["pred_label"]
                else:
                    if pred["pred_label"] == "S1":
                        cur_label = pred["pred_label"]

                vis = self.draw_overlay_from_arrays(
                    rgb_bgr.copy(),
                    sgg["bboxes"],
                    sgg["rels"],
                    cur_label,
                    pred["probs"],
                    planner_objects,
                )
                self.publish_outputs(vis, record, pred, planner_msg, rgb_msg, info_msg)
                last_run = time.time()

            except Exception as e:
                self.get_logger().error(f"Pipeline error: {e}")

    # def synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
    #     now = time.time()
    #     if self.args.max_publish_hz > 0.0:
    #         min_dt = 1.0 / self.args.max_publish_hz
    #         if now - self.last_pub_time < min_dt:
    #             return

    #     try:
    #         rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    #         depth = self.bridge.imgmsg_to_cv2(depth_msg)
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to convert ROS images: {e}")
    #         return

    #     try:
    #         intr = self.camera_info_to_intrinsics(info_msg)
    #         depth_m = self.depth_to_meters(depth, depth_msg.encoding)
    #         sgg = self.predict_graph(rgb_bgr)
    #         record, planner_objects = self.sgg_arrays_to_record_and_3d(
    #             sgg["bboxes"], sgg["rels"], rgb_msg.header.stamp, depth_m, intr
    #         )
    #         pred = self.predict_situation(record)
    #         planner_msg = self.build_scene_context_msg(
    #             record=record,
    #             planner_objects=planner_objects,
    #             pred=pred,
    #             rgb_msg=rgb_msg,
    #             info_msg=info_msg,
    #         )
            
    #         cur_label = 'S0'
    #         if self.args.situation_thres < pred["probs"][pred["pred_label"]]:
    #             cur_label = pred["pred_label"]
                
    #         vis = self.draw_overlay_from_arrays(
    #             rgb_bgr.copy(),
    #             sgg["bboxes"],
    #             sgg["rels"],
    #             cur_label,
    #             pred["probs"],
    #             planner_objects,
    #         )
    #         self.publish_outputs(vis, record, pred, planner_msg, rgb_msg, info_msg)
    #         self.last_pub_time = now
    #     except Exception as e:
    #         self.get_logger().error(f"Pipeline error: {e}")

    def camera_info_to_intrinsics(self, info_msg: CameraInfo) -> CameraIntrinsics:
        return CameraIntrinsics(
            fx=float(info_msg.k[0]),
            fy=float(info_msg.k[4]),
            cx=float(info_msg.k[2]),
            cy=float(info_msg.k[5]),
            width=int(info_msg.width),
            height=int(info_msg.height),
        )

    def depth_to_meters(self, depth: np.ndarray, encoding: str) -> np.ndarray:
        if depth.dtype == np.uint16 or encoding in ("16UC1", "mono16"):
            return depth.astype(np.float32) * self.args.depth_scale
        return depth.astype(np.float32)

    def predict_graph(self, image: np.ndarray) -> Dict[str, Any]:
        if self.is_onnx:
            bboxes, rels = self.sgg_model.predict(image, visu_type="raw")
        else:
            self.sgg_model.model.roi_heads.eval()
            self.sgg_model.model.backbone.eval()
            img_list, _ = self.sgg_model._pre_processing(image)
            img_list.image_sizes = [(image.shape[0], image.shape[1])]
            img_list = img_list.to(self.sgg_model.device)
            with torch.no_grad():
                predictions = self.sgg_model.model(img_list, None, return_attention=False)
            bboxes, rels = self.sgg_model._post_process2(
                predictions[0],
                orig_size=image.shape[:2],
                box_thres=self.args.box_conf,
                rel_threshold=self.args.rel_conf,
            )

        bboxes = bboxes.cpu().numpy() if hasattr(bboxes, "cpu") else np.asarray(bboxes)
        rels = rels.cpu().numpy() if hasattr(rels, "cpu") else np.asarray(rels)
        return {"bboxes": bboxes, "rels": rels}

    def sgg_arrays_to_record_and_3d(
        self,
        bboxes: np.ndarray,
        rels: np.ndarray,
        stamp,
        depth_m: np.ndarray,
        intr: CameraIntrinsics,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        objects: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        planner_objects: List[Dict[str, Any]] = []

        for i, b in enumerate(bboxes):
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            score = float(b[4]) if len(b) > 4 else 0.0
            cls_id = int(b[5]) if len(b) > 5 else -1
            cls_name = self.sgg_model.stats["obj_classes"].get(cls_id, f"obj_{cls_id}")

            objects.append({
                "id": f"O{i}",
                "class": cls_name,
                "bbox": [x1, y1, x2, y2],
                "score": score,
            })

            bbox3d = self.estimate_3d_bbox_from_depth(depth_m, intr, x1, y1, x2, y2)
            planner_objects.append({
                "id": f"O{i}",
                "class": cls_name,
                "score": score,
                "bbox_2d": [x1, y1, x2, y2],
                "bbox_3d": bbox3d,
            })

        if rels is not None and len(rels) > 0:
            for r in rels:
                subj_idx = int(r[0])
                obj_idx = int(r[1])
                rel_id = int(r[2])
                rel_score = float(r[3]) if len(r) > 3 else 0.0
                if subj_idx < 0 or obj_idx < 0 or subj_idx >= len(objects) or obj_idx >= len(objects):
                    continue
                rel_name = self.sgg_model.stats["rel_classes"].get(rel_id, f"rel_{rel_id}")
                relationships.append({
                    "subject": f"O{subj_idx}",
                    "predicate": rel_name,
                    "object": f"O{obj_idx}",
                    "score": rel_score,
                })

        record = {
            "scene_id": f"{stamp.sec}_{stamp.nanosec}",
            "situation": "S0",
            "objects": objects,
            "relationships": relationships,
        }
        return record, planner_objects

    def estimate_3d_bbox_from_depth(
        self,
        depth_m: np.ndarray,
        intr: CameraIntrinsics,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Dict[str, Any]:
        h, w = depth_m.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return self.empty_bbox3d()

        roi_depth = depth_m[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return self.empty_bbox3d()

        valid = np.isfinite(roi_depth) & (roi_depth > self.args.min_depth_m) & (roi_depth < self.args.max_depth_m)
        if not np.any(valid):
            return self.empty_bbox3d()

        ys, xs = np.where(valid)
        zs = roi_depth[ys, xs]

        if self.args.depth_bbox_mode == "center":
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            z = float(np.median(zs))
            pt = self.project_pixel_to_3d(cx, cy, z, intr)
            size_x = max((x2 - x1) * z / intr.fx, self.args.min_box_size_m)
            size_y = max((y2 - y1) * z / intr.fy, self.args.min_box_size_m)
            size_z = max(float(np.percentile(zs, 90) - np.percentile(zs, 10)), self.args.min_box_depth_m)
            return {
                "valid": True,
                "frame": self.args.planner_frame,
                "center": {"x": pt[0], "y": pt[1], "z": pt[2]},
                "size": {"x": float(size_x), "y": float(size_y), "z": float(size_z)},
                "min": {"x": float(pt[0] - size_x / 2), "y": float(pt[1] - size_y / 2), "z": float(pt[2] - size_z / 2)},
                "max": {"x": float(pt[0] + size_x / 2), "y": float(pt[1] + size_y / 2), "z": float(pt[2] + size_z / 2)},
                "depth_stats": self.depth_stats(zs),
                "method": "2d_bbox_center_depth",
            }

        sample_limit = self.args.max_points_per_object
        if zs.shape[0] > sample_limit:
            idx = np.random.choice(zs.shape[0], sample_limit, replace=False)
            xs = xs[idx]
            ys = ys[idx]
            zs = zs[idx]

        u = xs.astype(np.float32) + x1
        v = ys.astype(np.float32) + y1
        X = (u - intr.cx) * zs / intr.fx
        Y = (v - intr.cy) * zs / intr.fy
        Z = zs.astype(np.float32)
        pts = np.stack([X, Y, Z], axis=1)

        if self.args.use_depth_percentile_crop:
            z_lo = np.percentile(Z, self.args.depth_crop_percentile_low)
            z_hi = np.percentile(Z, self.args.depth_crop_percentile_high)
            keep = (Z >= z_lo) & (Z <= z_hi)
            pts = pts[keep]
            Z = Z[keep]
            if pts.shape[0] == 0:
                return self.empty_bbox3d()

        pmin = np.percentile(pts, self.args.xyz_percentile_low, axis=0)
        pmax = np.percentile(pts, self.args.xyz_percentile_high, axis=0)
        center = (pmin + pmax) / 2.0
        size = np.maximum(pmax - pmin, np.array([self.args.min_box_size_m, self.args.min_box_size_m, self.args.min_box_depth_m], dtype=np.float32))

        return {
            "valid": True,
            "frame": self.args.planner_frame,
            "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
            "size": {"x": float(size[0]), "y": float(size[1]), "z": float(size[2])},
            "min": {"x": float(center[0] - size[0] / 2), "y": float(center[1] - size[1] / 2), "z": float(center[2] - size[2] / 2)},
            "max": {"x": float(center[0] + size[0] / 2), "y": float(center[1] + size[1] / 2), "z": float(center[2] + size[2] / 2)},
            "depth_stats": self.depth_stats(Z),
            "method": "2d_bbox_depth_roi",
        }

    def project_pixel_to_3d(self, u: float, v: float, z: float, intr: CameraIntrinsics):
        x = (u - intr.cx) * z / intr.fx
        y = (v - intr.cy) * z / intr.fy
        return float(x), float(y), float(z)

    def depth_stats(self, z_values: np.ndarray) -> Dict[str, float]:
        return {
            "median": float(np.median(z_values)),
            "mean": float(np.mean(z_values)),
            "min": float(np.min(z_values)),
            "max": float(np.max(z_values)),
            "std": float(np.std(z_values)),
        }

    def empty_bbox3d(self) -> Dict[str, Any]:
        return {
            "valid": False,
            "frame": self.args.planner_frame,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "size": {"x": 0.0, "y": 0.0, "z": 0.0},
            "min": {"x": 0.0, "y": 0.0, "z": 0.0},
            "max": {"x": 0.0, "y": 0.0, "z": 0.0},
            "depth_stats": {"median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
            "method": "invalid",
        }

    def predict_situation(self, record: Dict[str, Any]) -> Dict[str, Any]:
        graphs = build_graphs([record], self.maps, task=self.task)
        loader = DataLoader(graphs, batch_size=1, shuffle=False)
        with torch.no_grad():
            batch = next(iter(loader)).to(self.device)
            logits = self.gnn_model(batch)
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_idx = int(np.argmax(probs))
        return {
            "pred_index": pred_idx,
            "pred_label": self.idx2sit[pred_idx],
            "probs": {self.idx2sit[i]: float(probs[i]) for i in range(len(probs))},
        }

    def build_scene_context_msg(
        self,
        record: Dict[str, Any],
        planner_objects: List[Dict[str, Any]],
        pred: Dict[str, Any],
        rgb_msg: Image,
        info_msg: CameraInfo,
    ) -> SceneContext:
        msg = SceneContext()
        msg.header = rgb_msg.header
        msg.scene_id = record["scene_id"]
        msg.planner_frame = self.args.planner_frame

        cam = CameraModel()
        cam.width = info_msg.width
        cam.height = info_msg.height
        cam.k = [float(v) for v in info_msg.k]
        cam.distortion_model = info_msg.distortion_model
        cam.d = [float(v) for v in info_msg.d]
        msg.camera = cam

        top_label = pred["pred_label"]
        sit = SituationHypothesis()
        sit.label = top_label
        sit.index = int(pred["pred_index"])
        sit.confidence = float(pred["probs"].get(top_label, 0.0))
        sit.labels = list(pred["probs"].keys())
        sit.probs = [float(v) for v in pred["probs"].values()]
        msg.situation = sit

        for obj in planner_objects:
            mobj = DetectedObject3D()
            mobj.id = obj["id"]
            mobj.class_name = obj["class"]
            mobj.score = float(obj["score"])
            mobj.bbox_2d_xyxy = [int(v) for v in obj["bbox_2d"]]
            mobj.bbox_3d = self.to_bbox3d_msg(obj["bbox_3d"])
            msg.objects.append(mobj)

        for rel in record["relationships"]:
            mrel = SceneRelation()
            mrel.subject_id = rel["subject"]
            mrel.predicate = rel["predicate"]
            mrel.object_id = rel["object"]
            mrel.score = float(rel["score"])
            msg.relationships.append(mrel)

        return msg

    def to_bbox3d_msg(self, bbox3d: Dict[str, Any]) -> BoundingBox3D:
        msg = BoundingBox3D()
        msg.valid = bool(bbox3d["valid"])
        msg.frame_id = str(bbox3d["frame"])
        msg.center = Point(
            x=float(bbox3d["center"]["x"]),
            y=float(bbox3d["center"]["y"]),
            z=float(bbox3d["center"]["z"]),
        )
        msg.size = Vector3(
            x=float(bbox3d["size"]["x"]),
            y=float(bbox3d["size"]["y"]),
            z=float(bbox3d["size"]["z"]),
        )
        msg.min_corner = Point(
            x=float(bbox3d["min"]["x"]),
            y=float(bbox3d["min"]["y"]),
            z=float(bbox3d["min"]["z"]),
        )
        msg.max_corner = Point(
            x=float(bbox3d["max"]["x"]),
            y=float(bbox3d["max"]["y"]),
            z=float(bbox3d["max"]["z"]),
        )
        stats = bbox3d["depth_stats"]
        msg.z_median = float(stats["median"])
        msg.z_mean = float(stats["mean"])
        msg.z_min = float(stats["min"])
        msg.z_max = float(stats["max"])
        msg.z_std = float(stats["std"])
        msg.method = str(bbox3d["method"])
        return msg

    def draw_overlay_from_arrays(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        rels: np.ndarray,
        situation_label: str,
        probs: Dict[str, float],
        planner_objects: List[Dict[str, Any]],
    ) -> np.ndarray:
        image = self.sgg_model.draw_full_graph(image, bboxes, rels)
        topk = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        title = f"Situation: {situation_label}"
        prob_text = " | ".join([f"{k}:{v:.2f}" for k, v in topk])
        panel_w = min(image.shape[1] - 20, 1000)
        cv2.rectangle(image, (10, 10), (10 + panel_w, 10 + 78), (20, 20, 20), -1)
        cv2.putText(image, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, prob_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        for obj in planner_objects:
            x1, y1, x2, y2 = obj["bbox_2d"]
            bbox3d = obj["bbox_3d"]
            if bbox3d["valid"]:
                z = bbox3d["center"]["z"]
                sx = bbox3d["size"]["x"]
                sy = bbox3d["size"]["y"]
                label = f'{obj["class"]} z={z:.2f}m size=({sx:.2f},{sy:.2f})'
                cv2.putText(image, label, (x1, max(15, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                label = f'{obj["class"]} z=NA'
                cv2.putText(image, label, (x1, max(15, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        return image

    def publish_outputs(
        self,
        vis: np.ndarray,
        record: Dict[str, Any],
        pred: Dict[str, Any],
        planner_msg: SceneContext,
        rgb_msg: Image,
        info_msg: CameraInfo,
    ):
        vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        vis_msg.header = rgb_msg.header
        self.viz_pub.publish(vis_msg)

        result = {
            "stamp_sec": rgb_msg.header.stamp.sec,
            "stamp_nanosec": rgb_msg.header.stamp.nanosec,
            "frame_id": rgb_msg.header.frame_id,
            "scene_id": record["scene_id"],
            "situation": pred["pred_label"],
            "probs": pred["probs"],
            "objects": record["objects"],
            "relationships": record["relationships"],
            "camera_info": {
                "width": info_msg.width,
                "height": info_msg.height,
                "k": list(info_msg.k),
                "distortion_model": info_msg.distortion_model,
                "d": list(info_msg.d),
            },
        }
        result_msg = String()
        result_msg.data = json.dumps(result, ensure_ascii=False)
        self.result_pub.publish(result_msg)

        self.planner_pub.publish(planner_msg)


def build_argparser():
    p = argparse.ArgumentParser(description="ROS2 RGB-D Scene Graph + Situation GNN node with typed planner output")
    p.add_argument("--sgg-config", type=str, required=True)
    p.add_argument("--sgg-weights", type=str, required=True)
    p.add_argument("--gnn-checkpoint", type=str, required=True)

    p.add_argument("--rgb-topic", type=str, default="/camera/color/image_raw")
    p.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info")
    p.add_argument("--viz-topic", type=str, default="/scene_graph/debug_image")
    p.add_argument("--result-topic", type=str, default="/scene_graph/result")
    p.add_argument("--planner-topic", type=str, default="/scene_graph/planner_context")

    p.add_argument("--sync-queue-size", type=int, default=10)
    p.add_argument("--sync-slop", type=float, default=0.1)

    p.add_argument("--box-conf", type=float, default=0.5)
    p.add_argument("--rel-conf", type=float, default=0.1)
    p.add_argument("--detections-per-img", type=int, default=100)
    p.add_argument("--enable-tracking", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--onnx-provider", type=str, default="CUDAExecutionProvider")

    p.add_argument("--planner-frame", type=str, default="camera_color_optical_frame")
    p.add_argument("--depth-scale", type=float, default=0.001)
    p.add_argument("--min-depth-m", type=float, default=0.15)
    p.add_argument("--max-depth-m", type=float, default=3.0)
    p.add_argument("--max-points-per-object", type=int, default=1500)
    p.add_argument("--depth-bbox-mode", type=str, default="roi", choices=["roi", "center"])
    p.add_argument("--use-depth-percentile-crop", action="store_true")
    p.add_argument("--depth-crop-percentile-low", type=float, default=10.0)
    p.add_argument("--depth-crop-percentile-high", type=float, default=90.0)
    p.add_argument("--xyz-percentile-low", type=float, default=5.0)
    p.add_argument("--xyz-percentile-high", type=float, default=95.0)
    p.add_argument("--min-box-size-m", type=float, default=0.03)
    p.add_argument("--min-box-depth-m", type=float, default=0.03)
    p.add_argument("--max-publish-hz", type=float, default=15.0)
    p.add_argument("--situation-thres", type=float, default=0.80)
    return p


def main():
    parser = build_argparser()
    args, ros_args = parser.parse_known_args()
    for path_arg in [args.sgg_config, args.sgg_weights, args.gnn_checkpoint]:
        if not Path(path_arg).exists():
            raise FileNotFoundError(f"Required file not found: {path_arg}")

    rclpy.init(args=ros_args)
    node = SceneUnderstandingNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
