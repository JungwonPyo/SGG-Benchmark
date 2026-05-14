from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

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


class SceneUnderstandingNode(Node):
    def __init__(self, args):
        super().__init__("scene_understanding_node")
        self.args = args
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

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
            
        self.get_logger().info(f"SGG model loaded with config: {args.sgg_config} and weights: {args.sgg_weights}")

        self.gnn_ckpt = torch.load(args.gnn_checkpoint, map_location=self.device, weights_only=False)
        self.get_logger().info(f"GNN checkpoint loaded: {args.gnn_checkpoint}")

        self.maps = self.gnn_ckpt["maps"]
        self.task = self.maps["task"]
        self.idx2sit = {i: s for i, s in enumerate(self.maps["sit_list"])}

        self.gnn_model = SceneSituationGNN(
            num_obj_classes=len(self.maps["obj_list"]),
            num_rel_classes=len(self.maps["rel_list"]),
            num_situation_classes=len(self.maps["sit_list"]),
            hidden_dim=self.gnn_ckpt["args"]["hidden_dim"],
            num_layers=self.gnn_ckpt["args"]["num_layers"],
            dropout=self.gnn_ckpt["args"]["dropout"],
        ).to(self.device)
        self.gnn_model.load_state_dict(self.gnn_ckpt["model_state"])
        self.gnn_model.eval()
        self.get_logger().info(f"GNN model initialized with {len(self.maps['obj_list'])} object classes, {len(self.maps['rel_list'])} relation classes, and {len(self.maps['sit_list'])} situation classes.")

        self.viz_pub = self.create_publisher(Image, args.viz_topic, 10)
        self.result_pub = self.create_publisher(String, args.result_topic, 10)

        self.rgb_sub = Subscriber(self, Image, args.rgb_topic)
        self.depth_sub = Subscriber(self, Image, args.depth_topic)
        self.info_sub = Subscriber(self, CameraInfo, args.camera_info_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=args.sync_queue_size,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

    def synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        try:
            rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS images: {e}")
            return

        try:
            sgg = self.predict_graph(rgb_bgr)
            record = self.sgg_arrays_to_record(sgg["bboxes"], sgg["rels"], rgb_msg.header.stamp)
            pred = self.predict_situation(record)
            vis = self.draw_overlay_from_arrays(
                rgb_bgr.copy(), sgg["bboxes"], sgg["rels"], pred["pred_label"], pred["probs"], depth
            )
            self.publish_outputs(vis, record, pred, rgb_msg, info_msg)
        except Exception as e:
            self.get_logger().error(f"Pipeline error: {e}")

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

        return {
            "bboxes": bboxes,
            "rels": rels,
        }

    def sgg_arrays_to_record(self, bboxes: np.ndarray, rels: np.ndarray, stamp) -> Dict[str, Any]:
        objects: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

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

        return {
            "scene_id": f"{stamp.sec}_{stamp.nanosec}",
            "situation": "S0",
            "objects": objects,
            "relationships": relationships,
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

    def draw_overlay_from_arrays(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        rels: np.ndarray,
        situation_label: str,
        probs: Dict[str, float],
        depth: np.ndarray | None = None,
    ) -> np.ndarray:
        image = self.sgg_model.draw_full_graph(image, bboxes, rels)

        topk = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        title = f"Situation: {situation_label}"
        prob_text = " | ".join([f"{k}:{v:.2f}" for k, v in topk])

        panel_w = min(image.shape[1] - 20, 900)
        cv2.rectangle(image, (10, 10), (10 + panel_w, 10 + 78), (20, 20, 20), -1)
        cv2.putText(image, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, prob_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def publish_outputs(self, vis: np.ndarray, record: Dict[str, Any], pred: Dict[str, Any], rgb_msg: Image, info_msg: CameraInfo):
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
        msg = String()
        msg.data = json.dumps(result, ensure_ascii=False)
        self.result_pub.publish(msg)


def build_argparser():
    p = argparse.ArgumentParser(description="ROS2 RGB-D Scene Graph + Situation GNN node")
    p.add_argument("--sgg-config", type=str, required=True)
    p.add_argument("--sgg-weights", type=str, required=True)
    p.add_argument("--gnn-checkpoint", type=str, required=True)

    p.add_argument("--rgb-topic", type=str, default="/camera/color/image_raw")
    p.add_argument("--depth-topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info")
    p.add_argument("--viz-topic", type=str, default="/scene_graph/debug_image")
    p.add_argument("--result-topic", type=str, default="/scene_graph/result")

    p.add_argument("--sync-queue-size", type=int, default=10)
    p.add_argument("--sync-slop", type=float, default=0.1)

    p.add_argument("--box-conf", type=float, default=0.5)
    p.add_argument("--rel-conf", type=float, default=0.1)
    p.add_argument("--detections-per-img", type=int, default=100)
    p.add_argument("--enable-tracking", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--onnx-provider", type=str, default="CUDAExecutionProvider")
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