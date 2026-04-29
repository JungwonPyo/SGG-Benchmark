import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsTextItem, QPushButton, QLabel, QListWidget, QComboBox, 
    QFileDialog, QMessageBox, QSplitter, QFrame
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QPolygonF, QBrush
)
from PySide6.QtCore import Qt, QPointF, Signal

# SAM2 Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
SAM_MODEL_PATH = "/home/dxr-labtop/DAS_Pick_and_Place/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt"
SAM_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

CLASSES = [
    "부품 박스", "플라스틱 트레이", "공정 부품", "드라이버", "작업자 손",
    "조립 지그", "폐기 박스", "렌치", "케이블 묶음", "보호 고글"
]
RELATIONS = ["on", "inside", "next_to", "above", "touching", "blocking", "near", "beside"]
SITUATIONS = ["S1: 손 진입", "S2: 접근로 점유", "S3: 팔 궤적 간섭", "S4: 인간 접촉", "S5: 배치로 점유"]
PATH_MODS = ["stop", "detour", "retarget", "wait", "normal"]

# Random colors for instances
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128),
    (0, 128, 128), (255, 20, 147), (139, 69, 19), (128, 128, 0), (0, 191, 255)
]

class ImageGraphicsView(QGraphicsView):
    mouseClicked = Signal(QPointF, int)  # int: 1 for left (positive), 0 for right (negative)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            zoom_in_factor = 1.15
            zoom_out_factor = 1.0 / zoom_in_factor
            zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
            self.scale(zoom_factor, zoom_factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.ShiftModifier:
            # Shift + Click for SAM prompts
            scene_pos = self.mapToScene(event.position().toPoint())
            label = 1 if event.button() == Qt.LeftButton else 0
            self.mouseClicked.emit(scene_pos, label)
            event.accept()
        else:
            super().mousePressEvent(event)


class SceneGraphLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scene Graph & Instance Labeling Tool")
        self.setGeometry(100, 100, 1400, 800)

        # State
        self.image_paths = []
        self.current_idx = -1
        self.current_image_pil = None

        # Instance & Scene State
        self.instances = []    # list of dicts: id, class, bbox, mask (np.array)
        self.relations = []    # list of dicts: subject, predicate, object

        # SAM State
        self.input_points = []
        self.input_labels = []
        self.current_sam_mask = None

        self.init_ui()
        self.init_sam()

    def init_sam(self):
        self.statusBar().showMessage("Loading SAM2 Model...")
        QApplication.processEvents()
        try:
            self.sam2_model = build_sam2(SAM_MODEL_CFG, SAM_MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            self.statusBar().showMessage("SAM2 Loaded Successfully!")
        except Exception as e:
            self.statusBar().showMessage(f"SAM2 Load Error: {e}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # --- LEFT PANEL: Image List ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        btn_load = QPushButton("📂 Load Image Folder")
        btn_load.clicked.connect(self.load_folder)
        left_layout.addWidget(btn_load)

        self.img_list_widget = QListWidget()
        self.img_list_widget.currentRowChanged.connect(self.select_image)
        left_layout.addWidget(self.img_list_widget)

        splitter.addWidget(left_panel)

        # --- CENTER PANEL: Image View ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        self.scene = QGraphicsScene()
        self.view = ImageGraphicsView(self.scene)
        self.view.mouseClicked.connect(self.handle_sam_click)
        center_layout.addWidget(self.view)

        # SAM Controls
        sam_ctrl_layout = QHBoxLayout()
        btn_clear_sam = QPushButton("🧹 Clear Current Prompts")
        btn_clear_sam.clicked.connect(self.clear_sam_prompts)
        sam_ctrl_layout.addWidget(QLabel("<i>Shift+LeftClick: Add. Shift+RightClick: Remove</i>"))
        sam_ctrl_layout.addWidget(btn_clear_sam)
        center_layout.addLayout(sam_ctrl_layout)

        splitter.addWidget(center_panel)

        # --- RIGHT PANEL: Scene Graph ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 1. Classes & Instances
        right_layout.addWidget(QLabel("<b>1. Object Class</b>"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(CLASSES)
        right_layout.addWidget(self.class_combo)

        btn_add_inst = QPushButton("➕ Add Mask as Instance")
        btn_add_inst.setStyleSheet("background-color: #2196F3; color: white;")
        btn_add_inst.clicked.connect(self.add_instance)
        right_layout.addWidget(btn_add_inst)

        self.inst_list_widget = QListWidget()
        right_layout.addWidget(QLabel("<b>Instances:</b>"))
        right_layout.addWidget(self.inst_list_widget)

        btn_del_inst = QPushButton("🗑️ Delete Selected Instance")
        btn_del_inst.clicked.connect(self.delete_instance)
        right_layout.addWidget(btn_del_inst)

        right_layout.addWidget(QFrame(frameShape=QFrame.HLine))

        # 2. Relations
        right_layout.addWidget(QLabel("<b>2. Relationships</b>"))
        rel_layout = QHBoxLayout()
        self.subj_combo = QComboBox()
        self.pred_combo = QComboBox()
        self.pred_combo.addItems(RELATIONS)
        self.obj_combo = QComboBox()
        rel_layout.addWidget(self.subj_combo)
        rel_layout.addWidget(self.pred_combo)
        rel_layout.addWidget(self.obj_combo)
        right_layout.addLayout(rel_layout)

        btn_add_rel = QPushButton("🔗 Add Relation")
        btn_add_rel.clicked.connect(self.add_relation)
        right_layout.addWidget(btn_add_rel)

        self.rel_list_widget = QListWidget()
        right_layout.addWidget(self.rel_list_widget)

        btn_del_rel = QPushButton("🗑️ Delete Selected Relation")
        btn_del_rel.clicked.connect(self.delete_relation)
        right_layout.addWidget(btn_del_rel)

        right_layout.addWidget(QFrame(frameShape=QFrame.HLine))

        # 3. Scene Meta & Save
        right_layout.addWidget(QLabel("<b>3. Scene Attributes</b>"))
        self.sit_combo = QComboBox()
        self.sit_combo.addItems(SITUATIONS)
        self.pmod_combo = QComboBox()
        self.pmod_combo.addItems(PATH_MODS)
        right_layout.addWidget(self.sit_combo)
        right_layout.addWidget(self.pmod_combo)

        btn_save = QPushButton("💾 Save Scene (JSONL & Mask)")
        btn_save.setFixedHeight(50)
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_scene)
        right_layout.addWidget(btn_save)

        splitter.addWidget(right_panel)

        # Splitter ratios
        splitter.setSizes([200, 800, 400])

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            self.image_paths = [os.path.join(folder, f) for f in files]

            self.img_list_widget.clear()
            self.img_list_widget.addItems(files)

            if self.image_paths:
                self.img_list_widget.setCurrentRow(0)

    def select_image(self, idx):
        if idx < 0 or idx >= len(self.image_paths): return
        self.current_idx = idx
        img_path = self.image_paths[idx]

        # Load Image
        self.current_image_pil = Image.open(img_path).convert("RGB")

        # Reset State
        self.instances.clear()
        self.relations.clear()
        self.clear_sam_prompts()

        # Set SAM image
        img_np = np.array(self.current_image_pil)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(img_np)

        self.update_ui_lists()
        self.redraw_scene()

        # Auto-load existing JSONL if exists
        self.try_load_existing_annotation(img_path)

    def handle_sam_click(self, pos, label):
        if not self.current_image_pil: return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self.current_image_pil.width and 0 <= y < self.current_image_pil.height:
            self.input_points.append([x, y])
            self.input_labels.append(label)
            self.run_sam()

    def run_sam(self):
        if not self.input_points: return
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(self.input_points),
                point_labels=np.array(self.input_labels),
                multimask_output=False
            )
        self.current_sam_mask = np.squeeze(masks)
        self.redraw_scene()

    def clear_sam_prompts(self):
        self.input_points.clear()
        self.input_labels.clear()
        self.current_sam_mask = None
        self.redraw_scene()

    def add_instance(self):
        if self.current_sam_mask is None or np.sum(self.current_sam_mask) == 0:
            QMessageBox.warning(self, "Warning", "No valid SAM mask to add.")
            return

        # Get BBox from mask
        y_indices, x_indices = np.where(self.current_sam_mask > 0)
        x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
        x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))

        cls_name = self.class_combo.currentText()
        # Find next available ID
        existing_ids = [int(inst["id"][1:]) for inst in self.instances]
        next_id_num = max(existing_ids) + 1 if existing_ids else 1
        inst_id = f"O{next_id_num}"

        self.instances.append({
            "id": inst_id,
            "class": cls_name,
            "bbox": [x1, y1, x2, y2],
            "mask": self.current_sam_mask.copy()
        })

        self.clear_sam_prompts()
        self.update_ui_lists()

    def delete_instance(self):
        row = self.inst_list_widget.currentRow()
        if row >= 0:
            inst_id = self.instances[row]["id"]
            # Remove relations involving this instance
            self.relations = [r for r in self.relations if r["subject"] != inst_id and r["object"] != inst_id]
            del self.instances[row]
            self.update_ui_lists()
            self.redraw_scene()

    def add_relation(self):
        if not self.instances: return
        subj = self.subj_combo.currentText().split(":")[0]
        obj = self.obj_combo.currentText().split(":")[0]
        pred = self.pred_combo.currentText()

        if subj == obj:
            QMessageBox.warning(self, "Warning", "Subject and Object must be different.")
            return

        rel = {"subject": subj, "predicate": pred, "object": obj}
        if rel not in self.relations:
            self.relations.append(rel)
            self.update_ui_lists()

    def delete_relation(self):
        row = self.rel_list_widget.currentRow()
        if row >= 0:
            del self.relations[row]
            self.update_ui_lists()

    def update_ui_lists(self):
        # Update Instance List
        self.inst_list_widget.clear()
        for inst in self.instances:
            self.inst_list_widget.addItem(f"{inst['id']}: {inst['class']} {inst['bbox']}")

        # Update Comboboxes
        self.subj_combo.clear()
        self.obj_combo.clear()
        items = [f"{inst['id']}: {inst['class']}" for inst in self.instances]
        self.subj_combo.addItems(items)
        self.obj_combo.addItems(items)

        # Update Relation List
        self.rel_list_widget.clear()
        for r in self.relations:
            self.rel_list_widget.addItem(f"{r['subject']}  - [{r['predicate']}] ->  {r['object']}")

        self.redraw_scene()

    def redraw_scene(self):
        if not self.current_image_pil: return
        self.scene.clear()

        # 1. Base Image
        img_q = self.pil_to_qimage(self.current_image_pil)

        # 2. Draw Committed Instance Masks & BBoxes
        painter = QPainter(img_q)
        for i, inst in enumerate(self.instances):
            color = COLORS[i % len(COLORS)]
            # Draw Mask
            if "mask" in inst and inst["mask"] is not None:
                mask = inst["mask"]
                mask_img = Image.fromarray((mask * 100).astype(np.uint8), mode='L')
                mask_rgba = Image.new("RGBA", mask_img.size, color + (0,))
                mask_rgba.putalpha(mask_img)
                q_mask = self.pil_to_qimage(mask_rgba)
                painter.drawImage(0, 0, q_mask)

            # Draw BBox
            b = inst["bbox"]
            painter.setPen(QPen(QColor(*color), 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(b[0], b[1], b[2]-b[0], b[3]-b[1])

            # Draw Label
            painter.setBrush(QColor(*color))
            painter.setPen(Qt.white)
            painter.drawRect(b[0], b[1]-15, 40, 15)
            painter.drawText(b[0]+2, b[1]-3, inst["id"])

        # 3. Draw Relations (Arrows & Text)
        id_to_bbox = {inst["id"]: inst["bbox"] for inst in self.instances}
        painter.setRenderHint(QPainter.Antialiasing)
        import math
        for rel in self.relations:
            sub_id = rel["subject"]
            obj_id = rel["object"]
            pred = rel["predicate"]

            if sub_id in id_to_bbox and obj_id in id_to_bbox:
                b1 = id_to_bbox[sub_id]
                b2 = id_to_bbox[obj_id]

                # Centers of the two bounding boxes
                p1 = QPointF((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                p2 = QPointF((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)

                # Draw Dashed Line connecting the two centers
                painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
                painter.drawLine(p1, p2)

                # Draw Arrowhead pointing to the object
                angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
                arrow_size = 15
                p3 = QPointF(p2.x() - arrow_size * math.cos(angle - math.pi / 6),
                             p2.y() - arrow_size * math.sin(angle - math.pi / 6))
                p4 = QPointF(p2.x() - arrow_size * math.cos(angle + math.pi / 6),
                             p2.y() - arrow_size * math.sin(angle + math.pi / 6))

                painter.setBrush(Qt.yellow)
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(QPolygonF([p2, p3, p4]))

                # Draw Predicate Text at the midpoint
                mid_p = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(pred)

                # Center the rectangle on the midpoint and add padding
                text_rect.translate(int(mid_p.x() - text_rect.width()/2), int(mid_p.y() - text_rect.height()/2))
                text_rect.adjust(-6, -4, 6, 4) 

                # Draw semi-transparent background for text readability
                painter.setBrush(QColor(0, 0, 0, 180)) 
                painter.setPen(Qt.NoPen)
                painter.drawRect(text_rect)

                # Draw the actual relationship text
                painter.setPen(Qt.yellow)
                painter.drawText(text_rect, Qt.AlignCenter, pred)

        # 4. Draw Current Active SAM Mask
        if self.current_sam_mask is not None:
            mask = self.current_sam_mask
            mask_img = Image.fromarray((mask * 128).astype(np.uint8), mode='L')
            mask_rgba = Image.new("RGBA", mask_img.size, (255, 255, 0, 0)) # Yellow for active
            mask_rgba.putalpha(mask_img)
            q_mask = self.pil_to_qimage(mask_rgba)
            painter.drawImage(0, 0, q_mask)

        # 5. Draw SAM Prompts
        for pt, lbl in zip(self.input_points, self.input_labels):
            color = Qt.green if lbl == 1 else Qt.red
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(pt[0]-4, pt[1]-4, 8, 8)

        painter.end()

        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(img_q)))

    def pil_to_qimage(self, image):
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        return QImage(data, image.size[0], image.size[1], QImage.Format_RGBA8888)

    def save_scene(self):
        if not self.current_image_pil: return
        img_path = self.image_paths[self.current_idx]

        sit_code = self.sit_combo.currentText().split(":")[0]

        # Prepare output dirs
        base_dir = os.path.dirname(img_path)
        dataset_dir = os.path.join(os.path.dirname(base_dir), "dataset")
        masks_dir = os.path.join(os.path.dirname(base_dir), "masks")
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # 1. Compile Instance Mask (Instance ID encoded in pixels)
        w, h = self.current_image_pil.size
        total_mask = np.zeros((h, w), dtype=np.uint8)

        clean_instances = []
        for inst in self.instances:
            inst_id_int = int(inst["id"][1:])
            if "mask" in inst and inst["mask"] is not None:
                total_mask = np.where(inst["mask"] > 0, inst_id_int, total_mask)

            clean_instances.append({
                "id": inst["id"],
                "class": inst["class"],
                "bbox": inst["bbox"]
            })

        # Save Mask PNG
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{basename}_mask.png")
        Image.fromarray(total_mask).save(mask_path)

        # 2. Save JSONL
        scene = {
            "scene_id": f"{sit_code}_{basename}",
            "situation": sit_code,
            "image_path": img_path,
            "mask_path": mask_path,
            "objects": clean_instances,
            "relationships": self.relations,
            "path_modification": self.pmod_combo.currentText(),
            "goal_position": [w//2, h//2],
            "goal_changed": False
        }

        jsonl_path = os.path.join(dataset_dir, "manual_labeled.jsonl")

        # Update if exists, else append
        lines = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        with open(jsonl_path, "w", encoding="utf-8") as f:
            replaced = False
            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                if data["image_path"] == img_path:
                    f.write(json.dumps(scene, ensure_ascii=False) + "\n")
                    replaced = True
                else:
                    f.write(line)
            if not replaced:
                f.write(json.dumps(scene, ensure_ascii=False) + "\n")

        self.statusBar().showMessage(f"Saved! {basename}", 3000)

        # Auto next image
        if self.current_idx < len(self.image_paths) - 1:
            self.img_list_widget.setCurrentRow(self.current_idx + 1)

    def try_load_existing_annotation(self, img_path):
        base_dir = os.path.dirname(img_path)
        jsonl_path = os.path.join(os.path.dirname(base_dir), "dataset", "manual_labeled.jsonl")
        mask_path = os.path.join(os.path.dirname(base_dir), "masks", f"{os.path.splitext(os.path.basename(img_path))[0]}_mask.png")

        if not os.path.exists(jsonl_path): return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if data.get("image_path") == img_path:
                    # Load Objects
                    self.instances = data.get("objects", [])
                    # Attempt to load mask to restore SAM masks per instance
                    if os.path.exists(mask_path):
                        total_mask = np.array(Image.open(mask_path))
                        for inst in self.instances:
                            inst_id_int = int(inst["id"][1:])
                            inst["mask"] = (total_mask == inst_id_int).astype(np.uint8)

                    # Load Relations
                    self.relations = data.get("relationships", [])

                    # Load Scene Info
                    sit = data.get("situation", "S1")
                    pmod = data.get("path_modification", "normal")

                    for i in range(self.sit_combo.count()):
                        if self.sit_combo.itemText(i).startswith(sit):
                            self.sit_combo.setCurrentIndex(i)
                            break

                    idx = self.pmod_combo.findText(pmod)
                    if idx >= 0: self.pmod_combo.setCurrentIndex(idx)

                    self.update_ui_lists()
                    break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SceneGraphLabeler()
    ex.show()
    sys.exit(app.exec())