import os
import sys
import json
import time
import re          
import copy        
import numpy as np
import torch
from PIL import Image, ImageDraw  
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsTextItem, QPushButton, QLabel, QListWidget, QComboBox, 
    QFileDialog, QMessageBox, QSplitter, QFrame, QDialog
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QPolygonF, QBrush,
    QShortcut, QKeySequence  
)
from PySide6.QtCore import Qt, QPointF, Signal

# SAM2 Imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
SAM_MODEL_PATH = "/home/dxr/RRT_Tools/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

CLASSES = [
    "부품 박스", "플라스틱 트레이", "공정 부품", "드라이버", "작업자 손",
    "검사 지그", "렌치", "케이블 묶음", "보호 고글", "그리퍼"
]
RELATIONS = ["on", "inside", "beside", "above", "touching","near", "gripping"]
SITUATIONS = ["S0: 상황 없음","S1: 대상 물체 없음", "S2: 손 접촉", "S3: 로봇 경로 간섭", "S4: 부품박스 없음", "S5: 배치로 점유"]
PATH_MODS = ["stop", "retarget", "wait", "delay", "normal"]

# Random colors for instances
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (255, 20, 147), # DeepPink
    (139, 69, 19),  # SaddleBrown
    (128, 128, 0),  # Olive
    (0, 191, 255),  # DeepSkyBlue
    (255, 215, 0),  # Gold
    (50, 205, 50),  # LimeGreen
    (186, 85, 211), # MediumOrchid
    (255, 69, 0),   # OrangeRed
    (70, 130, 180), # SteelBlue
    (210, 105, 30), # Chocolate
    (154, 205, 50), # YellowGreen
    (219, 112, 147),# PaleVioletRed
    (0, 250, 154),  # MediumSpringGreen
    (100, 149, 237) # CornflowerBlue
]

class ImageGraphicsView(QGraphicsView):
    mouseClicked = Signal(QPointF, int)  # int: 1 for left, 0 for right (SAM)
    polygonClicked = Signal(QPointF, int) # int: 1 for left, 0 for right (Polygon)

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

    def keyPressEvent(self, event):
        # Shift 키를 누르면 드래그 모드를 해제하고 마우스 커서를 일반 삼각형 화살표로 변경
        if event.key() == Qt.Key_Shift:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # Shift 키를 떼면 다시 화면 이동(손바닥) 모드로 복구
        if event.key() == Qt.Key_Shift:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            scene_pos = self.mapToScene(event.position().toPoint())
            label = 1 if event.button() == Qt.LeftButton else 0

            if event.modifiers() == Qt.ShiftModifier:
                # Shift + Click => Polygon Tool
                self.polygonClicked.emit(scene_pos, label)
                event.accept()
            else:
                # Normal Click => SAM Tool
                self.mouseClicked.emit(scene_pos, label)
                # ScrollHandDrag를 유지하기 위해 super() 호출
                super().mousePressEvent(event)
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
        self.show_masks = True  # 뷰어 마스크 표시 상태 토글 변수

        # Instance & Scene State
        self.instances = []    # list of dicts: id, class, bbox, mask (np.array)
        self.relations = []    # list of dicts: subject, predicate, object

        # SAM & Polygon 시퀀스 State
        self.input_points = []
        self.input_labels = []
        self.current_sam_mask = None
        self.polygon_points = [] # 직접 선을 따기 위한 좌표 리스트
        
        self.preview_dialog = None # 미리보기 창 상태 저장용 변수 추가

        self.init_ui()
        self.init_shortcuts()  # 단축키 초기화
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

    def init_shortcuts(self):
        # A 단축키: 이전 이미지
        shortcut_prev_img = QShortcut(QKeySequence(Qt.Key_A), self)
        shortcut_prev_img.activated.connect(self.prev_image)

        # D 단축키: 다음 이미지
        shortcut_next_img = QShortcut(QKeySequence(Qt.Key_D), self)
        shortcut_next_img.activated.connect(self.next_image)

        # Shift+S 단축키: 저장
        shortcut_save = QShortcut(QKeySequence("Shift+S"), self)
        shortcut_save.activated.connect(self.save_scene)

        # S 단축키: Add Mask as Instance
        shortcut_add_inst = QShortcut(QKeySequence(Qt.Key_S), self)
        shortcut_add_inst.activated.connect(self.add_instance)
        
        # E 단축키: Edit Selected Instance Mask (덮어쓰기)
        shortcut_edit_inst = QShortcut(QKeySequence(Qt.Key_E), self)
        shortcut_edit_inst.activated.connect(self.edit_instance)

        # R 단축키: Append to Selected Instance Mask (영역 추가/합치기)
        shortcut_append_inst = QShortcut(QKeySequence(Qt.Key_R), self)
        shortcut_append_inst.activated.connect(self.append_to_instance)
        
        # C 단축키: SAM/폴리곤 초기화
        shortcut_clear = QShortcut(QKeySequence(Qt.Key_C), self)
        shortcut_clear.activated.connect(self.clear_sam_prompts)

        # V 단축키: 마스크 보이기/숨기기 토글
        shortcut_toggle_masks = QShortcut(QKeySequence(Qt.Key_V), self)
        shortcut_toggle_masks.activated.connect(self.toggle_masks_visibility)

        # Shift+F 단축키: 이전 파일 데이터 불러오기
        shortcut_load_prev = QShortcut(QKeySequence("Shift+F"), self)
        shortcut_load_prev.activated.connect(self.load_previous_annotation)
        
        # Shift+V 단축키: 마스크 겹침 미리보기 토글
        shortcut_mask_preview = QShortcut(QKeySequence("Shift+V"), self)
        shortcut_mask_preview.setContext(Qt.ApplicationShortcut) # 창이 띄워져 있을 때도 인식하도록 설정
        shortcut_mask_preview.activated.connect(self.show_mask_preview)

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
        self.view.polygonClicked.connect(self.handle_polygon_click)
        center_layout.addWidget(self.view)

        # Controls Hint
        sam_ctrl_layout = QHBoxLayout()
        btn_clear_sam = QPushButton("🧹 Clear Prompts (C)")
        btn_clear_sam.clicked.connect(self.clear_sam_prompts)
        
        btn_toggle_masks = QPushButton("👁️ 마스크 숨기기/보기 (V)")
        btn_toggle_masks.clicked.connect(self.toggle_masks_visibility)

        hint_text = ("<i>좌/우클릭: SAM 추가/제거 | Shift+좌클릭: 폴리곤 추가 | Shift+우클릭: 닫기</i><br>"
                     "<i>이전 이미지: A | 다음 이미지: D</i>")
        sam_ctrl_layout.addWidget(QLabel(hint_text))
        sam_ctrl_layout.addWidget(btn_toggle_masks)
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

        btn_add_inst = QPushButton("➕ Add Mask as Instance (단축키: S)")
        btn_add_inst.setStyleSheet("background-color: #2196F3; color: white;")
        btn_add_inst.clicked.connect(self.add_instance)
        right_layout.addWidget(btn_add_inst)

        right_layout.addWidget(QLabel("<b>Instances: (리스트 하단일수록 앞 레이어)</b>"))
        self.inst_list_widget = QListWidget()
        # 리스트 선택이 바뀔 때마다 뷰어 새로고침 (선택된 항목 강조를 위함)
        self.inst_list_widget.itemSelectionChanged.connect(self.redraw_scene)
        right_layout.addWidget(self.inst_list_widget)

        # Edit / Append Buttons Layout
        edit_layout = QHBoxLayout()
        btn_edit_inst = QPushButton("✏️ 덮어쓰기 (E)")
        btn_edit_inst.setStyleSheet("background-color: #9C27B0; color: white;")
        btn_edit_inst.clicked.connect(self.edit_instance)

        btn_append_inst = QPushButton("🧩 영역 합치기 (R)")
        btn_append_inst.setStyleSheet("background-color: #E91E63; color: white;")
        btn_append_inst.clicked.connect(self.append_to_instance)

        edit_layout.addWidget(btn_edit_inst)
        edit_layout.addWidget(btn_append_inst)
        right_layout.addLayout(edit_layout)

        btn_del_inst = QPushButton("🗑️ Delete Selected Instance")
        btn_del_inst.clicked.connect(self.delete_instance)
        right_layout.addWidget(btn_del_inst)

        # Layer Ordering Buttons
        order_layout = QHBoxLayout()
        btn_up_inst = QPushButton("🔼 위로 (뒤로 보내기)")
        btn_down_inst = QPushButton("🔽 아래로 (앞으로 오기)")
        btn_up_inst.clicked.connect(self.move_instance_up)
        btn_down_inst.clicked.connect(self.move_instance_down)
        order_layout.addWidget(btn_up_inst)
        order_layout.addWidget(btn_down_inst)
        right_layout.addLayout(order_layout)

        # Previous File Load Button
        btn_load_prev = QPushButton("⏮️ 이전 데이터 불러오기 (Shift+F)")
        btn_load_prev.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        btn_load_prev.clicked.connect(self.load_previous_annotation)
        right_layout.addWidget(btn_load_prev)

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

        ## 3. Scene Meta & Save
        right_layout.addWidget(QLabel("<b>3. Scene Attributes</b>"))
        self.sit_combo = QComboBox()
        self.sit_combo.addItems(SITUATIONS)
        self.pmod_combo = QComboBox()
        self.pmod_combo.addItems(PATH_MODS)
        right_layout.addWidget(self.sit_combo)
        right_layout.addWidget(self.pmod_combo)

        btn_preview = QPushButton("👁️ 마스크 겹침 미리보기 (Shift+V)")
        btn_preview.clicked.connect(self.show_mask_preview)
        right_layout.addWidget(btn_preview)

        btn_save = QPushButton("💾 Save Scene (단축키: Shift+S)")
        btn_save.setFixedHeight(50)
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_scene)
        right_layout.addWidget(btn_save)

        splitter.addWidget(right_panel)

        # Splitter ratios
        splitter.setSizes([200, 800, 400])

    def keyPressEvent(self, event):
        # 뷰어가 아닌 메인 윈도우가 포커스를 가질 때도 Shift 키 감지
        if event.key() == Qt.Key_Shift:
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.view.viewport().setCursor(Qt.ArrowCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.viewport().unsetCursor()
        super().keyReleaseEvent(event)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 자연스러운 정렬 (Natural Sorting)
            def natural_keys(text):
                return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
            
            files.sort(key=natural_keys)
            
            self.image_paths = [os.path.join(folder, f) for f in files]

            self.img_list_widget.clear()
            self.img_list_widget.addItems(files)

            if self.image_paths:
                self.img_list_widget.setCurrentRow(0)

    def prev_image(self):
        """A 단축키: 이전 이미지로 이동"""
        if self.current_idx > 0:
            self.img_list_widget.setCurrentRow(self.current_idx - 1)

    def next_image(self):
        """D 단축키: 다음 이미지로 이동"""
        if self.current_idx < len(self.image_paths) - 1:
            self.img_list_widget.setCurrentRow(self.current_idx + 1)

    def toggle_masks_visibility(self):
        """V 단축키: 마스크 오버레이 보이기/숨기기 토글"""
        self.show_masks = not self.show_masks
        self.redraw_scene()
        if self.show_masks:
            self.statusBar().showMessage("👁️ 마스크 표시가 켜졌습니다.", 2000)
        else:
            self.statusBar().showMessage("👁️ 마스크가 일시적으로 숨겨졌습니다. (숨김 상태에서도 작업은 정상 저장됩니다)", 3000)

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
        
        # 선택된 인스턴스 초기화 (에러 방지)
        self.inst_list_widget.clearSelection()

        # Set SAM image
        img_np = np.array(self.current_image_pil)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(img_np)

        # 1. 먼저 빈 UI로 초기화
        self.update_ui_lists()
        
        # 2. 기존 Annotation 시도 (여기서 리스트가 채워짐)
        self.try_load_existing_annotation(img_path)
        
        # 3. 최종적으로 UI와 뷰어 동기화 업데이트
        self.update_ui_lists()
        self.redraw_scene()

    def load_previous_annotation(self):
        """이전 이미지의 마스크와 설정값들을 그대로 불러옵니다."""
        if self.current_idx <= 0:
            QMessageBox.warning(self, "알림", "첫 번째 이미지이거나 이전 이미지가 없습니다.")
            return

        prev_img_path = self.image_paths[self.current_idx - 1]
        base_dir = os.path.dirname(prev_img_path)
        jsonl_path = os.path.join(os.path.dirname(base_dir), "dataset", "manual_labeled.jsonl")

        if not os.path.exists(jsonl_path):
            QMessageBox.warning(self, "알림", "저장된 데이터 파일(manual_labeled.jsonl)이 존재하지 않습니다.")
            return

        found = False
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if data.get("image_path") == prev_img_path:
                    # 데이터 찾음. 현재 인스턴스에 복사
                    self.instances = copy.deepcopy(data.get("objects", []))
                    
                    prev_mask_path = data.get("mask_path", "")
                    if os.path.exists(prev_mask_path):
                        total_mask = np.array(Image.open(prev_mask_path))
                        for inst in self.instances:
                            inst_id_int = int(inst["id"][1:])
                            inst["mask"] = (total_mask == inst_id_int).astype(np.uint8)

                    self.relations = copy.deepcopy(data.get("relationships", []))

                    sit = data.get("situation", "S1")
                    pmod = data.get("path_modification", "normal")

                    for i in range(self.sit_combo.count()):
                        if self.sit_combo.itemText(i).startswith(sit):
                            self.sit_combo.setCurrentIndex(i)
                            break

                    idx = self.pmod_combo.findText(pmod)
                    if idx >= 0: self.pmod_combo.setCurrentIndex(idx)

                    found = True
                    break

        if not found:
            QMessageBox.warning(self, "알림", f"이전 파일({os.path.basename(prev_img_path)})에 대한 저장 기록을 찾을 수 없습니다.")
        else:
            self.update_ui_lists()
            self.redraw_scene()
            self.statusBar().showMessage("이전 파일의 데이터를 성공적으로 불러왔습니다.", 3000)

    def handle_sam_click(self, pos, label):
        if not self.current_image_pil: return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self.current_image_pil.width and 0 <= y < self.current_image_pil.height:
            # SAM 클릭 시 기존 폴리곤 진행상황은 초기화
            self.polygon_points.clear() 
            self.input_points.append([x, y])
            self.input_labels.append(label)
            self.run_sam()

    def handle_polygon_click(self, pos, label):
        if not self.current_image_pil: return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < self.current_image_pil.width and 0 <= y < self.current_image_pil.height:
            if label == 1: # Shift + Left Click (점 추가)
                self.polygon_points.append((x, y))
                self.redraw_scene()
            elif label == 0: # Shift + Right Click (폴리곤 닫기 및 마스크 생성)
                if len(self.polygon_points) > 2:
                    w, h = self.current_image_pil.size
                    mask_img = Image.new('L', (w, h), 0)
                    ImageDraw.Draw(mask_img).polygon(self.polygon_points, outline=1, fill=1)
                    
                    self.current_sam_mask = np.array(mask_img, dtype=np.uint8)
                    self.polygon_points.clear() # 폴리곤 상태 초기화
                    self.redraw_scene()
                else:
                    QMessageBox.warning(self, "Warning", "다각형 마스크를 생성하려면 최소 3개의 점을 찍어주세요.")
                    self.polygon_points.clear()
                    self.redraw_scene()

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
        self.polygon_points.clear()
        self.current_sam_mask = None
        self.redraw_scene()
        self.statusBar().showMessage("Prompts Cleared", 2000)

    def add_instance(self):
        if self.current_sam_mask is None or np.sum(self.current_sam_mask) == 0:
            QMessageBox.warning(self, "Warning", "No valid SAM/Polygon mask to add.")
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
        
        # 방금 추가된 인스턴스를 리스트에서 자동으로 선택
        self.inst_list_widget.setCurrentRow(len(self.instances) - 1)

    def edit_instance(self):
        """선택한 인스턴스의 마스크와 위치(BBox)만 새것으로 완전히 덮어씁니다."""
        row = self.inst_list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "알림", "리스트에서 덮어씌울(Edit) 인스턴스를 먼저 선택해주세요.")
            return

        if self.current_sam_mask is None or np.sum(self.current_sam_mask) == 0:
            QMessageBox.warning(self, "알림", "새로 업데이트할 유효한 SAM/Polygon 마스크가 없습니다.\n마스크를 먼저 그려주세요.")
            return

        # Get new BBox from new mask
        y_indices, x_indices = np.where(self.current_sam_mask > 0)
        x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
        x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))

        # 기존 인스턴스 정보 중 마스크와 bbox만 덮어쓰기 (id와 class는 유지)
        self.instances[row]["bbox"] = [x1, y1, x2, y2]
        self.instances[row]["mask"] = self.current_sam_mask.copy()

        # 프롬프트 정리 및 화면 업데이트
        self.clear_sam_prompts()
        self.update_ui_lists()
        self.inst_list_widget.setCurrentRow(row)  # 선택 상태 유지
        self.redraw_scene()
        self.statusBar().showMessage(f"[{self.instances[row]['id']}] 마스크가 새로운 영역으로 덮어쓰기 되었습니다.", 3000)

    def append_to_instance(self):
        """선택한 인스턴스의 기존 마스크에 현재 그린 마스크를 추가(합치기)합니다."""
        row = self.inst_list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "알림", "리스트에서 영역을 추가할 인스턴스를 먼저 선택해주세요.")
            return

        if self.current_sam_mask is None or np.sum(self.current_sam_mask) == 0:
            QMessageBox.warning(self, "알림", "추가할 유효한 SAM/Polygon 마스크가 없습니다.\n마스크를 먼저 그려주세요.")
            return

        # 기존 마스크 가져와서 합치기 (Logical OR)
        old_mask = self.instances[row].get("mask", None)
        if old_mask is None:
            new_combined_mask = self.current_sam_mask.copy()
        else:
            new_combined_mask = np.logical_or(old_mask > 0, self.current_sam_mask > 0).astype(np.uint8)

        # 합쳐진 마스크를 기준으로 새로운 BBox 계산
        y_indices, x_indices = np.where(new_combined_mask > 0)
        x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
        x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))

        # 정보 업데이트
        self.instances[row]["bbox"] = [x1, y1, x2, y2]
        self.instances[row]["mask"] = new_combined_mask

        # 프롬프트 정리 및 화면 업데이트
        self.clear_sam_prompts()
        self.update_ui_lists()
        self.inst_list_widget.setCurrentRow(row)  # 선택 상태 유지
        self.redraw_scene()
        self.statusBar().showMessage(f"[{self.instances[row]['id']}] 마스크 영역이 추가(병합)되었습니다.", 3000)

    def move_instance_up(self):
        row = self.inst_list_widget.currentRow()
        if row > 0:
            self.instances[row - 1], self.instances[row] = self.instances[row], self.instances[row - 1]
            self.update_ui_lists()
            self.inst_list_widget.setCurrentRow(row - 1)
            self.redraw_scene()

    def move_instance_down(self):
        row = self.inst_list_widget.currentRow()
        if row >= 0 and row < len(self.instances) - 1:
            self.instances[row + 1], self.instances[row] = self.instances[row], self.instances[row + 1]
            self.update_ui_lists()
            self.inst_list_widget.setCurrentRow(row + 1)
            self.redraw_scene()

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
        # 1. 콤보박스의 현재 선택 상태 저장
        curr_subj = self.subj_combo.currentText()
        curr_obj = self.obj_combo.currentText()
        curr_pred = self.pred_combo.currentText()

        # Update Instance List
        self.inst_list_widget.blockSignals(True) # 리스트 업데이트 중 이벤트 발생 방지
        self.inst_list_widget.clear()
        for inst in self.instances:
            self.inst_list_widget.addItem(f"{inst['id']}: {inst['class']} {inst['bbox']}")
        self.inst_list_widget.blockSignals(False)

        # Update Comboboxes
        self.subj_combo.blockSignals(True)
        self.obj_combo.blockSignals(True)
        self.subj_combo.clear()
        self.obj_combo.clear()
        
        items = [f"{inst['id']}: {inst['class']}" for inst in self.instances]
        self.subj_combo.addItems(items)
        self.obj_combo.addItems(items)

        # 2. 저장했던 선택 상태 복원 (리스트에 해당 항목이 여전히 있을 경우)
        if curr_subj in items:
            self.subj_combo.setCurrentText(curr_subj)
        if curr_obj in items:
            self.obj_combo.setCurrentText(curr_obj)
        if curr_pred:
            self.pred_combo.setCurrentText(curr_pred)

        self.subj_combo.blockSignals(False)
        self.obj_combo.blockSignals(False)

        # Update Relation List
        self.rel_list_widget.clear()
        for r in self.relations:
            self.rel_list_widget.addItem(f"{r['subject']}  - [{r['predicate']}] ->  {r['object']}")

        # 3. 데이터가 갱신되었으므로 화면 즉시 다시 그리기
        self.redraw_scene()

        # 4. 새로 추가된 부분: 리스트의 스크롤을 맨 아래로 고정
        self.inst_list_widget.scrollToBottom()
        self.rel_list_widget.scrollToBottom()

    def redraw_scene(self):
        if not self.current_image_pil: return
        self.scene.clear()

        # 1. Base Image
        img_q = self.pil_to_qimage(self.current_image_pil)
        selected_row = self.inst_list_widget.currentRow()
        painter = QPainter(img_q)

        # show_masks가 True일 때만 오버레이 요소를 그립니다.
        if self.show_masks:
            # 2. Draw Committed Instance Masks & BBoxes
            for i, inst in enumerate(self.instances):
                inst_id_int = int(inst["id"][1:])
                color = COLORS[inst_id_int % len(COLORS)]
                
                # 선택된 인스턴스면 불투명도(alpha)를 높여 찐하게 표시하고, 아니면 투명도를 낮춤
                if selected_row == -1:
                    alpha_val = 100
                    pen_width = 2
                else:
                    alpha_val = 220 if i == selected_row else 60
                    pen_width = 4 if i == selected_row else 1
                
                # Draw Mask
                if "mask" in inst and inst["mask"] is not None:
                    mask = inst["mask"]
                    mask_img = Image.fromarray((mask * alpha_val).astype(np.uint8), mode='L')
                    mask_rgba = Image.new("RGBA", mask_img.size, color + (0,))
                    mask_rgba.putalpha(mask_img)
                    q_mask = self.pil_to_qimage(mask_rgba)
                    painter.drawImage(0, 0, q_mask)

                # Draw BBox
                b = inst["bbox"]
                painter.setPen(QPen(QColor(*color), pen_width, Qt.SolidLine))
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

                    text_rect.translate(int(mid_p.x() - text_rect.width()/2), int(mid_p.y() - text_rect.height()/2))
                    text_rect.adjust(-6, -4, 6, 4) 

                    painter.setBrush(QColor(0, 0, 0, 180)) 
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(text_rect)

                    painter.setPen(Qt.yellow)
                    painter.drawText(text_rect, Qt.AlignCenter, pred)

            # 4. Draw Current Active SAM / Polygon Mask
            if self.current_sam_mask is not None:
                mask = self.current_sam_mask
                mask_img = Image.fromarray((mask * 128).astype(np.uint8), mode='L')
                mask_rgba = Image.new("RGBA", mask_img.size, (255, 255, 0, 0)) # Yellow for active
                mask_rgba.putalpha(mask_img)
                q_mask = self.pil_to_qimage(mask_rgba)
                painter.drawImage(0, 0, q_mask)

        # 5. Draw SAM Prompts (작업 진행 상황은 마스크를 숨겨도 항상 표시)
        for pt, lbl in zip(self.input_points, self.input_labels):
            color = Qt.green if lbl == 1 else Qt.red
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(pt[0]-4, pt[1]-4, 8, 8)
            
        # 6. Draw Polygon Lines/Points (작업 진행 상황은 항상 표시)
        if self.polygon_points:
            painter.setPen(QPen(Qt.magenta, 2, Qt.SolidLine))
            for i in range(len(self.polygon_points) - 1):
                p1 = QPointF(self.polygon_points[i][0], self.polygon_points[i][1])
                p2 = QPointF(self.polygon_points[i+1][0], self.polygon_points[i+1][1])
                painter.drawLine(p1, p2)
            painter.setBrush(Qt.magenta)
            painter.setPen(Qt.NoPen)
            for pt in self.polygon_points:
                painter.drawEllipse(pt[0]-3, pt[1]-3, 6, 6)

        painter.end()
        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(img_q)))

    def show_mask_preview(self):
        """저장될 마스크와 동일한 우선순위(레이어 순서)로 시각화된 RGB 팝업창을 띄우거나 닫습니다."""
        
        # 이미 창이 열려있다면 닫고 종료 (토글 기능)
        if self.preview_dialog is not None and self.preview_dialog.isVisible():
            self.preview_dialog.close()
            return

        if not self.current_image_pil: return
        w, h = self.current_image_pil.size
        
        # 검은 배경에 덮어쓸 이미지
        preview_img = np.zeros((h, w, 3), dtype=np.uint8)

        # 저장될 때와 똑같은 순서로 덮어쓰기 진행
        for i, inst in enumerate(self.instances):
            if "mask" in inst and inst["mask"] is not None:
                inst_id_int = int(inst["id"][1:])
                color = COLORS[inst_id_int % len(COLORS)] # 리스트 컬러와 매칭
                preview_img[inst["mask"] > 0] = color

        # QDialog 생성 (인스턴스 변수에 저장)
        self.preview_dialog = QDialog(self)
        self.preview_dialog.setWindowTitle("마스크 겹침 결과 미리보기 (우측 UI 리스트 하단일수록 앞 레이어)")
        layout = QVBoxLayout(self.preview_dialog)
        lbl = QLabel()

        bytes_per_line = 3 * w
        q_img = QImage(preview_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 이미지가 너무 크면 창 크기에 맞게 축소
        if w > 1000 or h > 800:
            pixmap = pixmap.scaled(1000, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        lbl.setPixmap(pixmap)
        layout.addWidget(lbl)
        
        # 기존 exec() 대신 show()를 사용해야 메인 창과 동시에 조작 가능
        self.preview_dialog.show()

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
                # np.where: mask가 1인 곳을 현재 inst_id로 덮어쓰므로 순서에 따라 덮어씌워짐.
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
                    break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SceneGraphLabeler()
    ex.show()
    sys.exit(app.exec())
