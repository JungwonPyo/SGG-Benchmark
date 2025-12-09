"""
Main Application for COCO Relations Tool - PyQt6 Version

This module contains the main application class for the PyQt6-based
COCO relationship annotation tool.
"""

import sys
import json
import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QFileDialog, QMessageBox, QDialog,
    QProgressDialog, QGroupBox, QApplication, QMenu,
    QInputDialog, QLineEdit, QTextEdit, QPushButton, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from .components import ModernLabel, ModernButton, ModernListWidget
from .canvas import ImageCanvas
from .dialogs import PredicateSelectionDialog


class COCORelationAnnotatorPyQt(QMainWindow):
    """
    Modern PyQt6-based COCO Relations Annotation Tool.
    """
    
    def __init__(self, output_filename=None):
        super().__init__()
        self.output_filename = output_filename or "annotations_with_relationships.json"
        
        # Core data structures
        self.coco_data = None
        self.images_data = {}
        self.annotations_data = {}
        self.categories_data = {}
        self.images_folder = ""
        
        # Current state
        self.current_image_id = None
        self.current_image_index = 0
        self.image_ids = []
        self.current_bboxes = []
        
        # Relationship state
        self.relationships = []
        self.relationships_mapping = {}
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self.relation_creation_mode = True
        self.awaiting_object_selection = False
        
        # Annotation editing state
        self.add_bbox_mode = False
        self.bbox_start_pos = None
        self.bbox_end_pos = None
        self.selected_bbox_for_editing = None
        
        # Output directory
        self.output_dir = Path("output_coco_relations")
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        self.apply_dark_theme()
        
    def setup_ui(self):
        """Setup the main UI."""
        self.setWindowTitle("SGDET-Annotate - COCO Relations Tool (PyQt6)")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (image)
        self.create_image_panel(splitter)
        
        # Right panel (controls)
        self.create_control_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([800, 400])
        
        # Status bar
        self.statusBar().showMessage("Ready - Load COCO data to begin")
        
        # Create menu bar
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_coco_action = QAction('Load COCO JSON', self)
        load_coco_action.setShortcut('Ctrl+O')
        load_coco_action.triggered.connect(self.load_coco_file)
        file_menu.addAction(load_coco_action)
        
        load_relationships_action = QAction('Load Relationships', self)
        load_relationships_action.setShortcut('Ctrl+R')
        load_relationships_action.triggered.connect(self.load_relationships)
        file_menu.addAction(load_relationships_action)
        
        file_menu.addSeparator()
        
        save_action = QAction('Save Relationships', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_relationships)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        prev_action = QAction('Previous Image', self)
        prev_action.setShortcut('Left')
        prev_action.triggered.connect(self.previous_image)
        view_menu.addAction(prev_action)
        
        next_action = QAction('Next Image', self)
        next_action.setShortcut('Right')
        next_action.triggered.connect(self.next_image)
        view_menu.addAction(next_action)
        
        view_menu.addSeparator()
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        toggle_relation_action = QAction('Toggle Relation Mode', self)
        toggle_relation_action.setShortcut('Ctrl+T')
        toggle_relation_action.triggered.connect(self.toggle_relation_mode)
        edit_menu.addAction(toggle_relation_action)
        
        cancel_action = QAction('Cancel Operation', self)
        cancel_action.setShortcut('Escape')
        cancel_action.triggered.connect(self.cancel_operation)
        edit_menu.addAction(cancel_action)
        
    def create_image_panel(self, parent):
        """Create the image display panel."""
        image_widget = QWidget()
        image_layout = QVBoxLayout()
        image_widget.setLayout(image_layout)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_btn = ModernButton("◀ Previous", "secondary")
        self.prev_btn.clicked.connect(self.previous_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.image_info_label = ModernLabel("No COCO data loaded", "status")
        nav_layout.addWidget(self.image_info_label)
        
        self.next_btn = ModernButton("Next ▶", "secondary")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        image_layout.addLayout(nav_layout)
        
        # Image canvas
        self.image_canvas = ImageCanvas()
        image_layout.addWidget(self.image_canvas)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.toggle_relation_btn = ModernButton("Create Relationship", "primary")
        self.toggle_relation_btn.clicked.connect(self.toggle_relation_mode)
        self.toggle_relation_btn.setEnabled(False)
        action_layout.addWidget(self.toggle_relation_btn)
        
        self.save_btn = ModernButton("Save Relations", "success")
        self.save_btn.clicked.connect(self.save_relationships)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)
        
        self.cancel_btn = ModernButton("Cancel", "danger")
        self.cancel_btn.clicked.connect(self.cancel_operation)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)
        
        image_layout.addLayout(action_layout)
        
        parent.addWidget(image_widget)
        
    def create_control_panel(self, parent):
        """Create the control panel."""
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        
        # Load data section
        data_group = QGroupBox("Data Loading")
        data_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #E5E9F0;
                border: 2px solid #4C566A;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        data_layout = QVBoxLayout()
        
        self.load_coco_btn = ModernButton("Load COCO JSON", "primary")
        self.load_coco_btn.clicked.connect(self.load_coco_file)
        data_layout.addWidget(self.load_coco_btn)
        
        self.load_images_btn = ModernButton("Load Images Only", "success")
        self.load_images_btn.clicked.connect(self.load_images_only)
        data_layout.addWidget(self.load_images_btn)
        
        self.load_rel_btn = ModernButton("Load Relationships", "secondary")
        self.load_rel_btn.clicked.connect(self.load_relationships)
        data_layout.addWidget(self.load_rel_btn)
        
        self.load_existing_rel_btn = ModernButton("Load Existing Relations", "warning")
        self.load_existing_rel_btn.clicked.connect(self.load_existing_relationships_from_file)
        data_layout.addWidget(self.load_existing_rel_btn)
        
        data_group.setLayout(data_layout)
        control_layout.addWidget(data_group)
        
        # Available predicates
        rel_group = QGroupBox("Available Predicates")
        rel_group.setStyleSheet(data_group.styleSheet())
        rel_layout = QVBoxLayout()
        
        self.relationships_list = ModernListWidget()
        self.relationships_list.itemDoubleClicked.connect(self.on_predicate_select)
        rel_layout.addWidget(self.relationships_list)
        
        # Add predicate button
        self.add_predicate_btn = ModernButton("+ Add predicate")
        self.add_predicate_btn.clicked.connect(self.add_new_predicate)
        rel_layout.addWidget(self.add_predicate_btn)
        
        rel_group.setLayout(rel_layout)
        control_layout.addWidget(rel_group)
        
        # Current image objects
        objects_group = QGroupBox("Objects in Current Image")
        objects_group.setStyleSheet(data_group.styleSheet())
        objects_layout = QVBoxLayout()
        
        objects_label = ModernLabel("Click objects to select for relationships:")
        objects_layout.addWidget(objects_label)
        
        self.objects_list = ModernListWidget()
        self.objects_list.itemClicked.connect(self.on_object_list_click)
        self.objects_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.objects_list.customContextMenuRequested.connect(self.show_object_context_menu)
        objects_layout.addWidget(self.objects_list)
        
        # Add category button
        self.add_category_btn = ModernButton("+ Add Category")
        self.add_category_btn.clicked.connect(self.add_new_category)
        objects_layout.addWidget(self.add_category_btn)
        
        objects_group.setLayout(objects_layout)
        control_layout.addWidget(objects_group)
        
        # Annotation editing
        annotation_group = QGroupBox("Annotation Editing")
        annotation_group.setStyleSheet(data_group.styleSheet())
        annotation_layout = QVBoxLayout()
        
        # Add bounding box button
        self.add_bbox_btn = ModernButton("+ Add Bounding Box")
        self.add_bbox_btn.clicked.connect(self.enter_add_bbox_mode)
        annotation_layout.addWidget(self.add_bbox_btn)
        
        # Edit label button
        self.edit_label_btn = ModernButton("✏️ Edit Label")
        self.edit_label_btn.clicked.connect(self.edit_selected_label)
        self.edit_label_btn.setEnabled(False)
        annotation_layout.addWidget(self.edit_label_btn)
        
        annotation_group.setLayout(annotation_layout)
        control_layout.addWidget(annotation_group)
        
        # Current relationships
        current_rel_group = QGroupBox("Current Image Relationships")
        current_rel_group.setStyleSheet(data_group.styleSheet())
        current_rel_layout = QVBoxLayout()
        
        self.current_relationships_list = ModernListWidget()
        self.current_relationships_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.current_relationships_list.customContextMenuRequested.connect(self.show_relationship_context_menu)
        current_rel_layout.addWidget(self.current_relationships_list)
        
        current_rel_group.setLayout(current_rel_layout)
        control_layout.addWidget(current_rel_group)
        
        # Add stretch to push everything up
        control_layout.addStretch()
        
        parent.addWidget(control_widget)
        
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
                color: #E5E9F0;
            }
            QMenuBar {
                background-color: #3B4252;
                color: #E5E9F0;
                border-bottom: 1px solid #4C566A;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
            }
            QMenuBar::item:selected {
                background-color: #434C5E;
            }
            QMenu {
                background-color: #3B4252;
                color: #E5E9F0;
                border: 1px solid #4C566A;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #434C5E;
            }
            QStatusBar {
                background-color: #3B4252;
                color: #E5E9F0;
                border-top: 1px solid #4C566A;
            }
            QSplitter::handle {
                background-color: #4C566A;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #5E81AC;
            }
        """)
        
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_operation()
        elif event.key() == Qt.Key.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_Space:
            self.save_relationships()
        else:
            super().keyPressEvent(event)
        
    # Data loading methods
    def load_coco_file(self):
        """Load a COCO format JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COCO JSON File",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # Show progress dialog
            progress = QProgressDialog("Loading COCO data...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            # Load and parse COCO data
            with open(file_path, 'r') as f:
                self.coco_data = json.load(f)
            
            self.parse_coco_data()
            progress.close()
            
            # Ask for images folder
            images_folder = QFileDialog.getExistingDirectory(
                self,
                "Select Images Folder",
                str(Path(file_path).parent)
            )
            
            if not images_folder:
                QMessageBox.warning(self, "Images Folder Required", 
                                  "Images folder is required to display images.")
                return
                
            self.images_folder = images_folder
            
            # Load first image
            if self.image_ids:
                self.current_image_index = 0
                self.load_image_at_index(0)
                self.update_navigation_state()
                
            QMessageBox.information(self, "Success", 
                                  f"Successfully loaded {len(self.image_ids)} images with annotations.")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading COCO file:\n{str(e)}")
            
    def parse_coco_data(self):
        """Parse COCO data into usable structures."""
        if not self.coco_data:
            return
            
        # Parse images
        self.images_data = {img['id']: img for img in self.coco_data.get('images', [])}
        
        # Parse categories
        self.categories_data = {cat['id']: cat for cat in self.coco_data.get('categories', [])}
        
        # Parse annotations by image
        self.annotations_data = {}
        for ann in self.coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in self.annotations_data:
                self.annotations_data[image_id] = []
            self.annotations_data[image_id].append(ann)
        
        # Get image IDs that have annotations
        self.image_ids = [img_id for img_id in self.images_data.keys() 
                         if img_id in self.annotations_data]
        self.image_ids.sort()
        
        # Load existing relationships if they exist
        self.load_existing_relationships_from_coco()
        
    def load_images_only(self):
        """Load images without COCO JSON - create annotations from scratch."""
        images_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Images Folder",
            ""
        )
        
        if not images_folder:
            return
            
        try:
            # Show progress dialog
            progress = QProgressDialog("Loading images...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            self.images_folder = images_folder
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(images_folder).glob(f'*{ext}'))
                image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))
            
            if not image_files:
                QMessageBox.warning(self, "No Images Found", 
                                  "No image files found in the selected folder.")
                progress.close()
                return
            
            # Create minimal COCO structure
            self.coco_data = {
                "images": [],
                "annotations": [],
                "categories": [],
                "info": {
                    "description": "Created with SGDET-Annotate",
                    "version": "1.0",
                    "year": 2025
                }
            }
            
            # Add images to COCO structure
            for i, img_path in enumerate(sorted(image_files)):
                # Get image dimensions (basic implementation)
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        width, height = img.size
                except:
                    width, height = 640, 480  # Default size if can't read
                
                image_entry = {
                    "id": i + 1,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height
                }
                self.coco_data["images"].append(image_entry)
            
            # Start with empty categories - user will add them as needed
            # Categories will be added dynamically using the "Add Category" button
            
            # Parse the created COCO data
            self.parse_coco_data()
            
            # Since there are no annotations yet, all images are available
            self.image_ids = [img['id'] for img in self.coco_data['images']]
            self.image_ids.sort()
            
            progress.close()
            
            # Load first image
            if self.image_ids:
                self.current_image_index = 0
                self.load_image_at_index(0)
                self.update_navigation_state()
                
            QMessageBox.information(self, "Success", 
                                  f"Successfully loaded {len(self.image_ids)} images for annotation.\n"
                                  f"You can now start adding bounding boxes and relationships.")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading images:\n{str(e)}")
            
    def get_initial_categories(self):
        """Get initial categories from user."""
        
        class CategoriesDialog(QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Define Object Categories")
                self.setModal(True)
                self.resize(400, 300)
                self.categories = []
                
                layout = QVBoxLayout()
                
                # Instructions
                label = QLabel("Enter object categories (one per line):\nExample:\nperson\ncar\nbicycle\ndog")
                layout.addWidget(label)
                
                # Text area
                self.text_edit = QTextEdit()
                self.text_edit.setPlainText("person\ncar\nbicycle")  # Default categories
                layout.addWidget(self.text_edit)
                
                # Buttons
                button_layout = QHBoxLayout()
                ok_btn = QPushButton("OK")
                ok_btn.clicked.connect(self.accept)
                cancel_btn = QPushButton("Cancel")
                cancel_btn.clicked.connect(self.reject)
                
                button_layout.addWidget(ok_btn)
                button_layout.addWidget(cancel_btn)
                layout.addLayout(button_layout)
                
                self.setLayout(layout)
                
            def get_categories(self):
                text = self.text_edit.toPlainText().strip()
                if text:
                    categories = [line.strip() for line in text.split('\n') if line.strip()]
                    return categories
                return []
        
        dialog = CategoriesDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_categories()
        return []
        
    def load_relationships(self):
        """Load relationship list from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Relationship List File",
            "",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            self.relationships_mapping = {}
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    relationship = line.strip()
                    if relationship:
                        self.relationships_mapping[relationship] = i
            
            # Update relationships list
            self.relationships_list.clear()
            for rel in sorted(self.relationships_mapping.keys()):
                self.relationships_list.addItem(rel)
                
            self.update_navigation_state()
            
            QMessageBox.information(self, "Success", 
                                  f"Successfully imported {len(self.relationships_mapping)} relationships.")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error importing relationships:\n{str(e)}")
    
    # Image loading and navigation methods
    def load_image_at_index(self, index):
        """Load image at specific index."""
        if index < 0 or index >= len(self.image_ids):
            return
            
        self.current_image_index = index
        image_id = self.image_ids[index]
        self.current_image_id = image_id
        
        # Get image info
        image_info = self.images_data[image_id]
        image_filename = image_info['file_name']
        
        # Load image
        image_path = os.path.join(self.images_folder, image_filename)
        
        try:
            success = self.image_canvas.set_image(image_path)
            if success:
                self.load_current_bboxes()
                self.update_image_info()
                self.load_existing_relationships()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
            
    def load_current_bboxes(self):
        """Load and display bounding boxes for current image."""
        self.current_bboxes = []
        
        if self.current_image_id not in self.annotations_data:
            self.update_objects_list()
            return
            
        annotations = self.annotations_data[self.current_image_id]
        
        for ann in annotations:
            category = self.categories_data.get(ann['category_id'], {'name': 'unknown'})
            
            bbox_info = {
                'id': ann['id'],
                'category_id': ann['category_id'],
                'category_name': category['name'],
                'original_bbox': ann['bbox'],
                'annotation': ann
            }
            
            self.current_bboxes.append(bbox_info)
        
        # Update canvas and objects list
        self.image_canvas.set_bboxes(self.current_bboxes)
        self.update_objects_list()
        
    def update_objects_list(self):
        """Update the objects list widget."""
        self.objects_list.clear()
        for bbox in self.current_bboxes:
            item_text = f"{bbox['category_name']}:{bbox['id']}"
            self.objects_list.addItem(item_text)
            
    def update_image_info(self):
        """Update image info label."""
        if not self.image_ids:
            self.image_info_label.setText("No COCO data loaded")
        else:
            current = self.current_image_index + 1
            total = len(self.image_ids)
            image_info = self.images_data[self.current_image_id]
            filename = image_info['file_name']
            self.image_info_label.setText(f"{current}/{total}: {filename}")
    
    def previous_image(self):
        """Navigate to previous image."""
        if not self.image_ids:
            return
            
        new_index = self.current_image_index - 1
        if new_index < 0:
            new_index = len(self.image_ids) - 1
        self.load_image_at_index(new_index)
        
    def next_image(self):
        """Navigate to next image."""
        if not self.image_ids:
            return
            
        new_index = self.current_image_index + 1
        if new_index >= len(self.image_ids):
            new_index = 0
        self.load_image_at_index(new_index)
    
    # State management methods
    def update_navigation_state(self):
        """Update navigation button states."""
        has_data = bool(self.image_ids)
        has_relations = bool(self.relationships_mapping)
        
        self.prev_btn.setEnabled(has_data)
        self.next_btn.setEnabled(has_data)
        # Enable relation button if we have data, even without relationships loaded
        self.toggle_relation_btn.setEnabled(has_data)
        self.save_btn.setEnabled(has_data)
        
        # Update relation mode button and cancel button
        if self.relation_creation_mode and has_data:
            if has_relations:
                if self.awaiting_object_selection:
                    self.toggle_relation_btn.setText("Relation Mode: Select Object")
                    status_msg = "Select Object & Predicate"
                else:
                    self.toggle_relation_btn.setText("Relation Mode: Select Subject")
                    status_msg = "Select Subject Object"
                self.cancel_btn.setEnabled(True)
                self.statusBar().showMessage(status_msg)
            else:
                self.toggle_relation_btn.setText("Relation Mode: Load Relationships First")
                self.cancel_btn.setEnabled(True)
                self.statusBar().showMessage("Load relationships to begin creating relations")
        else:
            if has_data and not has_relations:
                self.toggle_relation_btn.setText("Load Relationships First")
            else:
                self.toggle_relation_btn.setText("Create Relationship")
            self.cancel_btn.setEnabled(False)
            if has_data:
                self.statusBar().showMessage("Ready")
            else:
                self.statusBar().showMessage("Load COCO data to begin")
            
    def toggle_relation_mode(self):
        """Toggle relationship creation mode."""
        # Check if relationships are loaded
        if not self.relationships_mapping:
            QMessageBox.information(
                self, "Load Relationships First", 
                "Please load a relationships file first using 'Load Relationships' button.\n\n"
                "The relationships file should contain one relationship per line (e.g., 'on', 'under', 'next to')."
            )
            return
            
        self.relation_creation_mode = not self.relation_creation_mode
        self.reset_relation_mode()
        
    def reset_relation_mode(self):
        """Reset relationship creation mode."""
        self.awaiting_object_selection = False
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        
        # Clear visual selections on canvas
        self.image_canvas.set_selections(None, None)
        
        # Update UI state
        self.update_navigation_state()
        
        if self.relation_creation_mode:
            self.statusBar().showMessage("Select Subject Object")
        else:
            self.statusBar().showMessage("Ready")
        
    def cancel_operation(self):
        """Cancel current operation and disable relation mode."""
        self.relation_creation_mode = False
        self.awaiting_object_selection = False
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        
        # Clear visual selections on canvas
        self.image_canvas.set_selections()
        
        # Update UI state
        self.update_navigation_state()
        self.statusBar().showMessage("Operation cancelled - Ready")
    
    # Event handlers
    def on_bbox_clicked(self, bbox):
        """Handle bbox click from canvas."""
        # Handle different modes
        if self.add_bbox_mode:
            # In add bbox mode, ignore clicks on existing boxes
            return
            
        if self.relation_creation_mode:
            # Handle relationship creation
            if not self.relationships_mapping:
                QMessageBox.warning(
                    self, "No Relationships Loaded", 
                    "Please load a relationships file first using 'Load Relationships' button."
                )
                return
                
            if not self.awaiting_object_selection:
                # Select subject
                self.selected_subject_bbox = bbox
                self.awaiting_object_selection = True
                self.image_canvas.set_selections(subject_bbox=bbox)
                self.update_navigation_state()
            else:
                # Select object
                if bbox['id'] == self.selected_subject_bbox['id']:
                    QMessageBox.warning(self, "Invalid Selection", 
                                      "Subject and object cannot be the same.")
                    return
                    
                self.selected_object_bbox = bbox
                self.image_canvas.set_selections(
                    subject_bbox=self.selected_subject_bbox,
                    object_bbox=bbox
                )
                
                # Show predicate selection dialog
                self.show_predicate_selection_dialog()
        else:
            # Handle editing mode - select bbox for editing
            self.selected_bbox_for_editing = bbox
            self.image_canvas.set_editing_selection(bbox)
            self.update_annotation_buttons()
            
    def on_object_list_click(self, item):
        """Handle clicks on the objects list."""
        if not self.relation_creation_mode:
            return
            
        # Check if relationships are loaded
        if not self.relationships_mapping:
            QMessageBox.warning(
                self, "No Relationships Loaded", 
                "Please load a relationships file first using 'Load Relationships' button."
            )
            return
            
        # Parse item text to find bbox
        item_text = item.text()
        try:
            category_name, object_id = item_text.split(':')
            object_id = int(object_id)
        except (ValueError, IndexError):
            return
            
        # Find corresponding bbox
        selected_bbox = None
        for bbox in self.current_bboxes:
            if bbox['id'] == object_id and bbox['category_name'] == category_name:
                selected_bbox = bbox
                break
                
        if selected_bbox:
            self.on_bbox_clicked(selected_bbox)
            
    def on_predicate_select(self, item):
        """Handle predicate selection from list."""
        if not (self.selected_subject_bbox and self.selected_object_bbox):
            return
            
        predicate = item.text()
        self.create_relationship_with_predicate(predicate)
        
    # Annotation editing methods
    def enter_add_bbox_mode(self):
        """Enter bounding box addition mode."""
        self.add_bbox_mode = True
        self.relation_creation_mode = False
        self.reset_selection()
        
        # Update button states
        self.add_bbox_btn.setText("Cancel Add Mode")
        self.add_bbox_btn.clicked.disconnect()
        self.add_bbox_btn.clicked.connect(self.exit_add_bbox_mode)
        
        # Update canvas cursor and info
        self.image_canvas.setCursor(Qt.CursorShape.CrossCursor)
        QMessageBox.information(
            self, "Add Bounding Box", 
            "Click and drag on the image to create a new bounding box.\n"
            "Click 'Cancel Add Mode' to exit."
        )
        
    def exit_add_bbox_mode(self):
        """Exit bounding box addition mode."""
        self.add_bbox_mode = False
        self.relation_creation_mode = True
        self.bbox_start_pos = None
        self.bbox_end_pos = None
        
        # Reset button
        self.add_bbox_btn.setText("+ Add Bounding Box")
        self.add_bbox_btn.clicked.disconnect()
        self.add_bbox_btn.clicked.connect(self.enter_add_bbox_mode)
        
        # Reset cursor
        self.image_canvas.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Update UI
        self.update_annotation_buttons()
        
    def edit_selected_label(self):
        """Edit the label of the selected bounding box."""
        if not self.selected_bbox_for_editing:
            return
            
        # Get available categories
        if not self.categories_data:
            QMessageBox.warning(self, "No Categories", "No categories loaded.")
            return
            
        # Create category selection dialog
        from .dialogs import CategorySelectionDialog
        category_names = [cat['name'] for cat in self.categories_data.values()]
        dialog = CategorySelectionDialog(
            self, 
            category_names,
            self.selected_bbox_for_editing['category_name']
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_category = dialog.get_selected_category()
            
            if new_category and new_category != self.selected_bbox_for_editing['category_name']:
                # Find category ID by name
                category_id = None
                for cat_id, cat_data in self.categories_data.items():
                    if cat_data['name'] == new_category:
                        category_id = cat_id
                        break
                        
                if category_id is None:
                    QMessageBox.warning(self, "Error", f"Category '{new_category}' not found.")
                    return
                
                # Update bbox category
                old_category = self.selected_bbox_for_editing['category_name']
                self.selected_bbox_for_editing['category_name'] = new_category
                self.selected_bbox_for_editing['category_id'] = category_id
                
                # Update COCO annotations
                if self.coco_data and 'annotations' in self.coco_data:
                    for ann in self.coco_data['annotations']:
                        if ann['id'] == self.selected_bbox_for_editing['annotation_id']:
                            ann['category_id'] = category_id
                            break
                
                # Update image annotations
                if self.current_image_id in self.annotations_data:
                    for ann in self.annotations_data[self.current_image_id]:
                        if ann['id'] == self.selected_bbox_for_editing['annotation_id']:
                            ann['category_id'] = category_id
                            break
                
                # Update UI
                self.update_objects_list()
                self.image_canvas.set_bboxes(self.current_bboxes)
                
                QMessageBox.information(
                    self, "Success", 
                    f"Label changed from '{old_category}' to '{new_category}'."
                )
        
    def update_annotation_buttons(self):
        """Update the state of annotation editing buttons."""
        has_selection = self.selected_bbox_for_editing is not None
        self.edit_label_btn.setEnabled(has_selection)
        
    def on_new_bbox_drawn(self, x, y, w, h):
        """Handle new bounding box drawn on canvas."""
        if not self.categories_data:
            QMessageBox.warning(self, "No Categories", 
                              "Please load categories first before adding bounding boxes.")
            self.exit_add_bbox_mode()
            return
            
        # Show category selection dialog
        from .dialogs import CategorySelectionDialog
        category_names = [cat['name'] for cat in self.categories_data.values()]
        dialog = CategorySelectionDialog(self, category_names)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            category_name = dialog.get_selected_category()
            if category_name:
                # Find category ID by name
                category_id = None
                for cat_id, cat_data in self.categories_data.items():
                    if cat_data['name'] == category_name:
                        category_id = cat_id
                        break
                        
                if category_id is None:
                    QMessageBox.warning(self, "Error", f"Category '{category_name}' not found.")
                    self.exit_add_bbox_mode()
                    return
                
                # Create new annotation
                new_annotation_id = self._get_next_annotation_id()
                new_bbox_id = self._get_next_bbox_id()
                
                # Create COCO annotation
                coco_annotation = {
                    'id': new_annotation_id,
                    'image_id': self.current_image_id,
                    'category_id': category_id,
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': 0
                }
                
                # Add to COCO data
                if self.coco_data and 'annotations' in self.coco_data:
                    self.coco_data['annotations'].append(coco_annotation)
                
                # Add to image annotations
                if self.current_image_id not in self.annotations_data:
                    self.annotations_data[self.current_image_id] = []
                self.annotations_data[self.current_image_id].append(coco_annotation)
                
                # Create bbox for display
                new_bbox = {
                    'id': new_bbox_id,
                    'annotation_id': new_annotation_id,
                    'category_id': category_id,
                    'category_name': category_name,
                    'original_bbox': [x, y, w, h]
                }
                
                # Add to current bboxes
                self.current_bboxes.append(new_bbox)
                
                # Update UI
                self.update_objects_list()
                self.image_canvas.set_bboxes(self.current_bboxes)
                
                QMessageBox.information(self, "Success", 
                                      f"Added new {category_name} bounding box.")
                
        # Exit add mode
        self.exit_add_bbox_mode()
        
    def _get_next_annotation_id(self):
        """Get the next available annotation ID."""
        if not self.coco_data or 'annotations' not in self.coco_data:
            return 1
        existing_ids = [ann['id'] for ann in self.coco_data['annotations']]
        return max(existing_ids) + 1 if existing_ids else 1
        
    def _get_next_bbox_id(self):
        """Get the next available bbox ID."""
        if not self.current_bboxes:
            return 1
        existing_ids = [bbox['id'] for bbox in self.current_bboxes]
        return max(existing_ids) + 1 if existing_ids else 1
        
    def reset_selection(self):
        """Reset all selections."""
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self.selected_bbox_for_editing = None
        self.awaiting_object_selection = False
        self.image_canvas.clear_selections()
        self.update_annotation_buttons()
        
    def add_new_predicate(self):
        """Add a new predicate to the relationships list."""
        # Show input dialog
        text, ok = QInputDialog.getText(
            self, 
            'Add New Predicate', 
            'Enter predicate name:', 
            QLineEdit.EchoMode.Normal,
            ''
        )
        
        if ok and text.strip():
            predicate = text.strip()
            
            # Check if predicate already exists
            if hasattr(self, 'relationships_mapping') and predicate in self.relationships_mapping:
                QMessageBox.warning(self, "Duplicate Predicate", 
                                  f"The predicate '{predicate}' already exists.")
                return
            
            # Initialize relationships_mapping if it doesn't exist
            if not hasattr(self, 'relationships_mapping'):
                self.relationships_mapping = {}
            
            # Add new predicate
            next_id = len(self.relationships_mapping)
            self.relationships_mapping[predicate] = next_id
            
            # Update the relationships list
            self.relationships_list.clear()
            for rel in sorted(self.relationships_mapping.keys()):
                self.relationships_list.addItem(rel)
            
            # Save to relationships file
            self.save_relationships_to_file()
            
            QMessageBox.information(self, "Success", 
                                  f"Successfully added predicate '{predicate}'.")
    
    def add_new_category(self):
        """Add a new category to the objects list."""
        # Show input dialog
        text, ok = QInputDialog.getText(
            self, 
            'Add New Category', 
            'Enter category name:', 
            QLineEdit.EchoMode.Normal,
            ''
        )
        
        if ok and text.strip():
            category_name = text.strip()
            
            # Check if category already exists
            for category in self.coco_data.get("categories", []):
                if category["name"] == category_name:
                    QMessageBox.warning(self, "Duplicate Category", 
                                      f"The category '{category_name}' already exists.")
                    return
            
            # Generate new category ID (max existing ID + 1)
            max_id = 0
            for category in self.coco_data.get("categories", []):
                max_id = max(max_id, category.get("id", 0))
            new_id = max_id + 1
            
            # Add to COCO data structure
            category_entry = {
                "id": new_id,
                "name": category_name,
                "supercategory": "object"
            }
            self.coco_data["categories"].append(category_entry)
            
            # Update categories mapping
            if not hasattr(self, 'categories'):
                self.categories = {}
            self.categories[new_id] = category_name
            
            # Update the objects list
            self.object_list.clear()
            for cat_id, cat_name in self.categories.items():
                self.object_list.addItem(cat_name)
            
            QMessageBox.information(self, "Success", 
                                  f"Successfully added category '{category_name}'.")
    
    def save_relationships_to_file(self):
        """Save current relationships to the relationships file."""
        try:
            # Use default relationships file path
            relationships_file = os.path.join(os.path.dirname(__file__), '..', 'new_relationships.txt')
            
            with open(relationships_file, 'w') as f:
                for rel in sorted(self.relationships_mapping.keys()):
                    f.write(f"{rel}\n")
                    
        except Exception as e:
            QMessageBox.warning(self, "Warning", 
                              f"Could not save relationships to file: {str(e)}")
        
    def show_predicate_selection_dialog(self):
        """Show predicate selection dialog."""
        if not (self.selected_subject_bbox and self.selected_object_bbox):
            return
            
        if not self.relationships_mapping:
            QMessageBox.warning(self, "No Relationships", 
                              "No relationships have been loaded. Please load relationships first.")
            self.reset_relation_mode()
            return
            
        subject_info = f"{self.selected_subject_bbox['category_name']}:{self.selected_subject_bbox['id']}"
        object_info = f"{self.selected_object_bbox['category_name']}:{self.selected_object_bbox['id']}"
        
        dialog = PredicateSelectionDialog(
            self, 
            self.relationships_mapping.keys(),
            subject_info,
            object_info
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_predicate:
            self.create_relationship_with_predicate(dialog.selected_predicate)
        else:
            self.reset_relation_mode()
    
    # Relationship management
    def create_relationship_with_predicate(self, predicate):
        """Create relationship with the selected predicate."""
        if not (self.selected_subject_bbox and self.selected_object_bbox):
            return
            
        # Create relationship
        relationship = {
            'subject_id': self.selected_subject_bbox['id'],
            'subject_category': self.selected_subject_bbox['category_name'],
            'predicate': predicate,
            'object_id': self.selected_object_bbox['id'],
            'object_category': self.selected_object_bbox['category_name'],
            'image_id': self.current_image_id
        }
        
        # Check for duplicate
        for existing_rel in self.relationships:
            if (existing_rel['subject_id'] == relationship['subject_id'] and
                existing_rel['predicate'] == relationship['predicate'] and
                existing_rel['object_id'] == relationship['object_id'] and
                existing_rel['image_id'] == relationship['image_id']):
                QMessageBox.warning(self, "Duplicate Relationship", 
                                  "This relationship already exists.")
                self.reset_relation_mode()
                return
        
        self.relationships.append(relationship)
        self.update_relationships_view()
        
        # Show success message
        subject_info = f"{self.selected_subject_bbox['category_name']}:{self.selected_subject_bbox['id']}"
        object_info = f"{self.selected_object_bbox['category_name']}:{self.selected_object_bbox['id']}"
        self.statusBar().showMessage(f"Created: {subject_info} → {predicate} → {object_info}")
        
        # Reset for next relationship
        self.reset_relation_mode()
        
    def update_relationships_view(self):
        """Update the relationships view."""
        self.current_relationships_list.clear()
        
        # Show only relationships for current image
        current_image_relationships = [
            rel for rel in self.relationships 
            if rel['image_id'] == self.current_image_id
        ]
        
        for rel in current_image_relationships:
            item_text = (f"{rel['subject_category']}:{rel['subject_id']} → "
                        f"{rel['predicate']} → "
                        f"{rel['object_category']}:{rel['object_id']}")
            self.current_relationships_list.addItem(item_text)
            
    def show_relationship_context_menu(self, position):
        """Show context menu for relationships."""
        item = self.current_relationships_list.itemAt(position)
        if item:
            menu = QMenu()
            delete_action = menu.addAction("Delete Relationship")
            action = menu.exec(self.current_relationships_list.mapToGlobal(position))
            
            if action == delete_action:
                self.delete_relationship(self.current_relationships_list.row(item))
    
    def show_object_context_menu(self, position):
        """Show context menu for objects."""
        item = self.objects_list.itemAt(position)
        if item:
            menu = QMenu()
            
            # Add Edit Label action
            edit_action = menu.addAction("✏️ Edit Label")
            
            # Add Delete action
            delete_action = menu.addAction("🗑️ Delete Bounding Box")
            delete_action.setToolTip("Remove this bounding box from the image")
            
            # Execute menu
            action = menu.exec(self.objects_list.mapToGlobal(position))
            
            if action == edit_action:
                self.edit_object_from_list(item)
            elif action == delete_action:
                self.delete_object_from_list(item)
    
    def edit_object_from_list(self, item):
        """Edit object label from list item."""
        # Parse item text to find bbox
        item_text = item.text()
        try:
            category_name, object_id = item_text.split(':')
            object_id = int(object_id)
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Error", "Invalid item format.")
            return
            
        # Find corresponding bbox
        target_bbox = None
        for bbox in self.current_bboxes:
            if bbox['id'] == object_id and bbox['category_name'] == category_name:
                target_bbox = bbox
                break
                
        if target_bbox:
            # Set as selected for editing and call edit method
            self.selected_bbox_for_editing = target_bbox
            self.edit_selected_label()
    
    def delete_object_from_list(self, item):
        """Delete object from list item."""
        # Parse item text to find bbox
        item_text = item.text()
        try:
            category_name, object_id = item_text.split(':')
            object_id = int(object_id)
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Error", "Invalid item format.")
            return
            
        # Find corresponding bbox
        target_bbox = None
        for bbox in self.current_bboxes:
            if bbox['id'] == object_id and bbox['category_name'] == category_name:
                target_bbox = bbox
                break
                
        if target_bbox:
            # Confirm deletion
            reply = QMessageBox.question(
                self, "Delete Bounding Box",
                f"Are you sure you want to delete this bounding box?\n"
                f"Category: {target_bbox['category_name']}\n"
                f"ID: {target_bbox['id']}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove from current bboxes
                if target_bbox in self.current_bboxes:
                    self.current_bboxes.remove(target_bbox)
                
                # Remove from COCO annotations
                if self.coco_data and 'annotations' in self.coco_data:
                    self.coco_data['annotations'] = [
                        ann for ann in self.coco_data['annotations'] 
                        if ann['id'] != target_bbox['annotation_id']
                    ]
                
                # Remove from image annotations
                if self.current_image_id in self.annotations_data:
                    self.annotations_data[self.current_image_id] = [
                        ann for ann in self.annotations_data[self.current_image_id]
                        if ann['id'] != target_bbox['annotation_id']
                    ]
                
                # Clear any selections of this bbox
                if self.selected_bbox_for_editing == target_bbox:
                    self.selected_bbox_for_editing = None
                if self.selected_subject_bbox == target_bbox:
                    self.selected_subject_bbox = None
                    self.awaiting_object_selection = False
                if self.selected_object_bbox == target_bbox:
                    self.selected_object_bbox = None
                
                # Update UI
                self.update_objects_list()
                self.image_canvas.set_bboxes(self.current_bboxes)
                self.update_annotation_buttons()
                
                QMessageBox.information(self, "Success", "Bounding box deleted successfully.")
                
    def delete_relationship(self, index):
        """Delete relationship at index."""
        current_image_relationships = [
            rel for rel in self.relationships 
            if rel['image_id'] == self.current_image_id
        ]
        
        if 0 <= index < len(current_image_relationships):
            rel_to_delete = current_image_relationships[index]
            reply = QMessageBox.question(
                self, "Delete Relationship",
                "Are you sure you want to delete this relationship?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.relationships.remove(rel_to_delete)
                self.update_relationships_view()
                
    def load_existing_relationships(self):
        """Load existing relationships for current image."""
        self.update_relationships_view()
        
    def load_existing_relationships_from_coco(self):
        """Load existing relationships from COCO format if they exist."""
        if not self.coco_data:
            return
            
        existing_relationships = []
        
        # Load relationship categories
        rel_categories = {}
        if 'rel_categories' in self.coco_data:
            rel_categories = {cat['id']: cat['name'] for cat in self.coco_data['rel_categories']}
        elif 'relationship_categories' in self.coco_data:
            rel_categories = {cat['id']: cat['name'] for cat in self.coco_data['relationship_categories']}
            
        # Update relationships mapping
        for cat_id, cat_name in rel_categories.items():
            if cat_name not in self.relationships_mapping:
                self.relationships_mapping[cat_name] = cat_id
                
        # Load relationships
        if 'rel_annotations' in self.coco_data:
            for rel_ann in self.coco_data['rel_annotations']:
                # Find annotation categories
                subject_category = 'unknown'
                object_category = 'unknown'
                
                for ann in self.coco_data.get('annotations', []):
                    if ann['id'] == rel_ann['subject_id']:
                        subject_category = self.categories_data.get(ann['category_id'], {'name': 'unknown'})['name']
                    elif ann['id'] == rel_ann['object_id']:
                        object_category = self.categories_data.get(ann['category_id'], {'name': 'unknown'})['name']
                        
                predicate_name = rel_categories.get(rel_ann['predicate_id'], 'unknown')
                
                relationship = {
                    'subject_id': rel_ann['subject_id'],
                    'subject_category': subject_category,
                    'predicate': predicate_name,
                    'object_id': rel_ann['object_id'],
                    'object_category': object_category,
                    'image_id': rel_ann['image_id']
                }
                existing_relationships.append(relationship)
                
        if existing_relationships:
            self.relationships.extend(existing_relationships)
    
    def load_existing_relationships_from_file(self):
        """Load existing relationships from a previously saved JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSON File with Relationships",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # Show progress dialog
            progress = QProgressDialog("Loading relationships...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            loaded_relationships = []
            
            # Method 1: Try to load from our custom format
            if 'relationships' in data and isinstance(data['relationships'], list):
                loaded_relationships = data['relationships']
                
            # Method 2: Try to load from COCO format with relationships
            elif 'rel_annotations' in data or 'relationship_annotations' in data:
                # Load relationship categories first
                rel_categories = {}
                if 'rel_categories' in data:
                    rel_categories = {cat['id']: cat['name'] for cat in data['rel_categories']}
                elif 'relationship_categories' in data:
                    rel_categories = {cat['id']: cat['name'] for cat in data['relationship_categories']}
                
                # Update our relationships mapping
                for cat_id, cat_name in rel_categories.items():
                    if cat_name not in self.relationships_mapping:
                        self.relationships_mapping[cat_name] = cat_id
                
                # Load relationships from rel_annotations
                rel_anns = data.get('rel_annotations', data.get('relationship_annotations', []))
                categories_lookup = {cat['id']: cat['name'] for cat in data.get('categories', [])}
                
                for rel_ann in rel_anns:
                    # Find annotation details
                    subject_category = 'unknown'
                    object_category = 'unknown'
                    subject_bbox_id = None
                    object_bbox_id = None
                    
                    for ann in data.get('annotations', []):
                        if ann['id'] == rel_ann['subject_id']:
                            subject_category = categories_lookup.get(ann['category_id'], 'unknown')
                            # Create a display ID (could be same as annotation ID)
                            subject_bbox_id = ann['id']
                        elif ann['id'] == rel_ann['object_id']:
                            object_category = categories_lookup.get(ann['category_id'], 'unknown')
                            object_bbox_id = ann['id']
                    
                    predicate_name = rel_categories.get(rel_ann['predicate_id'], 'unknown')
                    
                    relationship = {
                        'subject_id': subject_bbox_id or rel_ann['subject_id'],
                        'subject_category': subject_category,
                        'predicate': predicate_name,
                        'object_id': object_bbox_id or rel_ann['object_id'],
                        'object_category': object_category,
                        'image_id': rel_ann['image_id']
                    }
                    loaded_relationships.append(relationship)
            
            else:
                QMessageBox.warning(self, "No Relationships Found", 
                                  "No relationships found in the selected JSON file.")
                progress.close()
                return
            
            # Filter relationships to only those for currently loaded images
            if self.image_ids:
                valid_relationships = [
                    rel for rel in loaded_relationships 
                    if rel['image_id'] in self.image_ids
                ]
                
                if len(valid_relationships) != len(loaded_relationships):
                    QMessageBox.information(
                        self, "Filtered Relationships",
                        f"Loaded {len(valid_relationships)} relationships "
                        f"(filtered {len(loaded_relationships) - len(valid_relationships)} "
                        f"for images not currently loaded)."
                    )
                
                loaded_relationships = valid_relationships
            
            # Add to existing relationships (avoiding duplicates)
            new_relationships_count = 0
            for rel in loaded_relationships:
                # Simple duplicate check based on image_id, subject_id, object_id, predicate
                is_duplicate = any(
                    existing_rel['image_id'] == rel['image_id'] and
                    existing_rel['subject_id'] == rel['subject_id'] and
                    existing_rel['object_id'] == rel['object_id'] and
                    existing_rel['predicate'] == rel['predicate']
                    for existing_rel in self.relationships
                )
                
                if not is_duplicate:
                    self.relationships.append(rel)
                    new_relationships_count += 1
            
            progress.close()
            
            # Update UI
            self.update_relationships_view()
            if hasattr(self, 'relationships_list'):
                self.relationships_list.clear()
                for rel in sorted(self.relationships_mapping.keys()):
                    self.relationships_list.addItem(rel)
            
            if new_relationships_count > 0:
                QMessageBox.information(
                    self, "Success", 
                    f"Successfully loaded {new_relationships_count} new relationships.\n"
                    f"Total relationships: {len(self.relationships)}"
                )
            else:
                QMessageBox.information(
                    self, "No New Relationships", 
                    "All relationships from the file were already loaded."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading relationships:\n{str(e)}")
    
    # Saving methods
    def save_relationships(self):
        """Save all relationships to files."""
        if not self.relationships:
            QMessageBox.warning(self, "No Relationships", "No relationships to save.")
            return
            
        try:
            self.save_relationships_coco_format()
            self.save_relationships_json()
            self.save_relationships_txt()
            
            QMessageBox.information(self, "Success", 
                                  f"Saved {len(self.relationships)} relationships in multiple formats.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving relationships:\n{str(e)}")
            
    def save_relationships_coco_format(self):
        """Save relationships in COCO format."""
        if not self.coco_data:
            return
            
        output_file = self.output_dir / self.output_filename
        
        # Create enhanced COCO data
        coco_with_relationships = copy.deepcopy(self.coco_data)
        
        # Create rel_annotations
        rel_annotations = []
        for i, rel in enumerate(self.relationships):
            relationship_entry = {
                'id': i,
                'subject_id': rel['subject_id'],
                'predicate_id': self.relationships_mapping.get(rel['predicate'], -1),
                'object_id': rel['object_id'],
                'image_id': rel['image_id']
            }
            rel_annotations.append(relationship_entry)
            
        coco_with_relationships['rel_annotations'] = rel_annotations
        
        # Add relationship categories
        if 'rel_categories' not in coco_with_relationships:
            rel_categories = []
            for predicate, pred_id in self.relationships_mapping.items():
                rel_categories.append({
                    'id': pred_id,
                    'name': predicate
                })
            coco_with_relationships['rel_categories'] = rel_categories
            
        # Save file
        with open(output_file, 'w') as f:
            json.dump(coco_with_relationships, f, indent=2)
            
    def save_relationships_json(self):
        """Save relationships in JSON format."""
        output_file = self.output_dir / "relationships.json"
        
        relationships_data = {
            'relationships': self.relationships,
            'relationship_mapping': self.relationships_mapping,
            'coco_info': {
                'total_images': len(self.image_ids),
                'images_with_relationships': len(set(rel['image_id'] for rel in self.relationships))
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(relationships_data, f, indent=2)
            
    def save_relationships_txt(self):
        """Save relationships in text format."""
        output_file = self.output_dir / "relationships.txt"
        
        with open(output_file, 'w') as f:
            f.write("COCO Relationship Annotations\n")
            f.write("=" * 50 + "\n\n")
            
            # Group by image
            by_image = {}
            for rel in self.relationships:
                image_id = rel['image_id']
                if image_id not in by_image:
                    by_image[image_id] = []
                by_image[image_id].append(rel)
                
            for image_id, rels in by_image.items():
                image_info = self.images_data[image_id]
                f.write(f"Image: {image_info['file_name']} (ID: {image_id})\n")
                f.write("-" * 40 + "\n")
                
                for rel in rels:
                    f.write(f"{rel['subject_category']}:{rel['subject_id']} → "
                           f"{rel['predicate']} → "
                           f"{rel['object_category']}:{rel['object_id']}\n")
                f.write("\n")
                
    def closeEvent(self, event):
        """Handle application closing."""
        if self.relationships:
            reply = QMessageBox.question(
                self, "Unsaved Relationships",
                "You have unsaved relationships. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self.save_relationships()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
