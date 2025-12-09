"""
Image Canvas Widget for COCO Relations Tool

This module contains the ImageCanvas widget for displaying images with
bounding boxes and handling click interactions.
"""

import os
from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImageCanvas(QLabel):
    """Custom image canvas for displaying images with bounding boxes."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QLabel {
                background-color: #2E3440;
                border: 2px solid #4C566A;
                border-radius: 8px;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        
        # Image and bbox data
        self.original_image = None
        self.current_pixmap = None
        self.bboxes = []
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self.selected_editing_bbox = None
        self.scale_factor = 1.0
        self.image_offset = (0, 0)
        
        # Drawing state for new bboxes
        self.drawing_bbox = False
        self.bbox_start_point = None
        self.bbox_current_point = None
        
    def set_image(self, image_path):
        """Load and display an image."""
        if not PIL_AVAILABLE:
            self.setText("PIL/Pillow not available")
            return False
            
        try:
            # Load image with PIL
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            self.original_image = pil_image
            self._update_display()
            return True
            
        except Exception as e:
            self.setText(f"Error loading image: {str(e)}")
            return False
    
    def set_bboxes(self, bboxes):
        """Set bounding boxes to display."""
        self.bboxes = bboxes
        self._update_display()
    
    def set_selections(self, subject_bbox=None, object_bbox=None):
        """Set selected bounding boxes."""
        self.selected_subject_bbox = subject_bbox
        self.selected_object_bbox = object_bbox
        self.selected_editing_bbox = None  # Clear editing selection
        self._update_display()
    
    def set_editing_selection(self, bbox):
        """Set selected bbox for editing."""
        self.selected_editing_bbox = bbox
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self._update_display()
    
    def clear_selections(self):
        """Clear all selections."""
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self.selected_editing_bbox = None
        self._update_display()
    
    def _update_display(self):
        """Update the display with current image and bboxes."""
        if not self.original_image:
            return
            
        # Calculate scaling
        widget_size = self.size()
        img_width, img_height = self.original_image.size
        
        # Scale to fit widget while maintaining aspect ratio
        scale_x = widget_size.width() / img_width
        scale_y = widget_size.height() / img_height
        self.scale_factor = min(scale_x, scale_y) * 0.9  # 90% to leave some margin
        
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        
        # Calculate offset to center image
        self.image_offset = (
            (widget_size.width() - display_width) // 2,
            (widget_size.height() - display_height) // 2
        )
        
        # Create pixmap
        pil_image_resized = self.original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Convert PIL to QPixmap
        pil_image_resized.save("/tmp/temp_image.jpg", "JPEG")
        pixmap = QPixmap("/tmp/temp_image.jpg")
        
        # Draw bounding boxes
        if self.bboxes:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            for bbox in self.bboxes:
                # Scale bbox coordinates
                x, y, w, h = bbox['original_bbox']
                screen_x = int(x * self.scale_factor)
                screen_y = int(y * self.scale_factor)
                screen_w = int(w * self.scale_factor)
                screen_h = int(h * self.scale_factor)
                
                # Choose color based on selection
                if bbox == self.selected_subject_bbox:
                    color = QColor("#5E81AC")  # Blue for subject
                    width = 3
                elif bbox == self.selected_object_bbox:
                    color = QColor("#A3BE8C")  # Green for object
                    width = 3
                elif bbox == self.selected_editing_bbox:
                    color = QColor("#EBCB8B")  # Yellow for editing
                    width = 3
                else:
                    color = QColor("#BF616A")  # Red for normal
                    width = 2
                
                # Draw rectangle
                pen = QPen(color, width)
                painter.setPen(pen)
                painter.drawRect(screen_x, screen_y, screen_w, screen_h)
                
                # Draw label
                label = f"{bbox['category_name']}:{bbox['id']}"
                painter.setPen(QPen(QColor("#EBCB8B"), 1))
                painter.fillRect(screen_x, screen_y - 20, len(label) * 8, 20, QBrush(QColor("#2E3440")))
                painter.drawText(screen_x + 2, screen_y - 5, label)
            
            # Draw current drawing bbox if applicable
            if self.drawing_bbox and self.bbox_start_point and self.bbox_current_point:
                start_x, start_y = self.bbox_start_point
                end_x, end_y = self.bbox_current_point
                
                # Draw dashed rectangle for new bbox
                pen = QPen(QColor("#D8DEE9"), 2)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                
                x = int(min(start_x, end_x))
                y = int(min(start_y, end_y))
                w = int(abs(end_x - start_x))
                h = int(abs(end_y - start_y))
                
                painter.drawRect(x, y, w, h)
            
            painter.end()
        
        self.current_pixmap = pixmap
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for bbox selection and drawing."""
        if not self.current_pixmap:
            return
            
        # Convert click position to image coordinates
        click_x = int(event.position().x() - self.image_offset[0])
        click_y = int(event.position().y() - self.image_offset[1])
        
        # Get parent to check mode
        parent = self.parent()
        while parent and not hasattr(parent, 'add_bbox_mode'):
            parent = parent.parent()
            
        if parent and hasattr(parent, 'add_bbox_mode') and parent.add_bbox_mode:
            # Start drawing new bbox
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing_bbox = True
                self.bbox_start_point = (click_x, click_y)
                self.bbox_current_point = (click_x, click_y)
        else:
            # Handle bbox selection (existing logic)
            if not self.bboxes:
                return
                
            # Find clicked bbox
            clicked_bbox = None
            for bbox in self.bboxes:
                x, y, w, h = bbox['original_bbox']
                screen_x = int(x * self.scale_factor)
                screen_y = int(y * self.scale_factor)
                screen_w = int(w * self.scale_factor)
                screen_h = int(h * self.scale_factor)
                
                if (screen_x <= click_x <= screen_x + screen_w and 
                    screen_y <= click_y <= screen_y + screen_h):
                    clicked_bbox = bbox
                    break
            
            if clicked_bbox and parent and hasattr(parent, 'on_bbox_clicked'):
                parent.on_bbox_clicked(clicked_bbox)
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for drawing."""
        if self.drawing_bbox:
            click_x = int(event.position().x() - self.image_offset[0])
            click_y = int(event.position().y() - self.image_offset[1])
            self.bbox_current_point = (click_x, click_y)
            self._update_display()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for completing bbox drawing."""
        if self.drawing_bbox and event.button() == Qt.MouseButton.LeftButton:
            self.drawing_bbox = False
            
            # Calculate bbox in original image coordinates
            start_x, start_y = self.bbox_start_point
            end_x, end_y = self.bbox_current_point
            
            # Ensure minimum size
            if abs(end_x - start_x) < 10 or abs(end_y - start_y) < 10:
                self.bbox_start_point = None
                self.bbox_current_point = None
                self._update_display()
                return
            
            # Convert to original image coordinates
            orig_x = min(start_x, end_x) / self.scale_factor
            orig_y = min(start_y, end_y) / self.scale_factor
            orig_w = abs(end_x - start_x) / self.scale_factor
            orig_h = abs(end_y - start_y) / self.scale_factor
            
            # Notify parent of new bbox
            parent = self.parent()
            while parent and not hasattr(parent, 'on_new_bbox_drawn'):
                parent = parent.parent()
                
            if parent and hasattr(parent, 'on_new_bbox_drawn'):
                parent.on_new_bbox_drawn(orig_x, orig_y, orig_w, orig_h)
            
            self.bbox_start_point = None
            self.bbox_current_point = None
            self._update_display()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        if self.original_image:
            QTimer.singleShot(100, self._update_display)
