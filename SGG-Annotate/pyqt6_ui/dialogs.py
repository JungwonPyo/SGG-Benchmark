"""
Dialog Components for COCO Relations Tool

This module contains dialog widgets for user interactions.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt

from .components import ModernLabel, ModernButton, ModernListWidget


class PredicateSelectionDialog(QDialog):
    """Modern dialog for selecting relationship predicates."""
    
    def __init__(self, parent, predicates, subject_info, object_info):
        super().__init__(parent)
        self.selected_predicate = None
        self.setWindowTitle("Select Relationship Predicate")
        self.setModal(True)
        self.resize(500, 400)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2E3440;
                color: #E5E9F0;
            }
        """)
        
        self._setup_ui(predicates, subject_info, object_info)
        
    def _setup_ui(self, predicates, subject_info, object_info):
        """Setup the dialog UI."""
        layout = QVBoxLayout()
        
        # Header
        header = ModernLabel("Creating relationship:", "subtitle")
        layout.addWidget(header)
        
        # Relationship info
        info_text = f"{subject_info} → [PREDICATE] → {object_info}"
        info_label = ModernLabel(info_text, "status")
        layout.addWidget(info_label)
        
        # Instructions
        instructions = ModernLabel("Double-click on a predicate to create the relationship:")
        layout.addWidget(instructions)
        
        # Predicate list
        self.predicate_list = ModernListWidget()
        for predicate in sorted(predicates):
            self.predicate_list.addItem(predicate)
        self.predicate_list.itemDoubleClicked.connect(self.on_predicate_selected)
        layout.addWidget(self.predicate_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        create_btn = ModernButton("Create Relationship", "success")
        create_btn.clicked.connect(self.on_create_clicked)
        button_layout.addWidget(create_btn)
        
        cancel_btn = ModernButton("Cancel", "secondary")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def on_predicate_selected(self, item):
        """Handle predicate selection."""
        self.selected_predicate = item.text()
        self.accept()
    
    def on_create_clicked(self):
        """Handle create button click."""
        current_item = self.predicate_list.currentItem()
        if current_item:
            self.selected_predicate = current_item.text()
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a predicate.")


class CategorySelectionDialog(QDialog):
    """Modern dialog for selecting object categories."""
    
    def __init__(self, parent, categories, current_category=None):
        super().__init__(parent)
        self.selected_category = current_category
        self.setWindowTitle("Select Object Category")
        self.setModal(True)
        self.resize(400, 300)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2E3440;
                color: #E5E9F0;
            }
        """)
        
        self.setup_ui(categories, current_category)
        
    def setup_ui(self, categories, current_category):
        """Setup the dialog UI."""
        layout = QVBoxLayout()
        
        # Title
        title = ModernLabel("Select a category for this object:", "title")
        layout.addWidget(title)
        
        # Category list
        self.category_list = ModernListWidget()
        for category in sorted(categories):
            self.category_list.addItem(category)
            
        # Select current category if provided
        if current_category:
            items = self.category_list.findItems(current_category, Qt.MatchFlag.MatchExactly)
            if items:
                self.category_list.setCurrentItem(items[0])
        
        self.category_list.itemDoubleClicked.connect(self.on_category_double_clicked)
        layout.addWidget(self.category_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_btn = ModernButton("OK", "primary")
        self.ok_btn.clicked.connect(self.on_ok_clicked)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = ModernButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def get_selected_category(self):
        """Get the selected category."""
        return self.selected_category
    
    def on_category_double_clicked(self, item):
        """Handle double-click on category."""
        self.selected_category = item.text()
        self.accept()
    
    def on_ok_clicked(self):
        """Handle OK button click."""
        current_item = self.category_list.currentItem()
        if current_item:
            self.selected_category = current_item.text()
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a category.")
