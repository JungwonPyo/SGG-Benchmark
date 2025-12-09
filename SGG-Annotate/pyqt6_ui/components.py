"""
Modern UI Components for PyQt6 COCO Relations Tool

This module contains custom styled widgets with dark theme support.
"""

from PyQt6.QtWidgets import QLabel, QPushButton, QListWidget
from PyQt6.QtCore import Qt


class ModernLabel(QLabel):
    """Custom label with modern styling."""
    
    def __init__(self, text="", style_class="normal"):
        super().__init__(text)
        self.setStyleSheet(self._get_style(style_class))
        
    def _get_style(self, style_class):
        styles = {
            "title": """
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: #FFFFFF;
                    padding: 10px;
                    background-color: #2E3440;
                    border-radius: 8px;
                    margin: 5px;
                }
            """,
            "subtitle": """
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    color: #D8DEE9;
                    padding: 5px;
                    margin: 2px;
                }
            """,
            "normal": """
                QLabel {
                    font-size: 12px;
                    color: #E5E9F0;
                    padding: 2px;
                }
            """,
            "status": """
                QLabel {
                    font-size: 12px;
                    color: #88C0D0;
                    padding: 8px;
                    background-color: #3B4252;
                    border-radius: 6px;
                    border: 1px solid #4C566A;
                }
            """
        }
        return styles.get(style_class, styles["normal"])


class ModernButton(QPushButton):
    """Custom button with modern styling."""
    
    def __init__(self, text="", style_class="primary"):
        super().__init__(text)
        self.setStyleSheet(self._get_style(style_class))
        self.setMinimumHeight(35)
        
    def _get_style(self, style_class):
        styles = {
            "primary": """
                QPushButton {
                    background-color: #5E81AC;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #81A1C1;
                }
                QPushButton:pressed {
                    background-color: #4C566A;
                }
                QPushButton:disabled {
                    background-color: #434C5E;
                    color: #6B7280;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: #4C566A;
                    color: white;
                    border: 1px solid #5E81AC;
                    padding: 8px 16px;
                    font-size: 12px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #5E81AC;
                }
                QPushButton:pressed {
                    background-color: #434C5E;
                }
                QPushButton:disabled {
                    background-color: #3B4252;
                    color: #6B7280;
                    border-color: #434C5E;
                }
            """,
            "success": """
                QPushButton {
                    background-color: #A3BE8C;
                    color: #2E3440;
                    border: none;
                    padding: 8px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #B6D7A8;
                }
                QPushButton:pressed {
                    background-color: #8FBC8F;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #BF616A;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #D08770;
                }
                QPushButton:pressed {
                    background-color: #A54A54;
                }
            """,
            "warning": """
                QPushButton {
                    background-color: #EBCB8B;
                    color: #2E3440;
                    border: none;
                    padding: 8px 16px;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #F0D97B;
                }
                QPushButton:pressed {
                    background-color: #E6C76B;
                }
            """
        }
        return styles.get(style_class, styles["primary"])


class ModernListWidget(QListWidget):
    """Custom list widget with modern styling."""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QListWidget {
                background-color: #3B4252;
                border: 1px solid #4C566A;
                border-radius: 6px;
                padding: 5px;
                font-size: 12px;
                color: #E5E9F0;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 1px;
            }
            QListWidget::item:hover {
                background-color: #434C5E;
            }
            QListWidget::item:selected {
                background-color: #5E81AC;
                color: white;
            }
        """)
