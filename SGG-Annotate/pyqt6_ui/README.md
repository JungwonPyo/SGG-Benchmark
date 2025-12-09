# PyQt6 UI Package for SGDET-Annotate

This package contains the modular PyQt6 implementation of the SGDET-Annotate COCO Relations Tool.

## Package Structure

```
pyqt6_ui/
├── __init__.py           # Package initialization
├── components.py         # UI components (ModernLabel, ModernButton, ModernListWidget)
├── canvas.py            # Image canvas with bbox rendering
├── dialogs.py           # Dialog windows (PredicateSelectionDialog)
└── main_window.py       # Main application window (COCORelationAnnotatorPyQt)
```

## Components

### UI Components (`components.py`)
- **ModernLabel**: Styled label with different themes (title, subtitle, normal, status)
- **ModernButton**: Styled button with different styles (primary, secondary, success, danger, warning)
- **ModernListWidget**: Styled list widget with dark theme

### Image Canvas (`canvas.py`)
- **ImageCanvas**: Custom widget for displaying images with bounding boxes
- Handles PIL image loading and scaling
- Renders bounding boxes with color coding for selections
- Manages mouse clicks for object selection

### Dialogs (`dialogs.py`)
- **PredicateSelectionDialog**: Modal dialog for selecting relationship predicates
- Modern dark theme styling
- Double-click and button selection support

### Main Window (`main_window.py`)
- **COCORelationAnnotatorPyQt**: Main application class
- COCO data loading and parsing
- Relationship management
- File I/O operations
- Keyboard shortcuts and menu system

## Usage

Import the main application class:

```python
from pyqt6_ui.main_window import COCORelationAnnotatorPyQt

app = QApplication(sys.argv)
window = COCORelationAnnotatorPyQt()
window.show()
app.exec()
```

## Dependencies

- PyQt6 >= 6.4.0
- Pillow >= 9.0.0

## Features

- Modern dark theme with Nord color scheme
- Modular architecture for easy maintenance
- Full COCO format support
- Relationship annotation workflow
- Multiple export formats (COCO, JSON, TXT)
