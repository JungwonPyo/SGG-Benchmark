#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGDET-Annotate COCO Relations Tool - PyQt6 Version (Refactored)

Modern PyQt6 implementation of the COCO relationship annotation tool with modular
architecture. This is the main entry point that imports the application from
the pyqt6_ui package.
"""

import sys
import argparse
from PyQt6.QtWidgets import QApplication

# Import the main application class from our modular package
from pyqt6_ui.main_window import COCORelationAnnotatorPyQt


def main():
    """Main entry point for the PyQt6 COCO Relations annotation tool."""
    parser = argparse.ArgumentParser(
        description="SGDET-Annotate COCO Relations Tool - PyQt6 Version (Modular)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_coco_pyqt_modular.py
  python main_coco_pyqt_modular.py --output my_annotations.json
  python main_coco_pyqt_modular.py -o enhanced_coco_data.json
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='annotations_with_relationships.json',
        help='Output filename for the COCO file with relationships'
    )
    
    args = parser.parse_args()
    
    # Validate output filename
    if not args.output.endswith('.json'):
        args.output += '.json'
        
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("SGDET-Annotate COCO Relations Tool")
    app.setApplicationVersion("2.0")
    
    # Create main window
    window = COCORelationAnnotatorPyQt(output_filename=args.output)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
