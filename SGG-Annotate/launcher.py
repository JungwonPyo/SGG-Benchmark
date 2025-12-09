#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGDET-Annotate Launcher

Auto-detects the best available UI framework and launches the appropriate version.
Tries PyQt6 first (modern), falls back to Tkinter (classic) if PyQt6 is not available.
"""

import sys
import argparse


def main():
    """Main launcher that auto-detects the best UI framework."""
    parser = argparse.ArgumentParser(
        description="SGDET-Annotate COCO Relations Tool - Auto Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This launcher automatically detects the best available UI framework:
1. PyQt6 (Modern, recommended) - if available
   Features: Dark theme, annotation editing, add/remove bounding boxes
2. Tkinter (Classic, fallback) - always available
   Features: Classic interface, maximum compatibility

Examples:
  python launcher.py
  python launcher.py --output my_annotations.json
  python launcher.py -o enhanced_coco_data.json --force-tkinter
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='annotations_with_relationships.json',
        help='Output filename for the COCO file with relationships'
    )
    
    parser.add_argument(
        '--force-tkinter',
        action='store_true',
        help='Force use of Tkinter version instead of PyQt6'
    )
    
    args = parser.parse_args()
    
    # Validate output filename
    if not args.output.endswith('.json'):
        args.output += '.json'
    
    # Try PyQt6 first (unless forced to use tkinter)
    if not args.force_tkinter:
        try:
            print("🚀 Trying PyQt6 version (modern UI)...")
            from pyqt6_ui.main_window import COCORelationAnnotatorPyQt
            from PyQt6.QtWidgets import QApplication
            
            # Create PyQt6 application
            app = QApplication(sys.argv)
            app.setApplicationName("SGDET-Annotate COCO Relations Tool")
            app.setApplicationVersion("2.0")
            
            # Create main window
            window = COCORelationAnnotatorPyQt(output_filename=args.output)
            window.show()
            
            print("✅ PyQt6 version launched successfully!")
            print("💡 Features: Modern dark theme, enhanced UI, better performance")
            
            # Run application
            sys.exit(app.exec())
            
        except ImportError as e:
            print(f"⚠️  PyQt6 not available: {e}")
            print("📦 Install PyQt6 with: pip install PyQt6")
            print("🔄 Falling back to Tkinter version...")
    
    # Fallback to Tkinter
    try:
        print("🔧 Launching Tkinter version (classic UI)...")
        import tkinter as tk
        import os
        sys.path.append(os.path.dirname(__file__))
        
        # Run the tkinter version
        from main_coco import COCORelationAnnotator, main as coco_main
        
        print("✅ Tkinter version launched successfully!")
        print("💡 Features: Classic interface, maximum compatibility")
        
        # Set the output filename in sys.argv for the tkinter version
        original_argv = sys.argv[:]
        sys.argv = ['main_coco.py']
        if args.output != 'annotations_with_relationships.json':
            sys.argv.extend(['--output', args.output])
        
        # Run the tkinter main
        coco_main()
        
    except Exception as e:
        print(f"❌ Error launching Tkinter version: {e}")
        print("💡 Please check your Python installation")
        sys.exit(1)


if __name__ == "__main__":
    main()
