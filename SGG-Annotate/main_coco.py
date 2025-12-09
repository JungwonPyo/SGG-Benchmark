#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGDET-Annotate COCO Relations Tool

This script implements a specialized annotation tool for adding relationship annotations
to existing COCO format bounding box annotations. It loads COCO JSON files and allows
users to create relationships between existing annotated objects.
"""

import tkinter as tk
import json
import os
import argparse
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import dialog classes from the main application
from ui.dialogs import WarningDialog, SuccessDialog, ConfirmationDialog, BaseDialog


class COCORelationAnnotator:
    """
    COCO Relations Annotation Tool.
    Specialized tool for adding relationship annotations to existing COCO format data.
    """

    def __init__(self, master, output_filename=None):
        """
        Initialize the COCO Relations annotation tool.
        
        Args:
            master: The root Tkinter window
            output_filename: Optional custom filename for the output COCO file
        """
        self.master = master
        self.master.title("SGDET-Annotate - COCO Relations Tool")
        
        # Store output filename
        self.output_filename = output_filename or "annotations_with_relationships.json"

        # Set window to full screen (cross-platform)
        try:
            self.master.state('zoomed')
        except tk.TclError:
            try:
                self.master.attributes('-zoomed', True)
            except tk.TclError:
                screen_width = self.master.winfo_screenwidth()
                screen_height = self.master.winfo_screenheight()
                self.master.geometry(f"{screen_width}x{screen_height}+0+0")

        # ---------------------------
        # Core Data Structures
        # ---------------------------
        # COCO data
        self.coco_data = None
        self.images_data = {}  # image_id -> image info
        self.annotations_data = {}  # image_id -> list of annotations
        self.categories_data = {}  # category_id -> category info
        self.images_folder = ""
        
        # Current image state
        self.current_image_id = None
        self.current_image_file = None
        self.loaded_image = None
        self.photo_image = None
        self.image_area = (0, 0, 0, 0)
        
        # Navigation
        self.image_ids = []
        self.current_image_index = 0
        
        # Bounding boxes (from COCO)
        self.current_bboxes = []  # Current image's bboxes with screen coordinates
        
        # Relationship annotations
        self.relationships = []  # List of relationship tuples
        self.relationships_mapping = {}  # predicate -> id mapping
        
        # UI state
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self.relation_creation_mode = True  # Start with relation mode enabled by default
        self.awaiting_object_selection = False
        self.unselected_bboxes = set()  # Track boxes that have been unselected
        
        # Visual state
        self.bbox_canvas_objects = {}  # bbox_id -> {rect_id, text_id}
        
        # Output
        self.output_dir = "output_coco_relations"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Setup UI
        self._setup_layout()
        self._setup_events()

    def _setup_layout(self):
        """Setup the main UI layout."""
        # Configure grid
        self.master.grid_rowconfigure(0, weight=4)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=3, minsize=800)
        self.master.grid_columnconfigure(1, weight=1, minsize=300)

        # Create frames
        self._create_image_frame()
        self._create_control_panel()
        self._create_bottom_panel()

    def _create_image_frame(self):
        """Create the image display area."""
        self.image_frame = tk.Frame(self.master, bd=2, relief=tk.SUNKEN)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)

    def _create_control_panel(self):
        """Create the right control panel."""
        self.control_panel = tk.Frame(self.master, bd=2, relief=tk.GROOVE)
        self.control_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.control_panel.grid_columnconfigure(0, weight=1)

        # Load COCO button
        self.load_coco_button = tk.Button(
            self.control_panel,
            text="Load COCO JSON",
            command=self.load_coco_file,
            font=("Arial", 14, "bold"),
            padx=10,
            pady=10,
            bg="#e6f3ff"
        )
        self.load_coco_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Import relationships button
        self.import_rel_button = tk.Button(
            self.control_panel,
            text="Import Relationship List",
            command=self.import_relationship_list,
            font=("Arial", 12, "bold"),
            padx=10,
            pady=8
        )
        self.import_rel_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Relationship list
        rel_label = tk.Label(self.control_panel, text="Available Predicates", 
                           font=("Arial", 12, "bold"))
        rel_label.grid(row=2, column=0, padx=5, pady=(10,2), sticky="ew")
        
        self.rel_list_frame = tk.Frame(self.control_panel)
        self.rel_list_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.rel_scrollbar = tk.Scrollbar(self.rel_list_frame, orient=tk.VERTICAL)
        self.rel_listbox = tk.Listbox(
            self.rel_list_frame,
            yscrollcommand=self.rel_scrollbar.set
        )
        self.rel_scrollbar.config(command=self.rel_listbox.yview)
        self.rel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.rel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add predicate button
        self.add_predicate_btn = tk.Button(
            self.control_panel, 
            text="+ Add predicate",
            command=self.add_new_predicate,
            font=("Arial", 10)
        )
        self.add_predicate_btn.grid(row=4, column=0, padx=5, pady=2, sticky="ew")

        # Current image objects
        objects_label = tk.Label(self.control_panel, text="Objects in Current Image\n(Click to select for relationships)", 
                               font=("Arial", 12, "bold"))
        objects_label.grid(row=5, column=0, padx=5, pady=(10,2), sticky="ew")
        
        self.objects_list_frame = tk.Frame(self.control_panel)
        self.objects_list_frame.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
        self.objects_scrollbar = tk.Scrollbar(self.objects_list_frame, orient=tk.VERTICAL)
        self.objects_listbox = tk.Listbox(
            self.objects_list_frame,
            yscrollcommand=self.objects_scrollbar.set
        )
        self.objects_scrollbar.config(command=self.objects_listbox.yview)
        self.objects_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.objects_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure weights
        self.control_panel.grid_rowconfigure(3, weight=1)
        self.control_panel.grid_rowconfigure(6, weight=1)

    def _create_bottom_panel(self):
        """Create the bottom control and navigation panel."""
        self.bottom_panel = tk.Frame(self.master, bd=2, relief=tk.GROOVE)
        self.bottom_panel.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Navigation row
        self.prev_button = tk.Button(
            self.bottom_panel,
            text="◀ Previous Image",
            command=self.previous_image,
            font=("Arial", 12, "bold"),
            padx=10,
            pady=8,
            state="disabled"
        )
        self.prev_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.image_info_label = tk.Label(
            self.bottom_panel,
            text="No COCO data loaded",
            font=("Arial", 11),
            bg="#f0f0f0",
            relief=tk.SUNKEN,
            bd=1
        )
        self.image_info_label.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        self.next_button = tk.Button(
            self.bottom_panel,
            text="Next Image ▶",
            command=self.next_image,
            font=("Arial", 12, "bold"),
            padx=10,
            pady=8,
            state="disabled"
        )
        self.next_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Action buttons row
        self.create_relation_button = tk.Button(
            self.bottom_panel,
            text="Create Relationship",
            command=self.toggle_relation_mode,
            font=("Arial", 14, "bold"),
            padx=15,
            pady=10,
            state="disabled"
        )
        self.create_relation_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.save_button = tk.Button(
            self.bottom_panel,
            text="Save Relations",
            command=self.save_relationships,
            font=("Arial", 14, "bold"),
            padx=15,
            pady=10,
            bg="#ccffcc",
            state="disabled"
        )
        self.save_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.cancel_button = tk.Button(
            self.bottom_panel,
            text="Cancel (ESC)",
            command=self.cancel_operation,
            font=("Arial", 12, "bold"),
            padx=10,
            pady=8,
            bg="#ffcccc",
            state="disabled"
        )
        self.cancel_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        self.validate_next_button = tk.Button(
            self.bottom_panel,
            text="Save & Next Image",
            command=self.save_and_next,
            font=("Arial", 14, "bold"),
            padx=15,
            pady=10,
            bg="#e6ffe6",
            state="disabled"
        )
        self.validate_next_button.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        # Relationships view
        rel_title = tk.Label(self.bottom_panel, text="Current Image Relationships", 
                           font=("Arial", 12, "bold"))
        rel_title.grid(row=2, column=0, columnspan=4, padx=5, pady=(10,2), sticky="ew")
        
        self.relationships_frame = tk.Frame(self.bottom_panel)
        self.relationships_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        
        self.relationships_scrollbar = tk.Scrollbar(self.relationships_frame, orient=tk.VERTICAL)
        self.relationships_listbox = tk.Listbox(
            self.relationships_frame,
            yscrollcommand=self.relationships_scrollbar.set
        )
        self.relationships_scrollbar.config(command=self.relationships_listbox.yview)
        self.relationships_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.relationships_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure grid weights
        for i in range(4):
            self.bottom_panel.grid_columnconfigure(i, weight=1)
        self.bottom_panel.grid_rowconfigure(3, weight=1)

    def _setup_events(self):
        """Setup event bindings."""
        # Canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Shift-Button-1>", self.on_shift_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Listbox events
        self.rel_listbox.bind("<Double-Button-1>", self.on_predicate_select)
        self.relationships_listbox.bind("<Delete>", self.on_delete_relationship)
        self.relationships_listbox.bind("<Button-3>", self.show_relationship_context_menu)
        self.objects_listbox.bind("<Button-1>", self.on_object_list_click)

        # Keyboard shortcuts
        self.master.bind("<Escape>", lambda e: self.cancel_operation())
        self.master.bind("<Left>", lambda e: self.previous_image())
        self.master.bind("<Right>", lambda e: self.next_image())
        self.master.bind("<space>", lambda e: self.save_and_next())

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.focus_set()

    def _create_loading_dialog(self, title: str, message: str):
        """Create a larger loading dialog for 4K screens."""
        # Calculate dialog size based on screen
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Make dialog larger for 4K screens
        dialog_width = max(int(screen_width * 0.2), 600)
        dialog_height = max(int(screen_height * 0.15), 250)
        
        # Create loading dialog
        loading_dialog = tk.Toplevel(self.master)
        loading_dialog.title(title)
        loading_dialog.geometry(f"{dialog_width}x{dialog_height}")
        loading_dialog.transient(self.master)
        loading_dialog.resizable(False, False)
        
        # Center the dialog
        x = (screen_width // 2) - (dialog_width // 2)
        y = (screen_height // 2) - (dialog_height // 2)
        loading_dialog.geometry(f"+{x}+{y}")
        
        # Calculate font size based on screen resolution
        base_font_size = max(int(screen_width / 100), 16)
        
        # Add loading message
        loading_label = tk.Label(
            loading_dialog,
            text=message,
            font=("Arial", base_font_size),
            wraplength=dialog_width-60,
            justify=tk.CENTER,
            fg="blue"
        )
        loading_label.pack(pady=40, padx=30, expand=True)
        
        # Add progress indicator
        progress_frame = tk.Frame(loading_dialog)
        progress_frame.pack(pady=10)
        
        progress_label = tk.Label(
            progress_frame,
            text="●●●",
            font=("Arial", base_font_size + 4, "bold"),
            fg="orange"
        )
        progress_label.pack()
        
        # Make dialog modal
        loading_dialog.grab_set()
        loading_dialog.update()
        
        return loading_dialog

    def load_coco_file(self):
        """Load a COCO format JSON file."""
        # Create a larger file dialog for 4K screens
        self.master.update()  # Ensure window is rendered
        
        file_path = filedialog.askopenfilename(
            title="Select COCO JSON File - SGDET Annotate",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.master
        )
        
        if not file_path:
            return

        # Show loading message with larger dialog
        loading_dialog = self._create_loading_dialog("Loading COCO Data", 
                                                   "Loading and parsing COCO JSON file...\nPlease wait.")
        
        try:
            self.master.update()  # Process pending events
            
            with open(file_path, 'r') as f:
                self.coco_data = json.load(f)
            
            # Parse COCO data
            self._parse_coco_data()
            
            loading_dialog.destroy()
            
            # Ask for images folder with larger dialog
            images_folder = filedialog.askdirectory(
                title="Select Images Folder - SGDET Annotate",
                initialdir=os.path.dirname(file_path),
                parent=self.master
            )
            
            if not images_folder:
                WarningDialog(self.master, "Images Folder Required", 
                             "Images folder is required to display images.\n\nPlease select the folder containing your images.").show()
                return
                
            self.images_folder = images_folder
            
            # Load first image
            if self.image_ids:
                self.current_image_index = 0
                self._load_image_at_index(0)
                self._update_navigation_state()
                
            SuccessDialog(self.master, "COCO Data Loaded Successfully", 
                         f"Successfully loaded {len(self.image_ids)} images with annotations.\n\nYou can now create relationships between objects.").show()
                         
        except Exception as e:
            if 'loading_dialog' in locals():
                loading_dialog.destroy()
            WarningDialog(self.master, "COCO Load Error", 
                         f"Error loading COCO file:\n{str(e)}\n\nPlease ensure the file is a valid COCO JSON format.").show()

    def _parse_coco_data(self):
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
        
        # Load existing relationships if they exist in the COCO file
        self._load_existing_relationships_from_coco()

    def import_relationship_list(self):
        """Import relationship list from file."""
        file_path = filedialog.askopenfilename(
            title="Select Relationship List File - SGDET Annotate",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=self.master
        )
        
        if not file_path:
            return

        # Show loading dialog
        loading_dialog = self._create_loading_dialog("Loading Relationships", 
                                                   "Loading relationship list...\nPlease wait.")

        try:
            self.master.update()  # Process pending events
            
            self.relationships_mapping = {}
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    relationship = line.strip()
                    if relationship:
                        self.relationships_mapping[relationship] = i

            # Update relationship listbox
            self.rel_listbox.delete(0, tk.END)
            for rel in sorted(self.relationships_mapping.keys()):
                self.rel_listbox.insert(tk.END, rel)
            
            loading_dialog.destroy()
            self._update_navigation_state()  # Update button states
                
            SuccessDialog(self.master, "Relationships Imported Successfully", 
                         f"Successfully imported {len(self.relationships_mapping)} relationships.\n\nYou can now create relationships between objects in your images.").show()
                         
        except Exception as e:
            if 'loading_dialog' in locals():
                loading_dialog.destroy()
            WarningDialog(self.master, "Relationship Import Error", 
                         f"Error importing relationships:\n{str(e)}\n\nPlease ensure the file contains one relationship per line.").show()

    def _load_image_at_index(self, index):
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
        self.current_image_file = image_path
        
        try:
            self.loaded_image = Image.open(image_path)
            self._display_image()
            self._load_current_bboxes()
            self._update_image_info()
            self._load_existing_relationships()
            
        except Exception as e:
            WarningDialog(self.master, "Image Load Error", f"Error loading image: {str(e)}").show()

    def _display_image(self):
        """Display current image on canvas."""
        if not self.loaded_image:
            return
            
        # Calculate display parameters
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.master.after(100, self._display_image)
            return
            
        img_width, img_height = self.loaded_image.size
        
        # Scale to fit entire image within canvas (preserve aspect ratio)
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y)  # Use min to fit entire image
        
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        
        x_offset = (canvas_width - display_width) // 2
        y_offset = (canvas_height - display_height) // 2
        
        self.image_area = (x_offset, y_offset, display_width, display_height)
        
        # Resize and display image
        display_image = self.loaded_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(display_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo_image)

    def _load_current_bboxes(self):
        """Load and display bounding boxes for current image."""
        self.current_bboxes = []
        self.bbox_canvas_objects = {}
        
        if self.current_image_id not in self.annotations_data:
            self._update_objects_list()
            return
            
        annotations = self.annotations_data[self.current_image_id]
        x_offset, y_offset, display_width, display_height = self.image_area
        
        # Calculate scale from original image to display
        orig_width, orig_height = self.loaded_image.size
        scale_x = display_width / orig_width
        scale_y = display_height / orig_height
        
        for ann in annotations:
            # Convert COCO bbox [x, y, width, height] to screen coordinates
            bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
            
            # Scale to display coordinates
            screen_x1 = int(bbox_x * scale_x) + x_offset
            screen_y1 = int(bbox_y * scale_y) + y_offset
            screen_x2 = screen_x1 + int(bbox_w * scale_x)
            screen_y2 = screen_y1 + int(bbox_h * scale_y)
            
            # Get category name
            category = self.categories_data.get(ann['category_id'], {'name': 'unknown'})
            
            bbox_info = {
                'id': ann['id'],
                'category_id': ann['category_id'],
                'category_name': category['name'],
                'screen_coords': (screen_x1, screen_y1, screen_x2, screen_y2),
                'original_bbox': ann['bbox'],
                'annotation': ann
            }
            
            self.current_bboxes.append(bbox_info)
            
            # Draw on canvas
            rect_id = self.canvas.create_rectangle(
                screen_x1, screen_y1, screen_x2, screen_y2,
                outline="red", width=2
            )
            
            text_id = self.canvas.create_text(
                screen_x1 + 5, screen_y1 + 15,  # Inside box at top-left corner
                text=f"{category['name']}:{ann['id']}",
                fill="yellow",
                font=("Arial", 12, "bold"),  # Increased font size
                anchor="w"  # Left-align text
            )
            
            self.bbox_canvas_objects[ann['id']] = {
                'rect_id': rect_id,
                'text_id': text_id
            }
        
        self._update_objects_list()

    def _update_objects_list(self):
        """Update the objects listbox."""
        self.objects_listbox.delete(0, tk.END)
        for bbox in self.current_bboxes:
            entry = f"{bbox['category_name']}:{bbox['id']}"
            self.objects_listbox.insert(tk.END, entry)

    def _update_image_info(self):
        """Update image info label."""
        if not self.image_ids:
            self.image_info_label.config(text="No COCO data loaded")
        else:
            current = self.current_image_index + 1
            total = len(self.image_ids)
            image_info = self.images_data[self.current_image_id]
            filename = image_info['file_name']
            self.image_info_label.config(text=f"{current}/{total}: {filename}")

    def _update_navigation_state(self):
        """Update navigation button states."""
        has_data = bool(self.image_ids)
        has_relations = bool(self.relationships_mapping)
        
        self.prev_button.config(state="normal" if has_data else "disabled")
        self.next_button.config(state="normal" if has_data else "disabled")
        self.create_relation_button.config(state="normal" if (has_data and has_relations) else "disabled")
        self.save_button.config(state="normal" if has_data else "disabled")
        self.validate_next_button.config(state="normal" if has_data else "disabled")
        
        # Set create relationship button to active state if conditions are met
        if has_data and has_relations and self.relation_creation_mode:
            self.create_relation_button.config(relief=tk.SUNKEN, bg="lightgreen")
            self.cancel_button.config(state="normal")
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Subject Object]")

    def previous_image(self):
        """Navigate to previous image."""
        if not self.image_ids:
            return
        self._save_current_relationships()
        new_index = self.current_image_index - 1
        if new_index < 0:
            new_index = len(self.image_ids) - 1
        self._load_image_at_index(new_index)

    def next_image(self):
        """Navigate to next image."""
        if not self.image_ids:
            return
        self._save_current_relationships()
        new_index = self.current_image_index + 1
        if new_index >= len(self.image_ids):
            new_index = 0
        self._load_image_at_index(new_index)

    def save_and_next(self):
        """Save current relationships and go to next image."""
        self._save_current_relationships()
        self.next_image()

    def toggle_relation_mode(self):
        """Toggle relationship creation mode."""
        self.relation_creation_mode = not self.relation_creation_mode
        
        if self.relation_creation_mode:
            self.create_relation_button.config(relief=tk.SUNKEN, bg="lightgreen")
            self.cancel_button.config(state="normal")
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Subject Object]")
        else:
            self._reset_relation_mode()

    def _reset_relation_mode(self):
        """Reset relationship creation mode."""
        self.awaiting_object_selection = False
        self.selected_subject_bbox = None
        self.selected_object_bbox = None
        self.unselected_bboxes.clear()  # Clear unselected boxes for new relationship
        
        # Keep relation mode active by default, but update UI
        if self.relation_creation_mode:
            self.create_relation_button.config(relief=tk.SUNKEN, bg="lightgreen")
            self.cancel_button.config(state="normal")
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Subject Object]")
        else:
            self.create_relation_button.config(relief=tk.RAISED, bg=self.master.cget('bg'))
            self.cancel_button.config(state="disabled")
            self.master.title("SGDET-Annotate - COCO Relations Tool")
        
        # Reset visual highlights
        for bbox_id, canvas_objs in self.bbox_canvas_objects.items():
            self.canvas.itemconfig(canvas_objs['rect_id'], outline="red", width=2)

    def cancel_operation(self):
        """Cancel current operation and disable relation mode."""
        self.relation_creation_mode = False
        self._reset_relation_mode()

    def on_canvas_click(self, event):
        """Handle canvas clicks for relationship creation."""
        if not self.relation_creation_mode:
            return
            
        # Find clicked bbox
        clicked_bbox = self._get_bbox_at_position(event.x, event.y)
        if not clicked_bbox:
            return
        
        # Check if this box has been unselected and cannot be reselected
        if clicked_bbox['id'] in self.unselected_bboxes:
            WarningDialog(self.master, "Create Relation", 
                         "This box has been unselected and cannot be used in the current relationship.\n"
                         "Start a new relationship or cancel to reuse this box.").show()
            return
            
        if not self.awaiting_object_selection:
            # Select subject
            self.selected_subject_bbox = clicked_bbox
            self.awaiting_object_selection = True
            
            # Highlight subject in blue
            canvas_obj = self.bbox_canvas_objects[clicked_bbox['id']]
            self.canvas.itemconfig(canvas_obj['rect_id'], outline="blue", width=3)
            
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Object & Predicate]")
            
        else:
            # Select object
            if clicked_bbox['id'] == self.selected_subject_bbox['id']:
                WarningDialog(self.master, "Invalid Selection", 
                             "Subject and object cannot be the same.").show()
                return
                
            self.selected_object_bbox = clicked_bbox
            
            # Highlight object in green
            canvas_obj = self.bbox_canvas_objects[clicked_bbox['id']]
            self.canvas.itemconfig(canvas_obj['rect_id'], outline="green", width=3)
            
            # Show predicate selection dialog
            self._show_predicate_selection_dialog()

    def on_shift_canvas_click(self, event):
        """Handle Shift+click events on the canvas - dedicated handler for unselection."""
        # Only handle in relationship creation mode
        if not self.relation_creation_mode:
            SuccessDialog(self.master, "Shift+Click", 
                         "Shift+click only works during relationship creation mode.").show()
            return
            
        target_bbox = self._get_bbox_at_position(event.x, event.y)
        if target_bbox is None:
            WarningDialog(self.master, "Create Relation", 
                         "Please Shift+click on a valid bounding box to unselect it.").show()
            return
        
        # Only allow unselecting currently selected boxes (blue subject or green object)
        is_selected_subject = (target_bbox == self.selected_subject_bbox)
        is_selected_object = (hasattr(self, 'selected_object_bbox') and 
                            target_bbox == self.selected_object_bbox)
        
        if not is_selected_subject and not is_selected_object:
            WarningDialog(self.master, "Create Relation", 
                         "Shift+click only works on currently selected boxes (blue subject or green object).\n"
                         "To block an unselected box, first select it normally, then Shift+click to unselect.").show()
            return
            
        self._handle_shift_click_unselect(target_bbox)

    def _handle_shift_click_unselect(self, target_bbox):
        """Handle Shift+click to unselect a bbox during relationship creation."""
        # Check if this is the currently selected subject (blue)
        if target_bbox == self.selected_subject_bbox:
            # Unselect subject
            canvas_obj = self.bbox_canvas_objects[target_bbox['id']]
            self.canvas.itemconfig(canvas_obj['rect_id'], width=2, outline="red")
            self.selected_subject_bbox = None
            self.awaiting_object_selection = False
            self.unselected_bboxes.add(target_bbox['id'])
            
            SuccessDialog(self.master, "Subject Unselected", 
                         f"Subject box {target_bbox['category_name']}:{target_bbox['id']} has been unselected.\n"
                         "It cannot be reselected for this relationship.").show()
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Subject Object]")
            return
            
        # Check if this is the currently selected object (green)
        if hasattr(self, 'selected_object_bbox') and target_bbox == self.selected_object_bbox:
            # Unselect object
            canvas_obj = self.bbox_canvas_objects[target_bbox['id']]
            self.canvas.itemconfig(canvas_obj['rect_id'], width=2, outline="red")
            self.selected_object_bbox = None
            # Stay in awaiting_object_selection mode
            self.unselected_bboxes.add(target_bbox['id'])
            
            SuccessDialog(self.master, "Object Unselected", 
                         f"Object box {target_bbox['category_name']}:{target_bbox['id']} has been unselected.\n"
                         "It cannot be reselected for this relationship.").show()
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Object & Predicate]")
            return
            
        # This should not happen if the caller checked properly
        WarningDialog(self.master, "Unselect Error", 
                     f"Box {target_bbox['category_name']}:{target_bbox['id']} is not currently selected.\n"
                     "Shift+click only works on blue (subject) or green (object) selected boxes.").show()

    def _get_bbox_at_position(self, x, y):
        """
        Get bbox at canvas position.
        When multiple boxes overlap, returns the smallest one (highest priority).
        """
        # Find all bounding boxes that contain the point
        overlapping_bboxes = []
        for bbox in self.current_bboxes:
            x1, y1, x2, y2 = bbox['screen_coords']
            if x1 <= x <= x2 and y1 <= y <= y2:
                overlapping_bboxes.append(bbox)
        
        if not overlapping_bboxes:
            return None
        
        # If only one bbox, return it
        if len(overlapping_bboxes) == 1:
            return overlapping_bboxes[0]
        
        # If multiple bboxes overlap, return the smallest one (by area)
        smallest_bbox = min(overlapping_bboxes, 
                           key=lambda bbox: (bbox['screen_coords'][2] - bbox['screen_coords'][0]) * 
                                          (bbox['screen_coords'][3] - bbox['screen_coords'][1]))
        return smallest_bbox

    def on_predicate_select(self, event):
        """Handle predicate selection for relationship creation."""
        if not (self.selected_subject_bbox and self.selected_object_bbox):
            return
            
        selection = self.rel_listbox.curselection()
        if not selection:
            return
            
        predicate = self.rel_listbox.get(selection[0])
        
        # Create relationship
        relationship = {
            'subject_id': self.selected_subject_bbox['id'],
            'subject_category': self.selected_subject_bbox['category_name'],
            'predicate': predicate,
            'object_id': self.selected_object_bbox['id'],
            'object_category': self.selected_object_bbox['category_name'],
            'image_id': self.current_image_id
        }
        
        self.relationships.append(relationship)
        self._update_relationships_view()
        
        # Show success and reset
        subject_info = f"{self.selected_subject_bbox['category_name']}:{self.selected_subject_bbox['id']}"
        object_info = f"{self.selected_object_bbox['category_name']}:{self.selected_object_bbox['id']}"
        
        SuccessDialog(self.master, "Relationship Created", 
                     f"Created: {subject_info} → {predicate} → {object_info}").show()
        
        self._reset_relation_mode()
        
    def add_new_predicate(self):
        """Add a new predicate to the relationships list."""
        from tkinter import simpledialog, messagebox
        
        # Show input dialog
        predicate = simpledialog.askstring(
            "Add New Predicate", 
            "Enter predicate name:",
            parent=self.master
        )
        
        if predicate and predicate.strip():
            predicate = predicate.strip()
            
            # Check if predicate already exists
            if hasattr(self, 'relationships_mapping') and predicate in self.relationships_mapping:
                messagebox.showwarning("Duplicate Predicate", 
                                     f"The predicate '{predicate}' already exists.")
                return
            
            # Initialize relationships_mapping if it doesn't exist
            if not hasattr(self, 'relationships_mapping'):
                self.relationships_mapping = {}
            
            # Add new predicate
            next_id = len(self.relationships_mapping)
            self.relationships_mapping[predicate] = next_id
            
            # Update the relationships list
            self.rel_listbox.delete(0, tk.END)
            for rel in sorted(self.relationships_mapping.keys()):
                self.rel_listbox.insert(tk.END, rel)
            
            # Save to relationships file
            self.save_relationships_to_file()
            
            messagebox.showinfo("Success", 
                              f"Successfully added predicate '{predicate}'.")
    
    def save_relationships_to_file(self):
        """Save current relationships to the relationships file."""
        try:
            # Use default relationships file path
            relationships_file = os.path.join(os.path.dirname(__file__), 'new_relationships.txt')
            
            with open(relationships_file, 'w') as f:
                for rel in sorted(self.relationships_mapping.keys()):
                    f.write(f"{rel}\n")
                    
        except Exception as e:
            from tkinter import messagebox
            messagebox.showwarning("Warning", 
                                 f"Could not save relationships to file: {str(e)}")
        
        self._reset_relation_mode()

    def _show_predicate_selection_dialog(self):
        """Show a dialog to select predicate for the relationship."""
        if not (self.selected_subject_bbox and self.selected_object_bbox):
            return
        
        # Check if relationships are loaded
        if not self.relationships_mapping:
            WarningDialog(self.master, "No Relationships", 
                         "No relationships have been imported yet.\n\nPlease use 'Import Relationship List' to load relationships first.").show()
            self._reset_relation_mode()
            return
            
        # Create dialog
        dialog = tk.Toplevel(self.master)
        dialog.title("Select Relationship Predicate")
        dialog.transient(self.master)
        
        # Calculate dialog size based on screen
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        dialog_width = max(int(screen_width * 0.25), 400)
        dialog_height = max(int(screen_height * 0.4), 300)
        
        # Center the dialog
        x = (screen_width // 2) - (dialog_width // 2)
        y = (screen_height // 2) - (dialog_height // 2)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Make sure dialog is visible before grabbing
        dialog.update_idletasks()
        dialog.deiconify()
        
        try:
            dialog.grab_set()
        except tk.TclError:
            # If grab fails, continue without it
            pass
        
        # Create main frame
        main_frame = tk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Subject and object info
        subject_info = f"{self.selected_subject_bbox['category_name']}:{self.selected_subject_bbox['id']}"
        object_info = f"{self.selected_object_bbox['category_name']}:{self.selected_object_bbox['id']}"
        
        info_label = tk.Label(
            main_frame,
            text=f"Creating relationship:\n{subject_info} → [PREDICATE] → {object_info}",
            font=("Arial", 12, "bold"),
            justify=tk.CENTER,
            wraplength=dialog_width-40
        )
        info_label.pack(pady=(0, 10))
        
        # Instructions
        instruction_label = tk.Label(
            main_frame,
            text="Double-click on a predicate to create the relationship:",
            font=("Arial", 10),
            fg="blue"
        )
        instruction_label.pack(pady=(0, 5))
        
        # Predicate listbox with scrollbar
        listbox_frame = tk.Frame(main_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        predicate_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 11),
            selectmode=tk.SINGLE
        )
        scrollbar.config(command=predicate_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        predicate_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Populate predicate list
        predicates = sorted(self.relationships_mapping.keys())
        if not predicates:
            WarningDialog(dialog, "No Predicates", 
                         "No predicates found in the relationship mapping.\n\nPlease check your relationship file.").show()
            dialog.destroy()
            self._reset_relation_mode()
            return
            
        for predicate in predicates:
            predicate_listbox.insert(tk.END, predicate)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        def on_create_relationship():
            selection = predicate_listbox.curselection()
            if not selection:
                WarningDialog(dialog, "No Selection", "Please select a predicate.").show()
                return
            
            predicate = predicate_listbox.get(selection[0])
            self._create_relationship_with_predicate(predicate)
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
            self._reset_relation_mode()
        
        # Buttons
        create_button = tk.Button(
            button_frame,
            text="Create Relationship",
            command=on_create_relationship,
            font=("Arial", 11, "bold"),
            bg="#ccffcc",
            padx=15,
            pady=5
        )
        create_button.pack(side=tk.LEFT, padx=(0, 5))
        
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            font=("Arial", 11),
            bg="#ffcccc",
            padx=15,
            pady=5
        )
        cancel_button.pack(side=tk.RIGHT)
        
        # Bind double-click to create relationship
        def on_double_click(event):
            on_create_relationship()
        
        predicate_listbox.bind("<Double-Button-1>", on_double_click)
        
        # Focus on listbox
        predicate_listbox.focus_set()
        if predicate_listbox.size() > 0:
            predicate_listbox.selection_set(0)

    def _create_relationship_with_predicate(self, predicate):
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
                WarningDialog(self.master, "Duplicate Relationship", 
                             "This relationship already exists.").show()
                self._reset_relation_mode()
                return
        
        self.relationships.append(relationship)
        self._update_relationships_view()
        
        # Reset for next relationship (no success popup)
        self._reset_relation_mode()

    def on_object_list_click(self, event):
        """Handle clicks on the objects list for relationship creation."""
        if not self.relation_creation_mode:
            return
            
        selection = self.objects_listbox.curselection()
        if not selection:
            return
            
        # Get the selected object text and parse it
        selected_text = self.objects_listbox.get(selection[0])
        # Format is "category_name:id"
        try:
            category_name, object_id = selected_text.split(':')
            object_id = int(object_id)
        except (ValueError, IndexError):
            return
            
        # Find the corresponding bbox
        selected_bbox = None
        for bbox in self.current_bboxes:
            if bbox['id'] == object_id and bbox['category_name'] == category_name:
                selected_bbox = bbox
                break
                
        if not selected_bbox:
            return
            
        # Check if this box has been unselected and cannot be reselected
        if selected_bbox['id'] in self.unselected_bboxes:
            WarningDialog(self.master, "Create Relation", 
                         "This box has been unselected and cannot be used in the current relationship.\n"
                         "Start a new relationship or cancel to reuse this box.").show()
            return
            
        if not self.awaiting_object_selection:
            # Select subject
            self.selected_subject_bbox = selected_bbox
            self.awaiting_object_selection = True
            
            # Highlight subject in blue
            canvas_obj = self.bbox_canvas_objects[selected_bbox['id']]
            self.canvas.itemconfig(canvas_obj['rect_id'], outline="blue", width=3)
            
            self.master.title("SGDET-Annotate - COCO Relations Tool - [Select Object & Predicate]")
            
        else:
            # Select object
            if selected_bbox['id'] == self.selected_subject_bbox['id']:
                WarningDialog(self.master, "Invalid Selection", 
                             "Subject and object cannot be the same.").show()
                return
                
            self.selected_object_bbox = selected_bbox
            
            # Highlight object in green
            canvas_obj = self.bbox_canvas_objects[selected_bbox['id']]
            self.canvas.itemconfig(canvas_obj['rect_id'], outline="green", width=3)
            
            # Show predicate selection dialog
            self._show_predicate_selection_dialog()

    def _update_relationships_view(self):
        """Update the relationships listbox."""
        self.relationships_listbox.delete(0, tk.END)
        
        # Show only relationships for current image
        current_image_relationships = [
            rel for rel in self.relationships 
            if rel['image_id'] == self.current_image_id
        ]
        
        for rel in current_image_relationships:
            entry = (f"{rel['subject_category']}:{rel['subject_id']} → "
                    f"{rel['predicate']} → "
                    f"{rel['object_category']}:{rel['object_id']}")
            self.relationships_listbox.insert(tk.END, entry)

    def on_delete_relationship(self, event):
        """Delete selected relationship."""
        selection = self.relationships_listbox.curselection()
        if not selection:
            return
            
        # Find the relationship to delete
        current_image_relationships = [
            rel for rel in self.relationships 
            if rel['image_id'] == self.current_image_id
        ]
        
        if selection[0] < len(current_image_relationships):
            rel_to_delete = current_image_relationships[selection[0]]
            
            if ConfirmationDialog(self.master, "Delete Relationship", 
                                "Are you sure you want to delete this relationship?").show():
                self.relationships.remove(rel_to_delete)
                self._update_relationships_view()

    def show_relationship_context_menu(self, event):
        """Show context menu for relationships."""
        index = self.relationships_listbox.nearest(event.y)
        if index >= 0:
            self.relationships_listbox.selection_clear(0, tk.END)
            self.relationships_listbox.selection_set(index)
            
            context_menu = tk.Menu(self.master, tearoff=0)
            context_menu.add_command(label="Delete Relationship", 
                                   command=lambda: self.on_delete_relationship(None))
            
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

    def _load_existing_relationships(self):
        """Load existing relationships for current image."""
        # Relationships are already loaded from COCO file or from previous sessions
        # Just update the view to show relationships for current image
        current_image_relationships = [
            rel for rel in self.relationships 
            if rel['image_id'] == self.current_image_id
        ]
        self._update_relationships_view()
        
        if current_image_relationships:
            print(f"Image {self.current_image_id} has {len(current_image_relationships)} existing relationships")

    def _save_current_relationships(self):
        """Save relationships for current image."""
        if not self.current_image_id:
            return
            
        # This saves in memory - actual file saving happens in save_relationships()
        pass

    def save_relationships(self):
        """Save all relationships to file."""
        if not self.relationships:
            WarningDialog(self.master, "No Relationships", 
                         "No relationships to save.").show()
            return
            
        # Save in multiple formats
        self._save_relationships_json()
        self._save_relationships_txt()
        self._save_relationships_coco_format()
        
        SuccessDialog(self.master, "Relationships Saved", 
                     f"Saved {len(self.relationships)} relationships in multiple formats.\n\nCOCO format saved as: {self.output_filename}").show()

    def _save_relationships_json(self):
        """Save relationships in JSON format."""
        output_file = os.path.join(self.output_dir, "relationships.json")
        
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

    def _save_relationships_txt(self):
        """Save relationships in readable text format."""
        output_file = os.path.join(self.output_dir, "relationships.txt")
        
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

    def _save_relationships_coco_format(self):
        """Save relationships in COCO format with global rel_annotations field."""
        if not self.coco_data:
            return
            
        output_file = os.path.join(self.output_dir, self.output_filename)
        
        # Create a deep copy of the original COCO data
        import copy
        coco_with_relationships = copy.deepcopy(self.coco_data)
        
        # Create rel_annotations array (similar to annotations)
        rel_annotations = []
        rel_annotation_id = 0
        
        for rel in self.relationships:
            relationship_entry = {
                'id': rel_annotation_id,
                'subject_id': rel['subject_id'],
                'predicate_id': self.relationships_mapping.get(rel['predicate'], -1),
                'object_id': rel['object_id'],
                'image_id': rel['image_id']
            }
            rel_annotations.append(relationship_entry)
            rel_annotation_id += 1
        
        # Add rel_annotations field after annotations
        coco_with_relationships['rel_annotations'] = rel_annotations
        
        # Add relationship categories to the COCO format (rename from relationship_categories)
        if 'rel_categories' not in coco_with_relationships:
            rel_categories = []
            for predicate, pred_id in self.relationships_mapping.items():
                rel_categories.append({
                    'id': pred_id,
                    'name': predicate
                })
            coco_with_relationships['rel_categories'] = rel_categories
        
        # Add metadata about relationships
        coco_with_relationships['info']['description'] += ' - Enhanced with relationship annotations'
        coco_with_relationships['info']['relationships_version'] = '1.0'
        coco_with_relationships['info']['total_relationships'] = len(self.relationships)
        
        # Save the enhanced COCO file
        with open(output_file, 'w') as f:
            json.dump(coco_with_relationships, f, indent=2)
        
        print(f"Saved COCO format with relationships to: {output_file}")

    def _load_existing_relationships_from_coco(self):
        """Load existing relationships from COCO format if they exist."""
        if not self.coco_data:
            return
            
        # Check if relationships exist in the COCO data
        existing_relationships = []
        
        # Load relationship categories if they exist (support both old and new format)
        rel_categories = {}
        if 'rel_categories' in self.coco_data:
            rel_categories = {cat['id']: cat['name'] for cat in self.coco_data['rel_categories']}
        elif 'relationship_categories' in self.coco_data:
            rel_categories = {cat['id']: cat['name'] for cat in self.coco_data['relationship_categories']}
        
        # Update relationships mapping
        for cat_id, cat_name in rel_categories.items():
            if cat_name not in self.relationships_mapping:
                self.relationships_mapping[cat_name] = cat_id
        
        # Check for new global rel_annotations format first
        if 'rel_annotations' in self.coco_data:
            for rel_ann in self.coco_data['rel_annotations']:
                # Get category names from annotation IDs
                subject_category = 'unknown'
                object_category = 'unknown'
                
                # Find the annotations to get category names
                for ann in self.coco_data.get('annotations', []):
                    if ann['id'] == rel_ann['subject_id']:
                        subject_category = self.categories_data.get(ann['category_id'], {'name': 'unknown'})['name']
                    elif ann['id'] == rel_ann['object_id']:
                        object_category = self.categories_data.get(ann['category_id'], {'name': 'unknown'})['name']
                
                # Get predicate name from ID
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
        else:
            # Fallback: Load relationships from images (old format)
            for image in self.coco_data.get('images', []):
                if 'relationships' in image:
                    image_id = image['id']
                    for rel in image['relationships']:
                        relationship = {
                            'subject_id': rel['subject_id'],
                            'subject_category': rel.get('subject_category', 'unknown'),
                            'predicate': rel.get('predicate', rel_categories.get(rel.get('predicate_id', -1), 'unknown')),
                            'object_id': rel['object_id'],
                            'object_category': rel.get('object_category', 'unknown'),
                            'image_id': image_id
                        }
                        existing_relationships.append(relationship)
        
        if existing_relationships:
            self.relationships.extend(existing_relationships)
            print(f"Loaded {len(existing_relationships)} existing relationships from COCO file")

    def on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self.loaded_image:
            self._display_image()
            self._load_current_bboxes()

    def on_closing(self):
        """Handle application closing."""
        if self.relationships:
            result = ConfirmationDialog(
                self.master, "Unsaved Relationships",
                "You have unsaved relationships. Do you want to save before exiting?"
            ).show()
            if result:
                self.save_relationships()
        self.master.destroy()


def main():
    """Main entry point for the COCO Relations annotation tool."""
    parser = argparse.ArgumentParser(
        description="SGDET-Annotate COCO Relations Tool - Add relationship annotations to COCO format data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_coco.py
  python main_coco.py --output my_annotations.json
  python main_coco.py -o enhanced_coco_data.json
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='annotations_with_relationships.json',
        help='Output filename for the COCO file with relationships (default: annotations_with_relationships.json)'
    )
    
    args = parser.parse_args()
    
    # Validate output filename
    if not args.output.endswith('.json'):
        args.output += '.json'
    
    root = tk.Tk()
    app = COCORelationAnnotator(root, output_filename=args.output)
    
    # Display output filename in the title
    root.title(f"SGDET-Annotate - COCO Relations Tool - Output: {args.output}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
