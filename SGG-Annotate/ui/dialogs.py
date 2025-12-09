"""
Custom dialog classes for the SGDET-Annotate application.
Provides scalable dialogs with consistent styling and behavior.
"""

import tkinter as tk
from typing import Optional, Tuple, List, Callable


class BaseDialog:
    """Base class for all custom dialogs with common functionality."""
    
    def __init__(self, parent: tk.Widget, title: str, width_ratio: float = 0.5, height_ratio: float = 0.4, 
                 min_width: int = 700, min_height: int = 300):
        """
        Initialize a base dialog with responsive sizing optimized for 4K screens.
        
        Args:
            parent: Parent window
            title: Dialog title
            width_ratio: Width as ratio of parent window (increased for 4K)
            height_ratio: Height as ratio of parent window (increased for 4K)
            min_width: Minimum dialog width (increased for 4K)
            min_height: Minimum dialog height (increased for 4K)
        """
        self.parent = parent
        self.result = None
        
        # Get screen dimensions for better 4K scaling
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        
        # Calculate dialog size based on parent window or screen
        parent_width = max(parent.winfo_width(), 1600)  # Ensure minimum size for calculation
        parent_height = max(parent.winfo_height(), 900)
        
        # Use larger ratios for high-resolution screens
        if screen_width >= 3840:  # 4K or higher
            width_ratio = min(width_ratio * 1.3, 0.8)
            height_ratio = min(height_ratio * 1.3, 0.7)
            min_width = max(min_width, 900)
            min_height = max(min_height, 400)
        
        dialog_width = max(int(parent_width * width_ratio), min_width)
        dialog_height = max(int(parent_height * height_ratio), min_height)
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry(f"{dialog_width}x{dialog_height}")
        self.dialog.transient(parent)
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (screen_width // 2) - (dialog_width // 2)
        y = (screen_height // 2) - (dialog_height // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        # Update dialog to ensure it's visible before grabbing focus
        self.dialog.update()
        self.dialog.grab_set()
        
        # Calculate font sizes based on screen resolution
        self.scale_factor = self._calculate_scale_factor(screen_width, screen_height)
        self.label_font_size = max(int(18 * self.scale_factor), 16)
        self.button_font_size = max(int(16 * self.scale_factor), 14)
        
        self.dialog_width = dialog_width
        self.dialog_height = dialog_height
        
    def _calculate_scale_factor(self, width: int, height: int) -> float:
        """Calculate scaling factor based on screen size for 4K support."""
        # Base calculation on screen resolution instead of window size
        width_scale = width / 1920  # Base on 1920x1080
        height_scale = height / 1080
        scale = max(min(width_scale, height_scale), 0.8)
        
        # Additional scaling for very high resolutions
        if width >= 3840:  # 4K
            scale = max(scale, 1.5)
        elif width >= 2560:  # QHD
            scale = max(scale, 1.2)
            
        return min(scale, 2.5)  # Cap at 2.5x for very large screens
        
    def show(self) -> any:
        """Show the dialog and return the result."""
        self.dialog.focus_set()
        self.dialog.wait_window()
        return self.result
        
    def close(self, result=None):
        """Close the dialog with optional result."""
        self.result = result
        self.dialog.destroy()


class ConfirmationDialog(BaseDialog):
    """Dialog for simple confirmation with OK/Cancel buttons."""
    
    def __init__(self, parent: tk.Widget, title: str, message: str, 
                 ok_text: str = "OK", cancel_text: str = "Cancel"):
        super().__init__(parent, title, width_ratio=0.4, height_ratio=0.3, min_width=600, min_height=250)
        
        # Add message label
        message_label = tk.Label(
            self.dialog,
            text=message,
            font=("Arial", self.label_font_size),
            wraplength=self.dialog_width-60,
            justify=tk.CENTER
        )
        message_label.pack(pady=30, padx=30, expand=True)
        
        # Add buttons frame
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=15)
        
        def on_ok():
            self.close(True)
            
        def on_cancel():
            self.close(False)
        
        ok_button = tk.Button(
            button_frame,
            text=ok_text,
            command=on_ok,
            font=("Arial", self.button_font_size, "bold"),
            bg="lightgreen",
            padx=30,
            pady=10,
            width=10
        )
        ok_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = tk.Button(
            button_frame,
            text=cancel_text,
            command=on_cancel,
            font=("Arial", self.button_font_size),
            padx=30,
            pady=10,
            width=10
        )
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Bind Enter and Escape keys
        self.dialog.bind('<Return>', lambda e: on_ok())
        self.dialog.bind('<Escape>', lambda e: on_cancel())


class WarningDialog(BaseDialog):
    """Dialog for showing warning messages."""
    
    def __init__(self, parent: tk.Widget, title: str, message: str):
        super().__init__(parent, title, width_ratio=0.45, height_ratio=0.3, min_width=600, min_height=250)
        
        # Warning message
        warning_label = tk.Label(
            self.dialog,
            text=message,
            font=("Arial", self.label_font_size),
            wraplength=self.dialog_width-60,
            justify=tk.CENTER,
            fg="red"
        )
        warning_label.pack(pady=30, padx=30, expand=True)
        
        def close_warning():
            self.close()
        
        warning_ok_button = tk.Button(
            self.dialog,
            text="OK",
            command=close_warning,
            font=("Arial", self.button_font_size, "bold"),
            bg="lightcoral",
            padx=40,
            pady=12,
            width=15
        )
        warning_ok_button.pack(pady=15)
        
        # Bind keys
        self.dialog.bind('<Return>', lambda e: close_warning())
        self.dialog.bind('<Escape>', lambda e: close_warning())


class SuccessDialog(BaseDialog):
    """Dialog for showing success messages."""
    
    def __init__(self, parent: tk.Widget, title: str, message: str):
        super().__init__(parent, title, width_ratio=0.45, height_ratio=0.3, min_width=600, min_height=250)
        
        # Success message
        success_label = tk.Label(
            self.dialog,
            text=message,
            font=("Arial", self.label_font_size, "bold"),
            wraplength=self.dialog_width-60,
            justify=tk.CENTER,
            fg="green"
        )
        success_label.pack(pady=30, padx=30, expand=True)
        
        def close_success():
            self.close()
        
        success_ok_button = tk.Button(
            self.dialog,
            text="OK",
            command=close_success,
            font=("Arial", self.button_font_size, "bold"),
            bg="lightgreen",
            padx=40,
            pady=12,
            width=15
        )
        success_ok_button.pack(pady=15)
        
        # Bind keys
        self.dialog.bind('<Return>', lambda e: close_success())
        self.dialog.bind('<Escape>', lambda e: close_success())


class PredicateSelectionDialog(BaseDialog):
    """Dialog for selecting predicates in relationship creation."""
    
    def __init__(self, parent: tk.Widget, subject_bbox: dict, object_bbox: dict, 
                 predicates: List[str], on_create_callback: Callable):
        super().__init__(parent, "Select Relationship Predicate", 
                        width_ratio=0.4, height_ratio=0.5, min_width=500, min_height=400)
        
        self.subject_bbox = subject_bbox
        self.object_bbox = object_bbox
        self.predicates = predicates
        self.on_create_callback = on_create_callback
        
        # Add label showing the relationship
        info_label = tk.Label(
            self.dialog,
            text=f"Create relationship:\n{subject_bbox['label_str']}:{subject_bbox['id']} → {object_bbox['label_str']}:{object_bbox['id']}",
            font=("Arial", max(int(14 * self.scale_factor), 12), "bold"),
            wraplength=self.dialog_width-50
        )
        info_label.pack(pady=15)
        
        # Add listbox for predicate selection
        listbox_frame = tk.Frame(self.dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.predicate_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", max(int(12 * self.scale_factor), 10))
        )
        scrollbar.config(command=self.predicate_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.predicate_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Populate with predicates
        for predicate in predicates:
            self.predicate_listbox.insert(tk.END, predicate)
        
        # Add buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        def on_create():
            selection = self.predicate_listbox.curselection()
            if not selection:
                WarningDialog(self.dialog, "Selection Required", "Please select a predicate.").show()
                return
            
            predicate = self.predicate_listbox.get(selection[0])
            self.close(predicate)
            
        def on_cancel():
            self.close(None)
        
        create_button = tk.Button(
            button_frame,
            text="Create Relationship",
            command=on_create,
            font=("Arial", self.button_font_size, "bold"),
            bg="lightgreen",
            padx=15,
            pady=5
        )
        create_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            font=("Arial", self.button_font_size),
            padx=15,
            pady=5
        )
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Allow double-click to create relationship
        self.predicate_listbox.bind("<Double-Button-1>", lambda e: on_create())
        
        # Add keyboard bindings
        self.dialog.bind('<Return>', lambda e: on_create())
        self.dialog.bind('<Escape>', lambda e: on_cancel())
