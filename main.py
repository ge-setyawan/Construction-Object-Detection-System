import os
import cv2
import time
import math
import numpy as np
import pandas as pd
import customtkinter as ctk

from PIL import Image, ImageTk, ImageEnhance
from tkinter import filedialog, messagebox


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Construction YOLOv11 Detection"
PREVIEW_PADDING = 18
VIDEO_DETECTION_STRIDE = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

CLASS_NAMES = [
    "Cangkul",
    "Helm Proyek",
    "Scaffolding Bambu",
    "Semen",
    "Tukang"
]

ENHANCEMENTS = [
    "Original",
    "Contrast Stretching",
    "Histogram Equalization",
    "CLAHE",
    "Brightness"
]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def enhance_image(image, method):

    if image is None:
        return image

    if method == "Original":
        return image.copy()

    if method == "Brightness":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # brightness +20%
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2].astype(np.float32) * 1.20,
            0,
            255
        ).astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if method == "Contrast Stretching":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        min_val = np.min(l)
        max_val = np.max(l)

        if max_val > min_val:
            l = ((l - min_val) * 255.0 / (max_val - min_val)).astype(
                np.uint8
            )

        result = cv2.merge([l, a, b])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    if method == "Histogram Equalization":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        y, cr, cb = cv2.split(ycrcb)

        y = cv2.equalizeHist(y)

        result = cv2.merge([y, cr, cb])

        return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)

    if method == "CLAHE":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        result = cv2.merge([l, a, b])

        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return image.copy()


# ============================================================
# MAIN APPLICATION
# ============================================================

class ConstructionYOLOApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1440x880")
        self.minsize(1080, 700)

        self.configure(fg_color="#151515")

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = None

        self.load_model()

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        self.current_file = None
        self.current_type = None

        self.original_image = None
        self.current_frame = None
        self.preview_source_frame = None
        self._preview_resize_job = None

        self.video_capture = None
        self.video_running = False
        self.video_thread = None
        self.video_after_job = None

        self.video_total_frames = 0
        self.video_current_frame = 0
        self.video_fps = 25
        self.video_last_detections = []

        # ----------------------------------------------------
        # REGION
        # ----------------------------------------------------

        self.region_enabled = False

        # Region in ORIGINAL IMAGE COORDINATES
        self.region = None

        # Canvas display information
        self.display_image_width = 1
        self.display_image_height = 1
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.display_scale = 1

        self.region_start = None
        self.region_temp = None

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        self.last_results = []

        self.last_stats = {
            "total": 0,
            "avg_conf": 0,
            "classes": 0,
            "counts": [0] * len(CLASS_NAMES),
            "region_total": 0,
            "region_counts": [0] * len(CLASS_NAMES)
        }

        self.export_rows = []

        # ----------------------------------------------------
        # BUILD UI
        # ----------------------------------------------------

        self.build_ui()

    # ========================================================
    # MODEL
    # ========================================================

    def load_model(self):

        try:
            from ultralytics import YOLO

            if not os.path.exists(MODEL_PATH):
                self.model = None
                return

            self.model = YOLO(MODEL_PATH)

        except Exception as e:
            self.model = None
            print("Model error:", e)

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        # ====================================================
        # HEADER
        # ====================================================

        header = ctk.CTkFrame(
            self,
            fg_color="#1b222c",
            corner_radius=0
        )

        header.pack(
            fill="x",
            padx=0,
            pady=0
        )

        title = ctk.CTkLabel(
            header,
            text="Construction Object Detection System",
            font=ctk.CTkFont(
                size=22,
                weight="bold"
            )
        )

        title.pack(pady=7)

        subtitle = ctk.CTkLabel(
            header,
            text="AI Object Detection • Image Enhancement • Counting • Region Analysis",
            font=ctk.CTkFont(size=12),
            text_color="#aeb9c8"
        )

        # Subtitle intentionally omitted to keep the header compact.

        # ====================================================
        # MAIN
        # ====================================================

        main = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12
        )

        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=0, minsize=260)
        main.grid_columnconfigure(1, weight=1)
        main.grid_columnconfigure(2, weight=0, minsize=265)

        # LEFT PANEL
        self.left_panel = ctk.CTkFrame(
            main,
            width=260,
            fg_color="#1d252f",
            corner_radius=14
        )

        self.left_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        # The centre stays dedicated to the media preview.
        centre = ctk.CTkFrame(
            main,
            fg_color="#1d252f",
            corner_radius=14
        )

        centre.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.results_panel = ctk.CTkFrame(
            main,
            width=265,
            fg_color="#1d252f",
            corner_radius=14
        )
        self.results_panel.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(12, 0)
        )

        self.build_left_panel()
        self.build_right_panel(centre)
        self.build_results_panel(self.results_panel)

    # ========================================================
    # LEFT PANEL
    # ========================================================

    def build_left_panel(self):

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        self.section_label(
            self.left_panel,
            "INPUT"
        )

        self.btn_image = ctk.CTkButton(
            self.left_panel,
            text="▣  Buka Gambar",
            height=36,
            font=ctk.CTkFont(size=13),
            command=self.open_image
        )

        self.btn_image.pack(
            fill="x",
            padx=12,
            pady=(2, 5)
        )

        self.btn_video = ctk.CTkButton(
            self.left_panel,
            text="▶  Buka Video",
            height=36,
            font=ctk.CTkFont(size=13),
            command=self.open_video
        )

        self.btn_video.pack(
            fill="x",
            padx=12,
            pady=(0, 5)
        )

        self.file_label = ctk.CTkLabel(
            self.left_panel,
            text="Belum ada file",
            anchor="w",
            justify="left",
            wraplength=230,
            text_color="#bdbdbd",
            font=ctk.CTkFont(size=12)
        )

        self.file_label.pack(
            fill="x",
            padx=12,
            pady=(0, 7)
        )

        # ----------------------------------------------------
        # ENHANCEMENT
        # ----------------------------------------------------

        self.section_label(
            self.left_panel,
            "IMAGE ENHANCEMENT"
        )

        self.enhancement_menu = ctk.CTkOptionMenu(
            self.left_panel,
            values=ENHANCEMENTS,
            height=36,
            command=self.on_enhancement_change
        )

        self.enhancement_menu.set("Original")

        self.enhancement_menu.pack(
            fill="x",
            padx=12,
            pady=(2, 9)
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.section_label(
            self.left_panel,
            "MODEL"
        )

        model_status = (
            "● Model siap"
            if self.model is not None
            else "● Model tidak ditemukan"
        )

        self.model_label = ctk.CTkLabel(
            self.left_panel,
            text=model_status,
            text_color="#00e676" if self.model else "#ff5252",
            font=ctk.CTkFont(size=13)
        )

        self.model_label.pack(
            anchor="w",
            padx=12,
            pady=(0, 9)
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        self.section_label(
            self.left_panel,
            "CONFIDENCE THRESHOLD"
        )

        self.conf_value = ctk.CTkLabel(
            self.left_panel,
            text="0.25",
            anchor="e"
        )

        self.conf_value.pack(
            fill="x",
            padx=12
        )

        self.conf_slider = ctk.CTkSlider(
            self.left_panel,
            from_=0.05,
            to=0.90,
            number_of_steps=85,
            command=self.change_confidence
        )

        self.conf_slider.set(0.25)

        self.conf_slider.pack(
            fill="x",
            padx=12,
            pady=(2, 9)
        )

        # ----------------------------------------------------
        # REGION
        # ----------------------------------------------------

        self.section_label(
            self.left_panel,
            "OBJECT COUNTING & REGION"
        )

        self.region_button = ctk.CTkButton(
            self.left_panel,
            text="□  Aktifkan Region",
            height=36,
            command=self.toggle_region
        )

        self.region_button.pack(
            fill="x",
            padx=12,
            pady=(0, 5)
        )

        self.clear_region_button = ctk.CTkButton(
            self.left_panel,
            text="×  Hapus Region",
            height=34,
            fg_color="#555555",
            hover_color="#666666",
            command=self.clear_region
        )

        self.clear_region_button.pack(
            fill="x",
            padx=12,
            pady=(0, 10)
        )

    # ========================================================
    # RIGHT PANEL
    # ========================================================

    def build_right_panel(self, parent):

        # ----------------------------------------------------
        # PREVIEW HEADER
        # ----------------------------------------------------

        preview_header = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        preview_header.pack(
            fill="x",
            padx=20,
            pady=(16, 10)
        )

        ctk.CTkLabel(
            preview_header,
            text="Pratinjau Deteksi",
            font=ctk.CTkFont(
                size=24,
                weight="bold"
            )
        ).pack(side="left")

        self.video_info = ctk.CTkLabel(
            preview_header,
            text="",
            text_color="#bdbdbd"
        )

        self.video_info.pack(
            side="right"
        )

        # ----------------------------------------------------
        # CANVAS
        # ----------------------------------------------------

        self.preview_frame = ctk.CTkFrame(
            parent,
            fg_color="#0b1016",
            corner_radius=12,
            border_width=1,
            border_color="#344151"
        )

        self.preview_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(0, 10)
        )

        self.canvas = ctk.CTkCanvas(
            self.preview_frame,
            bg="#0b1016",
            highlightthickness=0,
            cursor="crosshair"
        )

        self.canvas.pack(
            fill="both",
            expand=True
        )

        self.canvas.bind(
            "<ButtonPress-1>",
            self.region_mouse_down
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.region_mouse_drag
        )

        self.canvas.bind(
            "<ButtonRelease-1>",
            self.region_mouse_up
        )

        self.canvas.bind(
            "<Configure>",
            self.on_canvas_resize
        )

        # ----------------------------------------------------
        # VIDEO CONTROLS
        # ----------------------------------------------------

        controls = ctk.CTkFrame(
            parent,
            fg_color="#171e27",
            corner_radius=10
        )

        controls.pack(
            fill="x",
            padx=20,
            pady=(0, 16)
        )

        self.play_button = ctk.CTkButton(
            controls,
            text="▶  Play Video",
            width=150,
            height=42,
            command=self.play_video
        )

        self.play_button.pack(
            side="left",
            padx=(0, 8)
        )

        self.stop_button = ctk.CTkButton(
            controls,
            text="■  Stop",
            width=120,
            height=42,
            fg_color="#555555",
            hover_color="#666666",
            command=self.stop_video
        )

        self.stop_button.pack(
            side="left"
        )

        self.frame_info = ctk.CTkLabel(
            controls,
            text="",
            text_color="#bdbdbd"
        )

        self.frame_info.pack(
            side="left",
            padx=20
        )

        self.export_button = ctk.CTkButton(
            controls,
            text="↓  Export Excel",
            width=160,
            height=42,
            command=self.export_excel
        )

        self.export_button.pack(
            side="right"
        )

        self.show_empty_preview()

    def build_results_panel(self, parent):
        """Build a compact, fixed detection sidebar."""

        results = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )
        results.pack(fill="x", padx=9, pady=8)

        ctk.CTkLabel(
            results,
            text="HASIL DETEKSI",
            anchor="w",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(fill="x", pady=(1, 5))

        summary = ctk.CTkFrame(results, fg_color="transparent")
        summary.pack(fill="x")
        summary.grid_columnconfigure(0, weight=1)

        self.total_label = self.summary_card(
            summary, "Total Objects", "0", 0
        )
        self.conf_label = self.summary_card(
            summary, "Average Confidence", "0.0%", 1
        )
        self.classes_label = self.summary_card(
            summary, "Classes Detected", "0", 2
        )

        classes = ctk.CTkFrame(results, fg_color="transparent")
        classes.pack(fill="x", pady=(7, 0))

        ctk.CTkLabel(
            classes,
            text="OBJEK TERDETEKSI",
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#bdbdbd"
        ).pack(fill="x", pady=(0, 3))

        self.class_labels = []
        for name in CLASS_NAMES:
            class_row = ctk.CTkFrame(
                classes,
                fg_color="#252525",
                corner_radius=8,
            )
            class_row.pack(fill="x", pady=1)
            class_row.pack_propagate(False)
            class_row.configure(height=27)

            ctk.CTkLabel(
                class_row,
                text=name,
                anchor="w",
                font=ctk.CTkFont(size=12)
            ).pack(side="left", padx=(9, 4), pady=3)

            count_label = ctk.CTkLabel(
                class_row,
                text="0",
                anchor="e",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            count_label.pack(side="right", padx=(4, 9), pady=3)
            self.class_labels.append(count_label)

        region = ctk.CTkFrame(results, fg_color="#252525", corner_radius=8)
        region.pack(fill="x", pady=(8, 0))
        self.region_total_label = ctk.CTkLabel(
            region,
            text="Objects in Region: 0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00e676"
        )
        self.region_total_label.pack(anchor="w", padx=9, pady=(5, 1))

        self.region_class_labels = []
        for name in CLASS_NAMES:
            region_row = ctk.CTkFrame(
                region,
                fg_color="transparent"
            )
            region_row.pack(fill="x", padx=9, pady=0)

            ctk.CTkLabel(
                region_row,
                text=name,
                anchor="w",
                text_color="#bdbdbd",
                font=ctk.CTkFont(size=10)
            ).pack(side="left")

            count_label = ctk.CTkLabel(
                region_row,
                text="0",
                anchor="e",
                text_color="#bdbdbd",
                font=ctk.CTkFont(size=10, weight="bold")
            )
            count_label.pack(side="right")
            self.region_class_labels.append(count_label)

    def summary_card(self, parent, title, value, column):
        card = ctk.CTkFrame(parent, fg_color="#252525", corner_radius=9)
        card.grid(row=column, column=0, sticky="ew", pady=2)

        ctk.CTkLabel(
            card, text=title, text_color="#bdbdbd",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(3, 0))
        value_label = ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(size=17, weight="bold")
        )
        value_label.pack(pady=(0, 3))
        return value_label

    # ========================================================
    # UI HELPERS
    # ========================================================

    def section_label(self, parent, text):

        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            ),
            anchor="w"
        ).pack(
            fill="x",
            padx=12,
            pady=(6, 3)
        )

    def stat_card(self, parent, title, value):

        frame = ctk.CTkFrame(
            parent,
            fg_color="#292929",
            corner_radius=10
        )

        frame.pack(
            fill="x",
            padx=12,
            pady=5
        )

        ctk.CTkLabel(
            frame,
            text=title,
            text_color="#bdbdbd",
            anchor="w"
        ).pack(
            side="left",
            padx=12,
            pady=12
        )

        value_label = ctk.CTkLabel(
            frame,
            text=value,
            font=ctk.CTkFont(
                size=17,
                weight="bold"
            ),
            anchor="e"
        )

        value_label.pack(
            side="right",
            padx=12
        )

        return value_label

    # ========================================================
    # FILE
    # ========================================================

    @staticmethod
    def short_filename(path, max_length=38):
        """Keep the sidebar tidy while retaining the beginning and extension."""
        name = os.path.basename(path)
        if len(name) <= max_length:
            return name
        stem, extension = os.path.splitext(name)
        visible = max(12, max_length - len(extension) - 3)
        return f"{stem[:visible]}...{extension}"

    def open_image(self):

        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_video()

        self.current_file = path
        self.current_type = "image"

        self.file_label.configure(
            text=self.short_filename(path)
        )

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror(
                "Error",
                "Gambar tidak dapat dibuka."
            )
            return

        self.original_image = image
        self.current_frame = image.copy()

        self.region = None
        self.region_enabled = False

        self.region_button.configure(
            text="□  Aktifkan Region"
        )

        self.process_image()

    def open_video(self):

        path = filedialog.askopenfilename(
            title="Pilih video",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_video()

        self.current_file = path
        self.current_type = "video"

        self.file_label.configure(
            text=self.short_filename(path)
        )

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            messagebox.showerror(
                "Error",
                "Video tidak dapat dibuka."
            )
            return

        self.video_total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.video_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.video_fps <= 0:
            self.video_fps = 25

        cap.release()

        self.region = None
        self.region_enabled = False

        self.region_button.configure(
            text="□  Aktifkan Region"
        )

        self.video_info.configure(
            text=f"{self.video_total_frames} frames • {self.video_fps:.1f} FPS"
        )

        self.frame_info.configure(
            text=""
        )

        self.export_rows = []

        # Show first frame
        cap = cv2.VideoCapture(path)

        ret, frame = cap.read()

        cap.release()

        if ret:
            self.current_frame = frame
            self.process_frame(frame)

    # ========================================================
    # ENHANCEMENT
    # ========================================================

    def on_enhancement_change(self, value):

        if self.current_type == "image":
            self.process_image()

    def process_image(self):

        if self.original_image is None:
            return

        enhanced = enhance_image(
            self.original_image,
            self.enhancement_menu.get()
        )

        self.current_frame = enhanced

        self.process_frame(enhanced)

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def change_confidence(self, value):

        value = float(value)

        self.conf_value.configure(
            text=f"{value:.2f}"
        )

        if self.current_type == "image":
            self.process_image()

    # ========================================================
    # DETECTION
    # ========================================================

    def detect(self, frame, imgsz=640):

        if self.model is None:
            return frame, []

        confidence = float(
            self.conf_slider.get()
        )

        try:

            results = self.model.predict(
                source=frame,
                conf=confidence,
                imgsz=imgsz,
                device="cpu",
                verbose=False
            )

            result = results[0]

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    xyxy = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = map(
                        int,
                        xyxy
                    )

                    cls = int(
                        box.cls[0].cpu().item()
                    )

                    conf = float(
                        box.conf[0].cpu().item()
                    )

                    detections.append({
                        "class_id": cls,
                        "class_name": CLASS_NAMES[cls],
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2)
                    })

            return frame, detections

        except Exception as e:

            print("Detection error:", e)

            return frame, []

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    def draw_detections(
        self,
        frame,
        detections
    ):

        output = frame.copy()

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            cls = det["class_id"]

            conf = det["confidence"]

            name = det["class_name"]

            # Different color by class
            colors = [
                (0, 255, 255),
                (255, 100, 0),
                (255, 0, 255),
                (0, 255, 0),
                (0, 165, 255)
            ]

            color = colors[
                cls % len(colors)
            ]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text = f"{name} {conf:.2f}"

            font_scale = max(
                0.45,
                min(
                    0.9,
                    frame.shape[1] / 1600
                )
            )

            thickness = 2

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            label_y = max(
                y1,
                th + 5
            )

            cv2.rectangle(
                output,
                (x1, label_y - th - 8),
                (x1 + tw + 8, label_y),
                color,
                -1
            )

            cv2.putText(
                output,
                text,
                (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return output

    # ========================================================
    # REGION COUNTING
    # ========================================================

    def calculate_region(
        self,
        detections,
        frame
    ):

        region_counts = [0] * len(CLASS_NAMES)

        region_total = 0

        if self.region is None:
            return region_total, region_counts

        rx1, ry1, rx2, ry2 = self.region

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            inside = (
                rx1 <= cx <= rx2
                and
                ry1 <= cy <= ry2
            )

            if inside:

                region_total += 1

                cls = det["class_id"]

                region_counts[cls] += 1

                # Mark centroid
                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 0, 255),
                    -1
                )

        return region_total, region_counts

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):

        if frame is None:
            return

        display_frame = frame.copy()

        display_frame, detections = self.detect(
            display_frame
        )

        # Draw detections
        display_frame = self.draw_detections(
            display_frame,
            detections
        )

        # Region
        region_total, region_counts = \
            self.calculate_region(
                detections,
                display_frame
            )

        # Draw region
        if self.region is not None:

            rx1, ry1, rx2, ry2 = self.region

            cv2.rectangle(
                display_frame,
                (rx1, ry1),
                (rx2, ry2),
                (0, 255, 255),
                3
            )

            cv2.rectangle(
                display_frame,
                (rx1, ry1 - 38),
                (
                    min(
                        rx1 + 300,
                        display_frame.shape[1]
                    ),
                    ry1
                ),
                (0, 255, 255),
                -1
            )

            cv2.putText(
                display_frame,
                f"COUNTING REGION: {region_total}",
                (rx1 + 8, ry1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        self.last_results = detections

        self.update_statistics(
            detections,
            region_total,
            region_counts
        )

        self.display_frame(
            display_frame
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def update_statistics(
        self,
        detections,
        region_total,
        region_counts
    ):

        total = len(detections)

        if total > 0:

            avg_conf = (
                sum(
                    d["confidence"]
                    for d in detections
                ) / total
            )

        else:

            avg_conf = 0

        counts = [0] * len(CLASS_NAMES)

        for det in detections:

            cls = det["class_id"]

            counts[cls] += 1

        classes_detected = sum(
            1
            for c in counts
            if c > 0
        )

        self.total_label.configure(
            text=str(total)
        )

        self.conf_label.configure(
            text=f"{avg_conf * 100:.1f}%"
        )

        self.classes_label.configure(
            text=str(classes_detected)
        )

        for i, _ in enumerate(CLASS_NAMES):

            self.class_labels[i].configure(
                text=str(counts[i])
            )

        self.region_total_label.configure(
            text=f"Objects in Region: {region_total}"
        )

        for i, _ in enumerate(CLASS_NAMES):

            self.region_class_labels[i].configure(
                text=str(region_counts[i])
            )

        self.last_stats = {
            "total": total,
            "avg_conf": avg_conf,
            "classes": classes_detected,
            "counts": counts,
            "region_total": region_total,
            "region_counts": region_counts
        }

    # ========================================================
    # DISPLAY IMAGE - NO CROPPING
    # ========================================================

    def display_frame(self, frame):

        if frame is None:
            return

        # Keep the annotated frame so resizing only redraws the preview.
        self.preview_source_frame = frame

        canvas_w = max(
            100,
            self.canvas.winfo_width()
        )

        canvas_h = max(
            100,
            self.canvas.winfo_height()
        )

        h, w = frame.shape[:2]

        # Keep every edge visible and leave a small gutter around the media.
        available_w = max(1, canvas_w - (PREVIEW_PADDING * 2))
        available_h = max(1, canvas_h - (PREVIEW_PADDING * 2))
        scale = min(
            available_w / w,
            available_h / h
        )

        display_w = max(
            1,
            int(w * scale)
        )

        display_h = max(
            1,
            int(h * scale)
        )

        resized = cv2.resize(
            frame,
            (display_w, display_h),
            interpolation=(
                cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
            )
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb)

        self.photo = ImageTk.PhotoImage(
            pil_image
        )

        self.canvas.delete(
            "all"
        )

        offset_x = (
            canvas_w - display_w
        ) / 2

        offset_y = (
            canvas_h - display_h
        ) / 2

        self.canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self.photo
        )

        # Store mapping
        self.display_image_width = display_w
        self.display_image_height = display_h

        self.display_offset_x = offset_x
        self.display_offset_y = offset_y

        self.display_scale = scale

        # Redraw region overlay on canvas
        if self.region is not None:

            rx1, ry1, rx2, ry2 = self.region

            x1 = (
                offset_x +
                rx1 * scale
            )

            y1 = (
                offset_y +
                ry1 * scale
            )

            x2 = (
                offset_x +
                rx2 * scale
            )

            y2 = (
                offset_y +
                ry2 * scale
            )

            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline="#FFD600",
                width=4
            )

            self.canvas.create_rectangle(
                x1,
                max(0, y1 - 32),
                x1 + 260,
                y1,
                fill="#FFD600",
                outline=""
            )

            self.canvas.create_text(
                x1 + 8,
                y1 - 16,
                text=f"COUNTING REGION",
                anchor="w",
                fill="black",
                font=("Arial", 12, "bold")
            )

    # ========================================================
    # CANVAS RESIZE
    # ========================================================

    def on_canvas_resize(self, event):

        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.after(80, self.refresh_preview)

    def refresh_preview(self):
        self._preview_resize_job = None
        if self.preview_source_frame is not None:
            self.display_frame(self.preview_source_frame)
        else:
            self.show_empty_preview()

    # ========================================================
    # REGION MOUSE
    # ========================================================

    def canvas_to_original(
        self,
        x,
        y
    ):

        if self.display_scale <= 0:
            return None

        ox = (
            x -
            self.display_offset_x
        ) / self.display_scale

        oy = (
            y -
            self.display_offset_y
        ) / self.display_scale

        if self.current_frame is None:
            return None

        h, w = self.current_frame.shape[:2]

        ox = max(
            0,
            min(
                w - 1,
                int(ox)
            )
        )

        oy = max(
            0,
            min(
                h - 1,
                int(oy)
            )
        )

        return ox, oy

    def region_mouse_down(self, event):

        if not self.region_enabled:
            return

        point = self.canvas_to_original(
            event.x,
            event.y
        )

        if point is None:
            return

        self.region_start = point

        self.region_temp = None

    def region_mouse_drag(self, event):

        if not self.region_enabled:
            return

        if self.region_start is None:
            return

        point = self.canvas_to_original(
            event.x,
            event.y
        )

        if point is None:
            return

        x1, y1 = self.region_start
        x2, y2 = point

        self.region_temp = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        )

        self.redraw_region_preview()

    def region_mouse_up(self, event):

        if not self.region_enabled:
            return

        if self.region_start is None:
            return

        point = self.canvas_to_original(
            event.x,
            event.y
        )

        if point is None:
            return

        x1, y1 = self.region_start
        x2, y2 = point

        region = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        )

        # Ignore tiny region
        if (
            abs(region[2] - region[0]) < 10
            or
            abs(region[3] - region[1]) < 10
        ):
            self.region_start = None
            return

        self.region = region

        self.region_start = None
        self.region_temp = None

        self.region_button.configure(
            text="✓  Region Aktif"
        )

        # Re-process current frame
        if self.current_frame is not None:

            self.process_frame(
                self.current_frame
            )

    def redraw_region_preview(self):

        if self.region_temp is None:
            return

        # Retain the detection overlay while the temporary region is drawn.
        preview = self.preview_source_frame
        if preview is None:
            preview = self.current_frame
        self.display_frame(preview)

        rx1, ry1, rx2, ry2 = \
            self.region_temp

        scale = self.display_scale

        x1 = (
            self.display_offset_x +
            rx1 * scale
        )

        y1 = (
            self.display_offset_y +
            ry1 * scale
        )

        x2 = (
            self.display_offset_x +
            rx2 * scale
        )

        y2 = (
            self.display_offset_y +
            ry2 * scale
        )

        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline="#FFD600",
            width=4,
            dash=(8, 5)
        )

        self.canvas.create_text(
            x1 + 10,
            y1 + 10,
            text="COUNTING REGION",
            anchor="nw",
            fill="#FFD600",
            font=("Arial", 13, "bold")
        )

    # ========================================================
    # REGION BUTTON
    # ========================================================

    def toggle_region(self):

        if self.current_frame is None:
            messagebox.showinfo(
                "Region",
                "Buka gambar atau video terlebih dahulu."
            )
            return

        self.region_enabled = \
            not self.region_enabled

        if self.region_enabled:

            self.region_button.configure(
                text="✓  Region Aktif",
                fg_color="#168c4b",
                hover_color="#1ca65a"
            )

            messagebox.showinfo(
                "Counting Region",
                "Sekarang klik dan DRAG pada gambar/video "
                "untuk membuat kotak Counting Region."
            )

        else:

            self.region_button.configure(
                text="□  Aktifkan Region",
                fg_color="#1f6aa5",
                hover_color="#144870"
            )

    def clear_region(self):

        self.region = None
        self.region_temp = None

        self.region_button.configure(
            text="□  Aktifkan Region",
            fg_color="#1f6aa5",
            hover_color="#144870"
        )

        self.region_enabled = False

        if self.current_frame is not None:

            self.process_frame(
                self.current_frame
            )

    # ========================================================
    # VIDEO
    # ========================================================

    def play_video(self):

        if self.current_type != "video":
            messagebox.showinfo(
                "Video",
                "Buka video terlebih dahulu."
            )
            return

        if self.video_running:
            return

        self.video_capture = cv2.VideoCapture(
            self.current_file
        )

        if not self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None
            messagebox.showerror("Video", "Video tidak dapat diputar.")
            return

        # Start again from the first frame every time Play is pressed.
        self.video_current_frame = 0
        self.video_last_detections = []
        self.video_running = True
        self.play_button.configure(state="disabled")
        self.play_next_frame()

    def stop_video(self):

        self.video_running = False

        if self.video_after_job is not None:
            self.after_cancel(self.video_after_job)
            self.video_after_job = None

        if self.video_capture is not None:

            try:
                self.video_capture.release()
            except:
                pass

            self.video_capture = None

        if hasattr(self, "play_button"):
            self.play_button.configure(state="normal")

    def play_next_frame(self):
        """Read, infer, and schedule one frame on Tk's main thread.

        Tkinter widgets are not thread-safe.  The previous worker thread
        queued UI work faster than inference could finish, which could leave
        Play apparently stuck.  `after` keeps capture, inference, and UI
        updates in one safe event loop.
        """
        # A scheduled callback is now executing, so it is no longer pending.
        self.video_after_job = None

        if not self.video_running or self.video_capture is None:
            return

        start = time.perf_counter()
        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_video()
            return

        self.video_current_frame += 1
        enhanced = enhance_image(frame, self.enhancement_menu.get())
        self.current_frame = enhanced
        # Drawing is cheap; YOLO inference is not.  Detect periodically and
        # reuse the latest boxes in between so video starts immediately and
        # remains responsive on CPU-only machines.
        needs_detection = (
            self.video_current_frame % VIDEO_DETECTION_STRIDE == 0
        )
        if needs_detection:
            _, detections = self.detect(enhanced.copy(), imgsz=512)
            self.video_last_detections = detections

        self.process_video_frame(
            enhanced,
            detections=self.video_last_detections,
            update_metrics=needs_detection,
            save_result=needs_detection
        )

        elapsed = time.perf_counter() - start
        delay_ms = max(1, int(((1 / self.video_fps) - elapsed) * 1000))
        self.video_after_job = self.after(delay_ms, self.play_next_frame)

    def process_video_frame(
        self,
        frame,
        detections=None,
        update_metrics=True,
        save_result=True
    ):

        display_frame = frame.copy()

        if detections is None:
            display_frame, detections = self.detect(display_frame)

        display_frame = \
            self.draw_detections(
                display_frame,
                detections
            )

        region_total, region_counts = \
            self.calculate_region(
                detections,
                display_frame
            )

        # Draw region
        if self.region is not None:

            rx1, ry1, rx2, ry2 = self.region

            cv2.rectangle(
                display_frame,
                (rx1, ry1),
                (rx2, ry2),
                (0, 255, 255),
                3
            )

            cv2.rectangle(
                display_frame,
                (rx1, max(0, ry1 - 40)),
                (
                    min(
                        rx1 + 300,
                        display_frame.shape[1]
                    ),
                    ry1
                ),
                (0, 255, 255),
                -1
            )

            cv2.putText(
                display_frame,
                f"COUNTING REGION: {region_total}",
                (rx1 + 8, max(20, ry1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        self.last_results = detections

        if update_metrics:
            self.update_statistics(
                detections,
                region_total,
                region_counts
            )

        self.display_frame(
            display_frame
        )

        self.frame_info.configure(
            text=(
                f"Frame: "
                f"{self.video_current_frame} / "
                f"{self.video_total_frames}"
                f"  •  FPS: {self.video_fps:.1f}"
            )
        )

        # Save data for Excel
        if not save_result:
            return

        self.export_rows.append({
            "Frame": self.video_current_frame,
            "Enhancement": self.enhancement_menu.get(),
            "Total Objects": len(detections),
            "Average Confidence": (
                sum(
                    d["confidence"]
                    for d in detections
                ) / len(detections)
                if detections else 0
            ),
            "Cangkul": sum(
                d["class_id"] == 0
                for d in detections
            ),
            "Helm Proyek": sum(
                d["class_id"] == 1
                for d in detections
            ),
            "Scaffolding Bambu": sum(
                d["class_id"] == 2
                for d in detections
            ),
            "Semen": sum(
                d["class_id"] == 3
                for d in detections
            ),
            "Tukang": sum(
                d["class_id"] == 4
                for d in detections
            ),
            "Region Total": region_total,
            "Region Cangkul": region_counts[0],
            "Region Helm Proyek": region_counts[1],
            "Region Scaffolding Bambu": region_counts[2],
            "Region Semen": region_counts[3],
            "Region Tukang": region_counts[4]
        })

    # ========================================================
    # EXCEL
    # ========================================================

    def export_excel(self):

        if not self.export_rows:

            if self.current_type == "image":

                stats = self.last_stats

                row = {
                    "File": os.path.basename(
                        self.current_file
                    ),
                    "Enhancement":
                        self.enhancement_menu.get(),
                    "Total Objects":
                        stats["total"],
                    "Average Confidence":
                        stats["avg_conf"],
                    "Classes Detected":
                        stats["classes"],
                    "Cangkul":
                        stats["counts"][0],
                    "Helm Proyek":
                        stats["counts"][1],
                    "Scaffolding Bambu":
                        stats["counts"][2],
                    "Semen":
                        stats["counts"][3],
                    "Tukang":
                        stats["counts"][4],
                    "Region Total":
                        stats["region_total"],
                    "Region Cangkul":
                        stats["region_counts"][0],
                    "Region Helm Proyek":
                        stats["region_counts"][1],
                    "Region Scaffolding Bambu":
                        stats["region_counts"][2],
                    "Region Semen":
                        stats["region_counts"][3],
                    "Region Tukang":
                        stats["region_counts"][4]
                }

                df = pd.DataFrame(
                    [row]
                )

            else:

                messagebox.showinfo(
                    "Export Excel",
                    "Belum ada data untuk diekspor."
                )

                return

        else:

            df = pd.DataFrame(
                self.export_rows
            )

        path = filedialog.asksaveasfilename(
            title="Simpan hasil Excel",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel Files", "*.xlsx")
            ],
            initialfile="hasil_deteksi_yolo11.xlsx"
        )

        if not path:
            return

        try:

            with pd.ExcelWriter(
                path,
                engine="openpyxl"
            ) as writer:

                df.to_excel(
                    writer,
                    sheet_name="Detection Results",
                    index=False
                )

            messagebox.showinfo(
                "Berhasil",
                f"Data berhasil disimpan:\n\n{path}"
            )

        except Exception as e:

            messagebox.showerror(
                "Export Error",
                str(e)
            )

    # ========================================================
    # EMPTY PREVIEW
    # ========================================================

    def show_empty_preview(self):

        self.canvas.delete("all")

        w = max(
            100,
            self.canvas.winfo_width()
        )

        h = max(
            100,
            self.canvas.winfo_height()
        )

        self.canvas.create_text(
            w / 2,
            h / 2,
            text="Belum ada gambar / video",
            fill="#dddddd",
            font=("Arial", 20)
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app = ConstructionYOLOApp()

    app.mainloop()
