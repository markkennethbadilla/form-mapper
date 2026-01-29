import streamlit as st
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
from typing import List, Dict

try:
    import pytesseract
except Exception:
    pytesseract = None

# Page config
st.set_page_config(page_title="Form Mapper", page_icon="📄", layout="wide")

# Initialize session state
if 'fields' not in st.session_state:
    st.session_state.fields = {}
if 'warped_image' not in st.session_state:
    st.session_state.warped_image = None
if 'paper_width' not in st.session_state:
    st.session_state.paper_width = 210.0
if 'paper_height' not in st.session_state:
    st.session_state.paper_height = 297.0
if 'unit' not in st.session_state:
    st.session_state.unit = 'mm'
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
if 'auto_prefix' not in st.session_state:
    st.session_state.auto_prefix = 'field'

# Unit conversion functions
def to_mm(value, unit):
    """Convert value from given unit to millimeters"""
    if unit == 'mm':
        return value
    elif unit == 'cm':
        return value * 10
    elif unit == 'in':
        return value * 25.4
    return value

def from_mm(value_mm, unit):
    """Convert value from millimeters to given unit"""
    if unit == 'mm':
        return value_mm
    elif unit == 'cm':
        return value_mm / 10
    elif unit == 'in':
        return value_mm / 25.4
    return value_mm

def _dedupe_candidates(candidates: List[Dict[str, int]], iou_threshold: float = 0.4) -> List[Dict[str, int]]:
    """Remove near-duplicate rectangles using IoU."""
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / float(area_a + area_b - inter_area)

    kept = []
    for cand in sorted(candidates, key=lambda c: c["w"] * c["h"], reverse=True):
        if all(iou(cand, k) < iou_threshold for k in kept):
            kept.append(cand)
    return kept

def detect_blank_candidates(
    image_rgb: np.ndarray,
    min_area_ratio: float = 0.0003,
    max_area_ratio: float = 0.2,
    min_box_width: int = 30,
    min_box_height: int = 12,
    min_line_width: int = 60,
) -> List[Dict[str, int]]:
    """Detect likely blank fields (rectangles or underlines)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )

    candidates: List[Dict[str, int]] = []

    # Detect rectangular boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = gray.shape
    min_area = (img_w * img_h) * min_area_ratio
    max_area = (img_w * img_h) * max_area_ratio

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w < min_box_width or h < min_box_height:
                continue
            aspect = w / float(h)
            if 0.5 <= aspect <= 10:
                candidates.append({"x": x, "y": y, "w": w, "h": h, "type": "box"})

    # Detect underlines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    line_contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in line_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_line_width or h < 2:
            continue
        pad = 6
        y2 = max(y - pad, 0)
        h2 = min(h + pad * 2, img_h - y2)
        candidates.append({"x": x, "y": y2, "w": w, "h": h2, "type": "line"})

    return _dedupe_candidates(candidates)

def draw_candidates(image_rgb: np.ndarray, candidates: List[Dict[str, int]]) -> np.ndarray:
    """Overlay candidate rectangles for preview."""
    preview = image_rgb.copy()
    for idx, c in enumerate(candidates, start=1):
        color = (0, 255, 0) if c["type"] == "box" else (0, 128, 255)
        cv2.rectangle(preview, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), color, 2)
        cv2.putText(preview, str(idx), (c["x"], max(0, c["y"] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return preview

def add_fields_from_candidates(
    candidates: List[Dict[str, int]],
    display_width: int,
    display_height: int,
    prefix: str = "auto"
) -> None:
    """Add detected candidates to session fields using percentage coordinates."""
    start_index = len(st.session_state.fields) + 1
    for i, c in enumerate(candidates, start=start_index):
        name = f"{prefix}_{i}"
        st.session_state.fields[name] = {
            "x": round((c["x"] / display_width) * 100, 2),
            "y": round((c["y"] / display_height) * 100, 2),
            "w": round((c["w"] / display_width) * 100, 2),
            "h": round((c["h"] / display_height) * 100, 2)
        }

st.title("📄 Form Mapper")
st.markdown("Digitize physical pre-printed forms for Next.js printing")

# ==================== PHASE 1: SIDEBAR ====================
st.sidebar.header("Phase 1: Setup")

# Unit selector
unit = st.sidebar.radio(
    "Measurement Unit",
    options=['mm', 'cm', 'in'],
    format_func=lambda x: {'mm': 'Millimeters', 'cm': 'Centimeters', 'in': 'Inches'}[x],
    horizontal=True,
    key='unit_selector'
)

# Update unit if changed
if unit != st.session_state.unit:
    # Convert existing values to new unit
    st.session_state.paper_width = from_mm(st.session_state.paper_width, unit)
    st.session_state.paper_height = from_mm(st.session_state.paper_height, unit)
    st.session_state.unit = unit

# Set min/max/step based on unit
if unit == 'mm':
    min_val, max_val, step = 50.0, 500.0, 1.0
    default_width, default_height = 210.0, 297.0
elif unit == 'cm':
    min_val, max_val, step = 5.0, 50.0, 0.1
    default_width, default_height = 21.0, 29.7
else:  # inches
    min_val, max_val, step = 2.0, 20.0, 0.1
    default_width, default_height = 8.27, 11.69

paper_width = st.sidebar.number_input(
    f"Paper Width ({unit})",
    min_value=min_val,
    max_value=max_val,
    value=st.session_state.paper_width,
    step=step,
    format="%.2f"
)
paper_height = st.sidebar.number_input(
    f"Paper Height ({unit})",
    min_value=min_val,
    max_value=max_val,
    value=st.session_state.paper_height,
    step=step,
    format="%.2f"
)

st.session_state.paper_width = paper_width
st.session_state.paper_height = paper_height

def generate_calibration_pdf(width_mm, height_mm):
    """Generate PDF with ArUco markers at corners"""
    buffer = io.BytesIO()

    # Convert mm to points (1mm = 2.83465 points)
    width_pt = width_mm * mm
    height_pt = height_mm * mm

    c = canvas.Canvas(buffer, pagesize=(width_pt, height_pt))

    # Generate ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 20 * mm  # 20mm markers
    offset = 5 * mm  # 5mm from edge

    # Generate 4 markers (IDs 0-3 for each corner)
    marker_ids = [0, 1, 2, 3]
    positions = [
        (offset, height_pt - offset - marker_size),  # Top-left
        (width_pt - offset - marker_size, height_pt - offset - marker_size),  # Top-right
        (offset, offset),  # Bottom-left
        (width_pt - offset - marker_size, offset)  # Bottom-right
    ]

    for marker_id, (x, y) in zip(marker_ids, positions):
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)

        # Convert to PIL Image and wrap with ImageReader
        pil_img = Image.fromarray(marker_img)
        img_reader = ImageReader(pil_img)

        # Draw on PDF
        c.drawImage(img_reader, x, y, width=marker_size, height=marker_size)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if st.sidebar.button("📋 Generate Calibration Sheet", use_container_width=True):
    # Convert to mm for PDF generation
    width_mm = to_mm(paper_width, st.session_state.unit)
    height_mm = to_mm(paper_height, st.session_state.unit)
    pdf_buffer = generate_calibration_pdf(width_mm, height_mm)
    st.sidebar.download_button(
        label="⬇️ Download Calibration PDF",
        data=pdf_buffer,
        file_name="calibration_sheet.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    st.sidebar.success("✅ PDF generated!")
    st.sidebar.info("📝 Instructions:\n1. Print this PDF onto your pre-printed form\n2. Take a picture of the printed form\n3. Upload the photo below")

# ==================== PHASE 2: MAIN AREA ====================
st.header("Phase 2: Map Your Form")

uploaded_file = st.file_uploader("📸 Upload photo or scan of your form", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(img)

    if ids is not None and len(ids) >= 4:
        st.success(f"✅ Detected {len(ids)} ArUco markers")

        # Extract corner positions (expecting IDs 0, 1, 2, 3)
        marker_centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                # Get center of marker
                corner = corners[i][0]
                center = corner.mean(axis=0)
                marker_centers[marker_id] = center

        if len(marker_centers) == 4:
            # Define source points (marker centers in image)
            # Order: top-left (0), top-right (1), bottom-left (2), bottom-right (3)
            src_pts = np.float32([
                marker_centers[0],  # Top-left
                marker_centers[1],  # Top-right
                marker_centers[2],  # Bottom-left
                marker_centers[3]   # Bottom-right
            ])

            # Define destination points (ideal positions)
            # Calculate output dimensions maintaining aspect ratio
            # Use mm values for consistent aspect ratio calculation
            width_mm = to_mm(paper_width, st.session_state.unit)
            height_mm = to_mm(paper_height, st.session_state.unit)
            aspect_ratio = width_mm / height_mm
            output_height = 1000  # pixels
            output_width = int(output_height * aspect_ratio)

            dst_pts = np.float32([
                [0, 0],                          # Top-left
                [output_width, 0],               # Top-right
                [0, output_height],              # Bottom-left
                [output_width, output_height]    # Bottom-right
            ])

            # Calculate homography
            H, _ = cv2.findHomography(src_pts, dst_pts)

            # Warp image
            warped = cv2.warpPerspective(img, H, (output_width, output_height))
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

            # Ensure uint8 format for proper display
            warped_rgb = warped_rgb.astype(np.uint8)
            st.session_state.warped_image = warped_rgb

            st.info("✏️ Draw rectangles to define fields. Switch to Edit mode to resize or delete shapes.")

            # Resize image to a consistent display width so canvas and image align perfectly
            max_display_width = 900
            display_width = min(output_width, max_display_width)
            scale = display_width / output_width
            display_height = int(output_height * scale)
            display_image = cv2.resize(warped_rgb, (display_width, display_height), interpolation=cv2.INTER_AREA)

            # Optional OCR and auto-detection
            with st.expander("🔍 Auto-detect blanks (optional)", expanded=False):
                auto_detect = st.checkbox("Detect blank fields (lines & boxes)", value=False)
                run_ocr = st.checkbox("Run OCR (optional)", value=False)

                ocr_text_count = 0
                if run_ocr:
                    if pytesseract is None:
                        st.warning("OCR is unavailable. Install Tesseract and pytesseract to enable OCR.")
                    else:
                        ocr_data = pytesseract.image_to_data(display_image, output_type=pytesseract.Output.DICT)
                        ocr_text_count = sum(1 for t in ocr_data.get("text", []) if t.strip())
                        st.caption(f"Detected {ocr_text_count} text items via OCR.")

                detected_candidates = []
                if auto_detect:
                    detected_candidates = detect_blank_candidates(
                        display_image,
                        min_area_ratio=0.0003,
                        max_area_ratio=0.2,
                        min_box_width=30,
                        min_box_height=12,
                        min_line_width=60,
                    )
                    st.caption(f"Detected {len(detected_candidates)} possible blank fields.")
                    if detected_candidates:
                        preview = draw_candidates(display_image, detected_candidates)
                        st.image(preview, caption="Detected blanks preview", use_column_width=True)

                    if detected_candidates:
                        labels = [
                            f"{i + 1}: {c['type']} (x={c['x']}, y={c['y']}, w={c['w']}, h={c['h']})"
                            for i, c in enumerate(detected_candidates)
                        ]
                        selected = st.multiselect(
                            "Select candidates to add",
                            options=list(range(len(detected_candidates))),
                            format_func=lambda i: labels[i],
                        )

                        col_a, col_b = st.columns(2)
                        if col_a.button("➕ Add selected fields") and selected:
                            add_fields_from_candidates(
                                [detected_candidates[i] for i in selected],
                                display_width,
                                display_height,
                                prefix="auto"
                            )
                            st.success("✅ Added selected fields")
                            st.rerun()

                        if col_b.button("➕ Add all detected fields"):
                            add_fields_from_candidates(
                                detected_candidates,
                                display_width,
                                display_height,
                                prefix="auto"
                            )
                            st.success("✅ Added detected fields")
                            st.rerun()

            # Drawable canvas (single column to ensure perfect overlap)
            pil_image = Image.fromarray(display_image, mode='RGB')
            canvas_width, canvas_height = pil_image.size

            edit_mode = st.radio("Mode", options=["Draw", "Edit"], horizontal=True)
            drawing_mode = "rect" if edit_mode == "Draw" else "transform"

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key=f"canvas_{st.session_state.canvas_key}",
            )

            st.subheader("Field Manager")

            rects = []
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                rects = [obj for obj in objects if obj.get("type") == "rect"]

            if rects:
                st.markdown("**Draft rectangles**")
                st.session_state.auto_prefix = st.text_input(
                    "Auto-name prefix",
                    value=st.session_state.auto_prefix,
                    key="auto_prefix_input"
                )

                col_a, col_b = st.columns(2)
                if col_a.button("➕ Add all drawings (auto-name)"):
                    for rect in rects:
                        name = f"{st.session_state.auto_prefix}_{len(st.session_state.fields) + 1}"
                        x_percent = (rect["left"] / canvas_width) * 100
                        y_percent = (rect["top"] / canvas_height) * 100
                        w_percent = (rect["width"] / canvas_width) * 100
                        h_percent = (rect["height"] / canvas_height) * 100
                        st.session_state.fields[name] = {
                            "x": round(x_percent, 2),
                            "y": round(y_percent, 2),
                            "w": round(w_percent, 2),
                            "h": round(h_percent, 2)
                        }
                    st.session_state.canvas_key += 1
                    st.success("✅ Added all drawings")
                    st.rerun()

                if col_b.button("🧹 Clear drawings"):
                    st.session_state.canvas_key += 1
                    st.rerun()

            # Display and edit current fields
            if st.session_state.fields:
                st.markdown("**Current Fields:**")
                for fname in list(st.session_state.fields.keys()):
                    field = st.session_state.fields[fname]
                    with st.expander(f"{fname}", expanded=False):
                        new_name = st.text_input("Field name", value=fname, key=f"rename_{fname}")
                        col1, col2, col3, col4 = st.columns(4)
                        x_val = col1.number_input("x (%)", value=field["x"], min_value=0.0, max_value=100.0, step=0.1, key=f"x_{fname}")
                        y_val = col2.number_input("y (%)", value=field["y"], min_value=0.0, max_value=100.0, step=0.1, key=f"y_{fname}")
                        w_val = col3.number_input("w (%)", value=field["w"], min_value=0.0, max_value=100.0, step=0.1, key=f"w_{fname}")
                        h_val = col4.number_input("h (%)", value=field["h"], min_value=0.0, max_value=100.0, step=0.1, key=f"h_{fname}")

                        btn_col_a, btn_col_b = st.columns(2)
                        if btn_col_a.button("💾 Save", key=f"save_{fname}"):
                            if new_name != fname:
                                del st.session_state.fields[fname]
                            st.session_state.fields[new_name] = {
                                "x": round(x_val, 2),
                                "y": round(y_val, 2),
                                "w": round(w_val, 2),
                                "h": round(h_val, 2)
                            }
                            st.success("✅ Saved")
                            st.rerun()

                        if btn_col_b.button("🗑️ Delete", key=f"del_{fname}"):
                            del st.session_state.fields[fname]
                            st.rerun()
        else:
            st.error("❌ Could not find all 4 corner markers (IDs 0-3)")
    else:
        st.error("❌ No ArUco markers detected. Make sure you printed the calibration sheet correctly.")

# ==================== PHASE 3: OUTPUT ====================
if st.session_state.fields:
    st.header("Phase 3: Export")

    json_output = json.dumps(st.session_state.fields, indent=2)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.code(json_output, language="json")

    with col2:
        st.download_button(
            label="📥 Download JSON",
            data=json_output,
            file_name="form_fields.json",
            mime="application/json",
            use_container_width=True
        )

        if st.button("🗑️ Clear All Fields", use_container_width=True):
            st.session_state.fields = {}
            st.rerun()
