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

            st.info("✏️ Draw rectangles to define fields. Click 'Add Field' after each rectangle.")

            # Drawable canvas
            col1, col2 = st.columns([3, 1])

            with col1:
                # Convert to PIL Image
                pil_image = Image.fromarray(warped_rgb)

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=pil_image,
                    update_streamlit=True,
                    height=output_height,
                    width=output_width,
                    drawing_mode="rect",
                    key="canvas",
                )

            with col2:
                st.subheader("Field Manager")

                # Check if user drew a rectangle
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if len(objects) > 0:
                        last_rect = objects[-1]
                        if last_rect["type"] == "rect":
                            field_name = st.text_input("Field Name", key=f"field_{len(st.session_state.fields)}")

                            if st.button("➕ Add Field") and field_name:
                                # Convert to percentages
                                x_percent = (last_rect["left"] / output_width) * 100
                                y_percent = (last_rect["top"] / output_height) * 100
                                w_percent = (last_rect["width"] / output_width) * 100
                                h_percent = (last_rect["height"] / output_height) * 100

                                st.session_state.fields[field_name] = {
                                    "x": round(x_percent, 2),
                                    "y": round(y_percent, 2),
                                    "w": round(w_percent, 2),
                                    "h": round(h_percent, 2)
                                }
                                st.success(f"✅ Added '{field_name}'")
                                st.rerun()

                # Display current fields
                if st.session_state.fields:
                    st.markdown("**Current Fields:**")
                    for fname in list(st.session_state.fields.keys()):
                        col_a, col_b = st.columns([3, 1])
                        col_a.write(f"• {fname}")
                        if col_b.button("🗑️", key=f"del_{fname}"):
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
