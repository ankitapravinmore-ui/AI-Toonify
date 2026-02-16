import streamlit as st
import cv2
import numpy as np
from cartoon import cartoon, sketch, pencil
from PIL import Image
import tempfile
import os

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI Toonify • Creator Studio", page_icon="✨", layout="wide")

# =========================
# Attractive “Creator Studio” UI CSS
# =========================
st.markdown(
    """
<style>
/* ---------- Base ---------- */
.stApp{
  background: linear-gradient(180deg, #0b1020 0%, #0a0f1e 40%, #070a14 100%);
  color:#EAF0FF;
}
.block-container{ padding-top: 1.0rem; }

/* ---------- Top Gradient Bar ---------- */
.topbar{
  border-radius: 22px;
  padding: 18px 20px;
  background:
    radial-gradient(1100px 420px at 10% 20%, rgba(255,0,150,0.25) 0%, transparent 55%),
    radial-gradient(1000px 420px at 90% 10%, rgba(0,229,255,0.22) 0%, transparent 55%),
    linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: 0 20px 60px rgba(0,0,0,0.45);
  backdrop-filter: blur(10px);
}
.brand{
  display:flex; align-items:center; gap:12px;
}
.logo{
  width:44px; height:44px; border-radius:14px;
  background: linear-gradient(135deg, #ff4fd8, #00e5ff);
  box-shadow: 0 12px 30px rgba(0,229,255,0.18);
  display:flex; align-items:center; justify-content:center;
  font-weight:900; color:#071018;
}
.title{
  font-size: 34px; font-weight: 900; margin:0;
  letter-spacing: .2px;
}
.subtitle{ margin:2px 0 0 0; color: rgba(234,240,255,0.75); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.04);
  border-right: 1px solid rgba(255,255,255,0.10);
}
.sidebar-card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 14px;
  margin-bottom: 12px;
}

/* ---------- Main Cards ---------- */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.40);
  backdrop-filter: blur(10px);
}
.hr{ height:1px; background: rgba(255,255,255,0.12); margin: 14px 0; }
.muted{ color: rgba(234,240,255,0.72); font-size: 0.92rem; }

/* ---------- Buttons ---------- */
.stButton > button{
  width: 100%;
  border-radius: 14px;
  border: 0;
  padding: 0.82rem 1rem;
  font-weight: 900;
  color: #071018;
  background: linear-gradient(90deg, #ff4fd8, #00e5ff);
  box-shadow: 0 14px 28px rgba(0,229,255,0.12);
}
.stButton > button:hover{
  transform: translateY(-1px);
  filter: brightness(1.04);
}

/* ---------- Download Buttons ---------- */
.stDownloadButton > button{
  width: 100%;
  border-radius: 14px;
  border: 0;
  padding: 0.82rem 1rem;
  font-weight: 900;
  color: #071018;
  background: linear-gradient(90deg, #ffe259, #ffa751);
  box-shadow: 0 14px 28px rgba(255,226,89,0.14);
}

/* ---------- Inputs ---------- */
input, textarea{ border-radius: 12px !important; }

/* ---------- Tabs look a bit cleaner ---------- */
.stTabs [data-baseweb="tab-list"]{
  gap: 8px;
}
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 10px 14px;
}
.stTabs [aria-selected="true"]{
  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}

/* ---------- Rounded images ---------- */
img{ border-radius: 16px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Helpers
# =========================
def process_image(image_np, style: str):
    if style == "Cartoon":
        out_bgr = cartoon(image_np)
    elif style == "Sketch":
        out_bgr = sketch(image_np)
    else:
        out_bgr = pencil(image_np)
    return out_bgr


def image_to_video(image_rgb, style: str, seconds: int = 5, fps: int = 24, add_zoom: bool = True):
    out_bgr = process_image(image_rgb, style)
    h, w = out_bgr.shape[:2]
    total_frames = int(seconds * fps)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i in range(total_frames):
        frame = out_bgr.copy()
        if add_zoom:
            t = i / max(total_frames - 1, 1)
            scale = 1.0 + 0.08 * t
            nw, nh = int(w * scale), int(h * scale)
            zoom = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x1 = (nw - w) // 2
            y1 = (nh - h) // 2
            frame = zoom[y1:y1 + h, x1:x1 + w]
        vw.write(frame)

    vw.release()
    return out_path


def video_to_toonify(input_path: str, style: str, max_size: int = 720):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    frames_out = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        h, w = frame_rgb.shape[:2]
        scale = min(max_size / max(h, w), 1.0)
        nh, nw = int(h * scale), int(w * scale)
        frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        nh = max((nh // 8) * 8, 8)
        nw = max((nw // 8) * 8, 8)
        frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        out_bgr = process_image(frame_rgb, style)
        frames_out.append(out_bgr)

    cap.release()

    if len(frames_out) == 0:
        raise RuntimeError("No frames extracted from video.")

    out_h, out_w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = input_path.replace(".mp4", f"_{style.lower()}.mp4")

    vw = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    for fr in frames_out:
        vw.write(fr)
    vw.release()
    return output_path


# =========================
# Session State
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "paid" not in st.session_state:
    st.session_state.paid = False


# =========================
# LOGIN (clean centered)
# =========================
if not st.session_state.logged_in:
    st.markdown(
        """
<div class="topbar">
  <div class="brand">
    <div class="logo">AI</div>
    <div>
      <div class="title">AI Toonify</div>
      <div class="subtitle">Creator Studio • Login to continue</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Login")
        st.markdown('<div class="muted">Demo: <b>admin / admin123</b></div>', unsafe_allow_html=True)

        u = st.text_input("Username", placeholder="admin")
        p = st.text_input("Password", type="password", placeholder="admin123")

        if st.button("Login"):
            if u == "admin" and p == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =========================
# TOP BAR
# =========================
left, right = st.columns([5, 1.2])
with left:
    st.markdown(
        """
<div class="topbar">
  <div class="brand">
    <div class="logo">AI</div>
    <div>
      <div class="title">AI Toonify</div>
      <div class="subtitle">Cartoon • Sketch • Pencil • Image→Video • Video</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Session")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# =========================
# SIDEBAR (clean)
# =========================
st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("## ⚙ Controls")
mode = st.sidebar.radio("Mode", ["Image", "Video"], horizontal=True)
style = st.sidebar.selectbox("Style", ["Cartoon", "Sketch", "Pencil"])
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 📂 Upload")
if mode == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 💳 Dummy Payment")
p1, p2 = st.sidebar.columns(2)
with p1:
    if not st.session_state.paid:
        if st.button("Pay ₹10"):
            st.session_state.paid = True
            st.sidebar.success("Paid ✅")
    else:
        st.sidebar.success("Paid ✅")
with p2:
    if st.button("Reset"):
        st.session_state.paid = False
        st.rerun()
st.sidebar.caption("Tip: Use short video (10–20s) for fast demo.")
st.sidebar.markdown("</div>", unsafe_allow_html=True)


# =========================
# MAIN
# =========================
if not uploaded_file:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 👈 Start here")
    st.markdown(
        "<div class='muted'>Upload from the sidebar, choose a style, then export image or MP4.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("<span class='badge'>Pipeline</span> Preprocessing → Frames → Stylization → Post → Reconstruction",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    if mode == "Image":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tabs = st.tabs(["✨ Preview", "🎬 Image→Video", "⬇ Export"])

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        out_bgr = process_image(image_np, style)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        with tabs[0]:
            st.markdown("### Result Preview")
            a, b = st.columns(2)
            with a:
                st.image(image, caption="Original", use_container_width=True)
            with b:
                st.image(out_rgb, caption=f"{style}", use_container_width=True)

        with tabs[1]:
            st.markdown("### Create MP4 from Image")
            st.markdown("<div class='muted'>We create frames from the stylized image and rebuild an MP4.</div>",
                        unsafe_allow_html=True)
            sec = st.slider("Duration (seconds)", 2, 12, 5)
            fps_sel = st.selectbox("FPS", [12, 24, 30], index=1)
            zoom = st.checkbox("Smooth zoom effect", value=True)

            if st.button("Render MP4"):
                if not st.session_state.paid:
                    st.info("Please complete payment to download the video.")
                else:
                    with st.spinner("Rendering MP4..."):
                        vid_path = image_to_video(image_np, style, seconds=sec, fps=fps_sel, add_zoom=zoom)
                    st.success("MP4 created ✅")
                    with open(vid_path, "rb") as f:
                        st.download_button(
                            "⬇ Download MP4",
                            data=f,
                            file_name=f"image_to_video_{style.lower()}.mp4",
                            mime="video/mp4",
                        )

        with tabs[2]:
            st.markdown("### Export Image")
            if st.session_state.paid:
                st.download_button(
                    "⬇ Download PNG",
                    data=cv2.imencode(".png", out_bgr)[1].tobytes(),
                    file_name="toonify.png",
                    mime="image/png",
                )
            else:
                st.info("Please complete payment to download the image.")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tabs = st.tabs(["🎥 Preview", "⚡ Convert", "⬇ Export"])

        with tabs[0]:
            st.markdown("### Video Preview")
            st.video(uploaded_file)
            st.markdown("<span class='badge'>Pipeline</span> Preprocess → Frames → Stylize → Rebuild",
                        unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("### Convert Video")
            st.markdown("<div class='muted'>For faster demo, keep video short (10–20 seconds).</div>",
                        unsafe_allow_html=True)
            max_size = st.selectbox("Quality / Speed", [480, 720], index=1)

            if st.button("Convert Video → Toonify"):
                if not st.session_state.paid:
                    st.info("Please complete payment to convert/download the video.")
                else:
                    with st.spinner("Processing..."):
                        suffix = os.path.splitext(uploaded_file.name)[1].lower()
                        if suffix not in [".mp4", ".mov", ".avi", ".mkv"]:
                            suffix = ".mp4"

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                            tmp_in.write(uploaded_file.read())
                            in_path = tmp_in.name

                        if not in_path.lower().endswith(".mp4"):
                            mp4_path = in_path + ".mp4"
                            os.rename(in_path, mp4_path)
                            in_path = mp4_path

                        out_path = video_to_toonify(in_path, style, max_size=max_size)

                    st.session_state["last_video_out"] = out_path
                    st.success("Done ✅ Go to Export tab")

        with tabs[2]:
            st.markdown("### Export Video")
            out_path = st.session_state.get("last_video_out")
            if out_path and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇ Download MP4",
                        data=f,
                        file_name=f"toonify_{style.lower()}.mp4",
                        mime="video/mp4",
                    )
            else:
                st.info("Convert first to enable export")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Footer
# =========================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    "<div class='muted' style='text-align:center;'>© 2026 AI Toonify • Creator Studio</div>",
    unsafe_allow_html=True,
)
