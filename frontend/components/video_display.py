# frontend/components/video_display.py

import os
import streamlit as st
import cv2
from typing import Optional

def preview_video(video_path: str, title: Optional[str] = None):
    """
    Tampilkan pratinjau video di Streamlit.

    Args:
        video_path: Path ke file video yang akan dipratinjau.
        title: Judul opsional yang ditampilkan di atas video.
    """
    if title:
        st.subheader(title)

    # Log singkat saat menukar/menampilkan video.
    print(f"Menampilkan preview untuk: {os.path.basename(video_path)}")

    # Validasi path sebelum render.
    if not os.path.exists(video_path):
        st.error(f"File video tidak ditemukan: {video_path}")
        return

    st.caption(f"Video: {os.path.basename(video_path)}")

    # Render video menggunakan komponen Streamlit.
    try:
        # with memastikan file ditutup setelah dibaca.
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
    except Exception as e:
        st.error(f"Error menampilkan video: {str(e)}")

def display_video_with_controls(video_path: str):
    """
    Tampilkan video dengan kontrol sederhana (slider untuk memilih frame).

    Args:
        video_path: Path ke file video.
    """
    cap = cv2.VideoCapture(video_path)

    # Metadata dasar untuk slider dan timestamp.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Slider memilih frame tertentu.
    current_frame = st.slider("Frame", 0, frame_count - 1, 0)

    # Pindah ke frame terpilih lalu baca satu frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    success, frame = cap.read()

    if success:
        # OpenCV -> Streamlit: BGR ke RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tampilkan frame terkini.
        st.image(frame_rgb, use_container_width=True)

        # Informasi waktu (mm:ss) untuk konteks navigasi.
        timestamp = current_frame / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        st.text(f"Timestamp: {minutes:02d}:{seconds:02d} (Frame {current_frame}/{frame_count})")
    else:
        st.error("Gagal membaca frame")

    # Pastikan resource kamera dilepas.
    cap.release()