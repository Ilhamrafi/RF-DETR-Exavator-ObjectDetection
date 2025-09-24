# utils/video_utils.py

import cv2
import numpy as np
import tempfile
import supervision as sv
from typing import Generator, Tuple, Optional

def get_video_info(video_path: str) -> sv.VideoInfo:
    """
    Ambil metadata video dari path yang diberikan.
    
    Args:
        video_path: Path file video.
        
    Returns:
        Objek sv.VideoInfo (width, height, fps, total_frames).
    """
    return sv.VideoInfo.from_video_path(video_path)

def create_frame_generator(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Buat generator frame per frame dari video.
    
    Args:
        video_path: Path file video.
        
    Yields:
        Frame video dalam bentuk numpy.ndarray (BGR).
    """
    return sv.get_video_frames_generator(video_path)

def create_video_player(video_path: str) -> Tuple[cv2.VideoCapture, dict]:
    """
    Siapkan pembaca video (cv2.VideoCapture) dan informasi ringkasnya.
    
    Args:
        video_path: Path file video.
        
    Returns:
        Tuple berisi:
        - cv2.VideoCapture untuk membaca frame,
        - dict info: {"width", "height", "fps", "frame_count"}.
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    return cap, info

def process_uploaded_video(uploaded_file) -> str:
    """
    Simpan file video yang diunggah ke file sementara (.mp4) untuk diproses.
    
    Args:
        uploaded_file: Objek file hasil unggahan (mis. Streamlit).
        
    Returns:
        Path file sementara yang siap digunakan pipeline analisis.
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    return tfile.name