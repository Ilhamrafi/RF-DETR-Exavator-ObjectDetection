# frontend/view/detect_view.py

"""Halaman Streamlit untuk alur deteksi video excavator.

Fokus fungsi halaman:
- Mengunggah video baru.
- Memilih video yang tersedia untuk dianalisis.
- Menjalankan proses deteksi serta menampilkan progres dan ringkasan hasil.
"""

import streamlit as st
import os
import time
import sys
from typing import Dict, Optional, Any

# Menambahkan root project ke sys.path agar impor modul lintas-folder berfungsi.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.file_manager import FileManager
from utils.video_utils import process_uploaded_video, get_video_info
from frontend.components.video_display import preview_video
from backend.detector import ExcavatorDetector

file_manager = FileManager()

# Path model dan file kelas (relatif terhadap root project).
MODEL_PATH = "models/best_model.pth"
CLASSES_JSON = "classes.json"

# Nilai ambang (default) untuk tahap deteksi.
CONFIDENCE_THRESHOLD = 0.85
PASSING_THRESHOLD = 0.80
RITASE_THRESHOLD = 0.90

def get_detector(video_path=None):
    """
    Membuat instance `ExcavatorDetector` baru untuk setiap pemrosesan video.

    Args:
        video_path: Opsional—hanya untuk konteks logging.

    Returns:
        `ExcavatorDetector`: Objek detector siap pakai.
    """
    # Logging ringan untuk memudahkan pelacakan saat memproses banyak video.
    video_name = os.path.basename(video_path) if video_path else "Unknown"
    st.write(f"Inisialisasi detector untuk video: {video_name}")
    print(f"Inisialisasi detector untuk video: {video_name}")
    
    return ExcavatorDetector(
        model_path=MODEL_PATH, 
        classes_json=CLASSES_JSON
    )

def show_detect_page():
    """Merender halaman 'Deteksi Video Excavator' dengan tab Upload & Pilih Video."""
    
    st.header("Deteksi Video Excavator")
    
    # Dua tab: (1) Upload video, (2) Pilih video yang sudah tersedia untuk dianalisis.
    tab1, tab2 = st.tabs(["Upload Video Baru", "Pilih Video Tersedia"])
    
    # Tab 1 — unggah dan simpan video (analisis dilakukan di tab 2).
    with tab1:
        st.markdown("### Upload Video")
        st.write("Upload video untuk analisis. Setelah selesai, beralih ke tab 'Pilih Video Tersedia' untuk memulai deteksi.")
        
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Simpan sementara untuk membaca metadata.
            temp_video_path = process_uploaded_video(uploaded_file)
            
            # Info dasar file (nama & ukuran) ditampilkan selalu.
            file_size_bytes = uploaded_file.size
            file_size_mb = file_size_bytes/1048576
            st.markdown(f"**File terdeteksi:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            # Detail metadata video ditaruh pada expander.
            with st.expander("Lihat Detail Informasi Video", expanded=False):
                try:
                    # Ambil informasi video melalui utilitas yang ada.
                    video_info = get_video_info(temp_video_path)
                    
                    # Hitung durasi (mm:ss) dari total frame dan FPS.
                    duration_seconds = video_info.total_frames / video_info.fps
                    minutes = int(duration_seconds // 60)
                    seconds = int(duration_seconds % 60)
                    
                    # Tampilkan informasi dalam grid 2 kolom.
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Resolusi:** {video_info.width}x{video_info.height}")
                        st.markdown(f"**Total Frame:** {int(video_info.total_frames)}")
                    with col2:
                        st.markdown(f"**Durasi:** {minutes}:{seconds:02d}")
                        st.markdown(f"**FPS:** {video_info.fps:.2f}")
                except Exception as e:
                    st.error(f"Error membaca metadata video: {str(e)}")
            
            # Simpan permanen ke direktori input.
            if st.button("Simpan Video", type="primary"):
                with st.spinner("Menyimpan video..."):
                    saved_path = file_manager.save_uploaded_video(uploaded_file)
                    st.success(f"Video berhasil disimpan: {os.path.basename(saved_path)}")
                    st.info("Silakan pindah ke tab 'Pilih Video Tersedia' untuk memulai analisis")
                    # Tandai agar otomatis terseleksi di tab berikutnya.
                    st.session_state.new_video_path = saved_path
    
    # Tab 2 — pilih video yang tersedia dan jalankan deteksi.
    with tab2:
        st.markdown("### Pilih Video untuk Analisis")
        
        available_videos = file_manager.list_input_videos()
        
        if not available_videos:
            st.info("Tidak ada video tersedia. Silakan upload video terlebih dahulu di tab 'Upload Video Baru'.")
        else:
            video_options = [os.path.basename(v) for v in available_videos]
            
            # Jika ada video baru pada sesi ini, jadikan sebagai default pilihan.
            default_index = 0
            if 'new_video_path' in st.session_state:
                for i, video_path in enumerate(available_videos):
                    if video_path == st.session_state.new_video_path:
                        default_index = i
                        break
                # Hapus penanda agar tidak memengaruhi pilihan berikutnya.
                if 'new_video_path' in st.session_state:
                    del st.session_state.new_video_path
                    
            selected_idx = st.selectbox(
                "Pilih Video", 
                range(len(video_options)), 
                format_func=lambda x: video_options[x],
                index=default_index
            )
            
            if selected_idx is not None:
                selected_video = available_videos[selected_idx]
                
                # Simpan pilihan ke session state agar konsisten lintas halaman.
                st.session_state.selected_video = selected_video
                
                # Pratinjau video.
                preview_video(selected_video, "Video Preview")
                
                # Info ringkas video terpilih.
                st.info(f"Video yang akan diproses: {os.path.basename(selected_video)}")
                
                # Tombol untuk memulai proses deteksi.
                if st.button("Mulai Deteksi", type="primary"):
                    run_detection(
                        video_path=selected_video,
                        confidence_threshold=CONFIDENCE_THRESHOLD,
                        passing_threshold=PASSING_THRESHOLD,
                        ritase_threshold=RITASE_THRESHOLD
                    )
            
def run_detection(
    video_path: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    passing_threshold: float = PASSING_THRESHOLD,
    ritase_threshold: float = RITASE_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    Menjalankan pipeline deteksi pada video dan menampilkan progres serta ringkasan.

    Args:
        video_path: Path video input.
        confidence_threshold: Ambang deteksi objek.
        passing_threshold: Ambang perhitungan passing.
        ritase_threshold: Ambang perhitungan ritase.

    Returns:
        Optional[Dict[str, Any]]: Ringkasan hasil (statistik & info video) atau None jika terjadi kegagalan.
    """
    try:
        # Validasi ketersediaan file.
        if not os.path.exists(video_path):
            st.error(f"File tidak ditemukan: {video_path}")
            return None
            
        # Logging singkat untuk debugging di UI/terminal.
        video_name = os.path.basename(video_path)
        st.write(f"Memproses video: {video_name}")
        print(f"Memproses video: {video_name}")
        
        # Siapkan path output (video/CSV/Excel).
        output_paths = file_manager.get_output_paths(video_path)
        
        # Elemen progres pada UI: progress bar + status teks.
        st.markdown("### Proses Deteksi Sedang Berjalan")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Gunakan detector baru untuk tiap video (hindari cache state lama).
        detector = get_detector(video_path)
        
        # Informasi dasar video untuk status.
        video_info = get_video_info(video_path)
        
        # Callback untuk memperbarui progres pada UI.
        def update_progress(progress: int, current_frame: int, stats: Dict[str, int]):
            progress_bar.progress(progress / 100)  # Progress dalam rentang 0–1
            status_text.text(f"Memproses frame {current_frame}/{video_info.total_frames} ({progress}%)")
        
        # Konteks proses yang sedang berjalan.
        st.markdown(f"Memproses video: **{video_name}**")
        
        # Eksekusi deteksi dengan indikator spinner.
        with st.spinner(f"Menjalankan deteksi untuk {video_name}..."):
            result = detector.run_detection(
                video_path=video_path,
                output_paths=output_paths,
                confidence_threshold=confidence_threshold,
                passing_threshold=passing_threshold,
                ritase_threshold=ritase_threshold,
                progress_callback=update_progress
            )
        
        # Bersihkan elemen progres.
        status_text.empty()
        progress_bar.empty()
        
        # Ringkasan hasil akhir.
        st.success(f"Deteksi selesai untuk video: {video_name}!")
        st.subheader("Ringkasan Hasil Deteksi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Passing", result["passing_stats"]["total_passing"])
            st.metric("Total Frame", result["stats"]["total_frames"])
        with col2:
            st.metric("Total Ritase", result["ritase_stats"]["total_ritase"])
            # Konversi durasi (detik → mm:ss).
            durasi_detik = result["video_info"]["duration_seconds"]
            menit = int(durasi_detik // 60)
            detik = int(durasi_detik % 60)
            st.metric("Durasi Video", f"{menit:02d}:{detik:02d}")
        
        # Informasi nama file keluaran (video dan Excel).
        st.markdown(f"Video output: `{os.path.basename(output_paths['video'])}`")
        st.markdown(f"Excel report: `{os.path.basename(output_paths['excel'])}`")
        
        # Navigasi cepat ke halaman hasil.
        if st.button("Lihat Hasil Sekarang"):
            st.session_state.selected_result = output_paths
            st.session_state.page = "Hasil Deteksi"
            st.rerun()
            
        # Simpan jejak hasil terakhir untuk diprioritaskan di halaman hasil.
        st.session_state.latest_result = output_paths
        
        return result
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        print(f"Error dalam run_detection: {str(e)}")
        return None