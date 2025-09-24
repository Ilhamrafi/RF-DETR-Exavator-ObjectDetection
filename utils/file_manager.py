# utils/file_manager.py

import os
import shutil
import pandas as pd
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FileManager:
    """Utility pengelolaan file input/output untuk aplikasi (Streamlit).
    
    Fungsionalitas utama:
    - Menyimpan video yang diunggah.
    - Menyusun path output (video/CSV/Excel) berbasis nama file input.
    - Mendaftar file input/output yang tersedia.
    - Menyusun dan mereset laporan master tracking.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Inisialisasi direktori kerja.
        
        Args:
            base_dir: Direktori dasar untuk menyimpan input dan output.
        """
        self.base_dir = base_dir
        self.input_dir = os.path.join(base_dir, "input")
        self.output_dir = os.path.join(base_dir, "output")
        
        # Pastikan direktori tersedia
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_uploaded_video(self, uploaded_file) -> str:
        """
        Simpan file video yang diunggah ke direktori input dengan nama asli.
        Jika nama sudah dipakai, tambahkan akhiran counter (_1, _2, ...).
        
        Args:
            uploaded_file: Objek file dari komponen upload (mis. Streamlit).
            
        Returns:
            Path absolut file yang tersimpan.
        """
        # Gunakan nama file asli
        filename = uploaded_file.name
        file_path = os.path.join(self.input_dir, filename)
        
        # Tangani konflik nama (buat nama unik)
        if os.path.exists(file_path):
            base_name, extension = os.path.splitext(filename)
            counter = 1
            while os.path.exists(file_path):
                filename = f"{base_name}_{counter}{extension}"
                file_path = os.path.join(self.input_dir, filename)
                counter += 1
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    
    def get_output_paths(self, input_filename: str) -> Dict[str, str]:
        """
        Bangun path output untuk video, CSV, dan Excel dengan pola: 
        `[nama_file]_results.*`.
        
        Args:
            input_filename: Nama/path file input (digunakan sebagai basis).
        
        Returns:
            Dictionary berisi path untuk kunci: "video", "csv", dan "excel".
        """
        basename = os.path.splitext(os.path.basename(input_filename))[0]
        output_base = f"{basename}_results"
        
        return {
            "video": os.path.join(self.output_dir, f"{output_base}.mp4"),
            "csv": os.path.join(self.output_dir, f"{output_base}_tracking.csv"),
            "excel": os.path.join(self.output_dir, f"{output_base}_summary.xlsx")
        }
    
    def list_input_videos(self) -> List[str]:
        """
        Ambil daftar video input yang tersedia (format .mp4), diurutkan naik.
        
        Returns:
            List path file video dalam direktori input.
        """
        return sorted(glob.glob(os.path.join(self.input_dir, "*.mp4")))
    
    def list_output_results(self) -> List[Dict[str, str]]:
        """
        Ambil daftar hasil output (video/CSV/Excel) yang tersedia.
        
        Returns:
            List dict berisi path untuk "video", "csv" (bila ada), "excel" (bila ada),
            dan "timestamp" (diambil dari nama file, bila tersedia).
        """
        video_files = glob.glob(os.path.join(self.output_dir, "*.mp4"))
        results = []
        
        for video_path in sorted(video_files, reverse=True):
            if not os.path.exists(video_path):
                continue  # Lewati jika file video tidak ada
                
            basename = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(self.output_dir, f"{basename}_tracking.csv")
            excel_path = os.path.join(self.output_dir, f"{basename}_summary.xlsx")
            
            # Verifikasi ketersediaan file pendamping
            csv_exists = os.path.exists(csv_path)
            excel_exists = os.path.exists(excel_path)
            
            # Tetap tambahkan entri selama file video ada
            results.append({
                "video": video_path,
                "csv": csv_path if csv_exists else None,
                "excel": excel_path if excel_exists else None,
                # Catatan: "timestamp" diasumsikan terletak di bagian akhir nama file, dipisah "_"
                "timestamp": basename.split("_")[-1] if "_" in basename else ""
            })
                
        return results

    def append_to_master_report(self, result_paths: Dict[str, str], tracking_data: pd.DataFrame) -> str:
        """
        Tambahkan data tracking terbaru ke laporan master (Excel).
        Jika file master sudah ada, data baru digabungkan; jika belum, file dibuat.
        
        Args:
            result_paths: Peta path hasil (mengandung setidaknya kunci "video").
            tracking_data: DataFrame berisi data tracking terbaru.
            
        Returns:
            Path file laporan master (string kosong jika gagal).
        """
        master_path = os.path.join(self.output_dir, "tracking_reports.xlsx")
        
        try:
            # Jika file master sudah ada, gabungkan data lama + baru
            if os.path.exists(master_path):
                existing_data = pd.read_excel(master_path)
                combined_data = pd.concat([existing_data, tracking_data], ignore_index=True)
            else:
                # Jika belum ada, gunakan data baru
                combined_data = tracking_data
                
                # Pastikan ada kolom penanda sumber video
                if "Source Video" not in combined_data.columns:
                    video_name = os.path.basename(result_paths["video"])
                    combined_data["Source Video"] = video_name
            
            # Simpan kembali gabungan data
            combined_data.to_excel(master_path, index=False)
            return master_path
        except Exception as e:
            logger.error(f"Gagal menyimpan master report: {str(e)}")
            return ""
            
    def reset_master_report(self) -> bool:
        """
        Hapus file laporan master dan mulai dari kosong.
        
        Returns:
            True jika penghapusan berhasil, False jika terjadi kegagalan.
        """
        # Dibiarkan apa adanya agar tidak mengubah perilaku yang sudah ada.
        master_path = os.path.join(self.output_dir, "tracking_reports.xlsx")
        try:
            if os.path.exists(master_path):
                os.remove(master_path)
            return True
        except Exception as e:
            logger.error(f"Gagal mereset master report: {str(e)}")
            return False

    def get_master_report_path(self) -> Optional[str]:
        """
        Kembalikan path laporan master jika file ada; selain itu None.
        
        Returns:
            Path file laporan master atau None jika tidak ditemukan.
        """
        master_path = os.path.join(self.output_dir, "tracking_reports.xlsx")
        return master_path if os.path.exists(master_path) else None
