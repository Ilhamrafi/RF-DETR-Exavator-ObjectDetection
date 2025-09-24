# backend/detector.py

"""
Adapter backend untuk menghubungkan modul deteksi dengan antarmuka Streamlit.
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import supervision as sv
from tqdm import tqdm

# Modul backend
from backend.penghitung_passing import PenghitungPassing
from backend.penghitung_ritase import PenghitungRitase
from backend.excel_report import generate_excel_report

# Nonaktifkan peringatan yang tidak relevan
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="TracerWarning")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="torch.as_tensor")
warnings.filterwarnings("ignore", message="torch.tensor")
warnings.filterwarnings("ignore", message="Iterating over a tensor")

# Konfigurasi logging dasar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Reduksi kebisingan log dari library lain
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)


class ExcavatorDetector:
    """
    Detektor objek (bucket & truck) dengan perhitungan passing dan ritase.
    """

    def __init__(self, model_path: str, classes_json: str):
        """
        Inisialisasi detektor dan menyiapkan state dasar.

        Args:
            model_path: Path ke file bobot model (.pth).
            classes_json: Path ke file pemetaan label (classes.json).
        """
        self.model_path = model_path
        self.classes_json = classes_json
        self.model = None
        self.class_mapping = None

        # Status proses
        self.is_initialized = False
        self.progress_callback = None

        # Penomoran siklus dimulai dari 1
        self.siklus_dimulai_dari = 1

        # Status tampilan ritase pada overlay
        self.display_ritase = 0
        self.siklus_terakhir = 0

        logger.info("ExcavatorDetector dibuat, menunggu inisialisasi...")

    def load_resources(self) -> bool:
        """
        Memuat model dan resource terkait (lazy loading).

        Returns:
            True jika berhasil, False bila terjadi kegagalan.
        """
        try:
            # Import di sini untuk mempercepat start-up (lazy)
            from rfdetr import RFDETRNano
            import json

            # Muat pemetaan kelas
            with open(self.classes_json, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)
            logger.info("Class mapping berhasil dimuat.")

            # Muat dan optimasi model
            logger.info("Memuat model dari: %s", self.model_path)
            self.model = RFDETRNano(pretrain_weights=self.model_path)
            self.model.optimize_for_inference()
            logger.info("Model siap untuk inference.")

            self.is_initialized = True
            return True

        except Exception as exc:
            logger.error("Gagal memuat resources: %s", str(exc))
            return False

    def run_detection(
        self,
        video_path: str,
        output_paths: Dict[str, str],
        confidence_threshold: float = 0.85,
        passing_threshold: float = 0.8,
        ritase_threshold: float = 0.9,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Menjalankan deteksi objek pada video dan menghitung passing/ritase.

        Args:
            video_path: Path video input.
            output_paths: Dict path output untuk video, CSV, dan Excel.
            confidence_threshold: Ambang minimal confidence untuk deteksi.
            passing_threshold: Ambang minimal confidence pada perhitungan passing.
            ritase_threshold: Ambang minimal confidence pada perhitungan ritase.
            progress_callback: Callback untuk update progres (opsional).

        Returns:
            Dict berisi ringkasan statistik, event, dan metadata video.
        """
        if not self.is_initialized and not self.load_resources():
            raise RuntimeError("Gagal memuat resources, cek log untuk detail.")

        self.progress_callback = progress_callback

        try:
            # Inisialisasi pembacaan video
            frame_generator = sv.get_video_frames_generator(video_path)
            video_info = sv.VideoInfo.from_video_path(video_path)
            logger.info(
                "Info video: %sx%s, %s fps, %s frame.",
                video_info.width,
                video_info.height,
                video_info.fps,
                video_info.total_frames,
            )

            # Siapkan annotator & tracker
            annotators = self._initialize_annotators()

            # Siapkan penghitung
            penghitung_passing = PenghitungPassing(min_confidence=passing_threshold)
            penghitung_ritase = PenghitungRitase(min_confidence=ritase_threshold)

            # Statistik runtime
            stats: Dict[str, int] = {
                "total_frames": 0,
                "truck_detections": 0,
                "bucket_empty": 0,
                "bucket_full": 0,
                "bucket_digging": 0,
                "bucket_dumping": 0,
                "passing_detections": 0,
                "ritase_detections": 0,
            }

            # Kumpulan event untuk laporan
            passing_events: List[Dict[str, Union[float, int]]] = []
            ritase_events: List[Dict[str, Union[float, int]]] = []

            # Proses video & tulis output
            with sv.CSVSink(output_paths["csv"]) as csv_sink, sv.VideoSink(
                output_paths["video"], video_info
            ) as sink:
                logger.info("Memulai pemrosesan frame...")

                # Gunakan tqdm untuk progress bar terminal
                for frame_index, frame in enumerate(
                    tqdm(frame_generator, desc="Memproses video", total=video_info.total_frames)
                ):
                    stats["total_frames"] += 1

                    # Update progres ke UI (periodik)
                    progress_percent = int((frame_index / video_info.total_frames) * 100)
                    if self.progress_callback and frame_index % 5 == 0:
                        self.progress_callback(progress_percent, frame_index, stats)

                    # Log berkala untuk memantau agregat hitungan
                    if frame_index % 200 == 0:
                        cur_pass = penghitung_passing.dapatkan_statistik()
                        cur_rits = penghitung_ritase.dapatkan_statistik()
                        logger.info(
                            "Frame %s - Passing: %s, Ritase: %s",
                            frame_index,
                            cur_pass["total_passing"],
                            cur_rits["total_ritase"],
                        )

                    # Proses frame → tracking, pembaruan statistik, dan anotasi
                    annotated_frame, tracked_bucket, tracked_truck = self._process_frame(
                        frame,
                        annotators,
                        penghitung_passing,
                        penghitung_ritase,
                        frame_index,
                        video_info,
                        stats,
                        passing_events,
                        ritase_events,
                        confidence_threshold,
                    )

                    # Gabungkan deteksi & simpan ke CSV + video
                    all_detections = sv.Detections.merge([tracked_bucket, tracked_truck])
                    csv_sink.append(all_detections, {})
                    sink.write_frame(annotated_frame)

                # Final update progres
                if self.progress_callback:
                    self.progress_callback(100, video_info.total_frames, stats)

            # Rekap akhir
            final_passing = penghitung_passing.dapatkan_statistik()
            final_ritase = penghitung_ritase.dapatkan_statistik()

            logger.info("=" * 60)
            logger.info("HASIL PEMROSESAN VIDEO")
            logger.info("Total frame: %s", stats["total_frames"])
            logger.info("Total passing terdeteksi: %s", final_passing["total_passing"])
            logger.info("Total ritase terdeteksi: %s", final_ritase["total_ritase"])
            logger.info("=" * 60)

            # Laporan Excel
            generate_excel_report(
                output_paths["excel"],
                passing_events,
                ritase_events,
                video_info,
                video_path,
            )

            # Hasil untuk UI
            results: Dict[str, Any] = {
                "stats": stats,
                "passing_stats": final_passing,
                "ritase_stats": final_ritase,
                "passing_events": passing_events,
                "ritase_events": ritase_events,
                "output_paths": output_paths,
                "video_info": {
                    "width": video_info.width,
                    "height": video_info.height,
                    "fps": video_info.fps,
                    "total_frames": video_info.total_frames,
                    "duration_seconds": video_info.total_frames / video_info.fps,
                },
            }

            # Tambahkan data tracking ke master report (opsional)
            try:
                import pandas as pd
                from utils.file_manager import FileManager

                tracking_data = pd.read_excel(output_paths["excel"], sheet_name="Tracking")

                # Tambah kolom source video bila belum ada
                if "Source Video" not in tracking_data.columns:
                    tracking_data["Source Video"] = os.path.basename(video_path)

                # Append ke master report
                file_manager = FileManager()
                master_path = file_manager.append_to_master_report(output_paths, tracking_data)
                results["master_report_path"] = master_path
                logger.info("Data tracking ditambahkan ke master report: %s", master_path)
            except Exception as e:
                logger.error("Gagal menyimpan ke master report: %s", str(e))

            return results

        except Exception as exc:
            logger.error("Terjadi error selama pemrosesan: %s", str(exc))
            raise

    def _initialize_annotators(self) -> Dict[str, object]:
        """
        Menyiapkan tracker dan annotator untuk visualisasi.
        """
        # Tracker objek
        bucket_tracker = sv.ByteTrack()
        truck_tracker = sv.ByteTrack()

        # Box annotator
        box_annotator_truck = sv.BoxAnnotator()
        box_annotator_bucket = sv.BoxCornerAnnotator()

        # Label annotator
        label_annotator_truck = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=1.0,
            text_position=sv.Position.TOP_RIGHT,
            smart_position=True,
            text_color=sv.Color.BLACK,
        )
        label_annotator_bucket = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=1.0,
            text_position=sv.Position.CENTER,
            smart_position=True,
            text_color=sv.Color.BLACK,
        )
        label_annotator_stats = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=1,
            text_position=sv.Position.TOP_LEFT,
            smart_position=False,
            text_color=sv.Color.WHITE,
            text_padding=10,
        )

        return {
            "bucket_tracker": bucket_tracker,
            "truck_tracker": truck_tracker,
            "truck_box": box_annotator_truck,
            "bucket_box": box_annotator_bucket,
            "truck_label": label_annotator_truck,
            "bucket_label": label_annotator_bucket,
            "stats_label": label_annotator_stats,
        }

    def _process_frame(
        self,
        frame: np.ndarray,
        annotators: Dict[str, object],
        penghitung_passing: PenghitungPassing,
        penghitung_ritase: PenghitungRitase,
        frame_index: int,
        video_info: sv.VideoInfo,
        stats: Dict[str, int],
        passing_events: List[Dict[str, Union[float, int]]],
        ritase_events: List[Dict[str, Union[float, int]]],
        confidence_threshold: float = 0.85,
    ) -> Tuple[np.ndarray, sv.Detections, sv.Detections]:
        """
        Memproses satu frame: deteksi → tracking → update passing/ritase → anotasi.

        Returns:
            Tuple (frame_beranotasi, tracked_bucket, tracked_truck).
        """
        try:
            # Prediksi
            detections = self.model.predict(frame, threshold=confidence_threshold)

            # Pisahkan deteksi truck (class_id 5/6) vs bucket (lainnya)
            truck_mask = (detections.class_id == 5) | (detections.class_id == 6)
            truck_detections = detections[truck_mask]
            bucket_detections = detections[~truck_mask]

            # Statistik dasar bucket & truck
            stats["truck_detections"] += len(truck_detections)
            for cid in bucket_detections.class_id:
                cname = self.class_mapping.get(str(cid))
                if cname in stats:
                    stats[cname] += 1

            # Tracking
            tracked_bucket = annotators["bucket_tracker"].update_with_detections(bucket_detections)
            tracked_truck = annotators["truck_tracker"].update_with_detections(truck_detections)

            # Flag ritase baru pada frame ini
            ritase_baru_terdeteksi = False

            # Simpan nomor siklus sebelum update
            nomor_siklus_awal = penghitung_ritase.nomor_siklus

            # Ritase: dihitung dari truck_full (class_id == 6), berakhir saat bucket_dumping (class_id == 2)
            for i, tracker_id in enumerate(tracked_truck.tracker_id):
                class_id = int(tracked_truck.class_id[i])
                confidence = float(tracked_truck.confidence[i])

                ritase_baru = penghitung_ritase.proses_deteksi(
                    tracker_id, class_id, frame_index, confidence
                )
                if ritase_baru:
                    ritase_baru_terdeteksi = True
                    stats["ritase_detections"] += 1
                    timestamp = frame_index / float(video_info.fps)
                    ritase_events.append(
                        {
                            "Nomor": stats["ritase_detections"],
                            "Frame": frame_index,
                            "Detik": round(timestamp, 2),
                            "Confidence": round(confidence, 4),
                            "tracker_id": tracker_id,
                        }
                    )
                    logger.info(
                        "RITASE #%s - frame %s (detik %.2f).",
                        stats["ritase_detections"],
                        frame_index,
                        timestamp,
                    )

                    # Tampilkan indikator ritase untuk siklus aktif
                    self.display_ritase = 1
                    self.siklus_terakhir = nomor_siklus_awal

                    # Reset counter passing saat ada ritase baru
                    logger.info("Counter passing direset karena ritase baru terdeteksi")
                    penghitung_passing.reset_all_counters()

                    # Reset counter ritase jika tersedia
                    if hasattr(penghitung_ritase, "reset_counters"):
                        penghitung_ritase.reset_counters()
                        logger.info("Counter ritase direset untuk siklus baru")
                    else:
                        # Fallback: set total ritase ke 0
                        penghitung_ritase.total_ritase = 0

            # Kirim bucket_dumping (class_id == 2) sebagai pemicu akhir siklus ritase
            for i, tracker_id in enumerate(tracked_bucket.tracker_id):
                class_id = int(tracked_bucket.class_id[i])
                confidence = float(tracked_bucket.confidence[i])
                penghitung_ritase.proses_deteksi(tracker_id, class_id, frame_index, confidence)

            # Passing: akhiri siklus saat bucket_digging (class_id == 1), hitung saat dumping (class_id == 2)
            for i, tracker_id in enumerate(tracked_bucket.tracker_id):
                class_id = int(tracked_bucket.class_id[i])
                confidence = float(tracked_bucket.confidence[i])

                if class_id == 1:  # bucket_digging → akhiri siklus sebelumnya
                    penghitung_passing.selesaikan_siklus(tracker_id)

                passing_baru = penghitung_passing.proses_deteksi(
                    tracker_id, class_id, frame_index, confidence
                )
                if passing_baru:
                    stats["passing_detections"] += 1
                    timestamp = frame_index / float(video_info.fps)
                    passing_events.append(
                        {
                            "Nomor": stats["passing_detections"],
                            "Frame": frame_index,
                            "Detik": round(timestamp, 2),
                            "Confidence": round(confidence, 4),
                            "tracker_id": tracker_id,
                        }
                    )
                    logger.info(
                        "PASSING #%s - frame %s (detik %.2f).",
                        stats["passing_detections"],
                        frame_index,
                        timestamp,
                    )

            # Siapkan label untuk anotasi kotak
            truck_labels = [
                f"{self.class_mapping.get(str(c))} ({conf:.2f})"
                for c, conf in zip(tracked_truck.class_id, tracked_truck.confidence)
            ]
            bucket_labels = [
                f"{self.class_mapping.get(str(c))} ({conf:.2f})"
                for c, conf in zip(tracked_bucket.class_id, tracked_bucket.confidence)
            ]

            annotated = frame.copy()

            # Anotasi truck
            annotated = annotators["truck_box"].annotate(annotated, detections=tracked_truck)
            annotated = annotators["truck_label"].annotate(
                annotated, detections=tracked_truck, labels=truck_labels
            )

            # Anotasi bucket
            annotated = annotators["bucket_box"].annotate(annotated, detections=tracked_bucket)
            annotated = annotators["bucket_label"].annotate(
                annotated, detections=tracked_bucket, labels=bucket_labels
            )

            # Tambahkan ringkas statistik di pojok
            passing_stats = penghitung_passing.dapatkan_statistik()
            ritase_stats = penghitung_ritase.dapatkan_statistik()

            # Nomor siklus yang ditampilkan (offset dimulai dari 1)
            nomor_siklus = penghitung_ritase.nomor_siklus + self.siklus_dimulai_dari

            # Jika siklus meloncat lebih dari satu, matikan indikator ritase
            if nomor_siklus > self.siklus_terakhir + 1:
                self.display_ritase = 0
                self.siklus_terakhir = nomor_siklus - 1

            # Teks statistik (menggunakan dummy detection agar LabelAnnotator dapat menggambar)
            stats_text = [
                f"ID Siklus: {nomor_siklus} | Total Passing: {passing_stats['total_passing']} | Total Ritase: {self.display_ritase}",
            ]

            if len(tracked_bucket) > 0 or len(tracked_truck) > 0:
                stats_detection = sv.Detections(
                    xyxy=np.array([[10, 50, 500, 100]]),  # area peletakan teks statistik
                    confidence=np.array([1.0]),
                    class_id=np.array([0]),
                )
                annotated = annotators["stats_label"].annotate(
                    annotated, detections=stats_detection, labels=stats_text
                )

            return annotated, tracked_bucket, tracked_truck

        except Exception as exc:
            logger.error("Error memproses frame %s: %s", frame_index, str(exc))
            return frame, sv.Detections.empty(), sv.Detections.empty()