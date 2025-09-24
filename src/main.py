# main.py
"""
Script utama untuk deteksi objek excavator dan perhitungan passing serta ritase.

Fitur:
- Auto-increment nama output video (testing_01, testing_02, dst.)
- Output CSV untuk hasil tracking
- Output Excel ringkasan Passing & Ritase

Catatan:
- Tidak mengubah alur/logic, hanya perapian gaya tulis, docstring, dan penambahan type hints.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import openpyxl
import pandas as pd  # dipertahankan walau tidak dipakai langsung (kompatibilitas)
import supervision as sv
from openpyxl.styles import Alignment, Font, PatternFill
from rfdetr import RFDETRNano
from tqdm import tqdm

from penghitung_passing import PenghitungPassing
from penghitung_ritase import PenghitungRitase
from excel_report import generate_excel_report

# ---------------------------------------------------------------------------
# Konfigurasi & Konstanta
# ---------------------------------------------------------------------------

# Nonaktifkan peringatan yang tidak relevan (tidak memengaruhi hasil)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="TracerWarning")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="torch.as_tensor")
warnings.filterwarnings("ignore", message="torch.tensor")
warnings.filterwarnings("ignore", message="Iterating over a tensor")

# Logging dasar
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Batasi kebisingan log dari library lain
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)

# Path sumber & tujuan
SOURCE_VIDEO_PATH: str = ("D:/project-computer-vision/exavator-load-detection/data/raw/exavator_testing/CH1_250814_04.mp4")
BASE_OUTPUT_DIR: str = ("D:/project-computer-vision/exavator-load-detection/data/output")
MODEL_PATH: str = ("D:/project-computer-vision/exavator-load-detection/model/best_model.pth")
CLASSES_JSON: str = "classes.json"

# ---------------------------------------------------------------------------
# Utilitas Penamaan & Output
# ---------------------------------------------------------------------------


def generate_next_video_name(
    base_dir: str, prefix: str = "testing", extension: str = ".mp4"
) -> str:
    """
    Buat nama file video berikutnya (auto-increment): testing_01, testing_02, dst.

    Args:
        base_dir: Direktori penyimpanan file.
        prefix: Prefix nama file.
        extension: Ekstensi file.

    Returns:
        Path lengkap file video berikutnya.
    """
    os.makedirs(base_dir, exist_ok=True)
    pattern: str = os.path.join(base_dir, f"{prefix}_*{extension}")
    existing_files: List[str] = glob.glob(pattern)

    if not existing_files:
        next_number = 1
    else:
        numbers: List[int] = []
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            try:
                number_part = (
                    filename.replace(f"{prefix}_", "").replace(extension, "")
                )
                numbers.append(int(number_part))
            except ValueError:
                # Abaikan file yang tidak sesuai pola penomoran
                continue
        next_number = max(numbers) + 1 if numbers else 1

    filename: str = f"{prefix}_{next_number:02d}{extension}"
    return os.path.join(base_dir, filename)


def generate_output_paths(base_dir: str, video_prefix: str = "testing") -> Dict[str, str]:
    """
    Siapkan semua path output untuk satu run.

    Args:
        base_dir: Direktori dasar output.
        video_prefix: Prefix nama video.

    Returns:
        Dict berisi path untuk video, CSV, dan Excel.
    """
    video_path: str = generate_next_video_name(base_dir, video_prefix)
    video_basename: str = os.path.splitext(os.path.basename(video_path))[0]

    return {
        "video": video_path,
        "csv": os.path.join(base_dir, f"{video_basename}_tracking.csv"),
        "excel": os.path.join(base_dir, f"{video_basename}_summary.xlsx"),
    }


# ---------------------------------------------------------------------------
# Inisialisasi Komponen
# ---------------------------------------------------------------------------


def muat_class_mapping() -> Dict[str, str]:
    """
    Muat pemetaan class ID → class name dari berkas JSON.

    Returns:
        Dict pemetaan kelas.
    """
    try:
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            class_mapping: Dict[str, str] = json.load(f)
        logger.info("Class mapping berhasil dimuat.")
        return class_mapping
    except Exception as exc:  # pragma: no cover (passthrough)
        logger.error("Gagal memuat class mapping: %s", str(exc))
        raise


def inisialisasi_model() -> RFDETRNano:
    """
    Inisialisasi model RF-DETR Nano untuk deteksi objek.

    Returns:
        Instans model RFDETRNano yang siap inference.
    """
    try:
        logger.info("Memuat model dari: %s", MODEL_PATH)
        model = RFDETRNano(pretrain_weights=MODEL_PATH)
        model.optimize_for_inference()
        logger.info("Model siap untuk inference.")
        return model
    except Exception as exc:  # pragma: no cover (passthrough)
        logger.error("Gagal memuat model: %s", str(exc))
        raise


def inisialisasi_video() -> Tuple[sv.FrameGenerator, sv.VideoInfo]:
    """
    Siapkan generator frame dan info video dari path sumber.

    Returns:
        Tuple (frame_generator, video_info).
    """
    try:
        frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
        video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
        logger.info(
            "Info video: %sx%s, %s fps, %s frame.",
            video_info.width,
            video_info.height,
            video_info.fps,
            video_info.total_frames,
        )
        return frame_generator, video_info
    except Exception as exc:  # pragma: no cover (passthrough)
        logger.error("Gagal inisialisasi video: %s", str(exc))
        raise


def inisialisasi_annotator() -> Dict[str, object]:
    """
    Siapkan semua annotator (tracker, box, label) untuk visualisasi.

    Returns:
        Dict komponen annotator & tracker.
    """
    # Tracker objek
    bucket_tracker = sv.ByteTrack()
    truck_tracker = sv.ByteTrack()
    logger.info("Tracker dibuat untuk bucket dan truck.")

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
        text_thickness=3,
        text_scale=1.5,
        text_position=sv.Position.TOP_LEFT,
        smart_position=False,
        text_color=sv.Color.WHITE,
        text_padding=10,
    )

    logger.info("Annotator berhasil diinisialisasi.")

    return {
        "bucket_tracker": bucket_tracker,
        "truck_tracker": truck_tracker,
        "truck_box": box_annotator_truck,
        "bucket_box": box_annotator_bucket,
        "truck_label": label_annotator_truck,
        "bucket_label": label_annotator_bucket,
        "stats_label": label_annotator_stats,
    }


# ---------------------------------------------------------------------------
# Pemrosesan Frame
# ---------------------------------------------------------------------------


def proses_frame(
    frame: np.ndarray,
    model: RFDETRNano,
    trackers: Dict[str, object],
    annotators: Dict[str, object],
    class_mapping: Dict[str, str],
    penghitung_passing: PenghitungPassing,
    penghitung_ritase: PenghitungRitase,
    frame_index: int,
    video_info: sv.VideoInfo,
    stats: Dict[str, int],
    passing_events: List[Dict[str, float | int]],
    ritase_events: List[Dict[str, float | int]],
) -> Tuple[np.ndarray, sv.Detections, sv.Detections]:
    """
    Proses satu frame: deteksi → tracking → perbarui statistik passing/ritase → anotasi.

    Returns:
        (frame_beranotasi, tracked_bucket, tracked_truck)
    """
    try:
        # Prediksi
        detections: sv.Detections = model.predict(frame, threshold=0.85)

        # Pisahkan deteksi truck (kelas 5/6) dan bucket (lainnya)
        truck_mask = (detections.class_id == 5) | (detections.class_id == 6)
        truck_detections = detections[truck_mask]
        bucket_detections = detections[~truck_mask]

        # Statistik dasar
        stats["truck_detections"] += len(truck_detections)
        for cid in bucket_detections.class_id:
            cname = class_mapping.get(str(cid))
            if cname in stats:
                stats[cname] += 1

        # Tracking
        tracked_bucket: sv.Detections = trackers["bucket_tracker"].update_with_detections(
            bucket_detections
        )
        tracked_truck: sv.Detections = trackers["truck_tracker"].update_with_detections(
            truck_detections
        )

        # Passing: reset siklus saat bucket_digging (class_id == 1), dan hitung passing saat dumping (class_id == 2)
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

        # Ritase: dihitung dari truck_full (class_id == 6) per siklus, diakhiri saat bucket_dumping (class_id == 2)
        for i, tracker_id in enumerate(tracked_truck.tracker_id):
            class_id = int(tracked_truck.class_id[i])
            confidence = float(tracked_truck.confidence[i])

            ritase_baru = penghitung_ritase.proses_deteksi(
                tracker_id, class_id, frame_index, confidence
            )
            if ritase_baru:
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

        # Kirim bucket_dumping (class_id == 2) ke penghitung ritase sebagai pemicu akhir siklus
        for i, tracker_id in enumerate(tracked_bucket.tracker_id):
            class_id = int(tracked_bucket.class_id[i])
            confidence = float(tracked_bucket.confidence[i])
            penghitung_ritase.proses_deteksi(tracker_id, class_id, frame_index, confidence)

        # Siapkan label untuk anotasi boks
        truck_labels: List[str] = [
            f"{class_mapping.get(str(c))} ({conf:.2f})"
            for c, conf in zip(tracked_truck.class_id, tracked_truck.confidence)
        ]
        bucket_labels: List[str] = [
            f"{class_mapping.get(str(c))} ({conf:.2f})"
            for c, conf in zip(tracked_bucket.class_id, tracked_bucket.confidence)
        ]

        annotated: np.ndarray = frame.copy()

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
        stats_text = [
            f"Total Passing: {passing_stats['total_passing']}",
            f"Total Ritase: {ritase_stats['total_ritase']}",
        ]

        if len(tracked_bucket) > 0 or len(tracked_truck) > 0:
            # Gunakan detection dummy agar LabelAnnotator dapat menggambar teks
            stats_detection = sv.Detections(
                xyxy=np.array([[50, 50, 500, 150]]),
                confidence=np.array([1.0]),
                class_id=np.array([0]),
            )
            annotated = annotators["stats_label"].annotate(
                annotated, detections=stats_detection, labels=[" | ".join(stats_text)]
            )

        return annotated, tracked_bucket, tracked_truck

    except Exception as exc:  # pragma: no cover (passthrough)
        logger.error("Error memproses frame %s: %s", frame_index, str(exc))
        return frame, sv.Detections.empty(), sv.Detections.empty()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Jalankan pipeline deteksi + perhitungan passing/ritase untuk satu video.
    """
    # Siapkan path output auto-increment
    output_paths = generate_output_paths(BASE_OUTPUT_DIR, "testing")

    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE DETEKSI EXCAVATOR (PASSING & RITASE)")
    logger.info("Video sumber   : %s", SOURCE_VIDEO_PATH)
    logger.info("Video output   : %s", output_paths["video"])
    logger.info("Output CSV     : %s", output_paths["csv"])
    logger.info("Output Excel   : %s", output_paths["excel"])
    logger.info("=" * 60)

    # Inisialisasi komponen
    class_mapping = muat_class_mapping()
    model = inisialisasi_model()
    frame_generator, video_info = inisialisasi_video()
    trackers_annotators = inisialisasi_annotator()
    penghitung_passing = PenghitungPassing(min_confidence=0.8)
    penghitung_ritase = PenghitungRitase(min_confidence=0.9)

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

    # Event untuk laporan
    passing_events: List[Dict[str, float | int]] = []
    ritase_events: List[Dict[str, float | int]] = []

    # Proses video
    try:
        with sv.CSVSink(output_paths["csv"]) as csv_sink, sv.VideoSink(
            output_paths["video"], video_info
        ) as sink:
            logger.info("Memulai pemrosesan frame...")

            for frame_index, frame in enumerate(
                tqdm(frame_generator, desc="Memproses video")
            ):
                stats["total_frames"] += 1

                # Log berkala
                if frame_index % 200 == 0:
                    cur_pass = penghitung_passing.dapatkan_statistik()
                    cur_rits = penghitung_ritase.dapatkan_statistik()
                    logger.info(
                        "Frame %s - Passing: %s, Ritase: %s",
                        frame_index,
                        cur_pass["total_passing"],
                        cur_rits["total_ritase"],
                    )

                # Proses frame → anotasi & tracking
                annotated_frame, tracked_bucket, tracked_truck = proses_frame(
                    frame,
                    model,
                    trackers_annotators,
                    trackers_annotators,
                    class_mapping,
                    penghitung_passing,
                    penghitung_ritase,
                    frame_index,
                    video_info,
                    stats,
                    passing_events,
                    ritase_events,
                )

                # Gabungkan deteksi & simpan
                all_detections = sv.Detections.merge(
                    [tracked_bucket, tracked_truck]
                )
                csv_sink.append(all_detections, {})
                sink.write_frame(annotated_frame)

            # Rekap hasil akhir
            final_passing = penghitung_passing.dapatkan_statistik()
            final_ritase = penghitung_ritase.dapatkan_statistik()

            logger.info("=" * 60)
            logger.info("HASIL PEMROSESAN VIDEO")
            logger.info("Total frame: %s", stats["total_frames"])
            logger.info("Total passing terdeteksi: %s", final_passing["total_passing"])
            logger.info("Total ritase terdeteksi: %s", final_ritase["total_ritase"])
            logger.info("=" * 60)

            # Buat laporan Excel
            generate_excel_report(
                output_paths["excel"],
                passing_events,
                ritase_events,
                video_info,
                SOURCE_VIDEO_PATH,
            )

    except Exception as exc:  # pragma: no cover (passthrough)
        logger.error("Terjadi error selama pemrosesan: %s", str(exc))
        raise

    # Ringkasan di console (dipertahankan)
    print(f"\n{'=' * 60}")
    print("PEMROSESAN VIDEO SELESAI!")
    print(f"Video output   : {output_paths['video']}")
    print(f"CSV tracking   : {output_paths['csv']}")
    print(f"Laporan Excel  : {output_paths['excel']}")
    print(f"Total passing terdeteksi: {penghitung_passing.dapatkan_passing_count()}")
    print(f"Total ritase terdeteksi: {penghitung_ritase.dapatkan_ritase_count()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()