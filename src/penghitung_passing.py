"""
Modul penghitung passing untuk mendeteksi siklus excavator berbasis deteksi
`bucket_dumping`. Pendekatan per-ID, memilih deteksi dengan confidence tertinggi.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, TypedDict

logger = logging.getLogger(__name__)

# Konstanta ID kelas (sesuaikan dengan skema label)
CLASS_BUCKET_DUMPING = 2


class DumpDeteksi(TypedDict, total=False):
    """Tipe data untuk menyimpan informasi deteksi dump pada siklus aktif."""
    dump_id: str
    frame_index: int
    confidence: float


class BestDump(TypedDict, total=False):
    """Tipe data untuk menyimpan deteksi dump terbaik pada siklus berjalan."""
    dump_id: str
    frame_index: int
    confidence: float


class ExcavatorData(TypedDict, total=False):
    """Tipe data untuk menyimpan statistik per excavator."""
    passing_count: int
    best_dump: Optional[BestDump]


class PenghitungPassing:
    """
    Menghitung jumlah passing berdasarkan kemunculan `bucket_dumping`.

    Mekanisme:
    - Setiap siklus, pilih deteksi `bucket_dumping` dengan confidence tertinggi.
    - Passing bertambah saat deteksi pertama (best_dump sebelumnya None); perubahan
      ke deteksi yang lebih yakin (confidence lebih tinggi) tidak menambah passing.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Inisialisasi penghitung passing.

        Args:
            min_confidence: Ambang minimal confidence agar deteksi dianggap valid.
        """
        # Penyimpanan data excavator per tracker_id
        self.excavators: Dict[int, ExcavatorData] = {}

        # Set deteksi yang sudah dihitung (hindari duplikasi)
        self.counted_dumps: Set[str] = set()

        # Ambang dan total
        self.min_confidence: float = min_confidence
        self.total_passing: int = 0

        # Pelacakan deteksi per siklus aktif: {tracker_id: [deteksi_dump]}
        self.active_cycle: DefaultDict[int, List[DumpDeteksi]] = defaultdict(list)

        logger.info("Penghitung passing diinisialisasi.")

    def proses_deteksi(
        self,
        tracker_id: int,
        class_id: int,
        frame_index: int,
        confidence: float,
    ) -> bool:
        """
        Proses satu deteksi untuk kebutuhan penghitungan passing.

        Args:
            tracker_id: ID tracker excavator.
            class_id: ID kelas terdeteksi (2 = bucket_dumping).
            frame_index: Indeks frame saat deteksi terjadi.
            confidence: Skor keyakinan deteksi.

        Returns:
            True jika passing baru terhitung pada pemanggilan ini; selain itu False.
        """
        # Inisialisasi data excavator bila belum ada
        if tracker_id not in self.excavators:
            self._inisialisasi_excavator(tracker_id)

        excavator = self.excavators[tracker_id]

        # Hanya proses bila class adalah bucket_dumping
        if class_id == CLASS_BUCKET_DUMPING:
            dump_id = f"{tracker_id}_{frame_index}"

            # Validasi duplikasi dan ambang confidence
            if dump_id in self.counted_dumps or confidence < self.min_confidence:
                return False

            # Simpan deteksi dalam daftar siklus aktif
            self.active_cycle[tracker_id].append(
                {"dump_id": dump_id, "frame_index": frame_index, "confidence": confidence}
            )

            is_new_passing = False

            # Jika belum ada best_dump atau confidence sekarang lebih baik -> perbarui
            if excavator["best_dump"] is None or confidence > float(
                excavator["best_dump"]["confidence"]  # type: ignore[index]
            ):
                old_best = excavator["best_dump"]

                excavator["best_dump"] = {
                    "dump_id": dump_id,
                    "frame_index": frame_index,
                    "confidence": confidence,
                }

                # Jika sebelumnya belum ada best_dump, berarti memulai siklus -> tambah passing
                if old_best is None:
                    excavator["passing_count"] = int(excavator["passing_count"]) + 1  # type: ignore[index]
                    self.total_passing += 1
                    is_new_passing = True
                    logger.debug(
                        "Passing baru: excavator=%s, frame=%s, confidence=%.3f",
                        tracker_id,
                        frame_index,
                        confidence,
                    )
                else:
                    # Hanya memperbaiki kandidat terbaik (confidence lebih tinggi)
                    logger.debug(
                        "Perbaikan kandidat dump: excavator=%s, frame=%s, confidence=%.3f",
                        tracker_id,
                        frame_index,
                        confidence,
                    )

            # Tandai deteksi ini sudah dihitung
            self.counted_dumps.add(dump_id)
            return is_new_passing

        return False

    def _inisialisasi_excavator(self, tracker_id: int) -> None:
        """Inisialisasi struktur data untuk excavator baru."""
        self.excavators[tracker_id] = {"passing_count": 0, "best_dump": None}
        logger.debug("Data excavator dibuat: excavator=%s", tracker_id)

    def selesaikan_siklus(self, tracker_id: int) -> None:
        """
        Akhiri siklus aktif excavator tertentu dan reset state untuk memulai siklus baru.
        Panggil saat terdeteksi pergantian siklus (mis. muncul `bucket_digging` baru).

        Args:
            tracker_id: ID excavator terkait.
        """
        if tracker_id in self.active_cycle:
            # Kosongkan daftar deteksi siklus aktif excavator ini
            self.active_cycle[tracker_id] = []

            # Reset kandidat terbaik agar siap untuk siklus selanjutnya
            if tracker_id in self.excavators:
                self.excavators[tracker_id]["best_dump"] = None

            logger.debug(
                "Siklus selesai & direset: excavator=%s. Siap memulai siklus baru.",
                tracker_id,
            )

    def dapatkan_statistik(self) -> Dict[str, object]:
        """
        Ambil ringkasan statistik penghitungan passing.

        Returns:
            Dict berisi total passing dan rincian per-excavator.
        """
        return {
            "total_passing": self.total_passing,
            "detail_excavator": {
                tracker: {
                    "passing_count": data["passing_count"],
                    "best_dump": data["best_dump"],
                }
                for tracker, data in self.excavators.items()
            },
        }

    def dapatkan_passing_count(self, tracker_id: Optional[int] = None) -> int:
        """
        Ambil jumlah passing untuk excavator tertentu atau total keseluruhan.

        Args:
            tracker_id: Jika None mengembalikan total seluruh passing; selain itu per-excavator.

        Returns:
            Jumlah passing.
        """
        if tracker_id is None:
            return self.total_passing
        return int(self.excavators.get(tracker_id, {}).get("passing_count", 0))
