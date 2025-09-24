# backend/penghitung_passing.py

"""
Penghitung passing untuk mendeteksi siklus excavator berbasis event `bucket_dumping`.
Logika utama: per siklus, pilih deteksi dengan confidence tertinggi; perubahan kandidat
tidak menambah jumlah passing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, TypedDict

logger = logging.getLogger(__name__)

# ID kelas (samakan dengan skema label Anda)
CLASS_BUCKET_DUMPING = 2


class DumpDeteksi(TypedDict, total=False):
    """Payload deteksi dump yang tersimpan di siklus aktif."""
    dump_id: str
    frame_index: int
    confidence: float


class BestDump(TypedDict, total=False):
    """Deteksi dump terbaik pada siklus berjalan."""
    dump_id: str
    frame_index: int
    confidence: float


class ExcavatorData(TypedDict, total=False):
    """Statistik per-excavator."""
    passing_count: int
    best_dump: Optional[BestDump]


class PenghitungPassing:
    """
    Hitung passing berdasarkan kemunculan `bucket_dumping` (per tracker_id).

    Aturan:
    - Satu siklus memilih deteksi dengan confidence tertinggi sebagai kandidat terbaik.
    - Passing hanya bertambah saat kandidat pertama muncul pada siklus tersebut.
    - Peningkatan confidence kandidat tidak menambah passing.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Inisialisasi penghitung.

        Args:
            min_confidence: Ambang confidence minimum agar deteksi dianggap valid.
        """
        # Data per excavator (key = tracker_id)
        self.excavators: Dict[int, ExcavatorData] = {}

        # Set ID deteksi yang sudah dihitung untuk mencegah duplikasi
        self.counted_dumps: Set[str] = set()

        # Ambang dan agregat
        self.min_confidence: float = min_confidence
        self.total_passing: int = 0

        # Deteksi per siklus aktif: {tracker_id: [DumpDeteksi]}
        self.active_cycle: DefaultDict[int, List[DumpDeteksi]] = defaultdict(list)

        logger.info("Penghitung passing diinisialisasi")

    def proses_deteksi(
        self,
        tracker_id: int,
        class_id: int,
        frame_index: int,
        confidence: float,
    ) -> bool:
        """
        Proses satu deteksi.

        Args:
            tracker_id: ID tracker excavator.
            class_id: ID kelas terdeteksi (2 = `bucket_dumping`).
            frame_index: Indeks frame deteksi.
            confidence: Skor confidence.

        Returns:
            True jika passing baru tercatat; jika tidak, False.
        """
        if tracker_id not in self.excavators:
            self._inisialisasi_excavator(tracker_id)

        excavator = self.excavators[tracker_id]

        # Hanya proses untuk kelas `bucket_dumping`
        if class_id == CLASS_BUCKET_DUMPING:
            dump_id = f"{tracker_id}_{frame_index}"

            # Abaikan jika sudah dihitung atau di bawah ambang
            if dump_id in self.counted_dumps or confidence < self.min_confidence:
                return False

            # Simpan ke siklus aktif
            self.active_cycle[tracker_id].append(
                {"dump_id": dump_id, "frame_index": frame_index, "confidence": confidence}
            )

            is_new_passing = False

            # Jadikan kandidat terbaik jika belum ada atau confidence lebih tinggi
            if excavator["best_dump"] is None or confidence > float(
                excavator["best_dump"]["confidence"]  # type: ignore[index]
            ):
                old_best = excavator["best_dump"]
                excavator["best_dump"] = {
                    "dump_id": dump_id,
                    "frame_index": frame_index,
                    "confidence": confidence,
                }

                # Kandidat pertama pada siklus -> tambah passing
                if old_best is None:
                    excavator["passing_count"] = int(excavator["passing_count"]) + 1  # type: ignore[index]
                    self.total_passing += 1
                    is_new_passing = True
                    logger.debug(
                        "Passing bertambah: excavator=%s frame=%s conf=%.3f",
                        tracker_id, frame_index, confidence,
                    )
                else:
                    logger.debug(
                        "Kandidat terbaik diperbarui: excavator=%s frame=%s conf=%.3f",
                        tracker_id, frame_index, confidence,
                    )

            # Tandai deteksi sudah dihitung
            self.counted_dumps.add(dump_id)
            return is_new_passing

        return False

    def _inisialisasi_excavator(self, tracker_id: int) -> None:
        """Buat entri data baru untuk excavator tertentu."""
        self.excavators[tracker_id] = {"passing_count": 0, "best_dump": None}
        logger.debug("Data excavator dibuat: excavator=%s", tracker_id)

    def selesaikan_siklus(self, tracker_id: int) -> None:
        """
        Akhiri siklus aktif excavator dan reset state agar siap siklus berikutnya.
        Panggil saat terjadi pergantian siklus (mis. event lain muncul).
        """
        if tracker_id in self.active_cycle:
            self.active_cycle[tracker_id] = []  # kosongkan deteksi siklus aktif

            if tracker_id in self.excavators:
                self.excavators[tracker_id]["best_dump"] = None

            logger.debug("Siklus selesai & direset: excavator=%s", tracker_id)

    def dapatkan_statistik(self) -> Dict[str, object]:
        """Kembalikan ringkasan statistik global dan per-excavator."""
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
        Ambil jumlah passing per-excavator atau total.

        Args:
            tracker_id: None untuk total; selain itu untuk excavator tertentu.

        Returns:
            Jumlah passing.
        """
        if tracker_id is None:
            return self.total_passing
        return int(self.excavators.get(tracker_id, {}).get("passing_count", 0))

    def reset_all_counters(self) -> None:
        """
        Reset semua counter saat ritase baru terdeteksi agar memulai siklus baru.
        """
        # Reset setiap excavator
        for tracker_id in self.excavators:
            self.excavators[tracker_id]["passing_count"] = 0
            self.excavators[tracker_id]["best_dump"] = None

        # Reset agregat & state siklus
        self.total_passing = 0
        self.active_cycle = defaultdict(list)
        self.counted_dumps = set()

        logger.info("Counter passing direset (ritase baru terdeteksi)")