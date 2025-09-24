# backend/penghitung_ritase.py

"""
Penghitung ritase untuk siklus pemuatan dump truck.

Aturan utama:
- Deteksi berbasis tracker_id, memilih `truck_full` dengan confidence tertinggi.
- Ritase dihitung dari kelas 6 (`truck_full`).
- Satu siklus global hanya menghasilkan satu ritase (anti-duplikasi).
- Siklus global berakhir saat kelas 2 (`bucket_dumping`) terdeteksi.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, TypedDict

logger = logging.getLogger(__name__)

# ID kelas (samakan dengan label dataset)
CLASS_TRUCK_FULL = 6
CLASS_BUCKET_DUMPING = 2


class FullDeteksi(TypedDict, total=False):
    """Payload deteksi `truck_full` pada siklus aktif."""
    full_truck_id: str
    frame_index: int
    confidence: float


class BestFull(TypedDict, total=False):
    """Deteksi `truck_full` terbaik pada siklus berjalan."""
    full_truck_id: str
    frame_index: int
    confidence: float
    cycle_number: int


class TruckData(TypedDict, total=False):
    """Statistik per-truck."""
    ritase_count: int
    best_full: Optional[BestFull]
    last_cycle_frame: int
    last_cycle_number: int


class KandidatInfo(TypedDict, total=False):
    """Jejak kandidat untuk monitoring/debugging."""
    tracker_id: int
    frame_index: int
    confidence: float
    status: str


class PenghitungRitase:
    """
    Hitung ritase dari deteksi `truck_full` per siklus global.

    Mekanisme:
    - Satu siklus global hanya boleh menghasilkan satu ritase.
    - Jika ada beberapa kandidat, pilih confidence tertinggi sebagai yang terbaik.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Inisialisasi penghitung.

        Args:
            min_confidence: Ambang confidence minimum agar deteksi valid.
        """
        # Data per truck (key = tracker_id)
        self.data_truk: Dict[int, TruckData] = {}

        # Set ID `truck_full` yang sudah dihitung (anti-duplikasi)
        self.truk_full_terhitung: Set[str] = set()

        # Ambang & agregat
        self.ambang_kepercayaan: float = min_confidence
        self.total_ritase: int = 0

        # Deteksi per siklus aktif: {tracker_id: [FullDeteksi]}
        self.siklus_aktif: DefaultDict[int, List[FullDeteksi]] = defaultdict(list)

        # Statistik
        self.total_siklus_selesai: int = 0
        self.total_deteksi_truk_penuh: int = 0

        # State siklus global
        self.siklus_berjalan_punya_ritase: bool = False
        self.ritase_terbaik_di_siklus: Optional[BestFull] = None
        self.nomor_siklus: int = 0

        # Pencegahan multiple count
        self.total_pencegahan_ganda: int = 0
        self.kandidat_siklus: List[KandidatInfo] = []

        logger.info("Penghitung ritase diinisialisasi")

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
            tracker_id: ID tracker truck.
            class_id: 6 = `truck_full`, 2 = `bucket_dumping`.
            frame_index: Indeks frame deteksi.
            confidence: Skor confidence.

        Returns:
            True jika ritase baru tercatat pada pemanggilan ini; selain itu False.
        """
        if tracker_id not in self.data_truk:
            self._inisialisasi_truck(tracker_id)

        data_truk = self.data_truk[tracker_id]

        # Deteksi `truck_full`
        if class_id == CLASS_TRUCK_FULL:
            self.total_deteksi_truk_penuh += 1

            # Jika siklus sudah punya ritase: hanya perbarui kandidat terbaik (tanpa tambah total)
            if self.siklus_berjalan_punya_ritase:
                kandidat_lebih_baik = (
                    self.ritase_terbaik_di_siklus is None
                    or confidence > float(self.ritase_terbaik_di_siklus["confidence"])  # type: ignore[index]
                )

                if kandidat_lebih_baik:
                    kandidat_lama = self.ritase_terbaik_di_siklus
                    self.ritase_terbaik_di_siklus = {
                        "tracker_id": tracker_id,
                        "frame_index": frame_index,
                        "confidence": confidence,
                        "cycle_number": self.nomor_siklus,
                    }  # type: ignore[assignment]

                    # Koreksi ritase kandidat lama (jika sebelumnya dihitung)
                    if kandidat_lama:
                        id_truk_lama = int(kandidat_lama["tracker_id"])  # type: ignore[index]
                        if id_truk_lama in self.data_truk:
                            self.data_truk[id_truk_lama]["ritase_count"] = max(
                                0, int(self.data_truk[id_truk_lama]["ritase_count"]) - 1
                            )

                    # Tambahkan ritase pada kandidat baru
                    data_truk["ritase_count"] = int(data_truk["ritase_count"]) + 1
                    data_truk["best_full"] = {
                        "full_truck_id": f"{tracker_id}_{frame_index}",
                        "frame_index": frame_index,
                        "confidence": confidence,
                        "cycle_number": self.nomor_siklus,
                    }

                    logger.debug(
                        "Kandidat ritase diperbarui: truck=%s frame=%s conf=%.3f cycle=%s (conf_lama=%s)",
                        tracker_id,
                        frame_index,
                        confidence,
                        self.nomor_siklus,
                        kandidat_lama["confidence"] if kandidat_lama else "None",  # type: ignore[index]
                    )

                # Catat kandidat untuk monitoring
                self.kandidat_siklus.append(
                    {
                        "tracker_id": tracker_id,
                        "frame_index": frame_index,
                        "confidence": confidence,
                        "status": "better_candidate" if kandidat_lebih_baik else "rejected",
                    }
                )

                self.total_pencegahan_ganda += 1
                logger.debug(
                    "Deteksi ganda dicegah: truck=%s frame=%s conf=%.3f total_pencegahan=%s",
                    tracker_id, frame_index, confidence, self.total_pencegahan_ganda,
                )
                return False

            # Jika siklus belum punya ritase -> evaluasi sebagai ritase pertama
            id_truk_full = f"{tracker_id}_{frame_index}"
            if id_truk_full not in self.truk_full_terhitung and confidence >= self.ambang_kepercayaan:
                # Simpan pada siklus aktif
                self.siklus_aktif[tracker_id].append(
                    {"full_truck_id": id_truk_full, "frame_index": frame_index, "confidence": confidence}
                )

                # Tetapkan sebagai ritase pertama di siklus ini
                self.siklus_berjalan_punya_ritase = True
                self.ritase_terbaik_di_siklus = {
                    "tracker_id": tracker_id,
                    "frame_index": frame_index,
                    "confidence": confidence,
                    "cycle_number": self.nomor_siklus,
                }  # type: ignore[assignment]

                # Perbarui data truck
                data_truk["ritase_count"] = int(data_truk["ritase_count"]) + 1
                data_truk["best_full"] = {
                    "full_truck_id": id_truk_full,
                    "frame_index": frame_index,
                    "confidence": confidence,
                    "cycle_number": self.nomor_siklus,
                }

                # Total ritase hanya naik untuk ritase pertama
                self.total_ritase += 1

                # Simpan jejak kandidat
                self.kandidat_siklus.append(
                    {
                        "tracker_id": tracker_id,
                        "frame_index": frame_index,
                        "confidence": confidence,
                        "status": "accepted_as_primary",
                    }
                )

                # Tandai sudah dihitung
                self.truk_full_terhitung.add(id_truk_full)

                logger.debug(
                    "Ritase baru (pertama di siklus): truck=%s frame=%s conf=%.3f cycle=%s",
                    tracker_id, frame_index, confidence, self.nomor_siklus,
                )
                return True

        # Deteksi `bucket_dumping` -> akhiri siklus global
        elif class_id == CLASS_BUCKET_DUMPING:
            self._selesaikan_siklus_global()

        return False

    def _inisialisasi_truck(self, tracker_id: int) -> None:
        """Buat entri data baru untuk truck tertentu."""
        self.data_truk[tracker_id] = {
            "ritase_count": 0,
            "best_full": None,
            "last_cycle_frame": 0,
            "last_cycle_number": -1,
        }
        logger.debug("Data truck dibuat: truck=%s", tracker_id)

    def selesaikan_siklus(self, tracker_id: int) -> None:
        """
        Tutup siklus aktif untuk truck tertentu dan reset state.
        """
        if tracker_id in self.siklus_aktif:
            self.siklus_aktif[tracker_id] = []  # kosongkan deteksi aktif

            if tracker_id in self.data_truk:
                self.data_truk[tracker_id]["best_full"] = None
                self.data_truk[tracker_id]["last_cycle_frame"] = 0
                self.data_truk[tracker_id]["last_cycle_number"] = self.nomor_siklus

            logger.debug("Siklus selesai & direset: truck=%s", tracker_id)

    def _selesaikan_siklus_global(self) -> None:
        """
        Akhiri siklus global saat `bucket_dumping` terdeteksi.
        Menutup semua siklus aktif lalu reset state global.
        """
        truk_dengan_siklus_aktif = [tid for tid, dets in self.siklus_aktif.items() if len(dets) > 0]

        for id_truk in truk_dengan_siklus_aktif:
            self.selesaikan_siklus(id_truk)

        if truk_dengan_siklus_aktif or self.siklus_berjalan_punya_ritase:
            self.total_siklus_selesai += 1

            # Catatan ringkas siklus yang berakhir
            if self.siklus_berjalan_punya_ritase and self.ritase_terbaik_di_siklus:
                terbaik = self.ritase_terbaik_di_siklus
                logger.debug(
                    "Siklus global berakhir (bucket_dumping). affected=%s "
                    "best_ritase: truck=%s conf=%.3f frame=%s kandidat=%s pencegahan=%s",
                    truk_dengan_siklus_aktif,
                    terbaik["tracker_id"],  # type: ignore[index]
                    terbaik["confidence"],  # type: ignore[index]
                    terbaik["frame_index"],  # type: ignore[index]
                    len(self.kandidat_siklus),
                    self.total_pencegahan_ganda,
                )

            # Reset state global
            self.siklus_berjalan_punya_ritase = False
            self.ritase_terbaik_di_siklus = None
            self.kandidat_siklus = []
            self.nomor_siklus += 1

            logger.debug("State siklus global direset -> nomor_siklus=%s", self.nomor_siklus)

    def selesaikan_siklus_manual(self, tracker_id: int, frame_index: int) -> None:
        """
        Tutup siklus secara manual untuk kasus khusus.

        Args:
            tracker_id: ID truck.
            frame_index: Frame saat penutupan.
        """
        if tracker_id in self.data_truk:
            self.data_truk[tracker_id]["last_cycle_frame"] = frame_index
            self.data_truk[tracker_id]["last_cycle_number"] = self.nomor_siklus
            self.selesaikan_siklus(tracker_id)
            logger.info("Siklus manual ditutup: truck=%s frame=%s", tracker_id, frame_index)

    def dapatkan_statistik(self) -> Dict[str, object]:
        """Ringkasan statistik global dan per-truck."""
        return {
            "total_ritase": self.total_ritase,
            "total_truck_full_detections": self.total_deteksi_truk_penuh,
            "cycles_completed": self.total_siklus_selesai,
            "current_cycle_number": self.nomor_siklus,
            "current_cycle_has_ritase": self.siklus_berjalan_punya_ritase,
            "prevented_false_multiple": self.total_pencegahan_ganda,
            "cycle_candidates_count": len(self.kandidat_siklus),
            "active_trucks": len([t for t in self.data_truk.values() if int(t["ritase_count"]) > 0]),
            "detail_truck": {
                tracker_id: {
                    "ritase_count": data["ritase_count"],
                    "best_full": data["best_full"],
                    "last_cycle_frame": data["last_cycle_frame"],
                    "last_cycle_number": data["last_cycle_number"],
                }
                for tracker_id, data in self.data_truk.items()
            },
        }

    def dapatkan_ritase_count(self, tracker_id: Optional[int] = None) -> int:
        """
        Ambil jumlah ritase per-truck atau total.

        Args:
            tracker_id: None untuk total; selain itu untuk truck tertentu.

        Returns:
            Jumlah ritase.
        """
        if tracker_id is None:
            return self.total_ritase
        return int(self.data_truk.get(tracker_id, {}).get("ritase_count", 0))

    def dapatkan_truck_produktif(self) -> List[Dict[str, int | float]]:
        """
        Dapatkan daftar truck yang sudah menghasilkan ritase (diurutkan menurun).
        """
        produktif: List[Dict[str, int | float]] = []
        for tracker_id, data in self.data_truk.items():
            if int(data["ritase_count"]) > 0:
                best_conf = float(data["best_full"]["confidence"]) if data["best_full"] else 0.0  # type: ignore[index]
                last_frame = int(data["best_full"]["frame_index"]) if data["best_full"] else 0  # type: ignore[index]
                produktif.append(
                    {
                        "tracker_id": tracker_id,
                        "ritase_count": int(data["ritase_count"]),
                        "best_confidence": best_conf,
                        "last_frame": last_frame,
                        "last_cycle_number": int(data["last_cycle_number"]),
                    }
                )

        return sorted(produktif, key=lambda x: x["ritase_count"], reverse=True)

    def reset_statistik(self) -> None:
        """Reset seluruh statistik dan state siklus global (untuk run baru/testing)."""
        self.data_truk = {}
        self.truk_full_terhitung = set()
        self.total_ritase = 0
        self.siklus_aktif = defaultdict(list)
        self.total_siklus_selesai = 0
        self.total_deteksi_truk_penuh = 0

        # Reset state global
        self.siklus_berjalan_punya_ritase = False
        self.ritase_terbaik_di_siklus = None
        self.nomor_siklus = 0
        self.total_pencegahan_ganda = 0
        self.kandidat_siklus = []

        logger.info("Seluruh statistik direset (termasuk state siklus global)")

    def export_summary(self) -> Dict[str, object]:
        """
        Ekspor ringkasan hasil untuk laporan.

        Returns:
            Dict dengan summary utama, daftar truck produktif, dan detail statistik.
        """
        stats = self.dapatkan_statistik()
        produktif = self.dapatkan_truck_produktif()

        return {
            "summary": {
                "total_ritase": stats["total_ritase"],
                "total_truck_aktif": stats["active_trucks"],
                "rata_rata_ritase_per_truck": round(
                    int(stats["total_ritase"]) / max(int(stats["active_trucks"]), 1), 2
                ),
                "total_deteksi_truck_full": stats["total_truck_full_detections"],
                "efficiency_ratio": round(
                    int(stats["total_ritase"]) / max(int(stats["total_truck_full_detections"]), 1), 4
                ),
                "prevented_false_multiple": stats["prevented_false_multiple"],
                "false_multiple_prevention_rate": round(
                    int(stats["prevented_false_multiple"])
                    / max(int(stats["total_truck_full_detections"]), 1),
                    4,
                ),
            },
            "truck_produktif": produktif,
            "detail_statistik": stats,
        }

    def get_siklus_aktif(self) -> Dict[str, Any]:
        """Informasi ringkas tentang siklus aktif saat ini."""
        return {
            "nomor_siklus": self.nomor_siklus,
            "has_ritase": self.siklus_berjalan_punya_ritase,
            "active_trucks": len([t for t in self.data_truk.values() if int(t["ritase_count"]) > 0]),
            "total_ritase": self.total_ritase,
            "total_cycle_completed": self.total_siklus_selesai,
        }

    def dapatkan_status_siklus_aktif(self) -> Dict[str, object]:
        """Status rinci siklus berjalan (untuk monitoring real-time)."""
        return {
            "current_cycle_number": self.nomor_siklus,
            "current_cycle_has_ritase": self.siklus_berjalan_punya_ritase,
            "best_ritase_in_current_cycle": self.ritase_terbaik_di_siklus,
            "cycle_candidates": self.kandidat_siklus,
            "active_detections": {
                tracker_id: len(detections)
                for tracker_id, detections in self.siklus_aktif.items()
                if len(detections) > 0
            },
            "prevented_false_multiple_total": self.total_pencegahan_ganda,
            "cycles_completed": self.total_siklus_selesai,
        }

    def get_current_cycle_info(self) -> Dict[str, object]:
        """Informasi ringkas siklus aktif: status ritase dan kandidat."""
        return {
            "has_ritase": self.siklus_berjalan_punya_ritase,
            "best_ritase": self.ritase_terbaik_di_siklus,
            "candidates_count": len(self.kandidat_siklus),
            "candidates": self.kandidat_siklus,
        }

    def reset_counters(self) -> None:
        """
        Reset counter ritase saat memulai siklus baru.
        """
        self.total_ritase = 0
        for tracker_id in self.data_truk:
            self.data_truk[tracker_id]["ritase_count"] = 0
            self.data_truk[tracker_id]["best_full"] = None

        logger.info("Counter ritase direset untuk siklus baru")