# RF-DETR-Exavator-ObjectDetection

---

Sistem deteksi objek exavator untuk menghitung siklus ritase dan passing secara otomatis pada video, berbasis model RF-DETR.

## Struktur Folder

```
📁 backend/         # Logika deteksi, pelaporan, dan perhitungan
📁 data/
│   ├── 📁 input/       # Video input
│   └── 📁 output/      # Hasil deteksi, laporan, dan video hasil
📁 frontend/        # Aplikasi Streamlit (UI)
│   ├── 📁 components/  # Komponen visualisasi dan display video
│   └── 📁 view/        # Halaman tampilan utama
📁 models/          # File model hasil training (.pth/.pt)
📁 utils/           # Utilitas (file/video manager)
📄 requirements.txt # Daftar dependensi Python
📄 classes.json     # Daftar kelas objek deteksi
```

## Instalasi

1. Clone repository
2. Install dependensi Python:
   ```bash
   pip install -r requirements.txt
   ```
3. Pastikan file model (`models/best_model.pth`) tersedia. Jika belum ada, lakukan training model terlebih dahulu sesuai kebutuhan proyek.

## Cara Menjalankan

1. Jalankan backend dan frontend (Streamlit):
   ```bash
   cd frontend
   streamlit run app.py
   ```
2. Ikuti instruksi untuk melakukan deteksi dan melihat hasilnya

## Output

- Hasil deteksi video: `data/output/*.mp4`
- Laporan tracking: `data/output/*_results_tracking.csv`
- Laporan summary: `data/output/*_results_summary.xlsx`
