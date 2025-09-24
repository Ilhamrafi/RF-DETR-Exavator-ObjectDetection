# frontend/view/results_view.py

"""Halaman Streamlit untuk meninjau hasil deteksi.

Fokus fungsi halaman:
- Menampilkan hasil per video (individual) lengkap dengan pratinjau dan ringkasan.
- Menampilkan rekap gabungan (master) berupa tabel dan visualisasi.
- Menyediakan tombol unduh (Excel/CSV) dan opsi reset data gabungan.
"""

import streamlit as st
import os
import sys
from typing import Dict, Optional
import pandas as pd
import cv2
import plotly.express as px

# Menambahkan root project ke sys.path agar impor modul lintas-folder berfungsi.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.file_manager import FileManager
from frontend.components.video_display import display_video_with_controls
from frontend.components.data_visualisasi import plot_passing_ritase_time_series, display_excel_summary, plot_passing_per_cycle

file_manager = FileManager()

def show_results_page():
    """Merender halaman 'Hasil Deteksi Excavator' dengan dua mode: Individual & Gabungan."""
    
    st.header("Hasil Deteksi Excavator")
    
    # Mode tampilan: hasil individual per video, atau gabungan seluruh hasil.
    view_mode = st.radio(
        "Mode Tampilan",
        ["Hasil Individual", "Semua Hasil (Gabungan)"],
        horizontal=True
    )
    
    if view_mode == "Hasil Individual":
        show_individual_results()
    else:
        show_combined_results()

def show_individual_results():
    """Menampilkan hasil deteksi untuk satu video yang dipilih pengguna."""
    # Ambil daftar artefak hasil (video/CSV/Excel) yang tersedia.
    available_results = file_manager.list_output_results()
    
    if not available_results:
        st.info("Belum ada hasil deteksi. Silakan jalankan deteksi terlebih dahulu di menu 'Deteksi Video'.")
        return
    
    # Jika ada hasil terbaru pada sesi ini, jadikan default.
    default_idx = 0
    if "latest_result" in st.session_state:
        latest_path = st.session_state.latest_result["video"]
        for i, res in enumerate(available_results):
            if res["video"] == latest_path and os.path.exists(latest_path):
                default_idx = i
                break
    
    # Opsi hasil ditampilkan sebagai nama file + timestamp.
    result_options = [
        f"{os.path.basename(res['video'])} ({res['timestamp']})"
        for res in available_results
    ]
    
    selected_idx = st.selectbox(
        "Pilih Hasil Deteksi", 
        range(len(result_options)), 
        format_func=lambda x: result_options[x],
        index=default_idx
    )
    
    if selected_idx is not None:
        selected_result = available_results[selected_idx]
        
        # Validasi ketersediaan file sebelum render.
        if not os.path.exists(selected_result["video"]):
            st.error(f"File video tidak ditemukan: {selected_result['video']}")
            return
            
        display_detection_result(selected_result)

def show_combined_results():
    """Menampilkan rekap gabungan (master) dalam bentuk tabel dan grafik agregat."""
    
    master_path = file_manager.get_master_report_path()
    
    if not master_path:
        st.info("Belum ada data gabungan. Jalankan analisis terlebih dahulu.")
        return
    
    try:
        # Muat data gabungan dari Excel master.
        df_master = pd.read_excel(master_path)
        
        # Tabel data mentah gabungan.
        st.subheader("Data Tracking Gabungan")
        st.dataframe(df_master, use_container_width=True)
        
        # Visualisasi agregat per video.
        st.subheader("Visualisasi")
        
        if "Source Video" in df_master.columns:
            try:
                # Ringkas per ID (mengambil nilai maksimum per siklus).
                summary_by_video = df_master.groupby(["Source Video", "ID"]).agg({
                    "Ritase": "max",
                    "Passing": "max"
                }).reset_index()
                
                # Agregasi per video (jumlah siklus, total ritase & passing).
                video_summary = summary_by_video.groupby("Source Video").agg({
                    "ID": "nunique",
                    "Ritase": "sum",
                    "Passing": "sum"
                }).reset_index()
                
                video_summary.columns = ["Video", "Jumlah Siklus", "Total Ritase", "Total Passing"]
                
                st.subheader("Ringkasan Per Video")
                st.dataframe(video_summary, use_container_width=True)
                
                # Grafik perbandingan ritase vs passing antar video.
                chart_data = video_summary.melt(
                    id_vars=["Video"],
                    value_vars=["Total Ritase", "Total Passing"],
                    var_name="Metrik",
                    value_name="Nilai"
                )
                fig = px.bar(
                    chart_data,
                    x="Video",
                    y="Nilai",
                    color="Metrik",
                    barmode="group",
                    title="Perbandingan Ritase dan Passing Antar Video"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Analisis tambahan: passing per siklus (jika kolom tersedia).
                if "Siklus" in df_master.columns:
                    st.subheader("Analisis Per Siklus")
                    siklus_summary = df_master.groupby(["Source Video", "Siklus"])["Passing"].max().reset_index()
                    fig_siklus = px.bar(
                        siklus_summary,
                        x="Siklus",
                        y="Passing",
                        color="Source Video",
                        title="Passing per Siklus (Semua Video)",
                        labels={"Passing": "Jumlah Passing", "Siklus": "Nomor Siklus"}
                    )
                    st.plotly_chart(fig_siklus, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Gagal membuat visualisasi: {str(e)}")
        
        # Tombol unduh Excel master (safe read).
        with open(master_path, "rb") as file:
            excel_bytes = file.read()
            st.download_button(
                label="Unduh Laporan Master Excel",
                data=excel_bytes,
                file_name="master_tracking_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Opsi reset master report.
        if st.button("Reset Data Gabungan", type="secondary"):
            if file_manager.reset_master_report():
                st.success("Data gabungan berhasil direset")
                st.experimental_rerun()
            else:
                st.error("Gagal mereset data gabungan")
            
    except Exception as e:
        st.error(f"Gagal memuat data gabungan: {str(e)}")

def display_detection_result(result: Dict[str, Optional[str]]):
    """
    Menampilkan hasil deteksi untuk satu video: tabel, grafik, dan unduhan.

    Args:
        result: Kamus path file hasil (video, CSV, dan Excel).
    """
    # Periksa ketersediaan tiap artefak hasil.
    video_exists = result["video"] and os.path.exists(result["video"])
    excel_exists = result["excel"] and os.path.exists(result["excel"])
    csv_exists = result["csv"] and os.path.exists(result["csv"])
    
    # Dua tab: ringkasan di Excel dan detail statistik.
    excel_tab, detail_tab = st.tabs(["Laporan Excel", "Detail Statistik"])
    
    with excel_tab:
        if excel_exists:
            st.subheader("Laporan Excel")
            try:
                # Tampilkan sheet "Tracking" sebagai tabel.
                df_tracking = pd.read_excel(result["excel"], sheet_name="Tracking")
                st.subheader("Data Tracking")
                st.dataframe(df_tracking, use_container_width=True)
                
                # Visualisasi maksimum Ritase/Passing per ID (jika data tersedia).
                if not df_tracking.empty:
                    try:
                        id_stats = df_tracking.groupby("ID")[["Ritase", "Passing"]].max().reset_index()
                        st.subheader("Visualisasi")
                        chart_data = id_stats.set_index("ID")
                        st.bar_chart(chart_data)
                        
                        # Visualisasi passing per siklus (jika kolom tersedia).
                        if "Siklus" in df_tracking.columns:
                            st.subheader("Passing per Siklus")
                            plot_passing_per_cycle(df_tracking)
                            
                    except Exception as chart_e:
                        st.warning(f"Gagal membuat visualisasi: {str(chart_e)}")
                
                # Tombol unduh Excel (safe read).
                try:
                    with open(result["excel"], "rb") as file:
                        excel_bytes = file.read()
                        st.download_button(
                            label="Unduh Laporan Excel",
                            data=excel_bytes,
                            file_name=os.path.basename(result["excel"]),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error saat menyiapkan tombol unduh Excel: {str(e)}")
            except Exception as e:
                st.error(f"Gagal memuat data Excel: {str(e)}")
        else:
            st.error("Laporan Excel tidak ditemukan")
    
    with detail_tab:
        st.subheader("Detail Statistik dan Visualisasi")
        
        # Analisis lanjutan menggunakan sheet "Detail Events".
        if excel_exists:
            try:
                df_detail = pd.read_excel(result["excel"], sheet_name="Detail Events")
                df_tracking = pd.read_excel(result["excel"], sheet_name="Tracking")
                
                # Analisis per siklus (opsional bila kolom tersedia).
                st.subheader("Analisis Per Siklus")
                if "Siklus" in df_detail.columns:
                    siklus_passing = df_detail[df_detail["Tipe"] == "Passing"].groupby("Siklus").size().reset_index(name="Count")
                    fig = px.bar(
                        siklus_passing,
                        x="Siklus",
                        y="Count",
                        title="Jumlah Passing per Siklus",
                        labels={"Count": "Jumlah Passing", "Siklus": "Nomor Siklus"},
                        color="Count",
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Jumlah event per tipe.
                st.subheader("Jumlah Event Berdasarkan Tipe")
                type_counts = df_detail["Tipe"].value_counts()
                st.bar_chart(type_counts)
                
                # Perbandingan passing vs ritase (jika tersedia).
                passing_events = df_detail[df_detail["Tipe"] == "Passing"].to_dict('records')
                ritase_events = df_detail[df_detail["Tipe"] == "Ritase"].to_dict('records')
                if len(passing_events) > 0 or len(ritase_events) > 0:
                    st.subheader("Perbandingan Passing dan Ritase")
                    try:
                        plot_passing_ritase_time_series(passing_events, ritase_events)
                    except Exception as e:
                        st.error(f"Gagal membuat grafik: {str(e)}")
                
                # Opsi menampilkan data mentah.
                if st.checkbox("Tampilkan Data Mentah"):
                    st.dataframe(df_detail)
                    
            except Exception as e:
                st.error(f"Gagal memproses data Excel untuk visualisasi: {str(e)}")
        else:
            st.info("File Excel tidak tersedia untuk analisis detail.")
        
        # Tombol unduh CSV (jika ada).
        if csv_exists:
            try:
                with open(result["csv"], "rb") as file:
                    csv_bytes = file.read()
                    st.download_button(
                        label="Unduh Data Tracking CSV",
                        data=csv_bytes,
                        file_name=os.path.basename(result["csv"]),
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error saat menyiapkan tombol unduh CSV: {str(e)}")
